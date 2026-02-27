"""
control.py — BFMC Lane-Hold Controller (Safety-First)
======================================================
GOAL: "No matter what happens, the car stays in the lane."

5-Layer Defence:

  Layer 1 — Pure pursuit to target centre (V3 formula, unchanged)
  Layer 2 — IMU feed-forward: counteract measured yaw-rate drift
  Layer 3 — DividerGuard HARD forcefield: violent correction when
             car strays within 130 px of centre divider
  Layer 4 — Emergency boundary clamp: if |lateral_error| > 110 px
             the steer is OVERRIDDEN to max corrective angle
             regardless of what pure-pursuit says
  Layer 5 — Confidence-proportional speed: speed falls linearly
             as lane confidence drops; at DEAD_RECKONING → 18% PWM
             (slow enough to self-correct before leaving the lane)

Steering EMA coefficients (unchanged from V3):
  STEER_EMA_SLOW = 0.40  (accepts 60 % new signal per frame)
  STEER_EMA_FAST = 0.10  (for emergency / guard corrections)
  GUARD_EMA      = 0.30

DividerGuard versus V3 original:
  DIVIDER_SAFE_PX: 110 → 130  (trigger earlier)
  GAIN:            0.35 → 0.50 (43 % stronger repulsion)
  MAX_CORR:        25° → 45°   (full rack authority in emergency)
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class ControlOutput:
    steer_angle_deg: float
    speed_pwm:       float
    target_x:        float
    anchor:          str
    lookahead_px:    float


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 — DividerGuard (UPGRADED)
# ═══════════════════════════════════════════════════════════════════════════
class DividerGuard:
    """
    Repulsion forcefield around both lane boundaries.

    Compared to the V3 original:
     - Trigger zone raised: DIVIDER_SAFE_PX 110→130, EDGE_SAFE_PX 70→90
     - GAIN increased: 0.35→0.50  (43 % harder push)
     - MAX_CORR raised: 25→45° (full steering authority on emergency saves)
     - Speed braking is more aggressive when close to boundary
    """
    DIVIDER_SAFE_PX = 130    # px gap to centre line to trigger
    EDGE_SAFE_PX    =  90    # px gap to outer edge to trigger
    GAIN            = 0.50   # base repulsion gain
    MAX_CORR        = 45.0   # max emergency correction degrees
    DEADBAND_PX     =  2

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        """
        Returns (corrected_steer, speed_scale, triggered).
        speed_scale < 1.0 means we are braking because we're too close
        to a boundary.
        """
        div_corr = edge_corr = 0.0
        speed_scale = 1.0
        triggered   = False

        # ── Left boundary (centre divider) ────────────────────────────────
        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.DIVIDER_SAFE_PX - gap)
                # 3× multiplier: divider is a hard wall
                div_corr  = min((self.GAIN * 3.0) * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.15, 1.0 - err / 50.0))
                triggered = True

        # ── Right boundary (outer edge) ───────────────────────────────────
        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR * 0.5)
                speed_scale = min(speed_scale, max(0.30, 1.0 - err / 80.0))
                triggered = True

        # Resolve: divider repulsion dominates
        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered


# ═══════════════════════════════════════════════════════════════════════════
# Main controller
# ═══════════════════════════════════════════════════════════════════════════
class Controller:
    """
    Safety-first lane-hold controller.

    Steering pipeline (all 5 layers):
      1. Pure pursuit → raw_steer
      2. IMU gz feed-forward nudge
      3. Rate limiting (±15°/frame)
      4. EMA blend (SLOW normal, FAST for guard saves)
      5a. DividerGuard forcefield (Layer 3)
      5b. Emergency boundary override (Layer 4): if |error| > EMRG threshold,
          pure-pursuit is IGNORED and maximum corrective steer is applied.

    Speed pipeline:
      base_speed
        × curvature_scale   (0.60–1.00)
        × confidence_scale  (Layer 5: 0.25–1.00 proportional to lane conf)
        × guard_scale       (DividerGuard braking)
        × dead_reck_scale   (0.18 when both lines lost)
        × traffic_mult
      + zone floor after (MIN_PWM_CITY / MIN_PWM_HIGHWAY)
    """

    # ── EMA (V3 unchanged) ────────────────────────────────────────────────
    STEER_EMA_SLOW = 0.40
    STEER_EMA_FAST = 0.10
    GUARD_EMA      = 0.30

    # ── Limits ────────────────────────────────────────────────────────────
    MAX_STEER      = 45.0
    MAX_STEER_RATE = 15.0

    # ── Curvature thresholds (V3) ─────────────────────────────────────────
    HIGH_CURV_THRESH = 0.0025
    MED_CURV_THRESH  = 0.0010
    HIGH_CURV_SCALE  = 0.60
    MED_CURV_SCALE   = 0.80
    DUAL_SPEED_SCALE = 1.15

    # ── Layer 4: Emergency override thresholds ────────────────────────────
    # If the car's target_x is this many pixels from centre → override steer
    EMRG_BOUNDARY_PX   = 110   # |error| > this → emergency override
    EMRG_STEER_DEG     = 38.0  # steering angle applied in override (toward centre)

    # ── Layer 5: Confidence → speed scaling ──────────────────────────────
    # speed_scale = conf_scale_min + (1 - conf_scale_min) * confidence
    CONF_SPEED_MIN     = 0.35   # 35% PWM at zero confidence (dead-reckoning)
    DEAD_RECK_SCALE    = 0.18   # 18% when BOTH lines lost (very slow)

    # ── PWM references ────────────────────────────────────────────────────
    PWM_DEADBAND    = 14.0
    MIN_PWM_CITY    = 22.0   # reduced from 27 — confidence scaling now floors
    MIN_PWM_HIGHWAY = 38.0

    # ── Lookahead ─────────────────────────────────────────────────────────
    LOOKAHEAD_NORMAL   = 200
    LOOKAHEAD_JUNCTION = 150
    LOOKAHEAD_HIGHWAY  = 350

    WHEELBASE_M  = 0.23
    LANE_WIDTH_M = 0.35

    def __init__(self):
        self.smooth_steer = 0.0
        self.smooth_guard = 0.0
        self.prev_steer   = 0.0
        self.guard        = DividerGuard()
        # Integral term for persistent lateral error
        self._lateral_integral = 0.0
        self._INTEGRAL_GAIN    = 0.0008   # very gentle; resets on large errors
        self._INTEGRAL_MAX     = 15.0     # cap integral wind-up

    # ─────────────────────────────────────────────────────────────────────
    # Pure pursuit (V3 exact)
    # ─────────────────────────────────────────────────────────────────────
    def _pure_pursuit(self, target_x, lookahead_px, lane_width_px):
        lane_width_px = max(lane_width_px, 50)
        ppm   = lane_width_px / self.LANE_WIDTH_M
        dx    = target_x - 320.0
        dy    = max(float(lookahead_px), 1.0)
        ld    = math.sqrt(dx * dx + dy * dy)
        alpha = math.atan2(dx, dy)
        wb_px = self.WHEELBASE_M * ppm
        steer = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer)

    # ─────────────────────────────────────────────────────────────────────
    # Compute
    # ─────────────────────────────────────────────────────────────────────
    def compute(self,
                perc_res,
                nav_state:    str   = "NORMAL",
                traffic_state: str  = "SYS_GO",
                base_speed:   float = 50.0,
                traffic_mult: float = 1.0,
                zone_mode:    str   = "CITY",
                parking_state: str  = "NONE",
                steer_bias:   float = 0.0,
                upcoming_curve: str = "STRAIGHT",
                imu_yaw_rate_rps: float = 0.0,
                velocity_ms:  float = 0.0,
                dt:           float = 0.033,
                ) -> ControlOutput:

        sl        = perc_res.sl
        sr        = perc_res.sr
        error_px  = perc_res.lateral_error_px   # target_x - 320
        target_x  = 320.0 + error_px
        lw        = perc_res.lane_width_px
        curvature = perc_res.curvature
        confidence= perc_res.confidence
        anchor    = perc_res.anchor

        # ── Lookahead ─────────────────────────────────────────────────────
        if nav_state.startswith("JUNCTION") or nav_state == "JUNCTION_PROMPT":
            la_px = self.LOOKAHEAD_JUNCTION
        elif zone_mode == "HIGHWAY":
            la_px = self.LOOKAHEAD_HIGHWAY
        else:
            la_px = self.LOOKAHEAD_NORMAL

        # ─────────────────────────────────────────────────────────────────
        # ★ LAYER 4 — EMERGENCY BOUNDARY OVERRIDE
        # If |error| is dangerously large, skip pure pursuit entirely and
        # apply maximum corrective steer directly toward lane centre.
        # This fires before the EMA so response is instantaneous.
        # ─────────────────────────────────────────────────────────────────
        emergency_override = abs(error_px) > self.EMRG_BOUNDARY_PX

        if emergency_override:
            # Sign: positive error → car too far right → steer left (negative)
            emergency_steer = -math.copysign(self.EMRG_STEER_DEG, error_px)
            # Reset smoothing state to emergency value so EMA doesn't fight it
            self.smooth_steer = emergency_steer
            self.prev_steer   = emergency_steer
            raw_steer = emergency_steer
        else:
            # ── Layer 1: Pure Pursuit + Map Anticipation ──────────────────
            raw_steer = self._pure_pursuit(target_x, la_px, lw)

            # Proactive steering: if camera is blind, steer toward the map curve
            if confidence < 0.30:
                if upcoming_curve == "LEFT":
                    raw_steer -= 15.0  # gentle left pull
                elif upcoming_curve == "RIGHT":
                    raw_steer += 15.0  # gentle right pull

            # ── Layer 2: IMU feed-forward ─────────────────────────────────
            imu_ff    = math.degrees(imu_yaw_rate_rps) * dt * 0.35
            raw_steer += imu_ff

            # ── Integral term for persistent offset ───────────────────────
            # Accumulate only when error is moderate (not during emergencies)
            if abs(error_px) < self.EMRG_BOUNDARY_PX:
                self._lateral_integral += error_px * dt
                self._lateral_integral  = max(-self._INTEGRAL_MAX,
                                              min(self._INTEGRAL_MAX,
                                                  self._lateral_integral))
            else:
                self._lateral_integral *= 0.5   # bleed off during big swings

            raw_steer += self._INTEGRAL_GAIN * self._lateral_integral

            # ── Layer 3a: Rate limiting ───────────────────────────────────
            delta       = raw_steer - self.prev_steer
            delta       = max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, delta))
            rate_limited = self.prev_steer + delta

            # ── Layer 3b: EMA blend ───────────────────────────────────────
            new_steer        = (self.STEER_EMA_SLOW * rate_limited
                                + (1.0 - self.STEER_EMA_SLOW) * self.smooth_steer)
            self.smooth_steer = new_steer
            raw_steer         = new_steer

        # ── Layer 3c: DividerGuard ─────────────────────────────────────────
        guarded_steer, guard_speed_scale, guard_triggered = self.guard.apply(
            raw_steer, sl, sr, y_eval=440, car_x=320)

        if guard_triggered:
            # Fast-blend guard correction into output
            self.smooth_guard = (self.GUARD_EMA * guarded_steer
                                 + (1.0 - self.GUARD_EMA) * self.smooth_guard)
            final_steer = self.smooth_guard
        else:
            self.smooth_guard = guarded_steer
            final_steer = raw_steer

        # Parking bias
        final_steer += steer_bias

        # Clip and store
        final_steer   = max(-self.MAX_STEER, min(self.MAX_STEER, final_steer))
        self.prev_steer = final_steer

        # ─────────────────────────────────────────────────────────────────
        # Speed calculation
        # ─────────────────────────────────────────────────────────────────
        if traffic_state == "SYS_STOP" or parking_state == "WAIT":
            speed = 0.0
            self._lateral_integral = 0.0   # reset integral on stops
        else:
            speed = float(base_speed)

            # Curvature scaling (V3)
            if curvature > self.HIGH_CURV_THRESH:
                speed *= self.HIGH_CURV_SCALE
            elif curvature > self.MED_CURV_THRESH:
                speed *= self.MED_CURV_SCALE

            # Dual-line bonus
            if sl is not None and sr is not None and "DUAL" in anchor:
                speed *= self.DUAL_SPEED_SCALE

            # ── ★ LAYER 5: Confidence-proportional speed ──────────────────
            # speed falls from base all the way to DEAD_RECK_SCALE * base
            # when the car can see neither lane line.
            if "DEAD_RECKONING" in anchor:
                speed *= self.DEAD_RECK_SCALE   # both lines lost → very slow
            else:
                # Blend between CONF_SPEED_MIN and 1.0 based on confidence
                conf_scale = self.CONF_SPEED_MIN + (1.0 - self.CONF_SPEED_MIN) * confidence
                speed *= conf_scale

            # Guard speed braking
            speed *= guard_speed_scale

            # Emergency override: also slow down
            if emergency_override:
                speed *= 0.40

            # Traffic multiplier
            speed *= traffic_mult

            # Parking multiplier
            if parking_state not in ("NONE", "DONE"):
                park_scale = {"SEEK": 0.30, "ENTER": 0.20, "EXIT": 0.28}
                speed *= park_scale.get(parking_state, 1.0)

            speed = max(0.0, min(100.0, speed))

            # Zone speed floors  (only when SYS_GO and speed > 0)
            if speed > 0.0 and traffic_state == "SYS_GO":
                floor = (self.MIN_PWM_HIGHWAY if zone_mode == "HIGHWAY"
                         else self.MIN_PWM_CITY)
                # Floor is lower during dead-reckoning to allow crawl
                if "DEAD_RECKONING" in anchor:
                    floor = min(floor, 18.0)
                speed = max(speed, floor)

        # Note: PWM deadband guard applied in main.py AFTER this returns
        return ControlOutput(
            steer_angle_deg = final_steer,
            speed_pwm       = speed,
            target_x        = target_x,
            anchor          = anchor,
            lookahead_px    = la_px,
        )
