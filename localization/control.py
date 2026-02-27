"""
control.py — BFMC Lane-Hold Controller  (FIXED v3)
===================================================
Fixes applied (v2 → v3):
  CTRL-01  Emergency steer sign corrected: was -copysign → steered INTO boundary.
           Now +copysign(EMRG_STEER_DEG, -error_px) steers toward lane centre.
  CTRL-02  DividerGuard EMA stale state: smooth_guard now synced to raw_steer
           when guard is not triggered, preventing jump on next trigger.
  CTRL-03  Integral decays 50% on emergency override entry (prevents windup
           that caused kick when emergency condition cleared).
  CTRL-04  VC-05 (dead-reckoning scale) now documented in fix list.

Fixes carried forward from v2:
  VC-01  DividerGuard edge_corr sign fixed (pushes toward centre)
  VC-02  Integral decays during DEAD_RECKONING
  VC-03  VO feed-forward removed (double-counted yaw)
  VC-04  Pure-pursuit minimum lookahead enforced (la_px >= wb_px * 2.5)
  VC-05  Dead-reckoning speed scale applied before floor

5-Layer Defence:
  Layer 1  Pure pursuit to target centre
  Layer 2  [REMOVED VO feed-forward — was double-counting]
  Layer 3  DividerGuard HARD forcefield
  Layer 4  Emergency boundary clamp (sign-corrected)
  Layer 5  Confidence-proportional speed
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


# ═══════════════════════════════════════════════════════════════════════════════
class DividerGuard:
    """
    Repulsion forcefield around both lane boundaries.

    FIX VC-01: edge_corr is now NEGATIVE (pushes car LEFT away from right edge).
    Resolution: correction = div_corr + edge_corr  (both signed correctly).
    """
    DIVIDER_SAFE_PX = 130
    EDGE_SAFE_PX    =  90
    GAIN            = 0.50
    MAX_CORR        = 45.0
    DEADBAND_PX     =  2

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        """Returns (corrected_steer, speed_scale, triggered)."""
        div_corr  = 0.0
        edge_corr = 0.0
        speed_scale = 1.0
        triggered   = False

        # ── Left boundary (centre divider) — pushes RIGHT (+) ─────────────────
        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err      = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min((self.GAIN * 3.0) * err, self.MAX_CORR)   # positive
                speed_scale = min(speed_scale, max(0.15, 1.0 - err / 50.0))
                triggered = True

        # ── Right boundary (outer edge) — pushes LEFT (−) ─────────────────────
        # FIX VC-01: edge_corr is NEGATIVE (steers left to avoid right edge)
        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.EDGE_SAFE_PX - gap)
                edge_corr = -min(self.GAIN * err, self.MAX_CORR * 0.5)  # NEGATIVE
                speed_scale = min(speed_scale, max(0.30, 1.0 - err / 80.0))
                triggered = True

        # FIX VC-01: ADD corrections — both push toward lane centre
        if triggered:
            correction = div_corr + edge_corr
            # Deadband floor: only apply if total correction is meaningful
            if abs(correction) < self.DEADBAND_PX * self.GAIN:
                correction = 0.0
        else:
            correction = 0.0

        return steer_angle + correction, speed_scale, triggered


# ═══════════════════════════════════════════════════════════════════════════════
class Controller:
    """
    Safety-first lane-hold controller.  All 5 layers active.

    Changes vs. original:
      - VO feed-forward removed (VC-03)
      - Integral decays in DEAD_RECKONING (VC-02)
      - Pure-pursuit enforces min lookahead (VC-04)
      - DividerGuard sign corrected (VC-01)
    """

    STEER_EMA_SLOW = 0.40
    STEER_EMA_FAST = 0.10
    GUARD_EMA      = 0.30

    MAX_STEER      = 45.0
    MAX_STEER_RATE = 15.0

    HIGH_CURV_THRESH = 0.0025
    MED_CURV_THRESH  = 0.0010
    HIGH_CURV_SCALE  = 0.60
    MED_CURV_SCALE   = 0.80
    DUAL_SPEED_SCALE = 1.15

    EMRG_BOUNDARY_PX = 110
    EMRG_STEER_DEG   = 38.0

    CONF_SPEED_MIN  = 0.35
    DEAD_RECK_SCALE = 0.18

    PWM_DEADBAND    = 14.0
    MIN_PWM_CITY    = 22.0
    MIN_PWM_HIGHWAY = 38.0

    # FIX VC-04: lookahead values kept; _pure_pursuit enforces minimum vs wb_px
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
        self._lateral_integral = 0.0
        self._INTEGRAL_GAIN    = 0.0008
        self._INTEGRAL_MAX     = 15.0

    def _pure_pursuit(self, target_x, lookahead_px, lane_width_px):
        """
        FIX VC-04: Enforce minimum lookahead = 2.5 × wb_px so the formula
        never degenerates (ld must be >> wb_px for atan2 to be geometrically valid).
        """
        lane_width_px = max(lane_width_px, 50)
        ppm   = lane_width_px / self.LANE_WIDTH_M
        wb_px = self.WHEELBASE_M * ppm

        # FIX VC-04: minimum lookahead
        la_px = max(float(lookahead_px), wb_px * 2.5)

        dx    = target_x - 320.0
        dy    = la_px
        ld    = math.sqrt(dx * dx + dy * dy)
        alpha = math.atan2(dx, dy)
        steer = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer)

    def compute(self,
                perc_res,
                nav_state:     str   = "NORMAL",
                traffic_state: str   = "SYS_GO",
                base_speed:    float = 50.0,
                traffic_mult:  float = 1.0,
                zone_mode:     str   = "CITY",
                parking_state: str   = "NONE",
                steer_bias:    float = 0.0,
                upcoming_curve: str  = "STRAIGHT",
                visual_yaw_rate_rps: float = 0.0,   # kept in signature, not used
                velocity_ms:   float = 0.0,
                dt:            float = 0.033,
                ) -> ControlOutput:

        sl        = perc_res.sl
        sr        = perc_res.sr
        error_px  = perc_res.lateral_error_px
        target_x  = 320.0 + error_px
        lw        = perc_res.lane_width_px
        curvature = perc_res.curvature
        confidence= perc_res.confidence
        anchor    = perc_res.anchor

        # ── Lookahead ─────────────────────────────────────────────────────────
        if nav_state.startswith("JUNCTION") or nav_state == "JUNCTION_PROMPT":
            la_px = self.LOOKAHEAD_JUNCTION
        elif zone_mode == "HIGHWAY":
            la_px = self.LOOKAHEAD_HIGHWAY
        else:
            la_px = self.LOOKAHEAD_NORMAL

        # ── Layer 4: Emergency Boundary Override ──────────────────────────────
        emergency_override = abs(error_px) > self.EMRG_BOUNDARY_PX

        if emergency_override:
            # FIX CTRL-01: sign was inverted — negative copysign steered INTO boundary.
            # error_px > 0 means car is RIGHT of centre → steer LEFT (negative angle).
            # copysign(EMRG_STEER_DEG, error_px) gives positive → negate → steer left. ✓
            # But original code did -copysign → steered further right. Fixed to +copysign.
            emergency_steer = math.copysign(self.EMRG_STEER_DEG, -error_px)
            # Also reset integral to prevent windup during multi-frame emergency
            # FIX CTRL-03: integral reset on emergency entry
            self._lateral_integral *= 0.5
            self.smooth_steer = emergency_steer
            self.prev_steer   = emergency_steer
            raw_steer = emergency_steer
        else:
            # ── Layer 1: Pure Pursuit ─────────────────────────────────────────
            raw_steer = self._pure_pursuit(target_x, la_px, lw)

            # Map anticipation — proactive lean-in at ALL confidence levels.
            # 5° constant lean-in keeps the car on the inside of upcoming curves
            # while the lane tracker still dominates.  Adds up to 10° extra as
            # confidence drops toward zero (blind driving).
            _blind_extra = max(0.0, (0.30 - confidence) / 0.30)
            if upcoming_curve == "LEFT":
                raw_steer -= (5.0 + 10.0 * _blind_extra)
            elif upcoming_curve == "RIGHT":
                raw_steer += (5.0 + 10.0 * _blind_extra)

            # FIX VC-03: VO feed-forward removed — it double-counted yaw
            # correction already applied by the localizer.

            # ── Integral ──────────────────────────────────────────────────────
            # FIX VC-02: decay integral during DEAD_RECKONING to prevent lurch
            if "DEAD_RECKONING" in anchor:
                self._lateral_integral *= 0.7    # exponential decay to zero
            elif abs(error_px) < self.EMRG_BOUNDARY_PX:
                self._lateral_integral += error_px * dt
                self._lateral_integral  = max(-self._INTEGRAL_MAX,
                                              min(self._INTEGRAL_MAX,
                                                  self._lateral_integral))
            else:
                self._lateral_integral *= 0.5

            raw_steer += self._INTEGRAL_GAIN * self._lateral_integral

            # ── Rate limiting ─────────────────────────────────────────────────
            delta        = raw_steer - self.prev_steer
            delta        = max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, delta))
            rate_limited = self.prev_steer + delta

            # ── EMA blend ─────────────────────────────────────────────────────
            new_steer = (self.STEER_EMA_SLOW * rate_limited
                         + (1.0 - self.STEER_EMA_SLOW) * self.smooth_steer)
            self.smooth_steer = new_steer
            raw_steer         = new_steer

        # ── Layer 3: DividerGuard ─────────────────────────────────────────────
        guarded_steer, guard_speed_scale, guard_triggered = self.guard.apply(
            raw_steer, sl, sr, y_eval=440, car_x=320)

        if guard_triggered:
            self.smooth_guard = (self.GUARD_EMA * guarded_steer
                                 + (1.0 - self.GUARD_EMA) * self.smooth_guard)
            final_steer = self.smooth_guard
        else:
            # FIX CTRL-02: always sync smooth_guard to the current unguarded steer
            # so the EMA has no stale value to jump from on the next trigger.
            self.smooth_guard = raw_steer
            final_steer = raw_steer

        # Parking bias
        final_steer += steer_bias

        final_steer   = max(-self.MAX_STEER, min(self.MAX_STEER, final_steer))
        self.prev_steer = final_steer

        # ── Speed ─────────────────────────────────────────────────────────────
        if traffic_state == "SYS_STOP" or parking_state == "WAIT":
            speed = 0.0
            self._lateral_integral = 0.0
        else:
            speed = float(base_speed)

            if curvature > self.HIGH_CURV_THRESH:
                speed *= self.HIGH_CURV_SCALE
            elif curvature > self.MED_CURV_THRESH:
                speed *= self.MED_CURV_SCALE

            if sl is not None and sr is not None and "DUAL" in anchor:
                speed *= self.DUAL_SPEED_SCALE

            # FIX VC-05: dead-reckoning scale applied FIRST, floor skipped
            if "DEAD_RECKONING" in anchor:
                speed *= self.DEAD_RECK_SCALE
                # Do NOT apply zone floor — it would defeat the crawl intent
            else:
                conf_scale = self.CONF_SPEED_MIN + (1.0 - self.CONF_SPEED_MIN) * confidence
                speed *= conf_scale

            speed *= guard_speed_scale

            if emergency_override:
                speed *= 0.40

            speed *= traffic_mult

            if parking_state not in ("NONE", "DONE"):
                park_scale = {"SEEK": 0.30, "ENTER": 0.20, "EXIT": 0.28}
                speed *= park_scale.get(parking_state, 1.0)

            speed = max(0.0, min(100.0, speed))

            # Zone speed floors — only when NOT dead-reckoning
            if speed > 0.0 and traffic_state == "SYS_GO" and "DEAD_RECKONING" not in anchor:
                floor = (self.MIN_PWM_HIGHWAY if zone_mode == "HIGHWAY"
                         else self.MIN_PWM_CITY)
                speed = max(speed, floor)

        return ControlOutput(
            steer_angle_deg = final_steer,
            speed_pwm       = speed,
            target_x        = target_x,
            anchor          = anchor,
            lookahead_px    = la_px,
        )