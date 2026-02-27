"""
control.py — BFMC Lane-Hold Controller  (FIXED v4 — DEEP UPGRADE)
==================================================================
DEEP UPGRADES in v4:

  CTRL-A  VELOCITY-ADAPTIVE LOOKAHEAD: lookahead is now computed as
          la_px = clamp(velocity_ms * PPM * T_HORIZON, la_min, la_max)
          where T_HORIZON = 0.8 s.  At 0 m/s la = la_min (tight tracking).
          At 0.5 m/s la = ~350 px (smooth anticipation).  This directly
          fixes "steering not enough" — at speed the car was looking 200 px
          ahead (0.17 m) which left no time to react to curves.

  CTRL-B  CURVATURE FEED-FORWARD STEER: a direct feed-forward term
          ff_steer = K_FF * curvature * sign  is added before rate-limiting.
          K_FF is tuned so a lane curvature of 0.0025 adds ~8° of pre-steer.
          This makes the car lean into curves BEFORE lateral error builds up,
          fixing the "always late to react" behaviour.

  CTRL-C  STRONGER INTEGRAL (0.0008 → 0.0018) + FASTER MAX RATE (15 → 22 °/s):
          The old integral was too weak to correct sustained cross-track error
          on curves. Increased gain closes the loop faster. Rate limit raised
          so the controller can actually command the required angle change in
          one or two frames at 30 Hz.

  CTRL-D  CONFIDENCE-ADAPTIVE EMA: EMA alpha now scales with confidence.
          High confidence (both lanes): alpha = 0.55 (responsive).
          Low confidence (one lane):    alpha = 0.35 (smoother).
          Dead-reckoning:               alpha = 0.15 (very smooth hold).
          Old fixed alpha = 0.40 was too slow on clear road and too jittery
          when losing a lane.

  CTRL-E  DYNAMIC SPEED REDUCTION ON CURVE: speed is now also reduced when
          upcoming_curve != STRAIGHT and velocity is above a threshold,
          regardless of whether BEV curvature has been detected yet.
          This pre-slows the car before the BEV even sees the curve,
          complementing the existing curvature-based scaling.

  CTRL-F  HEADING-RATE STEERING CORRECTION: if the localizer's visual_yaw_rate
          is provided and non-trivial (> 0.05 rad/s), a small proportional
          correction K_YAW_RATE * yaw_rate_rps is added to raw_steer.
          This acts as a yaw-rate damper — reducing oscillation and improving
          curve tracking without double-counting (the localizer already
          integrates the heading; this is a derivative-like term only).

Fixes from v3 (all retained):
  CTRL-01  Emergency steer sign corrected
  CTRL-02  DividerGuard EMA synced when not triggered
  CTRL-03  Integral decays on emergency override
  VC-01/02/03/04/05 (boundary guard, integral decay, no VO ff, lookahead min)
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
    """Repulsion forcefield around both lane boundaries. (unchanged from v3)"""

    DIVIDER_SAFE_PX = 130
    EDGE_SAFE_PX    =  90
    GAIN            = 0.55      # slightly raised for faster response
    MAX_CORR        = 45.0
    DEADBAND_PX     =  2

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        div_corr = edge_corr = 0.0
        speed_scale = 1.0
        triggered   = False

        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err      = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min((self.GAIN * 3.0) * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.15, 1.0 - err / 50.0))
                triggered = True

        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.EDGE_SAFE_PX - gap)
                edge_corr = -min(self.GAIN * err, self.MAX_CORR * 0.5)
                speed_scale = min(speed_scale, max(0.30, 1.0 - err / 80.0))
                triggered = True

        if triggered:
            correction = div_corr + edge_corr
            if abs(correction) < self.DEADBAND_PX * self.GAIN:
                correction = 0.0
        else:
            correction = 0.0

        return steer_angle + correction, speed_scale, triggered


# ═══════════════════════════════════════════════════════════════════════════════
class Controller:
    """
    Safety-first lane-hold controller.  Deep-upgraded v4.
    """

    # ── EMA alphas (CTRL-D: confidence-adaptive, these are per-mode values) ──
    STEER_EMA_HIGH_CONF = 0.55   # both lanes visible — responsive
    STEER_EMA_LOW_CONF  = 0.35   # one lane — smoother
    STEER_EMA_DEAD_RECK = 0.15   # dead-reckoning — very smooth
    GUARD_EMA           = 0.30

    MAX_STEER      = 45.0
    MAX_STEER_RATE = 22.0        # CTRL-C: raised from 15 → 22 °/frame

    # Curvature speed scaling (unchanged)
    HIGH_CURV_THRESH = 0.0025
    MED_CURV_THRESH  = 0.0010
    HIGH_CURV_SCALE  = 0.58
    MED_CURV_SCALE   = 0.78
    DUAL_SPEED_SCALE = 1.10

    # Upcoming-curve pre-slow (CTRL-E)
    UPCOMING_CURVE_SCALE = 0.82  # multiply speed when curve predicted
    UPCOMING_CURVE_V_MIN = 0.20  # m/s — only apply if faster than this

    EMRG_BOUNDARY_PX = 110
    EMRG_STEER_DEG   = 38.0

    CONF_SPEED_MIN  = 0.35
    DEAD_RECK_SCALE = 0.18

    PWM_DEADBAND    = 14.0
    MIN_PWM_CITY    = 22.0
    MIN_PWM_HIGHWAY = 38.0

    # CTRL-A: velocity-adaptive lookahead
    T_HORIZON        = 0.80     # seconds of forward look
    LOOKAHEAD_MIN_CITY    = 130  # px — min at standstill city
    LOOKAHEAD_MAX_CITY    = 280  # px — max at top city speed
    LOOKAHEAD_MIN_JUNCTION= 100  # px — junction: look close
    LOOKAHEAD_MAX_JUNCTION= 180
    LOOKAHEAD_MIN_HIGHWAY = 200
    LOOKAHEAD_MAX_HIGHWAY = 450

    # CTRL-B: curvature feed-forward gain
    K_FF_CURV  = 3200.0  # deg per unit curvature; 0.0025 curv → ~8° ff

    # CTRL-F: yaw-rate correction gain
    K_YAW_RATE = 4.5     # deg per rad/s visual yaw rate

    WHEELBASE_M  = 0.23
    LANE_WIDTH_M = 0.35

    def __init__(self):
        self.smooth_steer = 0.0
        self.smooth_guard = 0.0
        self.prev_steer   = 0.0
        self.guard        = DividerGuard()
        self._lateral_integral = 0.0
        self._INTEGRAL_GAIN    = 0.0018   # CTRL-C: raised from 0.0008
        self._INTEGRAL_MAX     = 18.0

    # ── Pure pursuit ──────────────────────────────────────────────────────────
    def _pure_pursuit(self, target_x, lookahead_px, lane_width_px):
        lane_width_px = max(lane_width_px, 50)
        ppm   = lane_width_px / self.LANE_WIDTH_M
        wb_px = self.WHEELBASE_M * ppm
        la_px = max(float(lookahead_px), wb_px * 2.5)   # VC-04 minimum

        dx    = target_x - 320.0
        dy    = la_px
        ld    = math.sqrt(dx * dx + dy * dy)
        alpha = math.atan2(dx, dy)
        steer = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer)

    # ── CTRL-A: velocity-adaptive lookahead ───────────────────────────────────
    def _adaptive_lookahead(self, velocity_ms, lane_width_px, nav_state, zone_mode):
        """Compute lookahead in pixels based on speed and driving context."""
        lane_width_px = max(lane_width_px, 50)
        ppm = lane_width_px / self.LANE_WIDTH_M

        # Metres ahead at T_HORIZON seconds → pixels
        la_m  = velocity_ms * self.T_HORIZON
        la_px = la_m * ppm

        if nav_state.startswith("JUNCTION") or nav_state == "JUNCTION_PROMPT":
            la_px = max(self.LOOKAHEAD_MIN_JUNCTION,
                        min(self.LOOKAHEAD_MAX_JUNCTION, la_px))
        elif zone_mode == "HIGHWAY":
            la_px = max(self.LOOKAHEAD_MIN_HIGHWAY,
                        min(self.LOOKAHEAD_MAX_HIGHWAY, la_px))
        else:
            la_px = max(self.LOOKAHEAD_MIN_CITY,
                        min(self.LOOKAHEAD_MAX_CITY, la_px))

        return la_px

    # ── Main compute ──────────────────────────────────────────────────────────
    def compute(self,
                perc_res,
                nav_state:           str   = "NORMAL",
                traffic_state:       str   = "SYS_GO",
                base_speed:          float = 50.0,
                traffic_mult:        float = 1.0,
                zone_mode:           str   = "CITY",
                parking_state:       str   = "NONE",
                steer_bias:          float = 0.0,
                upcoming_curve:      str   = "STRAIGHT",
                visual_yaw_rate_rps: float = 0.0,
                velocity_ms:         float = 0.0,
                dt:                  float = 0.033,
                ) -> ControlOutput:

        sl        = perc_res.sl
        sr        = perc_res.sr
        error_px  = perc_res.lateral_error_px
        target_x  = 320.0 + error_px
        lw        = perc_res.lane_width_px
        curvature = perc_res.curvature
        confidence= perc_res.confidence
        anchor    = perc_res.anchor

        # CTRL-A: velocity-adaptive lookahead
        la_px = self._adaptive_lookahead(velocity_ms, lw, nav_state, zone_mode)

        # ── Layer 4: Emergency Boundary Override ──────────────────────────────
        emergency_override = abs(error_px) > self.EMRG_BOUNDARY_PX

        if emergency_override:
            # CTRL-01: sign corrected — push toward lane centre
            emergency_steer = math.copysign(self.EMRG_STEER_DEG, -error_px)
            self._lateral_integral *= 0.5   # CTRL-03: prevent windup
            self.smooth_steer = emergency_steer
            self.prev_steer   = emergency_steer
            raw_steer         = emergency_steer
        else:
            # ── Layer 1: Pure Pursuit ─────────────────────────────────────────
            raw_steer = self._pure_pursuit(target_x, la_px, lw)

            # CTRL-B: curvature feed-forward — pre-steer into the curve
            if curvature > 1e-5:
                # Determine sign: if target is left of centre → left curve (+)
                # The polynomial 'a' coefficient sign tells curvature direction:
                # negative 'a' → curve right; positive 'a' → curve left
                if sl is not None:
                    curve_sign = 1.0 if sl[0] > 0 else -1.0
                elif sr is not None:
                    curve_sign = 1.0 if sr[0] > 0 else -1.0
                else:
                    curve_sign = math.copysign(1.0, -error_px)
                ff_steer   = self.K_FF_CURV * curvature * curve_sign
                ff_steer   = max(-20.0, min(20.0, ff_steer))   # cap at ±20°
                raw_steer += ff_steer

            # CTRL-F: yaw-rate heading correction (derivative-like damper)
            if abs(visual_yaw_rate_rps) > 0.05:
                yaw_corr   = self.K_YAW_RATE * visual_yaw_rate_rps
                yaw_corr   = max(-10.0, min(10.0, yaw_corr))
                raw_steer += yaw_corr

            # Map anticipation: lean-in at ALL confidence levels
            _blind_extra = max(0.0, (0.30 - confidence) / 0.30)
            if upcoming_curve == "LEFT":
                raw_steer -= (6.0 + 10.0 * _blind_extra)
            elif upcoming_curve == "RIGHT":
                raw_steer += (6.0 + 10.0 * _blind_extra)

            # ── Integral ──────────────────────────────────────────────────────
            if "DEAD_RECKONING" in anchor:
                self._lateral_integral *= 0.7
            elif abs(error_px) < self.EMRG_BOUNDARY_PX:
                self._lateral_integral += error_px * dt
                self._lateral_integral  = max(-self._INTEGRAL_MAX,
                                              min(self._INTEGRAL_MAX,
                                                  self._lateral_integral))
            else:
                self._lateral_integral *= 0.5

            raw_steer += self._INTEGRAL_GAIN * self._lateral_integral

            # ── Rate limiting (CTRL-C: raised to 22 °/frame) ──────────────────
            delta        = raw_steer - self.prev_steer
            delta        = max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, delta))
            rate_limited = self.prev_steer + delta

            # CTRL-D: confidence-adaptive EMA
            if "DEAD_RECKONING" in anchor:
                ema_alpha = self.STEER_EMA_DEAD_RECK
            elif sl is not None and sr is not None:
                ema_alpha = self.STEER_EMA_HIGH_CONF
            else:
                ema_alpha = self.STEER_EMA_LOW_CONF

            new_steer         = (ema_alpha * rate_limited
                                 + (1.0 - ema_alpha) * self.smooth_steer)
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
            # CTRL-02: sync EMA state when guard not triggered
            self.smooth_guard = raw_steer
            final_steer       = raw_steer

        final_steer += steer_bias
        final_steer  = max(-self.MAX_STEER, min(self.MAX_STEER, final_steer))
        self.prev_steer = final_steer

        # ── Speed ─────────────────────────────────────────────────────────────
        if traffic_state == "SYS_STOP" or parking_state == "WAIT":
            speed = 0.0
            self._lateral_integral = 0.0
        else:
            speed = float(base_speed)

            # BEV curvature scaling
            if curvature > self.HIGH_CURV_THRESH:
                speed *= self.HIGH_CURV_SCALE
            elif curvature > self.MED_CURV_THRESH:
                speed *= self.MED_CURV_SCALE

            # CTRL-E: pre-slow on predicted curve (before BEV sees it)
            if (upcoming_curve != "STRAIGHT"
                    and velocity_ms > self.UPCOMING_CURVE_V_MIN):
                speed *= self.UPCOMING_CURVE_SCALE

            if sl is not None and sr is not None and "DUAL" in anchor:
                speed *= self.DUAL_SPEED_SCALE

            if "DEAD_RECKONING" in anchor:
                speed *= self.DEAD_RECK_SCALE
            else:
                conf_scale = self.CONF_SPEED_MIN + (1.0 - self.CONF_SPEED_MIN) * confidence
                speed *= conf_scale

            speed *= guard_speed_scale

            if emergency_override:
                speed *= 0.40

            speed *= traffic_mult

            if parking_state not in ("NONE", "DONE"):
                speed *= {"SEEK": 0.30, "ENTER": 0.20, "EXIT": 0.28}.get(
                    parking_state, 1.0)

            speed = max(0.0, min(100.0, speed))

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