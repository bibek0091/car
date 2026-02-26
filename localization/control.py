"""
control.py — V3 YOLO Pure Pursuit Controller
=============================================
Exact port of bfmc_pilot_v3_yolo.py steering logic:
  - _pure_pursuit: pure geometric atan2-based formula
  - Steering EMA:  STEER_EMA_SLOW=0.40, STEER_EMA_FAST=0.10
  - DividerGuard:  forcefield repulsion from centre line
  - Speed scaling: curvature-based HI(>0.0025)→0.60×, MED(>0.001)→0.80×
  - IMU yaw rate used for feed-forward heading correction
  - No A* path planning — pure camera lane following
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class ControlOutput:
    steer_angle_deg: float
    speed_pwm: float
    target_x: float
    anchor: str
    lookahead_px: float


class DividerGuard:
    """
    Forcefield repulsion from the centre-lane divider (left line).
    Direct V3 port: violent shove rightward when too close to left line.
    """
    DIVIDER_SAFE_PX = 110   # minimum gap from centre divider
    EDGE_SAFE_PX    =  70   # minimum gap from right edge
    GAIN            = 0.35  # repulsion gain
    MAX_CORR        = 25.0  # max emergency correction degrees
    DEADBAND_PX     =  2    # ignore tiny errors

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        correction, speed_scale, triggered = 0.0, 1.0, False
        div_corr = edge_corr = 0.0

        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err = float(self.DIVIDER_SAFE_PX - gap)
                div_corr  = min((self.GAIN * 3.0) * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.2, 1.0 - err / 60.0))
                triggered = True

        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap = edge_x - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR * 0.4)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 100.0))
                triggered = True

        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered


class Controller:
    """
    V3-exact pure-pursuit steering controller with IMU feed-forward.

    Key parameters (from bfmc_pilot_v3_yolo.py):
      STEER_EMA_SLOW  = 0.40   (60% new signal accepted per frame)
      STEER_EMA_FAST  = 0.10   (90% for emergency/guard saves)
      GUARD_EMA       = 0.30
      MAX_STEER       = 45.0
      MAX_STEER_RATE  = 15.0   degrees/frame
    """

    # ── EMA coefficients (V3 exact) ──────────────────────────────────────────
    STEER_EMA_SLOW = 0.40   # blend factor for NEW signal (0.40 = accept 60% new)
    STEER_EMA_FAST = 0.10   # for emergency saves
    GUARD_EMA      = 0.30   # for guard corrections

    # ── Limits ───────────────────────────────────────────────────────────────
    MAX_STEER      = 45.0
    MAX_STEER_RATE = 15.0   # max deg/frame rate limiting

    # ── Speed curvature thresholds (V3) ─────────────────────────────────────
    HIGH_CURV_THRESH = 0.0025
    MED_CURV_THRESH  = 0.0010
    HIGH_CURV_SCALE  = 0.60
    MED_CURV_SCALE   = 0.80
    DUAL_SPEED_SCALE = 1.15   # bonus when both lines visible

    # ── PWM deadband (applied in main.py AFTER traffic multiplier) ──────────
    PWM_DEADBAND     = 14.0   # reference only — do not guard here
    MIN_PWM_CITY     = 27.0
    MIN_PWM_HIGHWAY  = 41.0

    # ── Base lookahead (pixels) ──────────────────────────────────────────────
    LOOKAHEAD_NORMAL    = 200
    LOOKAHEAD_JUNCTION  = 150
    LOOKAHEAD_HIGHWAY   = 350

    # ── Physical constants ───────────────────────────────────────────────────
    WHEELBASE_M   = 0.23
    LANE_WIDTH_M  = 0.35

    def __init__(self):
        self.smooth_steer = 0.0
        self.smooth_guard = 0.0
        self.prev_steer   = 0.0
        self.guard = DividerGuard()

    # ─────────────────────────────────────────────────────────────────────────
    # Pure Pursuit (V3 exact)
    # ─────────────────────────────────────────────────────────────────────────
    def _pure_pursuit(self, target_x, lookahead_px, lane_width_px):
        """
        Direct V3 formula: steer = atan2(2 * wb_px * sin(alpha), ld)
        target_x    : pixel column in BEV frame
        lookahead_px: forward distance (pixels)
        lane_width_px: pixels per lane width (used as pixels-per-metre scaler)
        """
        lane_width_px = max(lane_width_px, 50)
        ppm   = lane_width_px / self.LANE_WIDTH_M
        dx    = target_x - 320.0
        dy    = max(float(lookahead_px), 1.0)
        ld    = math.sqrt(dx * dx + dy * dy)
        alpha = math.atan2(dx, dy)
        wb_px = self.WHEELBASE_M * ppm
        steer = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer)

    # ─────────────────────────────────────────────────────────────────────────
    # Main compute
    # ─────────────────────────────────────────────────────────────────────────
    def compute(self,
                perc_res,            # PerceptionResult from perception.py
                nav_state: str,      # "NORMAL"|"JUNCTION_LEFT"|"JUNCTION_RIGHT"|"ROUNDABOUT"
                traffic_state: str,  # "SYS_GO"|"SYS_STOP"|"SYS_SLOW"|"SYS_LIMIT"
                base_speed: float,   # raw base PWM (0-100)
                traffic_mult: float, # multiplier from TrafficDecisionEngine
                zone_mode: str = "CITY",     # "CITY"|"HIGHWAY"
                parking_state: str = "NONE", # "NONE"|"SEEK"|"ENTER"|"WAIT"|"EXIT"
                steer_bias: float = 0.0,     # extra degrees from parking FSM
                imu_yaw_rate_rps: float = 0.0,  # IMU gz in rad/s (feed-forward)
                velocity_ms: float = 0.0,   # from hardware_io speed estimate
                dt: float = 0.033,
                ) -> ControlOutput:
        """
        V3-style compute.  Returns ControlOutput.

        Steering pipeline (V3):
          1. pure pursuit →  raw_steer
          2. feed-forward IMU yaw rate nudge
          3. rate limiting ± MAX_STEER_RATE
          4. EMA blend (SLOW for normal, FAST for guard saves)
          5. DividerGuard forcefield
          6. clip to ± MAX_STEER

        Speed pipeline:
          base_speed × curvature_scale × traffic_mult × parking_mult
          + zone floor (MIN_PWM_CITY or MIN_PWM_HIGHWAY)
          - deadband guard NOT applied here (main.py does it after multiplier)
        """
        sl         = perc_res.sl
        sr         = perc_res.sr
        target_x   = 320.0 + perc_res.lateral_error_px  # convert error back to pixel
        lw         = perc_res.lane_width_px
        curvature  = perc_res.curvature
        anchor     = perc_res.anchor

        # ── Lookahead selection ───────────────────────────────────────────────
        if nav_state.startswith("JUNCTION") or nav_state == "JUNCTION_PROMPT":
            la_px = self.LOOKAHEAD_JUNCTION
        elif zone_mode == "HIGHWAY":
            la_px = self.LOOKAHEAD_HIGHWAY
        else:
            la_px = self.LOOKAHEAD_NORMAL

        # ── 1. Pure pursuit ───────────────────────────────────────────────────
        raw_steer = self._pure_pursuit(target_x, la_px, lw)

        # ── 2. IMU feed-forward heading correction ────────────────────────────
        # Convert gz (rad/s) to expected steering degrees per frame
        # A yaw-rate of +0.5 rad/s means car is already turning left;
        # we add a small counter-steer to stabilise.
        imu_ff = math.degrees(imu_yaw_rate_rps) * dt * 0.35
        raw_steer += imu_ff

        # ── 3. Rate limiting ─────────────────────────────────────────────────
        delta = raw_steer - self.prev_steer
        delta = max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, delta))
        rate_limited = self.prev_steer + delta

        # ── 4. EMA blend (V3 exact) ──────────────────────────────────────────
        # smooth_steer holds previous EMA state
        new_steer = (self.STEER_EMA_SLOW * rate_limited
                     + (1.0 - self.STEER_EMA_SLOW) * self.smooth_steer)
        self.smooth_steer = new_steer

        # ── 5. DividerGuard ──────────────────────────────────────────────────
        guarded_steer, guard_speed_scale, guard_triggered = self.guard.apply(
            new_steer, sl, sr, y_eval=440, car_x=320)

        if guard_triggered:
            # Blend guard correction with fast EMA
            self.smooth_guard = (self.GUARD_EMA * guarded_steer
                                 + (1.0 - self.GUARD_EMA) * self.smooth_guard)
            final_steer = self.smooth_guard
        else:
            self.smooth_guard = guarded_steer
            final_steer = new_steer

        # Add parking bias
        final_steer += steer_bias

        # ── 6. Clip and store ─────────────────────────────────────────────────
        final_steer = max(-self.MAX_STEER, min(self.MAX_STEER, final_steer))
        self.prev_steer = final_steer

        # ─────────────────────────────────────────────────────────────────────
        # Speed calculation (V3 curvature + traffic scaling)
        # ─────────────────────────────────────────────────────────────────────
        if traffic_state == "SYS_STOP" or parking_state == "WAIT":
            speed = 0.0
        else:
            speed = base_speed

            # V3 curvature scaling
            if curvature > self.HIGH_CURV_THRESH:
                speed *= self.HIGH_CURV_SCALE
            elif curvature > self.MED_CURV_THRESH:
                speed *= self.MED_CURV_SCALE

            # Bonus for dual lines (stable)
            if sl is not None and sr is not None and anchor.startswith("CENTERED_DUAL"):
                speed *= self.DUAL_SPEED_SCALE

            # Guard speed scaling
            speed *= guard_speed_scale

            # Dead-reckoning penalty
            if "DEAD_RECKONING" in anchor:
                speed *= 0.50

            # Apply traffic multiplier
            speed *= traffic_mult

            # Apply parking multiplier
            if parking_state not in ("NONE", "DONE"):
                park_mult_map = {
                    "SEEK": 0.30, "ENTER": 0.20, "EXIT": 0.28
                }
                speed *= park_mult_map.get(parking_state, 1.0)

            speed = max(0.0, min(100.0, speed))

            # Zone speed floors (only when actually driving)
            if speed > 0.0 and traffic_state == "SYS_GO":
                if zone_mode == "HIGHWAY":
                    speed = max(speed, self.MIN_PWM_HIGHWAY)
                else:
                    speed = max(speed, self.MIN_PWM_CITY)

        # BUG-03: Deadband NOT applied here — main.py applies after traffic mult

        return ControlOutput(
            steer_angle_deg = final_steer,
            speed_pwm       = speed,
            target_x        = target_x,
            anchor          = anchor,
            lookahead_px    = la_px,
        )
