"""
control.py — BFMC Controller (Stanley Algorithm & Divider Guard)
================================================================
Stanley controller replaces Pure Pursuit + PID. It fuses heading
error and cross-track error in one clean formula, eliminating the
oscillation caused by the old two-loop architecture.
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
class StanleyController:
    """
    Stanley method — the standard for path-tracking robots:

        δ = heading_err + atan2(k * e_ct, v + ks)

    where:
        heading_err  = lane tangent angle  (rad, from perception)
        e_ct         = signed cross-track error  (metres, +right)
        v            = forward speed  (m/s)
        k            = cross-track gain  (higher → tighter tracking)
        ks           = softening constant  (prevents atan singularity at v=0)
    """

    def __init__(self, k: float = 1.2, ks: float = 0.2):
        self.k  = k   # cross-track gain
        self.ks = ks  # softening constant

    def compute(self, target_x_px: float, heading_rad: float,
                velocity_ms: float, lane_width_px: float) -> float:
        """Returns steering angle in degrees."""
        ppm   = max(lane_width_px, 50) / 0.35   # pixels per metre (0.35 m lane)
        ce_px = 320.0 - target_x_px              # positive = target is left of centre
        ce_m  = ce_px / ppm                      # cross-track error in metres

        delta_rad = heading_rad + math.atan2(self.k * ce_m, velocity_ms + self.ks)
        return math.degrees(delta_rad)


# ═══════════════════════════════════════════════════════════════════════════════
class DividerGuard:
    """Repulsion force-field around lane boundaries — unchanged from previous version."""

    DIVIDER_SAFE_PX = 110
    EDGE_SAFE_PX    =  70
    GAIN            = 0.35
    MAX_CORR        = 25.0
    DEADBAND_PX     =  2

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        correction, speed_scale, triggered = 0.0, 1.0, False
        div_corr = edge_corr = 0.0

        # Left divider (centre line) — stronger repulsion
        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err      = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min((self.GAIN * 3.0) * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.2, 1.0 - err / 60.0))
                triggered   = True

        # Right edge — weaker repulsion
        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR * 0.4)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 100.0))
                triggered   = True

        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered


# ═══════════════════════════════════════════════════════════════════════════════
class Controller:

    MAX_STEER      = 45.0
    MAX_STEER_RATE = 20.0   # °/frame — hard rate limit

    HIGH_CURV_THRESH = 0.0025
    MED_CURV_THRESH  = 0.0010

    def __init__(self):
        self.prev_steer = 0.0
        self.guard      = DividerGuard()
        self.stanley    = StanleyController(k=1.2, ks=0.2)

    def compute(self, perc_res, nav_state: str = "NORMAL",
                velocity_ms: float = 0.0, dt: float = 0.033,
                base_speed: float = 50.0,
                traffic_mult: float = 1.0) -> ControlOutput:

        curvature = perc_res.curvature

        # ── 1. Stanley Steering ───────────────────────────────────────────────
        raw_steer = self.stanley.compute(
            perc_res.target_x, perc_res.heading_rad,
            velocity_ms, perc_res.lane_width_px)

        # ── 2. Hardware Rate Limiting ─────────────────────────────────────────
        rate_delta  = max(-self.MAX_STEER_RATE,
                          min(self.MAX_STEER_RATE, raw_steer - self.prev_steer))
        steer_angle = self.prev_steer + rate_delta
        self.prev_steer = steer_angle

        # ── 3. Divider Guard ──────────────────────────────────────────────────
        steer_guarded, guard_spd_mult, _ = self.guard.apply(
            steer_angle, perc_res.sl, perc_res.sr, y_eval=perc_res.y_eval)
        steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_guarded))

        # ── 4. Trajectory-Aware Speed Profiling ───────────────────────────────
        speed = float(base_speed)

        if nav_state == "ROUNDABOUT":
            speed *= 0.50
        elif curvature > self.HIGH_CURV_THRESH:
            speed *= 0.45
        elif curvature > self.MED_CURV_THRESH:
            speed *= 0.65
        elif abs(steer_angle) < 5:
            speed *= 1.15
        elif abs(steer_angle) > 15:
            speed *= 0.70

        if "DEAD_RECKONING" in perc_res.anchor:
            try:
                dr_conf = float(perc_res.anchor.split("_")[2])
            except Exception:
                dr_conf = 0.5
            speed *= (0.4 + 0.4 * dr_conf)

        final_speed = speed * traffic_mult * guard_spd_mult

        return ControlOutput(
            steer_angle_deg = steer_angle,
            speed_pwm       = final_speed,
            target_x        = perc_res.target_x,
            anchor          = perc_res.anchor,
            lookahead_px    = 0.0,   # Stanley doesn't use a fixed lookahead distance
        )