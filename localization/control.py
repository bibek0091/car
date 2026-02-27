"""
control.py — BFMC Controller (Stanley + Map Feed-Forward & Smooth Braking)
==========================================================================
Upgrade history:
  v1  Pure Pursuit + PID (replaced)
  v2  Stanley reactive controller
  v3  Stanley + map curvature feed-forward term (L * kappa_map)
      + smooth distance-to-curve braking profile
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
    steer_ff_deg:    float = 0.0   # Feed-forward contribution (map curvature)
    steer_react_deg: float = 0.0   # Reactive contribution (Stanley visual)


# ═══════════════════════════════════════════════════════════════════════════════
class StanleyController:
    """
    Full kinematic Stanley controller with map-curvature feed-forward:

        δ(t) = θ_e(t) + atan2(k·e(t), v(t)+ks) + atan(L·κ_map)

    where:
        θ_e      = lane tangent / heading error (rad, from perception)
        e        = signed cross-track error (metres, positive = car is right of line)
        v        = forward speed (m/s)
        k        = cross-track gain
        ks       = softening constant (prevents atan singularity at v≈0)
        L        = wheelbase (m)
        κ_map    = map path curvature ahead (1/m, from localizer)

    The feed-forward term atan(L·κ_map) pre-steers into the curve so the
    reactive terms only need to correct residual error, drastically reducing
    the phase lag that causes overshoot on sharp bends.
    """

    def __init__(self, k: float = 1.2, ks: float = 0.2, wheelbase_m: float = 0.23):
        self.k  = k
        self.ks = ks
        self.L  = wheelbase_m

    def compute(self, target_x_px: float, heading_rad: float,
                velocity_ms: float, lane_width_px: float,
                map_curvature: float = 0.0):
        """Returns (total_deg, reactive_deg, ff_deg) tuple for telemetry."""
        ppm  = max(lane_width_px, 50) / 0.35    # pixels per metre
        ce_m = (320.0 - target_x_px) / ppm      # cross-track error (metres)

        # F-04: velocity-scaled cross-track gain to prevent startup oscillation.
        # k ramps from 0 → full over 0–0.25 m/s so large CTE at v≈0 doesn't jerk.
        k_eff = self.k * min(1.0, velocity_ms / 0.25)

        # Reactive Stanley term (with velocity-scaled gain)
        reactive_rad = heading_rad + math.atan2(k_eff * ce_m, velocity_ms + self.ks)

        # Predictive map feed-forward term  (atan(L·κ) = Ackermann relationship)
        feed_forward_rad = math.atan(self.L * map_curvature)

        total_deg    = math.degrees(reactive_rad + feed_forward_rad)
        reactive_deg = math.degrees(reactive_rad)
        ff_deg       = math.degrees(feed_forward_rad)
        return total_deg, reactive_deg, ff_deg


# ═══════════════════════════════════════════════════════════════════════════════
class DividerGuard:
    """
    Repulsion force-field around lane boundaries.

    Right-lane driving convention:
      left_fit  = sl  = centre dashed line  (divider — car MUST stay right of it)
      right_fit = sr  = outer solid edge     (wall — car must not hit it)

    DIVIDER_SAFE_PX is set higher than EDGE_SAFE_PX because wandering over the
    centre line into oncoming traffic is worse than clipping the outer edge.
    """

    DIVIDER_SAFE_PX = 130   # raised from 110: stronger push away from centre divider
    EDGE_SAFE_PX    =  70
    GAIN            = 0.35
    MAX_CORR        = 25.0
    DEADBAND_PX     =  2

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        correction, speed_scale, triggered = 0.0, 1.0, False
        div_corr = edge_corr = 0.0

        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err      = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min((self.GAIN * 3.0) * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.2, 1.0 - err / 60.0))
                triggered   = True

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
    MAX_STEER_RATE = 20.0

    HIGH_CURV_THRESH = 0.0025
    MED_CURV_THRESH  = 0.0010

    # Smooth curve braking parameters
    BRAKING_DISTANCE_M = 1.8   # metres before apex to start braking
    MIN_CURVE_SPEED_F  = 0.45  # fraction of base_speed at apex (45%)

    def __init__(self):
        self.prev_steer = 0.0
        self.guard      = DividerGuard()
        self.stanley    = StanleyController(k=1.2, ks=0.2, wheelbase_m=0.23)

    def compute(self, perc_res,
                nav_state:      str   = "NORMAL",
                velocity_ms:    float = 0.0,
                dt:             float = 0.033,
                base_speed:     float = 50.0,
                traffic_mult:   float = 1.0,
                map_curvature:  float = 0.0,
                upcoming_curve: str   = "STRAIGHT",
                curve_dist_m:   float = 99.0) -> ControlOutput:

        curvature = perc_res.curvature

        # ── 1. Stanley Steering (with map feed-forward) ─────────────────────────
        raw_steer, react_steer_deg, ff_steer_deg = self.stanley.compute(
            perc_res.target_x, perc_res.heading_rad,
            velocity_ms, perc_res.lane_width_px,
            map_curvature=map_curvature)

        # ── 2. Hardware Rate Limiting ──────────────────────────────────────────
        rate_delta  = max(-self.MAX_STEER_RATE,
                          min(self.MAX_STEER_RATE, raw_steer - self.prev_steer))
        steer_angle = self.prev_steer + rate_delta
        self.prev_steer = steer_angle

        # ── 3. Divider Guard ────────────────────────────────────────────────
        steer_guarded, guard_spd_mult, _ = self.guard.apply(
            steer_angle, perc_res.sl, perc_res.sr, y_eval=perc_res.y_eval)
        steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_guarded))

        # ── 4. Speed Profiling ──────────────────────────────────────────────
        speed           = float(base_speed)
        min_curve_speed = base_speed * self.MIN_CURVE_SPEED_F

        # 4a. Roundabout override
        if nav_state == "ROUNDABOUT":
            speed = min(speed, base_speed * 0.50)

        # 4b. Smooth distance-to-curve braking (replaces hard curvature step-multipliers)
        # Ramp from base_speed → min_curve_speed linearly as curve approaches.
        if upcoming_curve != "STRAIGHT" and curve_dist_m < self.BRAKING_DISTANCE_M:
            decel_factor = max(0.0, curve_dist_m / self.BRAKING_DISTANCE_M)
            braked_speed = min_curve_speed + (base_speed - min_curve_speed) * decel_factor
            speed = min(speed, braked_speed)
        elif abs(steer_angle) < 5:
            speed = min(speed * 1.15, base_speed * 1.20)   # straight-line boost, capped

        # 4c. Dead-reckoning speed penalty
        if "DEAD_RECKONING" in perc_res.anchor:
            try:
                dr_conf = float(perc_res.anchor.split("_")[2])
            except Exception:
                dr_conf = 0.5
            speed *= (0.4 + 0.4 * dr_conf)

        final_speed = speed * traffic_mult * guard_spd_mult

        # F-10: minimum speed floor — prevents stacked multipliers stalling mid-track.
        # 18 PWM = just above the 12 PWM deadband. Only applies in normal driving.
        MINIMUM_DRIVE_PWM = 18.0
        if nav_state not in ("SYS_STOP", "STOPPED") and final_speed > 0:
            final_speed = max(final_speed, MINIMUM_DRIVE_PWM)

        return ControlOutput(
            steer_angle_deg = steer_angle,
            speed_pwm       = final_speed,
            target_x        = perc_res.target_x,
            anchor          = perc_res.anchor,
            lookahead_px    = 0.0,
            steer_ff_deg    = ff_steer_deg,
            steer_react_deg = react_steer_deg,
        )