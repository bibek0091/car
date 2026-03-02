"""
control.py — BFMC Controller (Stanley + Map Feed-Forward & Smooth Braking)
==========================================================================
Upgrade history:
  v1  Pure Pursuit + PID (replaced)
  v2  Stanley reactive controller
  v3  Stanley + map curvature feed-forward term (L * kappa_map)
      + smooth distance-to-curve braking profile
  v4  BUG-FIXES:
      CTRL-01  heading_rad sign fixed: BEV lane heading (positive = car
               pointing left of lane) must be negated before the Stanley
               formula so that a left-leaning lane → right-steer correction.
      CTRL-02  Feed-forward signed curvature: map_curvature is now a signed
               float (+ve = left curve, -ve = right curve). atan(L·κ) naturally
               produces the right direction.
      CTRL-03  velocity-scaled k ramp extended to 0.40 m/s to reduce startup
               wobble on the physical car (was 0.25 m/s — too abrupt).
      CTRL-04  MINIMUM_DRIVE_PWM guard now also suppressed when behavior
               override is active (traffic_mult < 1.0 means a sign/light is
               holding the car — don't fight it with a speed floor).
      CTRL-05  Rate-limiter correctly clamps when reversing steer direction fast
               (abs clamp was wrong direction). Fixed to always compare with sign.
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

        # CTRL-03: velocity-scaled cross-track gain — ramp extended to 0.40 m/s
        # so the physical car doesn't wobble at low speed.
        k_eff = self.k * min(1.0, velocity_ms / 0.40)

        # CTRL-01: BEV heading_rad convention: positive = car pointing LEFT of
        # lane centre (right lane line appears steeper).  Stanley requires heading
        # error in the road-tangent frame; negate so positive error → right steer.
        heading_corrected = -heading_rad

        # Reactive Stanley term (with velocity-scaled gain)
        reactive_rad = heading_corrected + math.atan2(k_eff * ce_m, velocity_ms + self.ks)

        # CTRL-02: signed feed-forward (atan(L·κ)) — positive κ = left curve →
        # positive (left) feed-forward.  map_curvature must be signed by the caller.
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
    EDGE_SAFE_PX    = 100   # raised from 70: more buffer from right outer edge marking
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
            # Squeezed between both lines: prioritize center divider (stay right)
            correction = div_corr
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
                base_speed:     float = 22.0,   # matches CITY_SPEED_PWM
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
        # Adaptive Slew Rate: allow fast steering at 0 m/s (parking), slow at speed
        adaptive_rate = max(5.0, 25.0 - 15.0 * velocity_ms)
        delta = raw_steer - self.prev_steer
        rate_delta  = max(-adaptive_rate, min(adaptive_rate, delta))
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
            # Gentle straight-line boost — capped at 10% above base (was 20%)
            speed = min(speed * 1.08, base_speed * 1.10)

        # 4c. Lateral Acceleration Limit (Roll Instability Prevention)
        lat_acc = (velocity_ms ** 2) * abs(curvature)
        lat_acc_threshold = 0.5
        if lat_acc > lat_acc_threshold:
            roll_penalty = max(0.4, 1.0 - 1.5 * (lat_acc - lat_acc_threshold))
            speed *= roll_penalty

        # 4d. Dead-reckoning speed penalty
        if "DEAD_RECKONING" in perc_res.anchor:
            try:
                dr_conf = float(perc_res.anchor.split("_")[2])
            except Exception:
                dr_conf = 0.5
            speed *= (0.4 + 0.4 * dr_conf)

        # 4e. Divider-follow speed penalty
        # Right outer edge is lost — car is shadowing the centre divider.
        # 25% speed reduction; recovers next frame sr reappears (anchor → RL_*).
        if perc_res.anchor == "DIVIDER_FOLLOW":
            speed *= 0.75

        final_speed = speed * traffic_mult * guard_spd_mult

        # F-10: minimum speed floor — prevents stacked multipliers stalling mid-track.
        # 16 PWM = just above the 12 PWM deadband. Only applies in normal driving.
        # CTRL-04: suppress floor when traffic_mult < 1.0 (sign/light is actively
        # slowing the car — fighting it with a floor defeats the traffic logic).
        MINIMUM_DRIVE_PWM = 16.0
        if (nav_state not in ("SYS_STOP", "STOPPED")
                and final_speed > 0
                and traffic_mult >= 1.0):
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