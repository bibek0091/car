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

class Controller:
    """
    Camera-only lane controller with PID lateral correction and adaptive speed.

    Improvements over v1:
    - Integral term (Ki=0.002): eliminates steady-state lane offset on long straights.
    - Derivative term (Kd=0.15): damps oscillation on curves.
    - DividerGuard tightened to 80 px.
    - DEAD_RECKONING: cuts lookahead to 120 px and speed to 0.3× base.
    - DUAL + high confidence: allows 1.2× base speed.
    """

    Ki = 0.002    # integral gain  — eliminates steady bias
    Kd = 0.15     # derivative gain — damps oscillation

    def __init__(self):
        self.last_steer   = 0.0
        self._err_integral = 0.0   # accumulated lateral error (px·s)
        self._last_err    = 0.0    # for derivative
        self._dt          = 0.033  # default dt for PID (updated each frame)

    def pure_pursuit(self, target_x_px, lookahead_px, lane_width_px,
                     wheelbase_m=0.23, lane_width_m=0.35):
        ppm = lane_width_px / lane_width_m
        if ppm <= 0: ppm = 1.0

        dx = target_x_px - 320.0
        dy = max(float(lookahead_px), 1.0)

        ld = math.hypot(dx, dy)
        alpha = math.atan2(dx, dy)
        wb_px = wheelbase_m * ppm

        steer_rad = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer_rad)

    def compute(self, perc_res, pose, waypoints, nav_state,
                traffic_state, base_speed, map_curvature=0.0,
                velocity_ms=0.0, dt=0.033):
        """Compute steering and speed commands."""
        self._dt = max(dt, 0.001)

        # 1. Target Fusion (Vision + Map)
        t_vis = 320.0 + perc_res.lateral_error_px
        t_map = 320.0
        ppm = perc_res.lane_width_px / 0.35

        # Adaptive lookahead
        la_px = 320.0 if velocity_ms < 0.15 else max(150.0, min(500.0, velocity_ms * ppm * 1.5))
        if map_curvature > 0.002:
            la_px *= 0.8  # Look closer on known curves

        # Dead-reckoning: cut lookahead drastically
        if perc_res.anchor == "DEAD_RECKONING":
            la_px = 120.0

        # Project first valid map waypoint > 0.2m ahead
        px, py, pyaw = pose
        map_wp_valid = False
        for wp in waypoints:
            dx = wp[0] - px
            dy = wp[1] - py
            lx = dx * math.cos(-pyaw) - dy * math.sin(-pyaw)
            ly = dx * math.sin(-pyaw) + dy * math.cos(-pyaw)
            if ly > 0.2:
                t_map = 320.0 + lx * ppm
                la_px = ly * ppm
                map_wp_valid = True
                break

        anchor = perc_res.anchor

        if perc_res.confidence >= 0.6:
            if map_wp_valid:
                target_x = 0.75 * t_vis + 0.25 * t_map
                anchor += "+MAP"
            else:
                target_x = t_vis
        else:
            if map_wp_valid:
                target_x = t_map
                anchor = "MAP_TAKEOVER"
            else:
                target_x = 320.0
                anchor = "HOLD"

        # Traffic lane shifting
        if traffic_state == "SYS_LANE_CHANGE_LEFT":
            target_x -= 0.4 * perc_res.lane_width_px
            anchor += "+LANE_SHIFT"

        # DividerGuard: tightened to 80 px
        if perc_res.sl is not None:
            lx_v = np.polyval(perc_res.sl, 400)
            if (t_vis - lx_v) < 80:   # too close to left line
                target_x += 40

        if perc_res.sr is not None:
            rx_v = np.polyval(perc_res.sr, 400)
            if (rx_v - t_vis) < 80:   # too close to right line
                target_x -= 40

        # 2. PID lateral error terms
        lateral_err_px = target_x - 320.0

        # Integral — wind-up clamp at ±200 px·s
        self._err_integral += lateral_err_px * self._dt
        self._err_integral = max(-200.0, min(200.0, self._err_integral))

        # Derivative
        err_deriv = (lateral_err_px - self._last_err) / self._dt if self._dt > 0 else 0.0
        self._last_err = lateral_err_px

        # PID correction added to target_x
        pid_correction = self.Ki * self._err_integral + self.Kd * err_deriv
        target_x += pid_correction

        # Reset integrator when near centre or lane is lost
        if abs(lateral_err_px) < 5.0 or perc_res.anchor == "DEAD_RECKONING":
            self._err_integral *= 0.90

        # 3. Pure Pursuit steering + smoothing
        raw_steer = self.pure_pursuit(target_x, la_px, perc_res.lane_width_px)

        blend = 0.10 if abs(raw_steer - self.last_steer) > 12.0 else 0.40
        steer = self.last_steer + blend * (raw_steer - self.last_steer)

        steer = np.clip(steer, self.last_steer - 15.0, self.last_steer + 15.0)
        steer = np.clip(steer, -45.0, 45.0)
        self.last_steer = steer

        # 4. Dynamic Speed Rules
        speed = base_speed

        if traffic_state == "SYS_STOP" or nav_state == "CALIBRATING":
            speed = 0.0
        elif traffic_state == "SYS_SLOW":
            speed *= 0.5

        # Pre-emptive map curvature slowdown
        if map_curvature > 0.0030:
            speed *= 0.40
        elif map_curvature > 0.0015:
            speed *= 0.65

        # Steering-based speed rules
        abs_steer = abs(steer)
        if abs_steer > 25.0:
            speed *= 0.6
        elif abs_steer > 12.0:
            speed *= 0.8
        elif abs_steer < 8.0 and perc_res.anchor.startswith("DUAL"):
            # High-confidence dual-lane lock-on: allow slight speed bonus
            if perc_res.confidence > 0.8:
                speed *= 1.20
            else:
                speed *= 1.15

        # Dead-reckoning: severe speed cut — car crawls to recover lane
        if perc_res.anchor == "DEAD_RECKONING":
            speed *= 0.30

        speed = max(0.0, min(100.0, speed))

        return ControlOutput(
            steer_angle_deg=steer,
            speed_pwm=speed,
            target_x=target_x,
            anchor=anchor,
            lookahead_px=la_px
        )
