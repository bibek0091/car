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
    def __init__(self):
        self.last_steer = 0.0
        
    def pure_pursuit(self, target_x_px, lookahead_px, lane_width_px, wheelbase_m=0.23, lane_width_m=0.35):
        ppm = lane_width_px / lane_width_m
        if ppm <= 0: ppm = 1.0
        
        dx = target_x_px - 320.0
        dy = max(float(lookahead_px), 1.0)
        
        ld = math.hypot(dx, dy)
        alpha = math.atan2(dx, dy)
        wb_px = wheelbase_m * ppm
        
        # Pure pursuit formula
        steer_rad = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer_rad)

    def compute(self, perc_res, pose, waypoints, nav_state, traffic_state, base_speed, map_curvature=0.0, velocity_ms=0.0):
        """Compute steering and speed commands."""
        # 1. Target Fusion (Vision + Map)
        t_vis = 320.0 + perc_res.lateral_error_px
        t_map = 320.0
        ppm = perc_res.lane_width_px / 0.35
        
        la_px = max(150.0, min(500.0, velocity_ms * ppm * 1.5))
        if map_curvature > 0.002:
            la_px *= 0.8  # Look closer to react tighter on known curves

        # Project first valid map waypoint > 0.2m ahead
        px, py, pyaw = pose
        map_wp_valid = False
        for wp in waypoints:
            dx = wp[0] - px
            dy = wp[1] - py
            lx = dx*math.cos(-pyaw) - dy*math.sin(-pyaw)
            ly = dx*math.sin(-pyaw) + dy*math.cos(-pyaw)
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

        # DividerGuard (Emergency boundary check)
        if perc_res.sl is not None:
            lx = np.polyval(perc_res.sl, 400)
            if (t_vis - lx) < 110: # Too close to left
                target_x += 40
                
        if perc_res.sr is not None:
            rx = np.polyval(perc_res.sr, 400)
            if (rx - t_vis) < 70: # Too close to right
                target_x -= 40

        # 2. Steering Computation & Smoothing
        raw_steer = self.pure_pursuit(target_x, la_px, perc_res.lane_width_px)
        
        # Anti-jitter & swing limit
        blend = 0.10 if abs(raw_steer - self.last_steer) > 12.0 else 0.40
        steer = self.last_steer + blend * (raw_steer - self.last_steer)
        
        # Rate limit
        steer = np.clip(steer, self.last_steer - 15.0, self.last_steer + 15.0)
        steer = np.clip(steer, -45.0, 45.0)
        self.last_steer = steer
        
        # 3. Dynamic Speed Rules & Scaling
        speed = base_speed
        
        if traffic_state == "SYS_STOP" or nav_state == "CALIBRATING":
            speed = 0.0
        elif traffic_state == "SYS_SLOW":
            speed *= 0.5

        # Pre-emptive Map Curvature slow down
        if map_curvature > 0.0030:
            speed *= 0.40
        elif map_curvature > 0.0015:
            speed *= 0.65
            
        # Rules based on steering / curvature / path
        abs_steer = abs(steer)
        if abs_steer > 25.0:
            speed *= 0.6
        elif abs_steer > 12.0:
            speed *= 0.8
        elif abs_steer < 8.0 and perc_res.anchor.startswith("DUAL"):
            speed *= 1.15
        
        speed = max(0.0, min(100.0, speed))

        return ControlOutput(
            steer_angle_deg=steer,
            speed_pwm=speed,
            target_x=target_x,
            anchor=anchor,
            lookahead_px=la_px
        )
