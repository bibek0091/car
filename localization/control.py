"""
control.py — BFMC Controller (PID Lane Centering & Lethal Zone Guard)
=====================================================================
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


class LanePositionController:
    def __init__(self):
        self.target_position = 0.5  # Center of lane (normalized 0-1)
        self.kP = 0.3  # Proportional gain
        self.kD = 0.1  # Derivative gain
        self.last_error = 0.0

    def compute_correction(self, left_fit, right_fit, current_y, lane_width):
        """ Computes a precise steering offset to maintain mathematical center. """
        if left_fit is None or right_fit is None:
            return 0.0
        
        lx = np.polyval(left_fit, current_y)
        car_x = 320.0 
        current_position = (car_x - lx) / max(lane_width, 1.0)
        
        error = self.target_position - current_position
        d_error = error - self.last_error
        self.last_error = error
        
        # PID output mapped back to pixels
        correction = (self.kP * error + self.kD * d_error) * lane_width
        return correction


class DividerGuard:
    # 80px Lethal Zone: The car MUST NOT ever touch the center divider.
    DIVIDER_SAFE_PX = 110 
    EDGE_SAFE_PX    = 70
    GAIN            = 0.35 
    MAX_CORR        = 25.0 
    DEADBAND_PX     = 2

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        correction, speed_scale, triggered = 0.0, 1.0, False
        div_corr = 0.0
        
        # Left Divider (Center line) - Overwhelming penalty forcefield
        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min((self.GAIN * 3.0) * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.2, 1.0 - err / 60.0))
                triggered   = True

        # Right Edge - Standard Weak Repulsion
        edge_corr = 0.0
        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR * 0.4)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 100.0))
                triggered   = True

        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered


class Controller:
    MAX_STEER      = 45.0
    MAX_STEER_RATE = 15.0 
    STEER_EMA_SLOW = 0.40   
    STEER_EMA_FAST = 0.10   
    GUARD_EMA      = 0.30   

    HIGH_CURV_THRESH = 0.0025 
    MED_CURV_THRESH  = 0.0010

    def __init__(self):
        self.smooth_steer  = 0.0
        self.smooth_guard  = 0.0
        self.prev_steer    = 0.0
        self.guard = DividerGuard()
        self.lane_pos_controller = LanePositionController()

    def _pure_pursuit(self, target_x, look_ahead_px, lane_width_px):
        lane_width_px = max(lane_width_px, 50)
        ppm   = lane_width_px / 0.35 # 0.35m lane width
        dx    = target_x - 320.0
        dy    = max(float(look_ahead_px), 1.0)
        ld    = math.sqrt(dx * dx + dy * dy)
        alpha = math.atan2(dx, dy)
        wb_px = 0.23 * ppm # 0.23m wheelbase
        steer = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer)

    def compute(self, perc_res, nav_state="NORMAL", velocity_ms=0.0, dt=0.033, base_speed=50.0, traffic_mult=1.0) -> ControlOutput:
        
        target_x = perc_res.target_x
        curvature = perc_res.curvature

        # Apply Lane Position PID Correction
        position_correction = self.lane_pos_controller.compute_correction(perc_res.sl, perc_res.sr, perc_res.y_eval, perc_res.lane_width_px)
        if target_x is not None:
            target_x += position_correction

        # Determine Adaptive Lookahead
        look_ahead = 150
        if nav_state == "ROUNDABOUT": eff_la = int(look_ahead * 0.55)
        elif nav_state.startswith("JUNCTION"): eff_la = int(look_ahead * 0.75)
        elif curvature > self.HIGH_CURV_THRESH: eff_la = int(look_ahead * 1.30)
        elif curvature > self.MED_CURV_THRESH: eff_la = int(look_ahead * 1.10)
        else: eff_la = look_ahead
        eff_la = max(60, eff_la)

        # 1. Pure Pursuit Base Steer
        raw_steer = self._pure_pursuit(target_x, eff_la, perc_res.lane_width_px)

        # 2. Adaptive Smoothing (Fast on emergencies, slow on straights)
        steer_delta_abs = abs(raw_steer - self.smooth_steer)
        if curvature > self.HIGH_CURV_THRESH: alpha_adaptive = 0.15
        elif curvature > self.MED_CURV_THRESH: alpha_adaptive = 0.25
        else: alpha_adaptive = self.STEER_EMA_SLOW
            
        alpha = (self.STEER_EMA_FAST if steer_delta_abs > 12.0 else alpha_adaptive)
        self.smooth_steer = alpha * self.smooth_steer + (1.0 - alpha) * raw_steer
        steer_angle = self.smooth_steer

        # Rate Limiting
        rate_delta = max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, steer_angle - self.prev_steer))
        steer_angle = self.prev_steer + rate_delta
        self.prev_steer = steer_angle

        # 3. Apply Lethal Boundary Divider Guard
        raw_steer_guarded, guard_spd_mult, guard_on = self.guard.apply(steer_angle, perc_res.sl, perc_res.sr, y_eval=perc_res.y_eval)

        if "DEAD_RECKONING" in perc_res.anchor: 
            self.smooth_guard, guard_on = 0.0, False
        else:
            guard_delta = raw_steer_guarded - steer_angle
            self.smooth_guard = (self.GUARD_EMA * guard_delta + (1.0 - self.GUARD_EMA) * self.smooth_guard)
            
        steer_angle = steer_angle + self.smooth_guard
        steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_angle))

        # 4. Determine Output Speed
        speed = base_speed
        if nav_state == "ROUNDABOUT": speed *= 0.50
        elif curvature > self.HIGH_CURV_THRESH: speed *= 0.45
        elif curvature > self.MED_CURV_THRESH: speed *= 0.65
        elif abs(steer_angle) < 8: speed *= 1.15 
        elif abs(steer_angle) > 18: speed *= 0.60
        elif abs(steer_angle) > 10: speed *= 0.80

        if "DEAD_RECKONING" in perc_res.anchor:
            try: conf = float(perc_res.anchor.split("_")[2])
            except: conf = 0.5
            speed *= (0.4 + 0.4 * conf)

        final_speed = speed * traffic_mult * guard_spd_mult

        return ControlOutput(
            steer_angle_deg=steer_angle,
            speed_pwm=final_speed,
            target_x=target_x,
            anchor=perc_res.anchor,
            lookahead_px=eff_la
        )