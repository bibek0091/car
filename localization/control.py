"""
control.py — BFMC Controller  (Localization-Based Autonomous Driving)
=======================================================================
Driving Modes (nav_mode):

  BASIC         Standard lane-keeping + map-snap.  City speeds.
  INTERSECTION  A* junction following: slow to 60 %, tight lookahead 150 px.
  AEB           Autonomous Emergency Braking: hard-stop immediately.
                Triggered externally by main.py (pedestrian / stop sign).
  HIGHWAY       High-speed right-lane driving: base_speed×1.4, la≥350 px.

Other features carried over from previous version:
  ZONE SPEED FLOORS, LINE-TYPE-GATED OVERTAKING, PARKING, BUS-LANE GUARD.
  PWM DEADBAND GUARD: speed is never in (0, 14) — motors are silent below ~12.
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


class Controller:
    Ki = 0.002    # integral gain
    Kd = 0.15     # derivative gain

    # ── Speed floor constants (PWM units, 0-100 scale) ────────────────────────
    # Using SPEED_CALIB = 0.014, deadband = 12:
    #   velocity_ms = max(0, (pwm - 12) * 0.014)
    #   For 0.40 m/s: pwm = 12 + 0.40/0.014 ≈ 41
    #   For 0.20 m/s: pwm = 12 + 0.20/0.014 ≈ 26
    MIN_PWM_HIGHWAY = 41.0   # 40 cm/s minimum
    MIN_PWM_CITY    = 27.0   # 20 cm/s minimum
    # PWM_DEADBAND is enforced in main.py AFTER speed_multiplier is applied.
    # Do NOT apply it here — the multiplier can drop speed back below 14
    # AFTER compute() returns, so a guard here would fire too early (BUG-03).
    PWM_DEADBAND    = 14.0   # kept as a reference constant only

    def __init__(self):
        self.last_steer    = 0.0
        self._err_integral = 0.0
        self._last_err     = 0.0
        self._dt           = 0.033

    def pure_pursuit(self, target_x_px, lookahead_px, lane_width_px,
                     wheelbase_m=0.23, lane_width_m=0.35):
        ppm = lane_width_px / lane_width_m
        if ppm <= 0:
            ppm = 1.0
        dx = target_x_px - 320.0
        dy = max(float(lookahead_px), 1.0)
        ld     = math.hypot(dx, dy)
        alpha  = math.atan2(dx, dy)
        wb_px  = wheelbase_m * ppm
        steer_rad = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer_rad)

    def compute(self, perc_res, pose, waypoints, nav_state,
                traffic_state, base_speed, map_curvature=0.0,
                velocity_ms=0.0, dt=0.033,
                zone_mode="CITY",
                line_type="UNKNOWN",
                parking_state="NONE",
                steer_bias=0.0,
                bus_lane=False,
                in_roundabout=False,
                nav_mode="BASIC"):
        """
        Compute steering and speed commands.

        nav_mode : "BASIC" | "INTERSECTION" | "AEB" | "HIGHWAY"
          BASIC        — standard lane-keep, city speed.
          INTERSECTION — slow to 60 %, tight lookahead, full A* guidance.
          AEB          — hard-stop (pedestrian / stop-sign emergency).
          HIGHWAY      — high-speed, right-lane bias, long lookahead.

        Other parameters (unchanged)
        zone_mode      : "CITY" | "HIGHWAY" — governs speed floors and lane bias.
        line_type      : "DASHED"|"CONTINUOUS"|"UNKNOWN" — gates overtaking.
        parking_state  : "NONE"|"SEEK"|"ENTER"|"WAIT"|"EXIT".
        steer_bias     : additional steering degrees from ParkingStateMachine.
        bus_lane       : True → rightward correction to avoid bus lane.
        in_roundabout  : True → tighter lookahead and slower speed.
        """
        self._dt = max(dt, 0.001)

        # ── Effective traffic state: gate lane change on line type ────────────
        eff_traffic_state = traffic_state
        if traffic_state == "SYS_LANE_CHANGE_LEFT" and line_type == "CONTINUOUS":
            # Continuous line: must NOT overtake — demote to slow/tail
            eff_traffic_state = "SYS_SLOW"

        # ── AEB: hard-stop immediately — no steering computed ─────────────────
        if nav_mode == "AEB":
            return ControlOutput(
                steer_angle_deg = 0.0,
                speed_pwm       = 0.0,
                target_x        = 320.0,
                anchor          = "AEB_STOP",
                lookahead_px    = 0.0,
            )

        # ── 1. Target computation (Vision + Map) ──────────────────────────────
        t_vis = 320.0 + perc_res.lateral_error_px
        t_map = 320.0
        ppm   = perc_res.lane_width_px / 0.35

        # Adaptive lookahead — tuned per nav_mode
        if nav_mode == "INTERSECTION":
            la_px = min(150.0, velocity_ms * ppm * 1.0 + 80.0)
        elif nav_mode == "HIGHWAY":
            la_px = max(350.0, min(600.0, velocity_ms * ppm * 2.0))
        else:
            la_px = 320.0 if velocity_ms < 0.15 else max(
                150.0, min(500.0, velocity_ms * ppm * 1.5)
            )
        if map_curvature > 0.002:
            la_px *= 0.8
        if perc_res.anchor == "DEAD_RECKONING":
            la_px = 120.0
        if in_roundabout:
            la_px = min(la_px, 180.0)   # tighter lookahead in roundabouts

        # Project first valid map waypoint > 0.2 m ahead
        px, py, pyaw = pose
        map_wp_valid = False
        for wp in waypoints:
            dx = wp[0] - px
            dy_ = wp[1] - py
            lx = dx * math.cos(-pyaw) - dy_ * math.sin(-pyaw)
            ly = dx * math.sin(-pyaw) + dy_ * math.cos(-pyaw)
            if ly > 0.2:
                t_map       = 320.0 + lx * ppm
                la_px       = ly * ppm
                map_wp_valid = True
                break

        anchor = perc_res.anchor

        if perc_res.confidence >= 0.6:
            target_x = (0.75 * t_vis + 0.25 * t_map) if map_wp_valid else t_vis
            if map_wp_valid:
                anchor += "+MAP"
        else:
            if map_wp_valid:
                target_x = t_map
                anchor   = "MAP_TAKEOVER"
            else:
                target_x = 320.0
                anchor   = "HOLD"

        # ── Right-side driving constant bias ──────────────────────────────────
        # REMOVED: RIGHT_LANE_BIAS_PX — perception.py already centers the lane
        # target correctly for RIGHT-only anchors. A second rightward shift here
        # caused hard weaving by fighting the perception target.

        # ── Highway nav_mode right-lane bias ─────────────────────────────────
        # In HIGHWAY mode apply a 10 % rightward nudge so the car stays in the
        # outer right lane even on sensor-drift frames.
        if nav_mode == "HIGHWAY" or zone_mode == "HIGHWAY":
            target_x += 0.10 * perc_res.lane_width_px
            anchor += "+HW"

        # ── Bus-lane guard (steer right to avoid it) ──────────────────────────
        if bus_lane:
            target_x += 0.30 * perc_res.lane_width_px
            anchor    += "+BUS_AVOID"

        # ── Overtaking lane shift (gated by line type) ────────────────────────
        if eff_traffic_state == "SYS_LANE_CHANGE_LEFT":
            target_x -= 0.40 * perc_res.lane_width_px
            anchor    += "+OVERTAKE"
        elif traffic_state == "SYS_LANE_CHANGE_LEFT" and eff_traffic_state == "SYS_SLOW":
            # Continuous line: stay in lane, no shift
            anchor += "+TAIL"

        # ── DividerGuard (80 px min margin from lane lines) ───────────────────
        if perc_res.sl is not None:
            lx_v = np.polyval(perc_res.sl, 400)
            if (t_vis - lx_v) < 80:
                target_x += 40
        if perc_res.sr is not None:
            rx_v = np.polyval(perc_res.sr, 400)
            if (rx_v - t_vis) < 80:
                target_x -= 40

        # ── 2. PID ───────────────────────────────────────────────────────────
        lateral_err_px = target_x - 320.0

        self._err_integral += lateral_err_px * self._dt
        self._err_integral  = max(-200.0, min(200.0, self._err_integral))

        err_deriv = (
            (lateral_err_px - self._last_err) / self._dt if self._dt > 0 else 0.0
        )
        self._last_err = lateral_err_px

        pid_correction = self.Ki * self._err_integral + self.Kd * err_deriv
        target_x      += pid_correction

        if abs(lateral_err_px) < 5.0 or perc_res.anchor == "DEAD_RECKONING":
            self._err_integral *= 0.90

        # ── 3. Pure Pursuit + smoothing ───────────────────────────────────────
        raw_steer = self.pure_pursuit(target_x, la_px, perc_res.lane_width_px)

        # Add parking steer bias BEFORE smoothing
        raw_steer += steer_bias

        blend = 0.10 if abs(raw_steer - self.last_steer) > 12.0 else 0.40
        steer = self.last_steer + blend * (raw_steer - self.last_steer)
        steer = np.clip(steer, self.last_steer - 15.0, self.last_steer + 15.0)
        steer = np.clip(steer, -45.0, 45.0)
        self.last_steer = steer

        # ── 4. Speed rules ────────────────────────────────────────────────────
        # nav_mode modifies base_speed BEFORE traffic / curvature multipliers
        if nav_mode == "HIGHWAY":
            base_speed = min(100.0, base_speed * 1.40)   # +40 % on highway
        elif nav_mode == "INTERSECTION":
            base_speed = base_speed * 0.60               # -40 % at junctions

        speed = base_speed

        if eff_traffic_state == "SYS_STOP" or nav_state == "CALIBRATING":
            speed = 0.0
        elif eff_traffic_state == "SYS_SLOW":
            speed *= 0.5
        elif eff_traffic_state == "SYS_LIMIT":
            speed *= 0.75

        # Roundabout: slow down for safety
        if in_roundabout:
            speed *= 0.70

        # Map curvature pre-emptive slowdown
        if map_curvature > 0.0030:
            speed *= 0.40
        elif map_curvature > 0.0015:
            speed *= 0.65

        # Steering-based slowdown
        abs_steer = abs(steer)
        if abs_steer > 25.0:
            speed *= 0.6
        elif abs_steer > 12.0:
            speed *= 0.8
        elif abs_steer < 8.0 and perc_res.anchor.startswith("DUAL"):
            speed *= 1.20 if perc_res.confidence > 0.8 else 1.15

        # Dead-reckoning crawl (only in BASIC/INTERSECTION, not HIGHWAY)
        if perc_res.anchor == "DEAD_RECKONING" and nav_mode != "HIGHWAY":
            speed *= 0.30

        speed = max(0.0, min(100.0, speed))

        # ── Zone speed floors (applied only when actively driving) ─────────────
        _freely_driving = (
            eff_traffic_state == "SYS_GO"
            and perc_res.anchor != "DEAD_RECKONING"
            and parking_state in ("NONE", "DONE")
            and speed > 0.0
        )
        if _freely_driving:
            if nav_mode == "HIGHWAY" or zone_mode == "HIGHWAY":
                speed = max(speed, self.MIN_PWM_HIGHWAY)
            else:
                speed = max(speed, self.MIN_PWM_CITY)

        # BUG-03: Deadband guard deliberately NOT applied here.
        # main.py multiplies by t_res.speed_multiplier AFTER this call,
        # which can push speed back below deadband.  The single authoritative
        # guard lives in main.py after the multiplier is applied.

        return ControlOutput(
            steer_angle_deg = steer,
            speed_pwm       = speed,
            target_x        = target_x,
            anchor          = anchor,
            lookahead_px    = la_px,
        )
