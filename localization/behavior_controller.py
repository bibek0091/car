"""
behavior_controller.py — BFMC Priority-Based Reactive Controller  (TRACK-AWARE v2)
===================================================================================
Upgrades from v1 → v2 based on BFMC track geography analysis:

  BC-01  HIGHWAY PROTOCOL: When zone_mode == "HIGHWAY" the car enforces the
         outermost-lane rule by applying a persistent right-lane steer bias
         (HIGHWAY_LANE_BIAS_DEG) on top of the Stanley output. Speed is raised
         to HIGHWAY_SPEED_PWM. This fires at priority 3 (Mission), below P0-P2
         safety rules but above normal city driving.

  BC-02  SPEED OVAL MODE: "SPEED_OVAL" zone detected by map_planner → sustained
         high speed with no pedestrian or sign distractions expected. Still
         respects P0 emergency and P1 red lights.

  BC-03  ROUNDABOUT CCW PROTOCOL improved:
         - Entry: hard left-bias steer on approach (was a fixed −8° regardless
           of approach angle). Now uses CCW_ENTRY_STEER_DEG if map action == "CCW".
         - Inside: speed capped at ROUNDABOUT_SPEED_PWM (was 65% of base).
         - Exit: roundabout_active clears only when map zone changes away from
           ROUNDABOUT and no "roundabout" sign has been seen for 4 s.

  BC-04  BUS LANE HARD WALL: If localizer confirms the car is inside the bus-lane
         bounding box (planner.is_in_bus_lane()), apply a stronger correction
         (BUS_LANE_HARD_CORRECTION_DEG) and also reduce speed — not just a
         steer nudge. The old code only acted on t_res labels; now it acts on
         actual map position.

  BC-05  CROSSWALK PROTOCOL: When map_planner.get_crosswalk_approach() is True,
         speed is reduced to CROSSWALK_SPEED_PWM even before YOLO detects a
         pedestrian, because the car must decelerate proactively at known zebra
         crossings (rule from track description). Pedestrian detection then
         escalates to full stop (P0).

  BC-06  START / PIT AREA: zone == "START" applies a hard speed cap
         (START_AREA_SPEED_PWM) to match the restricted-zone rules near the
         start line and parking area.

  BC-07  Zone transition debouncing: a 10-frame hysteresis prevents jitter when
         crossing zone boundaries at low speed or near GPS/localizer noise.

  BC-08  BehaviorOutput extended with `zone_speed_ms` so the caller can set
         base_speed dynamically from the zone without hard-coding PWM values.

Priority Hierarchy (unchanged):
  0  EMERGENCY — pedestrian blocking road
  1  MANDATORY — RED light · STOP sign (3 s non-blocking halt)
  2  LEGAL     — No-Entry · Bus-Lane hard wall
  3  MISSION   — Roundabout CCW · Parking FSM · Highway/Oval mode · Overtake
  4  NORMAL    — default right-lane city driving
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Output contract
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BehaviorOutput:
    """Single-frame output of BehaviorController.compute()."""
    speed_pwm     : float        # 0 = stopped, positive = forward
    steer_deg     : float        # negative = left, positive = right
    priority      : int          # which priority level fired (0-4)
    state         : str          # human-readable state label
    reason        : str          # why this state was chosen
    zone_mode     : str = "CITY" # "CITY" | "HIGHWAY" | "SPEED_OVAL" | "ROUNDABOUT" | etc.
    maneuver      : str = "NONE" # "NONE" | "OVERTAKE" | "PARKING" | "ROUNDABOUT"
    zone_speed_ms : float = 0.0  # BC-08: zone target speed in m/s (0 = not set)


# ══════════════════════════════════════════════════════════════════════════════
# Overtake FSM  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

class OvertakeStateMachine:
    """
    Dashed-line obstacle overtake: IDLE → CHANGE_LEFT → PASS → CHANGE_RIGHT → IDLE
    Timing is open-loop. Steer biases are additive on top of Stanley output.
    """
    CHANGE_DURATION = 1.5   # s per lane-change segment
    PASS_DURATION   = 2.0   # s to pass the obstacle
    STEER_BIAS_DEG  = 12.0  # extra steer during lane-change
    SPEED_MULT      = 0.70  # slow slightly during maneuver

    def __init__(self):
        self.state = "IDLE"
        self._ts   = 0.0

    @property
    def active(self):
        return self.state != "IDLE"

    def trigger(self, now: float):
        if self.state == "IDLE":
            self.state = "CHANGE_LEFT"
            self._ts   = now
            log.info("OVERTAKE: lane-change left started")

    def update(self, now: float, base_steer: float, base_speed: float):
        """Returns (steer_deg, speed_pwm, maneuver_label)."""
        if self.state == "IDLE":
            return base_steer, base_speed, "NONE"

        elapsed = now - self._ts

        if self.state == "CHANGE_LEFT":
            if elapsed > self.CHANGE_DURATION:
                self.state, self._ts = "PASS", now
            return base_steer - self.STEER_BIAS_DEG, base_speed * self.SPEED_MULT, "OVERTAKE"

        if self.state == "PASS":
            if elapsed > self.PASS_DURATION:
                self.state, self._ts = "CHANGE_RIGHT", now
            return base_steer, base_speed * self.SPEED_MULT, "OVERTAKE"

        if self.state == "CHANGE_RIGHT":
            if elapsed > self.CHANGE_DURATION:
                self.state = "IDLE"
                log.info("OVERTAKE: complete — back in right lane")
            return base_steer + self.STEER_BIAS_DEG, base_speed * self.SPEED_MULT, "OVERTAKE"

        return base_steer, base_speed, "NONE"


# ══════════════════════════════════════════════════════════════════════════════
# Parking FSM  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

class ParkingSequenceFSM:
    """
    Parallel-parking: IDLE → SEEK → ENTER → WAIT → EXIT → DONE → IDLE
    No time.sleep() used anywhere.
    """
    SEEK_TIMEOUT   = 6.0
    SLOW_SPEED     = 0.30
    ENTER_DURATION = 2.0
    ENTER_STEER    = 22.0
    WAIT_DURATION  = 3.0
    EXIT_DURATION  = 2.5
    EXIT_STEER     = -18.0

    def __init__(self):
        self.state = "IDLE"
        self._ts   = 0.0

    @property
    def active(self):
        return self.state not in ("IDLE", "DONE")

    def trigger(self, now: float):
        if self.state == "IDLE":
            self.state = "SEEK"
            self._ts   = now
            log.info("PARKING: seek started")

    def reset(self):
        self.state = "IDLE"

    def update(self, now: float, base_speed: float, spot_clear: bool = True):
        """Returns (speed_mult, steer_bias_deg, state_str)."""
        if self.state in ("IDLE", "DONE"):
            return 1.0, 0.0, "NONE"

        elapsed = now - self._ts

        if self.state == "SEEK":
            if spot_clear or elapsed > self.SEEK_TIMEOUT:
                self.state, self._ts = "ENTER", now
                log.info("PARKING: entering spot")
            return self.SLOW_SPEED, 0.0, "SEEK"

        if self.state == "ENTER":
            if elapsed > self.ENTER_DURATION:
                self.state, self._ts = "WAIT", now
                log.info("PARKING: in spot — waiting %.1fs", self.WAIT_DURATION)
            return 0.20, self.ENTER_STEER, "ENTER"

        if self.state == "WAIT":
            if elapsed > self.WAIT_DURATION:
                self.state, self._ts = "EXIT", now
                log.info("PARKING: exiting spot")
            return 0.0, 0.0, "WAIT"

        if self.state == "EXIT":
            if elapsed > self.EXIT_DURATION:
                self.state = "DONE"
                log.info("PARKING: done — returning to lane-follow")
            return 0.28, self.EXIT_STEER, "EXIT"

        return 1.0, 0.0, "IDLE"


# ══════════════════════════════════════════════════════════════════════════════
# Main BehaviorController
# ══════════════════════════════════════════════════════════════════════════════

class BehaviorController:
    """
    Priority-Based Reactive Controller — BFMC track-aware version.

    New required argument in compute():
        planner  : PathPlanner  (from map_planner.py)
        map_action: str         ("STRAIGHT"|"LEFT"|"RIGHT"|"CCW"|"HIGHWAY_MERGE")
        cursor    : int         (current path cursor index)
        path      : list        (current planned path)

    All other arguments unchanged from v1.
    """

    # ── Priority constants ────────────────────────────────────────────────────
    PRI_EMERGENCY = 0
    PRI_MANDATORY = 1
    PRI_LEGAL     = 2
    PRI_MISSION   = 3
    PRI_NORMAL    = 4

    # ── Speed constants (PWM units) ───────────────────────────────────────────
    # User request: car is still overspeeding. Doing an EXTREME SLOWdown.
    # Note: DEADBAND is 12.0. So PWM=22 means 10 effective drive units.
    # Highway = city × 1.20 (user: +20% on highway only).
    CITY_SPEED_PWM          = 22.0   # city base speed         (was 28)
    HIGHWAY_SPEED_PWM       = 26.0   # highway = city × 1.20   (was 34)
    SPEED_OVAL_PWM          = 26.0   # speed oval (same as highway) (was 34)
    ROUNDABOUT_SPEED_PWM    = 16.0   # inside roundabout        (was 20)
    PARKING_SPEED_PWM       = 14.0   # parking maneuver zone    (was 16)
    START_AREA_SPEED_PWM    = 16.0   # restricted start/pit cap (was 20)
    APPROACH_SPEED_PWM      = 14.0   # sign-approach decel floor(was 16)
    CROSSWALK_SPEED_PWM     = 14.0   # proactive crosswalk slow (was 16)
    SLOW_SPEED_PWM          = 14.0   # generic slow             (was 16)
    MIN_SPEED_PWM           = 14.0   # stall-prevention floor   (was 18)



    # ── Sign approach deceleration ────────────────────────────────────────────
    APPROACH_DECEL_M   = 2.5   # start slowing
    APPROACH_FULL_M    = 0.8   # reach floor speed

    # ── Mandatory STOP ────────────────────────────────────────────────────────
    STOP_SIGN_HOLD_S   = 3.0
    STOP_SIGN_COOLDOWN = 5.0

    # ── Bus-lane corrections ──────────────────────────────────────────────────
    BUS_LANE_STEER_CORRECTION      = -8.0   # soft nudge from sign detection
    BUS_LANE_HARD_CORRECTION_DEG   = -18.0  # BC-04: hard correction from map position
    BUS_LANE_SPEED_MULT            =  0.70  # slow down when inside bus lane

    # ── Highway protocol (BC-01) ──────────────────────────────────────────────
    HIGHWAY_LANE_BIAS_DEG   = 4.0    # extra rightward steer on highway (outermost lane)
    HIGHWAY_MERGE_SLOW_MULT = 0.75   # slow on merge from city → highway

    # ── Roundabout (BC-03) ────────────────────────────────────────────────────
    CCW_ENTRY_STEER_DEG     = -10.0  # left bias entering roundabout
    CCW_INSIDE_SPEED_MULT   =  0.55  # speed inside roundabout
    ROUNDABOUT_SIGN_TIMEOUT =  4.0   # seconds without sign before exit

    # ── Zone hysteresis (BC-07) ───────────────────────────────────────────────
    ZONE_DEBOUNCE_FRAMES = 10

    def __init__(self):
        self._zone_mode         : str   = "CITY"
        self._zone_candidate    : str   = "CITY"
        self._zone_debounce_cnt : int   = 0

        self._stop_timer        : float = 0.0
        self._stop_cooldown     : float = 0.0
        self._priority_until    : float = 0.0

        self._roundabout_active : bool  = False
        self._roundabout_sign_ts: float = 0.0
        self._no_entry_active   : bool  = False

        self.overtake_fsm  = OvertakeStateMachine()
        self.parking_fsm   = ParkingSequenceFSM()
        # Fix-3: highway outer-lane one-shot FSM
        self._hw_outer_state : str   = "IDLE"  # IDLE | STEER_RIGHT | HOLD
        self._hw_outer_ts    : float = 0.0
        self._hw_outer_last_zone: str = "CITY"  # detect CITY->HIGHWAY transition
        self._last_state   = "NORMAL"

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self,
                perc_res,
                t_res,
                dt: float,
                base_steer: float = 0.0,
                planner=None,
                map_action: str = "STRAIGHT",
                cursor: int = 0,
                path: list = None,
                loc_x: float = 0.0,
                loc_y: float = 0.0) -> BehaviorOutput:
        """
        Evaluate all priority layers and return the highest-priority command.

        Parameters
        ----------
        perc_res   : PerceptionResult from perception.py
        t_res      : TrafficResult from traffic_module.py
        dt         : elapsed seconds since last call
        base_steer : steering angle (deg) from StanleyController
        planner    : PathPlanner instance (enables map-aware behaviors)
        map_action : result of planner.get_next_action() this frame
        cursor     : current path cursor index
        path       : current A* path (list of node IDs)
        loc_x      : localizer X position in map metres (Fix-4: bus-lane check)
        loc_y      : localizer Y position in map metres (Fix-4: bus-lane check)
        """
        now = time.time()

        # ── Zone update (BC-07 debounced) ─────────────────────────────────────
        self._update_zone(t_res, planner, now)

        # ── Base speed for this zone ──────────────────────────────────────────
        base_speed = self._zone_base_speed()

        # ── Sign-approach deceleration (applied before priority checks) ───────
        base_speed *= self._sign_approach_mult(t_res)

        # ── Crosswalk proactive slow (BC-05) ─────────────────────────────────
        # Map-known crosswalks slow the car regardless of YOLO detection.
        if planner and path and planner.get_crosswalk_approach(path, cursor):
            base_speed = min(base_speed, self.CROSSWALK_SPEED_PWM)

        # ── P0: EMERGENCY ─────────────────────────────────────────────────────
        em_out = self._check_emergency(t_res, base_steer)
        if em_out:
            return em_out

        # ── P1: MANDATORY ────────────────────────────────────────────────────
        mand_out = self._check_mandatory(t_res, now, base_steer)
        if mand_out:
            return mand_out

        # ── P2: LEGAL ────────────────────────────────────────────────────────
        legal_out = self._check_legal(t_res, perc_res, base_steer,
                                      planner, now, loc_x, loc_y)
        if legal_out:
            return legal_out

        # ── P3: MISSION ──────────────────────────────────────────────────────
        mission_out = self._check_mission(t_res, perc_res, now,
                                          base_speed, base_steer,
                                          map_action, planner, cursor, path)
        if mission_out:
            return mission_out

        # ── P4: NORMAL ───────────────────────────────────────────────────────
        return self._normal_drive(t_res, perc_res, now, base_speed, base_steer)

    # ── P0: Emergency ─────────────────────────────────────────────────────────

    def _check_emergency(self, t_res, base_steer: float) -> Optional[BehaviorOutput]:
        if not t_res.pedestrian_blocking:
            return None
        return BehaviorOutput(
            speed_pwm=0.0, steer_deg=base_steer,
            priority=self.PRI_EMERGENCY,
            state="EMERGENCY_STOP", reason="PEDESTRIAN ON ROAD",
        )

    # ── P1: Mandatory ─────────────────────────────────────────────────────────

    def _check_mandatory(self, t_res, now: float,
                         base_steer: float) -> Optional[BehaviorOutput]:
        is_red_light = (t_res.light_status is not None and
                        "RED" in t_res.light_status)
        is_stop_sign = (t_res.state == "SYS_STOP" and
                        "STOP SIGN" in t_res.reason)

        if is_stop_sign and now > self._stop_cooldown:
            if now > self._priority_until:
                if self._stop_timer == 0.0:
                    self._stop_timer = now
                    log.info("MANDATORY: STOP sign — 3 s halt")

        if self._stop_timer > 0.0:
            held = now - self._stop_timer
            if held < self.STOP_SIGN_HOLD_S:
                return BehaviorOutput(
                    speed_pwm=0.0, steer_deg=base_steer,
                    priority=self.PRI_MANDATORY,
                    state="STOP_SIGN_HOLD",
                    reason=f"STOP SIGN — {held:.1f}/{self.STOP_SIGN_HOLD_S:.0f}s",
                )
            else:
                self._stop_timer    = 0.0
                self._stop_cooldown = now + self.STOP_SIGN_COOLDOWN
                log.info("MANDATORY: STOP sign released")

        if is_red_light:
            return BehaviorOutput(
                speed_pwm=0.0, steer_deg=base_steer,
                priority=self.PRI_MANDATORY,
                state="RED_LIGHT_STOP", reason=t_res.light_status,
            )
        return None

    # ── P2: Legal ─────────────────────────────────────────────────────────────

    def _check_legal(self, t_res, perc_res, base_steer: float,
                     planner, now: float,
                     loc_x: float = 0.0, loc_y: float = 0.0) -> Optional[BehaviorOutput]:
        """
        No-Entry: refuse to proceed.
        Bus Lane: hard map-position correction + soft sign correction.
        """
        # No-Entry sign
        if "NO-ENTRY" in t_res.reason.upper() or "NO_ENTRY" in t_res.reason.upper():
            self._no_entry_active = True
            log.warning("LEGAL: No-Entry — path refused")
            return BehaviorOutput(
                speed_pwm=0.0, steer_deg=base_steer,
                priority=self.PRI_LEGAL,
                state="NO_ENTRY", reason="NO-ENTRY SIGN — PATH REFUSED",
            )
        else:
            self._no_entry_active = False

        # BC-04 Fix-4: Bus lane — check MAP POSITION first (harder rule).
        # loc_x/loc_y passed explicitly from orchestrator via compute() —
        # fixes the old broken getattr(perc_res, '_loc_x', None) which always
        # returned None because PerceptionResult has no such field.
        if planner and hasattr(planner, 'is_in_bus_lane'):
            if planner.is_in_bus_lane(loc_x, loc_y):
                hard_steer = base_steer + self.BUS_LANE_HARD_CORRECTION_DEG
                log.warning("LEGAL: Inside bus-lane zone — hard correction")
                return BehaviorOutput(
                    speed_pwm=self._zone_base_speed() * self.BUS_LANE_SPEED_MULT,
                    steer_deg=hard_steer,
                    priority=self.PRI_LEGAL,
                    state="BUS_LANE_HARD",
                    reason="BUS LANE MAP POSITION — hard left correction",
                )

        # Soft bus-lane correction from YOLO sign label
        if "BUS" in " ".join(t_res.active_labels).upper():
            corrected_steer = base_steer + self.BUS_LANE_STEER_CORRECTION
            return BehaviorOutput(
                speed_pwm=self._zone_base_speed() * 0.80,
                steer_deg=corrected_steer,
                priority=self.PRI_LEGAL,
                state="BUS_LANE_AVOID",
                reason="BUS LANE SIGN — soft correction",
            )

        return None

    # ── P3: Mission ───────────────────────────────────────────────────────────

    def _check_mission(self, t_res, perc_res, now: float,
                       base_speed: float, base_steer: float,
                       map_action: str, planner, cursor: int,
                       path: list) -> Optional[BehaviorOutput]:
        active_lower = " ".join(t_res.active_labels).lower()

        # ── BC-01: Highway protocol ────────────────────────────────────────────
        if self._zone_mode == "HIGHWAY":
            # Fix-3: LaneChangeToOuter - one-shot on CITY->HIGHWAY entry.
            # +12 deg right steer for 1.5 s then normal bias for 1.5 s.
            if self._hw_outer_last_zone != "HIGHWAY":   # fresh entry
                if self._hw_outer_state == "IDLE":
                    self._hw_outer_state = "STEER_RIGHT"
                    self._hw_outer_ts    = time.time()
                    log.info("HIGHWAY: LaneChangeToOuter triggered")
            self._hw_outer_last_zone = "HIGHWAY"

            hw_steer = base_steer + self.HIGHWAY_LANE_BIAS_DEG   # default

            if self._hw_outer_state == "STEER_RIGHT":
                if time.time() - self._hw_outer_ts < 1.5:
                    hw_steer = base_steer + 12.0
                else:
                    self._hw_outer_state = "HOLD"
                    self._hw_outer_ts    = time.time()
            elif self._hw_outer_state == "HOLD":
                if time.time() - self._hw_outer_ts >= 1.5:
                    self._hw_outer_state = "IDLE"
                    log.info("HIGHWAY: LaneChangeToOuter complete")

            hw_speed = max(base_speed, self.HIGHWAY_SPEED_PWM)
            if map_action == "HIGHWAY_MERGE":
                hw_speed *= self.HIGHWAY_MERGE_SLOW_MULT
                state  = "HIGHWAY_MERGE"
                reason = "HIGHWAY MERGE"
            else:
                state  = "HIGHWAY_CRUISE"
                reason = "HIGHWAY - outermost lane protocol"
            return BehaviorOutput(
                speed_pwm=hw_speed, steer_deg=hw_steer,
                priority=self.PRI_MISSION,
                state=state, reason=reason,
                zone_mode="HIGHWAY", maneuver="NONE",
            )
        else:
            self._hw_outer_last_zone = self._zone_mode

        # ── BC-02: Speed oval ─────────────────────────────────────────────────
        if self._zone_mode == "SPEED_OVAL":
            return BehaviorOutput(
                speed_pwm=max(base_speed, self.SPEED_OVAL_PWM),
                steer_deg=base_steer,
                priority=self.PRI_MISSION,
                state="SPEED_OVAL_CRUISE",
                reason="SPEED OVAL — sustained high-speed loop",
                zone_mode="SPEED_OVAL", maneuver="NONE",
            )

        # ── BC-06: START / PIT area speed cap ────────────────────────────────
        if self._zone_mode == "START":
            return BehaviorOutput(
                speed_pwm=min(base_speed, self.START_AREA_SPEED_PWM),
                steer_deg=base_steer,
                priority=self.PRI_MISSION,
                state="START_AREA",
                reason="START AREA — restricted zone speed cap",
                zone_mode="START",
            )

        # ── BC-03: Roundabout CCW ─────────────────────────────────────────────
        roundabout_sign = "roundabout" in active_lower
        if roundabout_sign:
            self._roundabout_sign_ts = now

        approaching_ccw = (map_action == "CCW")
        sign_recent     = (now - self._roundabout_sign_ts < self.ROUNDABOUT_SIGN_TIMEOUT)

        if approaching_ccw or self._roundabout_active or (roundabout_sign and sign_recent):
            self._roundabout_active = True

            # Inside CCW: left steer bias + speed cap
            ccw_steer = base_steer + self.CCW_ENTRY_STEER_DEG
            ccw_speed = min(base_speed, self.ROUNDABOUT_SPEED_PWM)

            # Exit condition: zone changed away from ROUNDABOUT AND no recent sign
            if (self._zone_mode != "ROUNDABOUT" and
                    not approaching_ccw and
                    not sign_recent):
                self._roundabout_active = False
                log.info("ROUNDABOUT: exited — resuming city drive")
            else:
                return BehaviorOutput(
                    speed_pwm=ccw_speed, steer_deg=ccw_steer,
                    priority=self.PRI_MISSION,
                    state="ROUNDABOUT_CCW",
                    reason="ROUNDABOUT — CCW navigation",
                    zone_mode="ROUNDABOUT", maneuver="ROUNDABOUT",
                )

        # ── Parking ───────────────────────────────────────────────────────────
        parking_sign = any(k in active_lower
                           for k in ("parking", "park-sign", "park_sign"))
        # F-01: zone-based trigger removed to prevent accidental drift-triggered parking.
        # Car now strictly requires seeing a parking sign to trigger the FSM.

        if parking_sign and not self.parking_fsm.active:
            self.parking_fsm.trigger(now)

        if self.parking_fsm.active:
            spot_clear = True
            if t_res.parking_state in ("SEEK",):
                spot_clear = (t_res.parking_state != "SEEK")
            speed_mult, steer_bias, park_label = self.parking_fsm.update(
                now, base_speed, spot_clear=spot_clear
            )
            if park_label == "DONE":
                self.parking_fsm.reset()
                return None
            return BehaviorOutput(
                speed_pwm=base_speed * speed_mult,
                steer_deg=base_steer + steer_bias,
                priority=self.PRI_MISSION,
                state=f"PARKING_{park_label}",
                reason=f"PARKING — phase: {park_label}",
                maneuver="PARKING",
            )

        # ── Overtake (dashed line + obstacle) ────────────────────────────────
        # STRICT RIGHT LANE COMPLIANCE: Overtaking logic has been removed.
        # The car will no longer swerve into the left lane.
        
        return None

    # ── P4: Normal ────────────────────────────────────────────────────────────

    def _normal_drive(self, t_res, perc_res, now: float,
                      base_speed: float,
                      base_steer: float) -> BehaviorOutput:
        speed  = base_speed
        steer  = base_steer
        reason = "NORMAL DRIVE"
        state  = "RL_DRIVE"

        if (t_res.state == "SYS_SLOW" and
                "TAILING" in t_res.reason.upper()):
            speed  *= 0.55
            state   = "TAILING"
            reason  = "CONTINUOUS LINE — tailing obstacle"

        elif "CROSSWALK" in t_res.reason.upper():
            speed = min(speed, self.CROSSWALK_SPEED_PWM)
            state  = "CROSSWALK_SLOW"
            reason = "CROSSWALK AHEAD — slowing"

        elif any(k in " ".join(t_res.active_labels).lower()
                 for k in ("parking", "park-sign")):
            speed = min(speed, self.SLOW_SPEED_PWM)
            state  = "PARKING_SCAN"
            reason = "PARKING SIGN — scanning for spot"

        elif "YELLOW" in (t_res.light_status or ""):
            speed *= 0.65
            state  = "YELLOW_SLOW"
            reason = "YELLOW LIGHT — prepare to stop"

        elif t_res.state == "SYS_LIMIT":
            speed *= 0.75
            state  = "SPEED_LIMIT"
            reason = "SPEED LIMIT ZONE"

        if 0 < speed < self.MIN_SPEED_PWM:
            speed = self.MIN_SPEED_PWM

        return BehaviorOutput(
            speed_pwm=speed, steer_deg=steer,
            priority=self.PRI_NORMAL,
            state=state, reason=reason,
            zone_mode=self._zone_mode,
        )

    # ── Zone management ───────────────────────────────────────────────────────

    def _update_zone(self, t_res, planner, now: float):
        """
        BC-07: Debounced zone update.
        Prefer the map-planner zone (from localizer) over sign-based detection.
        Falls back to sign-based if planner unavailable.
        """
        # Primary: planner-derived zone (set by orchestrator each tick via t_res or direct)
        new_zone = getattr(t_res, 'zone_mode', None)
        if new_zone in ("HIGHWAY", "SPEED_OVAL", "ROUNDABOUT", "PARKING", "START", "CITY"):
            candidate = new_zone
        else:
            # Fallback: sign-based zone transitions (v1 logic)
            active_lower = " ".join(t_res.active_labels).lower()
            if any(k in active_lower for k in ("highway-entry", "highway_entry",
                                                "highway_start")):
                candidate = "HIGHWAY"
            elif any(k in active_lower for k in ("highway-exit", "highway_exit",
                                                  "highway_end")):
                candidate = "CITY"
            else:
                candidate = self._zone_mode  # no change

        # BC-07: debounce — only commit after ZONE_DEBOUNCE_FRAMES consistent frames
        if candidate != self._zone_candidate:
            self._zone_candidate    = candidate
            self._zone_debounce_cnt = 0
        else:
            self._zone_debounce_cnt += 1
            if self._zone_debounce_cnt >= self.ZONE_DEBOUNCE_FRAMES:
                if candidate != self._zone_mode:
                    log.info("ZONE: %s → %s", self._zone_mode, candidate)
                self._zone_mode = candidate

    def _zone_base_speed(self) -> float:
        """Map zone_mode → base PWM speed."""
        return {
            "CITY"       : self.CITY_SPEED_PWM,
            "HIGHWAY"    : self.HIGHWAY_SPEED_PWM,
            "SPEED_OVAL" : self.SPEED_OVAL_PWM,
            "ROUNDABOUT" : self.ROUNDABOUT_SPEED_PWM,
            "PARKING"    : self.PARKING_SPEED_PWM,
            "START"      : self.START_AREA_SPEED_PWM,
        }.get(self._zone_mode, self.CITY_SPEED_PWM)

    def _sign_approach_mult(self, t_res) -> float:
        """Linear decel ramp: 1.0 (far) → 0.60 (at sign)."""
        dist = getattr(t_res, 'sign_approach_m', 99.0)
        if dist >= self.APPROACH_DECEL_M:
            return 1.0
        if dist <= self.APPROACH_FULL_M:
            return 0.60
        ratio = ((dist - self.APPROACH_FULL_M) /
                 (self.APPROACH_DECEL_M - self.APPROACH_FULL_M))
        return 0.60 + 0.40 * ratio

    # ── External API ──────────────────────────────────────────────────────────

    def set_priority_road(self, duration_s: float = 8.0):
        """Called when a priority/right-of-way sign is detected."""
        self._priority_until = time.time() + duration_s
        log.info("LEGAL: priority road active for %.1f s", duration_s)

    @property
    def zone_mode(self) -> str:
        return self._zone_mode