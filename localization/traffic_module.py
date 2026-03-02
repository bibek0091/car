"""
traffic_module.py — BFMC Traffic Decision Engine
=================================================
Handles all required traffic signs and rules:

  Traffic Lights : Red (stop) / Yellow (slow) / Green (go)
  Signs          : Stop, Parking, Crosswalk, Priority, Highway-Entry/Exit,
                   One-Way, Roundabout, No-Entry, Speed-Limit
  General rules  : Pedestrian-at-crosswalk stop, pedestrian-on-road stop,
                   obstacle overtake gated by dashed/continuous line
  Zone tracking  : CITY / HIGHWAY with appropriate speed multipliers
  Parking FSM    : SEEK -> ENTER -> WAIT -> EXIT -> DONE
"""

import threading
import queue
import time
import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("WARNING: ultralytics (YOLO) not found.")


# ══════════════════════════════════════════════════════════════════════════════
# Data contract
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrafficResult:
    """
    Single output produced by TrafficDecisionEngine.process().

    state           : primary driving command for the control layer.
    reason          : human-readable string for the dashboard.
    speed_multiplier: applied to base_speed in Controller.compute().
    zone_mode       : "CITY" | "HIGHWAY"   — governs speed floors.
    parking_state   : "NONE"|"SEEK"|"ENTER"|"WAIT"|"EXIT"|"DONE"
    steer_bias      : additional steer angle (deg) requested by parking FSM.
    pedestrian_blocking : True when pedestrian is holding us at a crosswalk.
    light_status    : dashboard string for the traffic-light colour.
    active_labels   : all YOLO class names seen this frame.
    yolo_debug_frame: BGR frame with detection overlays.
    sign_approach_m : estimated approach distance to nearest sign (for dash)
    """
    state: str              # SYS_GO | SYS_STOP | SYS_SLOW | SYS_APPROACH | SYS_LANE_CHANGE_LEFT | SYS_LIMIT
    reason: str
    speed_multiplier: float
    zone_mode: str = "CITY"
    parking_state: str = "NONE"
    steer_bias: float = 0.0
    pedestrian_blocking: bool = False
    light_status: str = "NONE"
    active_labels: List[str] = field(default_factory=list)
    yolo_debug_frame: np.ndarray = None
    sign_approach_m: float = 99.0   # estimated distance to nearest active sign


# ══════════════════════════════════════════════════════════════════════════════
# Threaded YOLO detector (unchanged from original, minus imports)
# ══════════════════════════════════════════════════════════════════════════════

class ThreadedYOLODetector:
    def __init__(self, model_path="best.pt"):
        self.frame_queue   = queue.Queue(maxsize=1)
        self.result_queue  = queue.Queue(maxsize=1)
        self.running       = True
        self.active_detections = []
        self.yolo_ok       = False
        self.model_path_used = None
        self.model         = None

        if _YOLO_AVAILABLE:
            resolved = self._resolve_model_path(model_path)
            if resolved:
                try:
                    import torch
                    _orig = torch.load
                    def _patched(*args, **kwargs):
                        kwargs['weights_only'] = False
                        return _orig(*args, **kwargs)
                    torch.load = _patched
                    try:
                        self.model = YOLO(resolved)
                        self.yolo_ok = True
                        self.model_path_used = resolved
                        print(f"[YOLO] Loaded: {resolved}")
                    finally:
                        torch.load = _orig
                except Exception as e:
                    print(f"[YOLO] Load failed: {e}")
            else:
                print(f"[YOLO] '{model_path}' not found — detection disabled.")

        self._latch_dets  : list = []     # latched detections
        self._latch_frames : int  = 0      # frames since last real detection
        self.LATCH_FRAMES  : int  = 5      # hold detections for this many frames

        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    @staticmethod
    def _resolve_model_path(model_path):
        import os
        candidates = [
            model_path,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", model_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", model_path),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
        return None

    def _run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if self.model is None:
                    continue

                # ─ YOLO inference ─────────────────────────────────────────────
                # Note: CLAHE image enhancement was removed. A standard YOLO 
                # model expects raw RGB/BGR input matching its training data.
                # Aggressive contrast filters often break the feature matching.
                results = self.model.predict(
                    source=frame, conf=0.25, verbose=False)
                detections = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = self.model.names[int(box.cls[0].item())]
                    conf  = box.conf[0].item()
                    detections.append({"label": label, "confidence": conf,
                                       "bbox": (x1, y1, x2, y2)})

                # ─ Detection latch: hold non-empty results for LATCH_FRAMES ──
                if detections:
                    self._latch_dets   = detections
                    self._latch_frames = 0
                else:
                    self._latch_frames += 1
                    if self._latch_frames <= self.LATCH_FRAMES:
                        detections = self._latch_dets  # reuse last good frame
                    else:
                        self._latch_dets = []          # latch expired

                if not self.result_queue.full():
                    self.result_queue.put(detections)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"YOLO Thread Error: {e}")

    def update_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())

    def get_detections(self):
        if not self.result_queue.empty():
            self.active_detections = self.result_queue.get()
        return self.active_detections

    def stop(self):
        self.running = False
        self.worker.join()


# ══════════════════════════════════════════════════════════════════════════════
# Traffic-light state machine  (added YELLOW)
# ══════════════════════════════════════════════════════════════════════════════

class TrafficLightStateMachine:
    """
    States: NO_LIGHT | LIGHT_DETECTED_FAR | LIGHT_RED_STOPPING |
            LIGHT_RED_STOPPED | LIGHT_YELLOW_SLOW | LIGHT_GREEN_GO
    """
    def __init__(self):
        self.state = "NO_LIGHT"
        self.last_seen_red = 0.0
        self.red_start_time = 0.0

    def update(self, is_red, is_yellow, is_green, dist_cat):
        now = time.time()

        if is_red:
            self.last_seen_red = now
            if self.red_start_time == 0.0:
                self.red_start_time = now
                
            # Traffic Light Stability Window (0.5s Debounce)
            if now - self.red_start_time > 0.5:
                if dist_cat == "HALT":
                    self.state = "LIGHT_RED_STOPPED"
                elif dist_cat == "APPROACH":
                    self.state = "LIGHT_RED_STOPPING"
                elif self.state == "NO_LIGHT":
                    self.state = "LIGHT_DETECTED_FAR"

        elif is_yellow:
            self.red_start_time = 0.0
            # Yellow: slow down and prepare to stop or proceed with caution
            self.state = "LIGHT_YELLOW_SLOW"

        elif is_green:
            self.red_start_time = 0.0
            if self.state in ["LIGHT_RED_STOPPED", "LIGHT_RED_STOPPING"]:
                if now - self.last_seen_red > 1.0:   # 1 s delay before moving
                    self.state = "LIGHT_GREEN_GO"
            else:
                self.state = "LIGHT_GREEN_GO"

        else:
            self.red_start_time = 0.0
            # Light lost
            if self.state in ["LIGHT_RED_STOPPED", "LIGHT_RED_STOPPING"]:
                if now - self.last_seen_red > 4.0:
                    self.state = "NO_LIGHT"
            elif self.state not in ["LIGHT_RED_STOPPED", "LIGHT_RED_STOPPING"]:
                self.state = "NO_LIGHT"

        return self.state


# ══════════════════════════════════════════════════════════════════════════════
# Collision predictor (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class CollisionPredictor:
    def __init__(self):
        self.history = {}

    def update_and_predict(self, detections, dt):
        critical = []
        current  = {}
        for det in detections:
            lbl = det["label"]
            if lbl not in ["car", "pedestrian", "closed-road-stand",
                           "roadblock", "obstacle"]:
                continue
            x1, y1, x2, y2 = det["bbox"]
            h  = y2 - y1
            cx = (x1 + x2) / 2

            matched = None
            best_dist = 50   # pixel proximity threshold
            for tid, data in self.history.items():
                if data["label"] == lbl:
                    dist = abs(data["cx"] - cx)
                    if dist < best_dist:   # BUG-05: pick CLOSEST, not first match
                        best_dist = dist
                        matched   = tid
            if not matched:
                self._next_id = getattr(self, '_next_id', 0) + 1
                matched = f"{lbl}_{self._next_id}"

            current[matched] = {"label": lbl, "cx": cx, "h": h}
            if matched in self.history:
                last_h = self.history[matched]["h"]
                growth = (h - last_h) / max(dt, 0.01)
                if growth > 5.0 and h > 40:
                    ttc = h / growth if growth > 0 else 999
                    if ttc < 3.0:
                        critical.append(det)

        self.history = current
        return critical


# ══════════════════════════════════════════════════════════════════════════════
# Pedestrian-at-crosswalk monitor
# ══════════════════════════════════════════════════════════════════════════════

class PedestrianCrosswalkMonitor:
    """
    Tracks when the car is near a crosswalk AND a pedestrian is present.
    Rule: stop and WAIT until the pedestrian has cleared the road.
    """
    CROSSWALK_MEMORY_S = 6.0    # hold crosswalk-detected state for 6 s

    def __init__(self):
        self._crosswalk_ts  = 0.0   # time last crosswalk sign was seen
        self._blocking      = False
        self._clear_ts      = 0.0   # time pedestrian last disappeared

    def update(self, dets, h, w, now):
        """
        Returns (blocking: bool, reason: str).
        blocking=True means SYS_STOP should be committed.
        """
        # Refresh crosswalk memory
        for d in dets:
            lbl = d["label"].lower()
            if any(k in lbl for k in ("crosswalk", "pedestrian_crossing",
                                      "crosswalk-sign")):
                self._crosswalk_ts = now

        near_crosswalk = (now - self._crosswalk_ts) < self.CROSSWALK_MEMORY_S

        if not near_crosswalk:
            self._blocking = False
            return False, ""

        # Check for pedestrian in the lower-centre road area
        ped_on_road = False
        for d in dets:
            if d["label"].lower() not in ("pedestrian", "person"):
                continue
            x1, y1, x2, y2 = d["bbox"]
            # Pedestrian is considered "on road" when their feet are in the
            # lower 60 % of the frame and their body straddles the road centre
            if y2 > h * 0.40 and x1 < w * 0.75 and x2 > w * 0.25:
                ped_on_road = True
                break

        if ped_on_road:
            self._blocking  = True
            self._clear_ts  = 0.0
        elif self._blocking:
            # Wait 1.5 s after pedestrian clears before releasing
            if self._clear_ts == 0.0:
                self._clear_ts = now
            if now - self._clear_ts > 1.5:
                self._blocking = False

        return self._blocking, "PEDESTRIAN AT CROSSWALK" if self._blocking else ""


# ══════════════════════════════════════════════════════════════════════════════
# Main decision engine
# ══════════════════════════════════════════════════════════════════════════════

class TrafficDecisionEngine:
    """
    Processes one frame per call and returns a TrafficResult.

    Sign / rule coverage
    --------------------
    Traffic Light : Red (stop) / Yellow (slow) / Green (go)
    Stop sign     : 3-second halt then cooldown
    Crosswalk     : slow; stop if pedestrian detected at crossing
    Priority      : skip intersection stop (right-of-way)
    Highway Entry : switch zone_mode -> HIGHWAY
    Highway Exit  : switch zone_mode -> CITY
    No-Entry      : commit SYS_STOP; planner should reroute
    One-Way       : informational log; routing handled by A* graph
    Roundabout    : slow on entry; routing handled by A* graph (CCW)
    Speed-limit   : reduce speed_multiplier to match sign value
    Parking       : trigger ParkingStateMachine
    Obstacle/car  : lane-change LEFT if on dashed segment (caller provides
                    line_type); just follow on continuous segment
    Pedestrian    : TTC-based emergency stop (CollisionPredictor)
                    + crosswalk-specific wait (PedestrianCrosswalkMonitor)
    """

    def __init__(self, threaded_detector):
        self.threaded_detector  = threaded_detector
        self.state              = "SYS_GO"
        self.reason             = "CLEAR"

        # Stop-sign FSM
        self.stop_timer  = 0.0   # time stop sign was first triggered
        self.stop_cd     = 0.0   # cooldown — don't re-trigger for 5 s

        # Priority: skip next stop-sign check
        self._priority_until = 0.0

        self.last_t  = time.time()

        self.tl_fsm     = TrafficLightStateMachine()
        self.col_pred   = CollisionPredictor()
        self.ped_xwalk  = PedestrianCrosswalkMonitor()
        # NOTE: ParkingStateMachine removed — ParkingSequenceFSM in
        # behavior_controller.py is the single source of truth.

        # Zone tracking
        self._zone_mode = "CITY"   # "CITY" | "HIGHWAY"
        
        # Stability tracking (Traffic Hardening)
        self.class_stability = {}
        self.last_det_areas  = {}

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _is_glowing(self, frame, x1, y1, x2, y2):
        """
        Analyse the HSV colour mass inside a traffic-light bounding box.
        Returns ("RED"|"YELLOW"|"GREEN"|"NONE", pixel_mass).
        """
        fh, fw = frame.shape[:2]
        # Clamp to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        
        # Zero-size check to prevent cv2.cvtColor crash
        if x1 >= x2 or y1 >= y2 or (y2 - y1) < 15 or (x2 - x1) < 10:
            return "NONE", 0
        crop = frame[y1:y2, x1:x2]
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Red (wraps around hue wheel)
        mr1 = cv2.inRange(hsv, np.array([0,  50, 150]), np.array([10, 255, 255]))
        mr2 = cv2.inRange(hsv, np.array([170, 50, 150]), np.array([180,255,255]))
        # Yellow
        my  = cv2.inRange(hsv, np.array([15, 100, 150]), np.array([35, 255, 255]))
        # Green
        mg  = cv2.inRange(hsv, np.array([50,  50, 150]), np.array([90, 255, 255]))

        red    = cv2.countNonZero(cv2.bitwise_or(mr1, mr2))
        yellow = cv2.countNonZero(my)
        green  = cv2.countNonZero(mg)
        MIN_MASS = 30

        # Highest mass wins, with a minimum threshold
        best = max(red, yellow, green)
        if best < MIN_MASS:
            return "NONE", best
        if red >= yellow and red >= green:
            return "RED", red
        if yellow >= green:
            return "YELLOW", yellow
        return "GREEN", green

    def _dist_cat(self, box_h):
        """Classify distance from bounding-box height.
        FAR      : box_h < 30  px  — sign detected at long range, pre-decelerate
        APPROACH : 30–70 px      — sign coming up, sign-specific slow/stop fires
        HALT     : > 70 px       — right at sign, full action
        """
        if box_h < 30:
            return "FAR"
        elif box_h < 70:
            return "APPROACH"
        return "HALT"

    def _approx_dist_m(self, box_h, focal_px: float = 400.0):
        """Distance estimate from bbox height.
        sign_h_m = 0.06 m — BFMC 1:10 scale sign real height.
        Formula: dist = (sign_h_m * focal_px) / box_h
        Example: 24 px box  → 0.06 * 400 / 24 = 1.0 m  (was 5× too large at 5.0 m)
        """
        sign_h_m = 0.06
        return (sign_h_m * focal_px) / max(box_h, 1)

    # ── Main process call ─────────────────────────────────────────────────────

    def process(self, frame, line_type="UNKNOWN"):
        """
        Parameters
        ----------
        frame      : BGR camera frame (640×480).
        line_type  : "DASHED" | "CONTINUOUS" | "UNKNOWN"
                     Injected from map_planner so we know whether
                     overtaking is permitted.

        Returns
        -------
        TrafficResult
        """
        fh, fw = frame.shape[:2]
        dbg    = frame.copy()
        now    = time.time()
        dt     = max(now - self.last_t, 0.001)
        self.last_t = now

        self.threaded_detector.update_frame(frame)
        dets = self.threaded_detector.get_detections()

        # ── Priority queue helpers ────────────────────────────────────────────
        pri     = 99
        p_state = "SYS_GO"
        p_res   = "CLEAR PATH"

        def commit(pr, st, rs):
            nonlocal pri, p_state, p_res
            if pr < pri:
                pri, p_state, p_res = pr, st, rs

        # ── Fix-7: Stationary pedestrian in road path (growth=0, TTC=∞) ─────────
        # CollisionPredictor only fires on box growth > 5 px/s. A pedestrian
        # standing still produces zero growth and infinite TTC — this check
        # catches that case before TTC logic runs.
        for _det7 in dets:
            if _det7["label"].lower() in ("pedestrian", "person"):
                _x1, _y1, _x2, _y2 = _det7["bbox"]
                _in_centre = (_x1 < fw * 0.75 and _x2 > fw * 0.25
                              and _y2 > fh * 0.50)
                if _in_centre:
                    commit(0, "SYS_STOP", "PEDESTRIAN IN ROAD PATH")
                    break

        # ── Collision predictor (TTC) ─────────────────────────────────────────
        crit = self.col_pred.update_and_predict(dets, dt)
        if crit:
            commit(1, "SYS_STOP", "COLLISION IMMINENT")

        # ── Stop-sign FSM: release after 3 s ─────────────────────────────────
        if self.stop_timer > 0.0:
            if now - self.stop_timer >= 3.0:
                self.stop_timer = 0.0
                self.stop_cd    = now + 5.0

        # ── Pedestrian at crosswalk ───────────────────────────────────────────
        ped_blocking, ped_reason = self.ped_xwalk.update(dets, fh, fw, now)
        if ped_blocking:
            commit(1, "SYS_STOP", ped_reason)

        light_st = "NONE"
        act_lbl  = []
        _nearest_sign_dist_m = 99.0   # for TrafficResult.sign_approach_m

        # ── Per-detection logic ───────────────────────────────────────────────
        current_labels = set()
        for d in dets:
            lbl       = d["label"]
            lbl_lower = lbl.lower()
            x1, y1, x2, y2 = d["bbox"]
            box_h = y2 - y1
            box_w = x2 - x1
            box_area = float(box_w * box_h)
            dist_cat = self._dist_cat(box_h)
            current_labels.add(lbl)
            
            # --- Class Stability Update ---
            prev_area = self.last_det_areas.get(lbl, 0.0)
            if lbl in self.class_stability:
                # Area must be within +/- 30% to count as stable
                if prev_area > 0 and (abs(box_area - prev_area) / prev_area) <= 0.3:
                    self.class_stability[lbl] += 1
                else:
                    self.class_stability[lbl] = 1 # Reset if box jumped massively
            else:
                self.class_stability[lbl] = 1
                
            self.last_det_areas[lbl] = box_area
            
            # If not seen for 3 consecutive frames, ignore it to prevent flicker
            if self.class_stability[lbl] < 3:
                continue

            # Estimate approach distance for dashboard
            approx_m = self._approx_dist_m(box_h)

            # Draw detection on debug frame
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(dbg, f"{lbl} ~{approx_m:.1f}m", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Do not register FAR signs as active labels; this prevents main.py 
            # from attempting to map-snap to them while they are still distant.
            _is_sign = not ("car" in lbl_lower or "pedestrian" in lbl_lower
                            or "person" in lbl_lower or "obstacle" in lbl_lower
                            or "roadblock" in lbl_lower
                            or "traffic" in lbl_lower)
            if not (_is_sign and dist_cat == "FAR"):
                act_lbl.append(lbl)

            # ── Traffic light ─────────────────────────────────────────────────
            if "traffic" in lbl_lower and "light" in lbl_lower:
                clr, mass = self._is_glowing(frame, x1, y1, x2, y2)
                dist = self._dist_cat(box_h)
                fsm_st = self.tl_fsm.update(
                    clr == "RED", clr == "YELLOW", clr == "GREEN", dist
                )
                if fsm_st in ("LIGHT_RED_STOPPING", "LIGHT_RED_STOPPED"):
                    light_st = f"[RED] {dist}"
                    commit(1, "SYS_STOP", "RED LIGHT")
                elif fsm_st == "LIGHT_YELLOW_SLOW":
                    light_st = "[YELLOW] CAUTION"
                    commit(3, "SYS_SLOW", "YELLOW LIGHT")
                elif fsm_st == "LIGHT_GREEN_GO":
                    light_st = "[GREEN] GO"
                continue   # done with this detection

            # ── APPROACH PRE-DECELERATION (DISABLED) ────────────────────────────
            # The user requested no reaction to far away signs. The car will
            # ignore signs until they enter the NEAR/APPROACH threshold.
            is_sign = not ("car" in lbl_lower or "pedestrian" in lbl_lower
                           or "person" in lbl_lower or "obstacle" in lbl_lower
                           or "roadblock" in lbl_lower)
            if is_sign and dist_cat == "FAR":
                # Only track distance for the dashboard, do NOT slow down or react
                _nearest_sign_dist_m = min(_nearest_sign_dist_m, approx_m)
                continue   # ignore until APPROACH range

            # Track distance for dashboard
            if is_sign:
                _nearest_sign_dist_m = min(_nearest_sign_dist_m, approx_m)

            # Skip tiny detections for all non-light signs
            if box_h < 30:
                continue

            # ── Stop sign ─────────────────────────────────────────────────────
            if any(k in lbl_lower for k in ("stop-sign", "stop_sign",)) or lbl_lower == "stop":
                if now > self.stop_cd and now > self._priority_until:
                    if self.stop_timer == 0.0:
                        self.stop_timer = now
                    
                    # If we've been stopped for 3 seconds, release and set 5s cooldown
                    if now - self.stop_timer > 3.0:
                        self.stop_cd = now + 5.0
                        self.stop_timer = 0.0
                        # Auto-release: next frame will be CLEAR unless another sign overrides
                    else:
                        commit(2, "SYS_STOP", "STOP SIGN (3 s)")

            # ── Crosswalk sign ────────────────────────────────────────────────
            elif any(k in lbl_lower for k in ("crosswalk", "pedestrian_crossing",
                                               "zebra")):
                commit(4, "SYS_SLOW", "CROSSWALK AHEAD")

            # ── Priority / right-of-way ───────────────────────────────────────
            elif any(k in lbl_lower for k in ("priority", "right-of-way",
                                               "priority-road", "give-way")):
                # Right-of-way: no stop required at next intersection
                self._priority_until = now + 8.0
                # No commit needed — car continues at SYS_GO

            # ── Highway entry ─────────────────────────────────────────────────
            elif any(k in lbl_lower for k in ("highway-entry", "highway_entry",
                                               "highway_start", "motorway-entry")):
                self._zone_mode = "HIGHWAY"
                commit(5, "SYS_GO", "HIGHWAY MODE ON")

            # ── Highway exit ──────────────────────────────────────────────────
            elif any(k in lbl_lower for k in ("highway-exit", "highway_exit",
                                               "highway_end", "motorway-exit")):
                self._zone_mode = "CITY"
                commit(5, "SYS_GO", "HIGHWAY MODE OFF")

            # ── No-entry ─────────────────────────────────────────────────────
            elif any(k in lbl_lower for k in ("no-entry", "no_entry",
                                               "no entry", "do-not-enter")):
                commit(1, "SYS_STOP", "NO-ENTRY SIGN — REROUTE")

            # ── One-way (informational — routing handled by A* graph) ─────────
            elif any(k in lbl_lower for k in ("one-way", "oneway", "one_way")):
                pass   # A* graph edge directions already enforce one-way

            # ── Roundabout ────────────────────────────────────────────────────
            elif any(k in lbl_lower for k in ("roundabout",)):
                commit(4, "SYS_SLOW", "ROUNDABOUT ENTRY")

            # ── Speed-limit signs (BUG-09: exact class name match) ────────────
            elif lbl_lower in {
                "speed_30", "speed_50", "speed_80",
                "speed-limit-30", "speed-limit-50", "speed-limit-80",
                "speedlimit_30", "speedlimit_50", "speedlimit_80",
                "speed-limit", "speedlimit", "speed_limit",
            } or (
                lbl_lower.startswith(("speed_", "speed-limit-", "speedlimit_"))
                and any(ch.isdigit() for ch in lbl_lower)
            ):
                commit(4, "SYS_LIMIT", "SPEED LIMIT ZONE")


            # ── Parking sign ──────────────────────────────────────────────────
            elif any(k in lbl_lower for k in ("parking", "park-sign", "park_sign",
                                               "car-park")):
                pass # self.parking_fsm.trigger(now) - Disabled per user request

            elif lbl_lower in ("car", "closed-road-stand", "roadblock", "obstacle"):
                in_path = (x1 < fw * 0.80 and x2 > fw * 0.20 and y2 > fh * 0.60)
                if in_path:
                    # STRICT RIGHT LANE COMPLIANCE: Do not swerve into oncoming traffic.
                    # Always brake if an obstacle is in the path.
                    commit(1, "SYS_STOP", f"OBSTACLE IN PATH ({lbl})")

        # ── Apply active stop-sign hold ───────────────────────────────────────
        if self.stop_timer > 0.0 and now - self.stop_timer < 3.0:
            commit(2, "SYS_STOP", "STOP SIGN (holding)")

        # --- Clean up lost labels from stability tracker ---
        lost_labels = [k for k in self.class_stability.keys() if k not in current_labels]
        for k in lost_labels:
            del self.class_stability[k]
            if k in self.last_det_areas:
                del self.last_det_areas[k]

        # ── Override: active red light beats everything else ──────────────────
        if "RED" in light_st:
            commit(1, "SYS_STOP", "RED LIGHT")

        # NOTE: Parking FSM removed — behavior_controller.ParkingSequenceFSM
        # is the single source of truth for parking state.

        self.state  = p_state
        self.reason = p_res

        # ── Speed multiplier ──────────────────────────────────────────────────
        mult = 1.0
        if self.state == "SYS_STOP":
            mult = 0.0
        elif self.state in ("SYS_SLOW", "SYS_LANE_CHANGE_LEFT"):
            mult = 0.55
        elif self.state == "SYS_APPROACH":
            mult = 0.85   # gentle pre-decel on sign approach
        elif self.state == "SYS_LIMIT":
            mult = 0.75

        return TrafficResult(
            state            = self.state,
            reason           = self.reason,
            speed_multiplier = mult,
            zone_mode        = self._zone_mode,
            pedestrian_blocking = ped_blocking,
            light_status     = light_st,
            active_labels    = act_lbl,
            yolo_debug_frame = dbg,
            sign_approach_m  = _nearest_sign_dist_m,
        )