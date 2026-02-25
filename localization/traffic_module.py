import threading
import queue
import time
import math
import numpy as np
import cv2
from dataclasses import dataclass, field

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("WARNING: ultralytics (YOLO) not found.")

@dataclass
class TrafficResult:
    state: str          # "SYS_GO" | "SYS_STOP" | "SYS_SLOW" | "SYS_LANE_CHANGE_LEFT" | "SYS_LIMIT"
    reason: str
    speed_multiplier: float
    light_status: str = "NONE"
    active_labels: list[str] = field(default_factory=list)
    yolo_debug_frame: np.ndarray = None


class ThreadedYOLODetector:
    def __init__(self, model_path="best.pt"):
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = True
        self.active_detections = []
        
        self.model = None
        if _YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                self.model = None

        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if self.model is None:
                    continue
                
                # YOLO limits on roi
                results = self.model.predict(source=frame, conf=0.25, verbose=False)
                detections = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = self.model.names[int(box.cls[0].item())]
                    conf = box.conf[0].item()
                    detections.append({"label": label, "confidence": conf, "bbox": (x1, y1, x2, y2)})
                
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


class TrafficLightStateMachine:
    def __init__(self):
        self.state = "NO_LIGHT"
        self.last_seen_red = 0.0
        
    def update(self, is_red, is_green, dist_cat):
        now = time.time()
        if is_red:
            self.last_seen_red = now
            if dist_cat == "HALT": self.state = "LIGHT_RED_STOPPED"
            elif dist_cat == "APPROACH": self.state = "LIGHT_RED_STOPPING"
            elif self.state == "NO_LIGHT": self.state = "LIGHT_DETECTED_FAR"
        elif is_green:
            if self.state in ["LIGHT_RED_STOPPED", "LIGHT_RED_STOPPING"]:
                if now - self.last_seen_red > 1.0: # 1s delay
                    self.state = "LIGHT_GREEN_GO"
            else:
                self.state = "LIGHT_GREEN_GO"
        else:
            if self.state in ["LIGHT_RED_STOPPED", "LIGHT_RED_STOPPING"] and (now - self.last_seen_red > 4.0):
                self.state = "NO_LIGHT"
            elif self.state not in ["LIGHT_RED_STOPPED", "LIGHT_RED_STOPPING"]:
                self.state = "NO_LIGHT"
                
        return self.state


class CollisionPredictor:
    def __init__(self):
        self.history = {}
        
    def update_and_predict(self, detections, dt):
        critical = []
        current = {}
        now = time.time()
        for det in detections:
            lbl = det["label"]
            if lbl not in ["car", "pedestrian", "closed-road-stand"]:
                continue
            x1, y1, x2, y2 = det["bbox"]
            h = y2 - y1
            cx = (x1 + x2) / 2
            
            matched = None
            for tid, data in self.history.items():
                if data["label"] == lbl and abs(data["cx"] - cx) < 50:
                    matched = tid
                    break
            if not matched: 
                self._next_id = getattr(self, '_next_id', 0) + 1
                matched = f"{lbl}_{self._next_id}"
            
            current[matched] = {"label":lbl, "cx":cx, "h":h, "last_h":h}
            if matched in self.history:
                last_h = self.history[matched]["h"]
                current[matched]["last_h"] = last_h
                growth = (h - last_h)/max(dt, 0.01)
                if growth > 5.0 and h > 40:
                    ttc = h / growth if growth > 0 else 999
                    if ttc < 3.0:
                        critical.append(det)
                        
        self.history = current
        return critical


class TrafficDecisionEngine:
    def __init__(self, threaded_detector):
        self.threaded_detector = threaded_detector
        self.state = "SYS_GO"
        self.reason = "CLEAR"
        self.stop_timer = 0.0
        self.stop_cd = 0.0
        self.last_t = time.time()
        self.tl_fsm = TrafficLightStateMachine()
        self.col_pred = CollisionPredictor()
        
    def _is_glowing(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if y2-y1 < 15 or x2-x1 < 10: return "NONE", 0
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_r1 = cv2.inRange(hsv, np.array([0,50,150]), np.array([10,255,255]))
        mask_r2 = cv2.inRange(hsv, np.array([170,50,150]), np.array([180,255,255]))
        mask_g = cv2.inRange(hsv, np.array([50,50,150]), np.array([90,255,255]))
        red = cv2.countNonZero(cv2.bitwise_or(mask_r1, mask_r2))
        green = cv2.countNonZero(mask_g)
        MIN_MASS = 30
        if red > MIN_MASS and red > green: return "RED", red
        if green > MIN_MASS: return "GREEN", green
        return "NONE", max(red, green)

    def process(self, frame):
        h, w = frame.shape[:2]
        dbg = frame.copy()
        now = time.time()
        dt = now - self.last_t
        self.last_t = now
        
        self.threaded_detector.update_frame(frame)
        dets = self.threaded_detector.get_detections()
        
        crit = self.col_pred.update_and_predict(dets, dt)
        # Priority 0: Active Priority Timers (e.g. Stop sign cooldown, intersection clear delay)
        if self.stop_timer > 0.0:
            if now - self.stop_timer >= 3.0:
                self.stop_timer, self.stop_cd = 0.0, now + 5.0
                if self.state == "SYS_STOP" and "STOP" in self.reason:
                    self.state, self.reason = "SYS_GO", "STOP CLEARED"
        
        # Build priority queue of critical new observations
        pri = 99
        p_state = "SYS_GO"
        p_res = "CLEAR PATH"
        
        def commit(pr, st, rs):
            nonlocal pri, p_state, p_res
            if pr < pri: pri, p_state, p_res = pr, st, rs
            
        if crit: commit(1, "SYS_STOP", "COLLISION IMMINENT")
        
        light_st = "NONE"
        act_lbl = []
        
        for d in dets:
            lbl = d["label"]
            x1,y1,x2,y2 = d["bbox"]
            cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(dbg, lbl, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            act_lbl.append(lbl)
            
            box_h = y2-y1
            if "traffic" in lbl.lower() and "light" in lbl.lower():
                clr, mass = self._is_glowing(frame, x1, y1, x2, y2)
                dist = "UNKNOWN"
                if box_h < 40: dist = "FAR"
                elif box_h < 70: dist = "APPROACH"
                else: dist = "HALT"
                
                fsm_st = self.tl_fsm.update(clr=="RED", clr=="GREEN", dist)
                
                # BUG 6: FSM has wrong state name here. Should be LIGHT_DETECTED_FAR
                if fsm_st == "LIGHT_DETECTED_FAR":
                    light_st = "[RED] APPROACH"
                    commit(4, "SYS_SLOW", "RED LIGHT AHEAD")
                elif fsm_st in ["LIGHT_RED_STOPPING", "LIGHT_RED_STOPPED"]:
                    light_st = "[RED] HALT"
                    commit(1, "SYS_STOP", "RED LIGHT CAUGHT")
                elif fsm_st == "LIGHT_GREEN_GO":
                    light_st = "[GREEN] CLEAR"
            else:
                if box_h < 90: continue
                if lbl == "stop-sign":
                    if now > self.stop_cd:
                        if self.stop_timer == 0.0: self.stop_timer = now
                        commit(2, "SYS_STOP", "STOP SIGN")
                elif lbl == "crosswalk-sign": commit(4, "SYS_SLOW", "CROSSWALK ZONE")
                elif "speed-limit" in lbl: commit(4, "SYS_LIMIT", "SPEED LIMIT ZONE")
                elif lbl in ["car", "closed-road-stand"]:
                    if (x1 < w*0.8) and (x2 > w*0.2) and y2 > h*0.6:
                        commit(3, "SYS_LANE_CHANGE_LEFT", "EVADING OBSTACLE")

        # If an active stop block is holding us, ignore lesser detections
        if self.stop_timer > 0.0 and now - self.stop_timer < 3.0:
            p_state = "SYS_STOP"
            p_res = "STOP SIGN"

        # If a red light is observed, it overrides everything including cooldowns
        if "RED" in light_st and p_state != "SYS_STOP":
            p_state = "SYS_STOP"
            p_res = "TRAFFIC LIGHT RED"
                    
        self.state, self.reason = p_state, p_res
        
        mult = 1.0
        if self.state == "SYS_STOP": mult = 0.0
        elif self.state in ["SYS_SLOW", "SYS_LANE_CHANGE_LEFT"]: mult = 0.6
        elif self.state == "SYS_LIMIT": mult = 0.75
        
        return TrafficResult(
            state=self.state, 
            reason=self.reason, 
            speed_multiplier=mult, 
            yolo_debug_frame=dbg, 
            light_status=light_st, 
            active_labels=act_lbl
        )
