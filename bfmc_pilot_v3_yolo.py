"""
BFMC Hybrid Pilot - Version 2 (Modular Architecture for Raw Camera Sharing)
=============================================================================
Improvements over base V2:
  - ARCH: Deep Modularization based on User Request.
  - ARCH: Single raw camera frame captured and dispatched to perception modules.
  - MOD: TrafficDecisionModule handles YOLO object/signal detection on the raw frame.
  - MOD: LanePerceptionModule handles the BEV transform and delegates to HybridLaneTracker.
  - ARCH: Steering and speed logic decoupled cleanly in BFMC_Pilot.
"""

import cv2
import numpy as np
import math
import time
import logging
import argparse
import sys

from ultralytics import YOLO
from map_parser import GlobalMap
from localization_engine import LocalizationEngine
from trajectory_planner import MapTrajectoryPlanner
import tkinter as tk
from tkinter_dashboard import BFMCDashboardApp
import threading

class PreTrainedYoloDetector:
    def __init__(self, global_model="best.pt"):
        """
        Initializes a single-stage YOLO pipeline.
        - global_model: The original Bosch model (detects cars, stop signs, pedestrians, track bounds)
        """
        print(f"Loading Global YOLO Scanner '{global_model}'...")
        self.global_model = YOLO(global_model)

    def detect_traffic_signals(self, frame_bgr, conf_threshold=0.3):
        # 1. RUN GLOBAL SCAN (Finds signs, cars, and the base structure of the traffic light)
        results = self.global_model.predict(
            source=frame_bgr, 
            conf=conf_threshold, 
            verbose=False
        )
        
        filtered_detections = []
        result = results[0]
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = self.global_model.names[cls_id]
            
            # Append standard global detections (Stop Sign, Car, Crosswalk, etc)
            filtered_detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })
            
        return filtered_detections

# ---------------------------------------------------------------------------
# Serial handler - graceful fallback
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, "..")          # allow running from the sub-folder
    from serial_handler import STM32_SerialHandler
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    print("WARNING: serial_handler not found - running in simulation mode")

    class STM32_SerialHandler:
        def connect(self):      return False
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def disconnect(self):   pass

# ---------------------------------------------------------------------------
# Camera - graceful fallback
# ---------------------------------------------------------------------------
_CAM_AVAILABLE = False
try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    print("WARNING: picamera2 not found - camera disabled")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ===========================================================================
# PHYSICAL CONSTANTS  (measure your car carefully)
# ===========================================================================
WHEELBASE_M          = 0.23    # front-to-rear axle distance (m)
LANE_WIDTH_M         = 0.35    # one-lane physical width (m)

# ===========================================================================
# CAMERA - Bird's Eye View calibration
# ===========================================================================
# SRC: [TL, TR, BL, BR] - Reverted to original dimensions
SRC_PTS = np.float32([[200, 260], [440, 260], [40,  450], [600, 450]])
DST_PTS = np.float32([[150,   0], [490,   0], [150, 480], [490, 480]])

# ===========================================================================
# RIGHT-LANE FINE TUNING DEFAULT
# ===========================================================================
DUAL_OFFSET_PX       = 0
SINGLE_DIV_OFFSET_PX = 40
SINGLE_EDGE_OFFSET_PX = -40

# ===========================================================================
# TIMING & LOSS RECOVERY
# ===========================================================================
TARGET_FPS    = 30
FRAME_PERIOD  = 1.0 / TARGET_FPS   # seconds
LOST_GRACE_FRAMES = 8


import threading
import queue

# ===========================================================================
# PERFORMANCE: THREADED YOLO DETECTION
# ===========================================================================
class ThreadedYOLODetector:
    def __init__(self, detector):
        self.detector = detector
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = True
        self.active_detections = []
        
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # ROI Cropping for Speed (skip clouds/sky and ego car hood)
                # Traffic signs and lights are mostly in top 60%
                # Obstacles are mostly below horizon
                roi_frame = frame.copy()
                
                # LOWERED: 0.25 confidence to ensure we never miss faint LEDs
                detections = self.detector.detect_traffic_signals(roi_frame, conf_threshold=0.25)
                
                if not self.result_queue.full():
                    self.result_queue.put(detections)
            except queue.Empty:
                pass
            except Exception as e:
                log.error(f"YOLO Thread Error: {e}")

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

# ===========================================================================
# PERCEPTION: TRAFFIC DECISIONS (Operating on RAW Frame)
# ===========================================================================
class TrafficDecisionModule:
    """ Handles decision state logic based on raw camera YOLO bounding boxes. """
    def __init__(self, threaded_detector):
        self.threaded_detector = threaded_detector
        self.state = "SYS_GO" 
        self.reason = ""
        self.stop_sign_timer = 0.0
        self.stop_sign_cooldown = 0.0
        self.halt_duration = 3.0 
        self.cooldown_duration = 5.0 
        self.active_detections = []
        self.tl_fsm = TrafficLightStateMachine()
        self.collision_predictor = CollisionPredictor()
        self.last_process_time = time.time()
        
    def _is_light_glowing(self, frame, x1, y1, x2, y2):
        """
        Uses Classical OpenCV HSV masking to find blooming LEDs inside the traffic light bounding box.
        Returns 'RED', 'GREEN', or 'NONE' based on absolute pixel mass (to eliminate far away lights).
        """
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_h, box_w = y2 - y1, x2 - x1
        if box_h < 15 or box_w < 10: return "NONE"
        
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Check for glowing Red or Green. S > 50 avoids white/grey, V > 150 ensures brightness
        mask_red1  = cv2.inRange(hsv, np.array([0, 50, 150]), np.array([10, 255, 255]))
        mask_red2  = cv2.inRange(hsv, np.array([170, 50, 150]), np.array([180, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([50, 50, 150]), np.array([90, 255, 255]))
        
        red_mask   = cv2.bitwise_or(mask_red1, mask_red2)
        
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(mask_green)
        
        # Absolute Mass Thresholds: Eliminates "small red color or far away red color"
        # Since a 10x10 LED is 100 pixels, 150 was way too high. Lowered to 30 pixels.
        MIN_GLOW_MASS = 30 
        
        if red_pixels > MIN_GLOW_MASS and red_pixels > green_pixels:
            return "RED", red_pixels
        elif green_pixels > MIN_GLOW_MASS:
            return "GREEN", green_pixels
            
        return "NONE", max(red_pixels, green_pixels)

    def _is_light_red(self, label):
        l = label.lower()
        return "red" in l

    def _is_light_green(self, label):
        l = label.lower()
        return "green" in l

    def _is_light_yellow(self, label):
        l = label.lower()
        return "yellow" in l or "orange" in l

    def _is_obstacle_in_path(self, x1, y1, x2, y2, frame_w, frame_h):
        # BUG 8: Check if ANY part of bbox overlaps path region instead of just center
        in_horizontal_path = (x1 < frame_w * 0.80) and (x2 > frame_w * 0.20)
        is_close = y2 > (frame_h * 0.60)
        return in_horizontal_path and is_close

    def process_raw_frame(self, raw_frame):
        h, w = raw_frame.shape[:2]
        yolo_dbg = raw_frame.copy()
        now = time.time()
        
        # Dispatch to worker thread, fetch latest result asynchronously
        self.threaded_detector.update_frame(raw_frame)
        self.active_detections = self.threaded_detector.get_detections()
        
        light_status = "NONE"
        active_labels = []
        
        # Priority System: Lower number = Higher Priority.
        # 1: Hard STOP (Red Light / Obstacle in Crosswalk)
        # 2: Timed STOP (Stop Sign)
        # 3: EVASION (Lane Change)
        # 4: SLOW (Crosswalk / Speed Limit / Yellow Light)
        # 99: GO (Default)
        highest_priority = 99
        proposed_state = "SYS_GO"
        proposed_reason = "CLEAR PATH"
        
        def commit_state(priority, state, reason):
            nonlocal highest_priority, proposed_state, proposed_reason
            if priority < highest_priority:
                highest_priority = priority
                proposed_state = state
                proposed_reason = reason

        dt = now - self.last_process_time
        self.last_process_time = now
        
        critical_obs = self.collision_predictor.update_and_predict(self.active_detections, dt)
        if critical_obs:
            commit_state(1, "SYS_STOP", f"PREDICTIVE COLLISION: {critical_obs[0]['label']}")
            
        for det in self.active_detections:
            label, (x1, y1, x2, y2), conf = det["label"], det["bbox"], det["confidence"]
            # --- 0. NO PRE-FILTER FOR TRAFFIC LIGHTS ---
            # (Deleted glowing pre-filter since it causes false negatives)
            box_h = y2 - y1
            active_labels.append(label)
            
            # --- 1. ALWAYS DRAW DETECTIONS ---
            color = (0, 255, 0)
            if label in ["stop-sign", "no-entry-road-sign"]: color = (0, 0, 255)
            elif any(c in label.lower() for c in ["red", "green", "yellow", "traffic"]): color = (0, 255, 255)
            elif label in ["car", "pedestrian", "closed-road-stand"]: color = (255, 0, 255)
            elif "speed-limit" in label: color = (255, 255, 0)
            elif label in ["crosswalk-sign", "parking-sign", "highway-sign", "priority-sign"]: color = (255, 128, 0)
                
            cv2.rectangle(yolo_dbg, (x1, y1), (x2, y2), color, 2)
            cv2.putText(yolo_dbg, f"{label} {conf:.2f} [{box_h}px]", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # --- 2. LOGICAL PROXIMITY RULES ---
            label_lower = label.lower()
            if "traffic" in label_lower and "light" in label_lower:
                # We revert to classical OpenCV HSV math. Look inside the giant plastic
                # YOLO Traffic Light bounding box for actual RED/GREEN blooming pixels.
                color_state, pixel_count = self._is_light_glowing(raw_frame, x1, y1, x2, y2)
                
                is_red = (color_state == "RED")
                is_green = (color_state == "GREEN")
                
                # We have reverted to checking distance! Because the Secondary LED 
                # model inherits the giant bounding box from the Primary Bosch model, 
                # we can accurately scale distance again based on the plastic base.
                distance_cat = "UNKNOWN"
                if box_h < 40: distance_cat = "FAR"
                elif box_h < 70: distance_cat = "APPROACH"
                elif box_h >= 70: distance_cat = "HALT"
                
                current_tl_state = self.tl_fsm.update(is_red, is_green, distance_cat)
                print(f"TL DETECTION: color='{color_state}' pixels={pixel_count} dist={distance_cat} fsm={current_tl_state}")
                
                # React based on state machine
                if current_tl_state == "LIGHT_APPROACHING":
                    light_status = "[RED] APPROACH"
                    commit_state(4, "SYS_SLOW", "RED LIGHT AHEAD")
                elif current_tl_state in ["LIGHT_RED_STOPPING", "LIGHT_RED_STOPPED"]:
                    light_status = "[RED] HALT" 
                    commit_state(1, "SYS_STOP", "RED LIGHT (PRIORITY)")
                elif current_tl_state == "LIGHT_GREEN_GO":
                    light_status = "[GREEN] CLEAR"
                    commit_state(99, "SYS_GO", "GREEN LIGHT CLEAR")
                else:
                    light_status = "[GREEN] CLEAR"
            
            else:
                if box_h < 90: # React to signs from a further distance
                    continue

                if label == "stop-sign":
                    # BUG 13: Silent during cooldown (It is still drawn in step 1, but we skip state change here if cooldown active)
                    if now > self.stop_sign_cooldown:
                        if self.stop_sign_timer == 0.0:
                            self.stop_sign_timer = now
                        commit_state(2, "SYS_STOP", "STOP SIGN")
                elif label in ["car", "closed-road-stand", "no-entry-road-sign"] and self._is_obstacle_in_path(x1, y1, x2, y2, w, h):
                    # BUG 12: Use already built active_labels
                    if "crosswalk-sign" in active_labels:
                        commit_state(1, "SYS_STOP", f"OBSTACLE AT CROSSWALK")
                    else:
                        commit_state(3, "SYS_LANE_CHANGE_LEFT", f"EVADING ({label})")
                elif label == "pedestrian" and self._is_obstacle_in_path(x1, y1, x2, y2, w, h):
                    commit_state(1, "SYS_STOP", "PEDESTRIAN IN PATH")
                elif label == "crosswalk-sign":
                    commit_state(4, "SYS_SLOW", "CROSSWALK ZONE")
                elif "speed-limit" in label:
                    commit_state(4, "SYS_LIMIT", f"SPEED LIMIT ZONE")
                elif label in ["parking-sign", "highway-sign", "priority-sign"]:
                    cv2.putText(yolo_dbg, f"INFO: {label}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        # Stop sign logic handles timers overriding immediate frame detections
        if self.stop_sign_timer > 0.0:
            if now - self.stop_sign_timer < self.halt_duration:
                # Still waiting
                pass
            else:
                self.stop_sign_timer, self.stop_sign_cooldown = 0.0, now + self.cooldown_duration
                if "STOP SIGN" in proposed_reason and proposed_state == "SYS_STOP":
                    self.state, self.reason = "SYS_GO", "STOP SIGN (CLEARED)"

        print(f"[TL_DEBUG] proposed_state={proposed_state} proposed_reason={proposed_reason} stop_timer={self.stop_sign_timer:.2f}")
        # Commit the mathematically highest priority state found in the frame
        self.state = proposed_state
        self.reason = proposed_reason

        cv2.putText(yolo_dbg, f"TRAFFIC: {self.state} | {self.reason}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if self.state == "SYS_STOP" else (0,255,0), 3)
        return self.state, self.get_speed_multiplier(), light_status, active_labels, yolo_dbg
        
    def get_speed_multiplier(self):
        if self.state == "SYS_STOP": return 0.0
        elif self.state == "SYS_SLOW": return 0.60 
        elif self.state == "SYS_LANE_CHANGE_LEFT": return 0.60 # Slow down during the swerve
        elif self.state == "SYS_LIMIT": return 0.75
        return 1.0


class LanePositionController:
    def __init__(self):
        self.target_position = 0.5  # Center of lane (normalized 0-1)
        self.kP = 0.3  # Proportional gain
        self.kD = 0.1  # Derivative gain
        self.last_error = 0.0

    def compute_correction(self, left_fit, right_fit, current_y):
        """
        Compute steering correction to maintain center
        """
        if left_fit is None or right_fit is None:
            return 0.0
        
        lx = np.polyval(left_fit, current_y)
        rx = np.polyval(right_fit, current_y)
        lane_width = rx - lx
        
        # Car position relative to lane (0=left edge, 1=right edge)
        car_x = 320  # Assume car at image center
        current_position = (car_x - lx) / lane_width
        
        # Error from target (0.5 = center)
        error = self.target_position - current_position
        
        # Derivative term (rate of change)
        d_error = error - self.last_error
        self.last_error = error
        
        # PID output in pixels
        correction = (self.kP * error + self.kD * d_error) * lane_width
        
        return correction

class TrafficLightStateMachine:
    def __init__(self):
        self.state = "NO_LIGHT"
        self.frames_in_state = 0
        self.frames_stopping = 0
        self.last_seen_red = 0.0
        
    def update(self, is_red_detected, is_green_detected, distance_category):
        # Extremely robust and forgiving FSM for small YOLO boxes
        if is_red_detected:
            self.last_seen_red = time.time()
            if distance_category == "HALT":
                self.state = "LIGHT_RED_STOPPED"
            elif distance_category == "APPROACH":
                if self.state not in ["LIGHT_RED_STOPPED"]:
                    self.state = "LIGHT_RED_STOPPING"
            elif self.state == "NO_LIGHT":
                self.state = "LIGHT_DETECTED_FAR"
                
        elif is_green_detected:
            if self.state in ["LIGHT_RED_STOPPED", "LIGHT_RED_STOPPING"]:
                if time.time() - self.last_seen_red > 1.0: # 1 sec delay before taking off
                    self.state = "LIGHT_GREEN_GO"
            else:
                self.state = "LIGHT_GREEN_GO"
                
        else:
            # Grace period for dropped YOLO frames (holds the red light if camera blinks)
            if self.state in ["LIGHT_RED_STOPPED", "LIGHT_RED_STOPPING"] and (time.time() - self.last_seen_red > 4.0):
                self.state = "NO_LIGHT"
            elif self.state not in ["LIGHT_RED_STOPPED", "LIGHT_RED_STOPPING"]:
                self.state = "NO_LIGHT"

        return self.state

class CollisionPredictor:
    def __init__(self):
        self.history = {} # track by label and rough position
        
    def update_and_predict(self, detections, dt):
        # Extremely simplified TTC (Time To Collision) 
        # using bounding box height expansion rate
        critical_obstacles = []
        current_seen = {}
        
        for det in detections:
            label, (x1, y1, x2, y2), conf = det["label"], det["bbox"], det["confidence"]
            if label not in ["car", "pedestrian", "closed-road-stand"]:
                continue
                
            box_h = y2 - y1
            center_x = (x1 + x2) / 2
            
            # Match with history (simple center spatial matching)
            matched_id = None
            for track_id, track_data in self.history.items():
                if track_data["label"] == label and abs(track_data["cx"] - center_x) < 50:
                    matched_id = track_id
                    break
                    
            if matched_id is None:
                matched_id = f"{label}_{time.time()}"
                
            current_seen[matched_id] = {"label": label, "cx": center_x, "h": box_h, "last_h": box_h}
            
            if matched_id in self.history:
                last_h = self.history[matched_id]["h"]
                current_seen[matched_id]["last_h"] = last_h
                
                # If box is growing, it's approaching
                growth_rate = (box_h - last_h) / max(dt, 0.01)
                if growth_rate > 5.0 and box_h > 40: # Growing fast and reasonably close
                    estimated_ttc = box_h / growth_rate if growth_rate > 0 else 999
                    if estimated_ttc < 3.0: # Critical threshold
                        critical_obstacles.append({"label": label, "ttc": estimated_ttc, "bbox": (x1, y1, x2, y2)})
                        
        self.history = current_seen
        return critical_obstacles

class TrajectoryPlanner:
    def __init__(self):
        self.evasion_offset = 0.0
        self.evasion_target = 0.0
        self.is_evading = False
        
    def compute_trajectory_offset(self, needs_evasion, lane_width_px, obstacle_center_x=None):
        # Smoothly transition evasion offset
        if needs_evasion:
            self.is_evading = True
            # BUG 9: Smart Evasion based on obstacle position
            if obstacle_center_x and obstacle_center_x > 320:
                self.evasion_target = -lane_width_px * 0.9 # Move left
            else:
                self.evasion_target = lane_width_px * 0.9 # Move right
        else:
            self.evasion_target = 0.0
            
        # EMA for smooth lane change
        self.evasion_offset = 0.1 * self.evasion_target + 0.9 * self.evasion_offset
        
        if abs(self.evasion_offset) < 5.0 and not needs_evasion:
            self.is_evading = False
            self.evasion_offset = 0.0
            
        return self.evasion_offset, self.is_evading

class AutonomousJunctionPlanner:
    def decide_junction_direction(self, warped_binary, left_fit, right_fit, lane_width_px):
        h, w = warped_binary.shape
        left_roi = warped_binary[0:240, 0:320]
        right_roi = warped_binary[0:240, 320:640]
        straight_roi = warped_binary[0:240, 200:440]
        
        weights = np.linspace(2.0, 0.5, 240).reshape(-1, 1)
        left_score = np.sum(left_roi * weights) / (320 * 240)
        right_score = np.sum(right_roi * weights) / (320 * 240)
        straight_score = np.sum(straight_roi * weights) / (240 * 240)
        
        scores = {"LEFT": left_score, "RIGHT": right_score, "STRAIGHT": straight_score}
        best_dir = max(scores, key=scores.get)
        
        total = sum(scores.values())
        confidence = scores[best_dir] / max(total, 1e-6)
        
        if confidence < 0.4: return "RIGHT", 0.3
        return best_dir, confidence

class DeadReckoningNavigator:
    def __init__(self):
        self.last_valid_target = 320.0
        self.last_valid_curvature = 0.0

    def predict_target(self, frames_lost, last_speed, last_steering):
        time_lost = frames_lost / max(TARGET_FPS, 1)
        lateral_drift = last_steering * 2.0 * time_lost
        predicted_target = self.last_valid_target + lateral_drift
        if abs(self.last_valid_curvature) > 0.001:
            predicted_target += self.last_valid_curvature * 5000 * time_lost
        predicted_target = np.clip(predicted_target, 150, 490)
        confidence = max(0.0, 1.0 - frames_lost / 30.0)
        return predicted_target, confidence

# ===========================================================================
# HYBRID LANE TRACKER V2 LOGIC
# ===========================================================================
class HybridLaneTracker:
    NWINDOWS         = 9
    SW_MARGIN        = 60
    MINPIX           = 50
    POLY_MARGIN_BASE = 60
    POLY_MARGIN_CURV = 120
    MIN_PIX_OK       = 200
    EMA_ALPHA        = 0.50
    STALE_FIT_FRAMES = 5

    def __init__(self, img_shape=(480, 640)):
        self.h, self.w = img_shape
        self.mode       = "SEARCH"
        self.left_fit   = None
        self.right_fit  = None
        self.sl         = None
        self.sr         = None
        self.left_conf  = 0
        self.right_conf = 0
        self.left_stale  = 0
        self.right_stale = 0
        self.dead_reckoner = DeadReckoningNavigator()
        self.estimated_lane_width = 280.0

    def update(self, warped_binary):
        nz  = warped_binary.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])

        if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
            curv = self.get_curvature(self.h // 2)
            li, ri, dbg = self._poly_search(warped_binary, nzx, nzy, curvature=curv)
            mode_label  = "POLY"
        else:
            li, ri, dbg = self._sliding_window(warped_binary, nzx, nzy)
            mode_label  = "SLIDE"

        self.left_conf  = len(li)
        self.right_conf = len(ri)
        has_l = self.left_conf  >= self.MIN_PIX_OK
        has_r = self.right_conf >= self.MIN_PIX_OK

        if has_l:
            fl = np.polyfit(nzy[li], nzx[li], 2)
            self.left_fit  = fl
            self.sl        = self._ema(self.sl, fl)
            self.left_stale = 0
        else:
            self.left_stale += 1
            if self.left_stale > self.STALE_FIT_FRAMES:
                self.left_fit = None
                self.sl       = None

        if has_r:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            self.right_fit  = fr
            self.sr         = self._ema(self.sr, fr)
            self.right_stale = 0
        else:
            self.right_stale += 1
            if self.right_stale > self.STALE_FIT_FRAMES:
                self.right_fit = None
                self.sr        = None

        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                if self.left_conf < self.right_conf:
                    self.left_fit  = None
                    self.sl        = None
                    self.left_stale = self.STALE_FIT_FRAMES
                    has_l          = False
                else:
                    self.right_fit  = None
                    self.sr         = None
                    self.right_stale = self.STALE_FIT_FRAMES
                    has_r           = False
            else:
                y_positions = [100, 200, 300, 400]
                widths = []
                for y in y_positions:
                    lx = np.polyval(self.sl, y)
                    rx = np.polyval(self.sr, y)
                    widths.append(rx - lx)
                weights = [4, 3, 2, 1]
                weighted_avg_width = np.average(widths, weights=weights)
                self.estimated_lane_width = 0.8 * self.estimated_lane_width + 0.2 * weighted_avg_width

        self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"
        return self.sl, self.sr, dbg, mode_label

    def get_target_x(self, y_eval, lane_width_px, extra_offset_px=0, nav_state="NORMAL", frames_lost=0, last_speed=0.0, last_steering=0.0):
        sl = self.sl
        sr = self.sr
        hw = lane_width_px / 2.0

        def ev(fit):
            return float(np.polyval(fit, y_eval))

        if nav_state == "ROUNDABOUT":
            if sl is not None: return ev(sl) + hw + extra_offset_px, "RBT_INNER"
            if sr is not None: return ev(sr) - hw + extra_offset_px, "RBT_OUTER"
            return None, "RBT_LOST"

        if nav_state.startswith("JUNCTION"):
            
            # Interactive Junction State: Look for user 'L' or 'R' choice
            base_x = 320.0
            
            if nav_state == "JUNCTION_RIGHT":
                # Brute force turn Right
                if sr is not None:
                    base_x = ev(sr) - (lane_width_px * 0.40)
                    anchor_type = "JCT_RIGHT_EDGE"
                elif sl is not None:
                    base_x = ev(sl) + (lane_width_px * 1.5)
                    anchor_type = "JCT_RIGHT_GHOST"
                else: 
                    base_x = 320.0 + (lane_width_px * 0.8)
                    anchor_type = "JCT_RIGHT_BLIND"
                    
            elif nav_state == "JUNCTION_LEFT":
                # Brute force turn Left
                if sl is not None:
                    base_x = ev(sl) + (lane_width_px * 0.40)
                    anchor_type = "JCT_LEFT_EDGE"
                elif sr is not None:
                    base_x = ev(sr) - (lane_width_px * 1.5)
                    anchor_type = "JCT_LEFT_GHOST"
                else:
                    base_x = 320.0 - (lane_width_px * 0.8)
                    anchor_type = "JCT_LEFT_BLIND"
                    
            else:
                # Default logic if no exact choice was processed (e.g. JUNCTION_PROMPT state)
                anchor_type = "JCT_WAITING_CHOICE"
                
            return base_x + extra_offset_px, anchor_type

        # ---------------------------------------------------------
        # BRUTE FORCE MIDDLE-LANE PRIORITY (NORMAL DRIVING)
        # ---------------------------------------------------------
        # The user requested PERFECT center-lane tracking.
        # We must never hug to the right or drift to the left.
        
        # BLIND CORNER FALLBACK
        if sl is None and sr is None:
            predicted_x, confidence = self.dead_reckoner.predict_target(
                frames_lost, last_speed, last_steering
            )
            return predicted_x + extra_offset_px, f"DEAD_RECKONING_{confidence:.2f}"
        
        # Scenario 1: Both lines perfectly visible. Center it mathematically.
        if sl is not None and sr is not None:
            base_x, anchor = (ev(sl) + ev(sr)) / 2.0, "CENTERED_DUAL"
            
        # Scenario 2: Only right line visible. Project center leftwards by hw.
        elif sr is not None:
            base_x, anchor = ev(sr) - hw, "CENTERED_FROM_RIGHT"
            
        # Scenario 3: Only left divider visible. Project center rightwards by hw.
        elif sl is not None:
            base_x, anchor = ev(sl) + hw, "CENTERED_FROM_LEFT"
            
        current_curvature = self.get_curvature(y_eval)
        # BUG 6: PID already handles centering. We leave the curvature calculation here 
        # to feed the topology memory, but we remove the manual `base_x += curvature_offset` 
        # translation to prevent fighting the LanePositionController later.
            
        self.dead_reckoner.last_valid_target = base_x
        self.dead_reckoner.last_valid_curvature = self.get_curvature(y_eval)
        return base_x + extra_offset_px, anchor

    def get_curvature(self, y_eval):
        fit = self.sr if self.sr is not None else self.sl
        if fit is None: return 0.0
        a, b = fit[0], fit[1]
        num   = abs(2.0 * a)
        denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return num / max(denom, 1e-6)

    def _sliding_window(self, warped, nzx, nzy):
        dbg  = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        hist = np.sum(warped[self.h // 2:, :], axis=0)

        mid    = int(self.w * 0.40)
        margin = self.SW_MARGIN

        lb = int(np.argmax(hist[margin : mid - margin])) + margin
        rb = int(np.argmax(hist[mid + margin : self.w - margin])) + mid + margin

        if abs(rb - lb) < 100:
            smoothed = np.convolve(hist.astype(float), np.ones(20) / 20, mode='same')
            p1 = int(np.argmax(smoothed))
            tmp = smoothed.copy()
            tmp[max(0, p1-40):min(self.w, p1+40)] = 0
            p2 = int(np.argmax(tmp))
            lb, rb = (min(p1, p2), max(p1, p2))

        wh = self.h // self.NWINDOWS
        lx, rx = lb, rb
        li, ri = [], []

        for win in range(self.NWINDOWS):
            y_lo, y_hi = self.h - (win + 1) * wh, self.h - win * wh
            xl0, xl1 = max(0, lx - self.SW_MARGIN), min(self.w, lx + self.SW_MARGIN)
            xr0, xr1 = max(0, rx - self.SW_MARGIN), min(self.w, rx + self.SW_MARGIN)

            cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 255, 0), 2)
            cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 255, 0), 2)

            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0)  & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0)  & (nzx < xr1)).nonzero()[0]

            li.append(gl); ri.append(gr)

            if len(gl) > self.MINPIX: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.MINPIX: rx = int(np.mean(nzx[gr]))

        li, ri = np.concatenate(li), np.concatenate(ri)
        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _poly_search(self, warped, nzx, nzy, curvature=0.0):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        m = (self.POLY_MARGIN_CURV if curvature > 0.0015 else self.POLY_MARGIN_BASE)

        def band(fit):
            cx = np.polyval(fit, nzy)
            return ((nzx > cx - m) & (nzx < cx + m)).nonzero()[0]

        li = band(self.sl) if self.sl is not None else np.array([], dtype=int)
        ri = band(self.sr) if self.sr is not None else np.array([], dtype=int)

        if len(li) < self.MIN_PIX_OK and len(ri) < self.MIN_PIX_OK:
            self.mode = "SEARCH"
            return self._sliding_window(warped, nzx, nzy)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _width_sane(self, lf, rf, y=400):
        w = np.polyval(rf, y) - np.polyval(lf, y)
        return 80 < w < 560

    def _ema(self, prev, new):
        if prev is None: return new.copy()
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev


# ===========================================================================
# PERCEPTION: LANE MODULAR WRAPPER
# ===========================================================================
class LanePerceptionModule:
    """Takes a raw camera frame, generates the BEV, and tracks the lanes."""
    def __init__(self, src_pts, dst_pts, h=480, w=640):
        self.M_forward = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.tracker = HybridLaneTracker(img_shape=(h, w))
        
    def process_raw_frame(self, raw_frame):
        # Resize high-res raw frame back to 640x480 for lane processing
        if raw_frame.shape[:2] != (480, 640):
            process_frame = cv2.resize(raw_frame, (640, 480))
        else:
            process_frame = raw_frame
            
        warped_colour = cv2.warpPerspective(process_frame, self.M_forward, (640, 480))
        # Use LAB color space, L channel (Lightness)
        lab = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2LAB)
        L = self.clahe.apply(lab[:, :, 0])
        
        # Adaptive Lighting Compensation
        # BUG 11: Gradual interpolation prevents harsh contrast snapping
        mean_l = np.mean(L)
        if mean_l < 100:
            a = 1.0 + (100 - mean_l) / 200
            b = (100 - mean_l) * 0.6
            L = cv2.convertScaleAbs(L, alpha=a, beta=int(b))
        elif mean_l > 180:
            a = 1.0 - (mean_l - 180) / 350
            b = -(mean_l - 180) * 0.4
            L = cv2.convertScaleAbs(L, alpha=a, beta=int(b))

        # Track is WHITE, lines are BLACK (dark spots).
        # We need to adaptively threshold looking for the darkest areas
        # cv2.THRESH_BINARY_INV will make the dark spots white (255)
        # We use a positive C value (+15) so it aggressively filters out shadows
        binary = cv2.adaptiveThreshold(
            L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 15)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        warped_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        sl, sr, line_dbg, mode_label = self.tracker.update(warped_binary)
        return warped_binary, sl, sr, line_dbg, mode_label


# ===========================================================================
# NAVIGATORS AND SAFETY
# ===========================================================================
class JunctionDetector:
    ENTRY_FRAMES       = 5
    EXIT_FRAMES        = 8
    RATIO_EARLY_WARN   = 1.7 # If upper BEV lane width is 1.7x normal, it's a junction approaching
    MIN_BOT_ENERGY     = 500

    def __init__(self):
        self.state, self.entry_count, self.exit_count, self.frames_in_jct = "NORMAL", 0, 0, 0
        self.user_choice = None
        self.prompt_timer = 0.0
        self.planner = AutonomousJunctionPlanner()

    def update(self, warped_binary, left_conf, right_conf, left_fit, right_fit, lane_width_px, active_labels):
        h, w = warped_binary.shape
        both_lost = (left_conf < 200) and (right_conf < 200)
        
        # Look far ahead (y=150 is the top 30% of the BEV frame)
        approaching_wide_gap = False
        if left_fit is not None and right_fit is not None:
            lx_far = np.polyval(left_fit,  150)
            rx_far = np.polyval(right_fit, 150)
            if (rx_far - lx_far) > lane_width_px * self.RATIO_EARLY_WARN: 
                approaching_wide_gap = True
        # BUG 7: Also check if single lane diverges significantly from expected position
        elif left_fit is not None:
            lx_far = np.polyval(left_fit, 150)
            if lx_far < max(0, 320 - lane_width_px * self.RATIO_EARLY_WARN):
                approaching_wide_gap = True
        elif right_fit is not None:
            rx_far = np.polyval(right_fit, 150)
            if rx_far > min(640, 320 + lane_width_px * self.RATIO_EARLY_WARN):
                approaching_wide_gap = True

        hist_bot = float(np.sum(warped_binary[h // 2:, :]))
        hist_top = float(np.sum(warped_binary[:h // 2, :]))
        cross_energy = False
        if hist_bot > self.MIN_BOT_ENERGY:
            # Standard intersection geometry
            cross_energy = (hist_top / hist_bot) > 1.4

        # BUG 6: A zebra crossing generates massive pixel energy in the BEV frame.
        # Only suppress cross_energy flag, not entire evidence calculation
        if "crosswalk-sign" in active_labels:
            cross_energy = False
            
        evidence = approaching_wide_gap or cross_energy or both_lost

        if self.state == "NORMAL":
            self.entry_count = self.entry_count + 1 if evidence else 0
            if self.entry_count >= self.ENTRY_FRAMES:
                # We detected a junction geometry! Pause and PROMPT user.
                self.state, self.exit_count, self.frames_in_jct = "JUNCTION_PROMPT", 0, 0
                self.user_choice = None
                self.prompt_timer = time.time() # BUG 4: Start timeout timer
                
        elif self.state == "JUNCTION_PROMPT":
            direction, confidence = self.planner.decide_junction_direction(
                warped_binary, left_fit, right_fit, lane_width_px
            )
            if confidence > 0.6:
                self.state = f"JUNCTION_{direction}"
                self.user_choice = direction
            else:
                self.state = "JUNCTION_RIGHT"
                self.user_choice = "RIGHT"
                
        elif self.state.startswith("JUNCTION_"):
            # We are actively executing a branch
            self.frames_in_jct += 1
            self.exit_count = self.exit_count + 1 if not evidence else 0
            
            # Ensure we've driven a minimum amount through the junction before trusting clear evidence
            if self.exit_count >= self.EXIT_FRAMES and self.frames_in_jct > 25:
                self.state, self.entry_count = "NORMAL", 0
                self.user_choice = None
                
        return self.state


class RoundaboutNavigator:
    ENTRY_WIDTH_RATIO  = 0.60
    EXIT_WIDTH_RATIO   = 0.82
    MIN_CIRCLE_FRAMES  = 25
    MAX_CIRCLE_FRAMES  = 120
    SPEED_SCALE        = 0.50
    LOOKAHEAD_SCALE    = 0.55

    def __init__(self):
        self.state, self.frames = "NORMAL", 0

    def update(self, left_fit, right_fit, lane_width_px, img_h=480):
        y = img_h - 50
        if left_fit is not None and right_fit is not None:
            lx = np.polyval(left_fit,  y)
            rx = np.polyval(right_fit, y)
            ratio = (rx - lx) / max(float(lane_width_px), 1.0)
            if self.state == "NORMAL":
                if ratio < self.ENTRY_WIDTH_RATIO:
                    self.state, self.frames = "ROUNDABOUT", 0
            elif self.state == "ROUNDABOUT":
                self.frames += 1
                normal_exit = (self.frames > self.MIN_CIRCLE_FRAMES and ratio > self.EXIT_WIDTH_RATIO)
                timeout_exit = self.frames > self.MAX_CIRCLE_FRAMES
                if normal_exit or timeout_exit:
                    self.state, self.frames = "NORMAL", 0
        return self.state


class DividerGuard:
    # 80px Lethal Zone: The car MUST NOT ever touch the center divider.
    DIVIDER_SAFE_PX = 110 
    EDGE_SAFE_PX    = 70
    GAIN            = 0.35 # Massive correction gain
    MAX_CORR        = 25.0 # Allowing enormous steering spikes for emergency saves
    DEADBAND_PX     = 2

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        # We want the car in the middle, but heavily penalize touching the center divider
        correction, speed_scale, triggered = 0.0, 1.0, False
        div_corr = 0.0
        
        # Left Divider (Center line) - Overwhelming penalty forcefield
        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err = float(self.DIVIDER_SAFE_PX - gap)
                # Triple the impact of the error to violently shove the car rightwards
                div_corr = min((self.GAIN * 3.0) * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.2, 1.0 - err / 60.0)) # Brake drastically during save
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
            # If trapped between both (super narrow lane), let the massive divider repulsion win
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered

# ===========================================================================
# TRACK TOPOLOGY MEMORY SYSTEM
# ===========================================================================
class TopologyMemory:
    def __init__(self):
        self.track_features = []
        self.current_distance = 0.0
        self.last_update_time = time.time()
        self.active_feature = "STRAIGHT"
        self.feature_start_dist = 0.0

    def update(self, speed, nav_state, curvature):
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        
        # Approximate distance traveled
        dist_delta = (speed * 0.01) * dt # Scale speed to m/s approximation
        self.current_distance += dist_delta

        # Detect feature type
        feature = "STRAIGHT"
        if nav_state == "ROUNDABOUT": feature = "ROUNDABOUT"
        elif nav_state.startswith("JUNCTION"): feature = "JUNCTION"
        elif curvature > 0.002: feature = "CURVE"
        
        if feature != self.active_feature:
            if self.active_feature != "STRAIGHT" or (self.current_distance - self.feature_start_dist) > 0.5:
                self.track_features.append({
                    "type": self.active_feature,
                    "start": self.feature_start_dist,
                    "end": self.current_distance
                })
            self.active_feature = feature
            self.feature_start_dist = self.current_distance
            
    def get_next_feature(self, lookahead=1.0):
        if len(self.track_features) < 5: return None
        track_length = self.track_features[-1]["end"]
        if track_length == 0: return None
        
        search_dist = (self.current_distance + lookahead) % track_length
        for f in self.track_features:
            if f["start"] <= search_dist <= f["end"]:
                return f["type"]
        return None

# ===========================================================================
# BEHAVIOR TREE DECISION SYSTEM
# ===========================================================================
class BehaviorStatus:
    SUCCESS, FAILURE, RUNNING = "SUCCESS", "FAILURE", "RUNNING"

class BehaviorNode:
    def tick(self, blackboard): raise NotImplementedError

class SequenceNode(BehaviorNode):
    def __init__(self, children): self.children = children
    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status != BehaviorStatus.SUCCESS: return status
        return BehaviorStatus.SUCCESS

class SelectorNode(BehaviorNode):
    def __init__(self, children): self.children = children
    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status != BehaviorStatus.FAILURE: return status
        return BehaviorStatus.FAILURE

class CheckTrafficLightNode(BehaviorNode):
    def tick(self, blackboard):
        if blackboard.get("traffic_state") == "SYS_STOP":
            blackboard["action"], blackboard["reason"] = "HALT", "RED LIGHT"
            return BehaviorStatus.SUCCESS
        elif blackboard.get("traffic_state") == "SYS_SLOW":
            blackboard["action"], blackboard["reason"] = "SLOW", "TRAFFIC LIGHT APPROACH"
            return BehaviorStatus.SUCCESS
        return BehaviorStatus.FAILURE

class CheckObstacleNode(BehaviorNode):
    def tick(self, blackboard):
        state = blackboard.get("traffic_state")
        if state == "SYS_STOP" and "OBSTACLE" in blackboard.get("traffic_reason", ""):
            blackboard["action"], blackboard["reason"] = "HALT", "OBSTACLE"
            return BehaviorStatus.SUCCESS
        elif state == "SYS_LANE_CHANGE_LEFT":
            blackboard["action"], blackboard["reason"] = "LANE_CHANGE", "EVADING"
            return BehaviorStatus.SUCCESS
        return BehaviorStatus.FAILURE

class CheckStopSignNode(BehaviorNode):
    def tick(self, blackboard):
        if blackboard.get("traffic_state") == "SYS_STOP" and blackboard.get("traffic_reason", "") == "STOP SIGN":
            blackboard["action"], blackboard["reason"] = "HALT", "STOP SIGN"
            return BehaviorStatus.SUCCESS
        return BehaviorStatus.FAILURE
        
class NavigateJunctionNode(BehaviorNode):
    def tick(self, blackboard):
        nav_state = blackboard.get("nav_state")
        if nav_state == "JUNCTION_PROMPT":
            blackboard["action"], blackboard["reason"] = "HALT", "JUNCTION_PROMPT"
            return BehaviorStatus.SUCCESS
        elif nav_state.startswith("JUNCTION"):
            blackboard["action"], blackboard["reason"] = "JUNCTION_NAV", nav_state
            return BehaviorStatus.SUCCESS
        return BehaviorStatus.FAILURE

class NormalDrivingNode(BehaviorNode):
    def tick(self, blackboard):
        blackboard["action"], blackboard["reason"] = "DRIVE", "NORMAL TRACING"
        return BehaviorStatus.SUCCESS

class BFMC_BehaviorTree:
    def __init__(self):
        self.root = SelectorNode([
            CheckTrafficLightNode(),
            CheckObstacleNode(),
            CheckStopSignNode(),
            NavigateJunctionNode(),
            NormalDrivingNode()
        ])
    def evaluate(self, blackboard):
        self.root.tick(blackboard)
        return blackboard.get("action", "DRIVE"), blackboard.get("reason", "")


# ===========================================================================
# DATA LOGGING AND DIAGNOSTICS
# ===========================================================================
import csv
import datetime

class ComprehensiveDataLogger:
    def __init__(self):
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"bfmc_telemetry_{self.session_id}.csv"
        self.headers = [
            "timestamp", "fps", "speed", "steering", "nav_state", 
            "traffic_state", "curvature", "degradation_level", "estop"
        ]
        
        try:
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            self.active = True
            log.info(f"Data Logger Initialized: {self.filename}")
        except Exception as e:
            log.error(f"Failed to initialize Data Logger: {e}")
            self.active = False
            
    def log_telemetry(self, fps, speed, steering, nav_state, traffic_state, curvature, degrad_level, estop):
        if not self.active: return
        
        try:
            with open(self.filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    round(time.time(), 3),
                    round(fps, 1),
                    round(speed, 2),
                    round(steering, 2),
                    nav_state,
                    traffic_state,
                    round(curvature, 4),
                    degrad_level,
                    1 if estop else 0
                ])
        except Exception as e:
            self.error_count = getattr(self, 'error_count', 0) + 1
            if self.error_count == 1:
                log.warning(f"Data logging failing (suppressing future errors): {e}")


# ===========================================================================
# SAFETY: HARDWARE WATCHDOG & GRACEFUL DEGRADATION
# ===========================================================================
class SystemWatchdog:
    def __init__(self, timeout_sec=2.0):
        self.timeout_sec = timeout_sec
        self.last_heartbeat = time.time()
        self.tripped = False
        
    def ping(self):
        self.last_heartbeat = time.time()
        self.tripped = False
        
    def check(self):
        if time.time() - self.last_heartbeat > self.timeout_sec:
            self.tripped = True
        return self.tripped

class DegradationManager:
    # Levels: 0=Perfect, 1=No YOLO, 2=Lane Blind (Deadreckon only), 3=FATAL STOP
    def __init__(self):
        self.degrad_level = 0
        # BUG 14: Relax watchdog limits for Raspberry Pi limits
        self.yolo_watchdog = SystemWatchdog(timeout_sec=5.0)
        self.cam_watchdog = SystemWatchdog(timeout_sec=3.0)
        
    def update(self, yolo_active, cam_active):
        if not cam_active or self.cam_watchdog.check():
            self.degrad_level = 3
            return self.degrad_level
            
        if not yolo_active or self.yolo_watchdog.check():
            self.degrad_level = max(self.degrad_level, 1)
        else:
            self.degrad_level = 0
            
        return self.degrad_level


# ===========================================================================
# MAIN PILOT ORCHESTRATOR
# ===========================================================================
class BFMC_Pilot:

    # -------------------------------------------------------------
    # AGGRESSIVE STEERING DYNAMICS (High Performance)
    # -------------------------------------------------------------
    STEER_EMA_SLOW = 0.40   # Accepts 60% of new steering signal instantly
    STEER_EMA_FAST = 0.10   # Accepts 90% of emergency steering instantly
    GUARD_EMA      = 0.30   # Apply divider guard saves rapidly
    MAX_STEER      = 45.0   # Expanded physical steering rack limit
    MAX_STEER_RATE = 15.0   # Allow violent 15-deg/frame snaps instead of sluggish 2-deg loops

    HIGH_CURV_THRESH = 0.0025 # Engage high-curve speed slow down earlier
    MED_CURV_THRESH  = 0.0010

    HIGH_CURV_SCALE  = 0.60
    MED_CURV_SCALE   = 0.80
    DUAL_SPEED_SCALE = 1.15

    def __init__(self, sim_mode=False, ui=None):
        self.sim_mode = sim_mode
        self.ui = ui
        self.running = True
        self.handler   = STM32_SerialHandler()
        self.connected = False if sim_mode else self.handler.connect()

        self.cam_ok = False
        if not sim_mode and _CAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                cfg = self.picam2.create_video_configuration(main={"size": (1280, 720), "format": "BGR888"})
                self.picam2.configure(cfg)
                self.picam2.start()
                self.cam_ok = True
            except Exception as e:
                log.warning(f"Camera init failed: {e} — using blank frames")

        # System Perceptual Modules
        self.lane_module = LanePerceptionModule(SRC_PTS, DST_PTS)
        
        try:
            raw_detector = PreTrainedYoloDetector("best.pt")  # Loading the newly trained YOLO Nano model
            self.threaded_yolo = ThreadedYOLODetector(raw_detector)
            self.traffic_module = TrafficDecisionModule(self.threaded_yolo)
        except Exception:
            self.traffic_module = None
            self.threaded_yolo = None
            print("[WARN] YOLO disabled. TrafficDecisionModule inactive.")

        # Sub-systems
        self.rbt     = RoundaboutNavigator()
        self.jct     = JunctionDetector()
        self.guard   = DividerGuard()

        # State
        self.smooth_steer  = 0.0
        self.smooth_guard  = 0.0
        self.prev_steer    = 0.0
        self.last_target   = 320.0 + 100.0  # Safe initial right-bias
        self.lost_frames   = 0
        self.lane_pos_controller = LanePositionController()
        self.topology_memory = TopologyMemory()
        self.behavior_tree = BFMC_BehaviorTree()
        self.safety_manager = DegradationManager()
        self.logger = ComprehensiveDataLogger()

        self._fps_t, self._fps = time.time(), 0.0

        # MAP AWARENESS INTEGRATION
        self.gmap = GlobalMap()
        self.global_pose = None # [x, y, yaw, spline_idx]
        self.localizer = LocalizationEngine()
        self.planner = MapTrajectoryPlanner(self.gmap, lookahead_meters=0.6)
        
        if self.ui:
            self.ui.gmap = self.gmap
            self.ui._draw_static_map_cache()

    def _pure_pursuit(self, target_x, look_ahead_px, lane_width_px):
        lane_width_px = max(lane_width_px, 50)
        ppm   = lane_width_px / LANE_WIDTH_M
        dx    = target_x - 320.0
        dy    = max(float(look_ahead_px), 1.0)
        ld    = math.sqrt(dx * dx + dy * dy)
        alpha = math.atan2(dx, dy)
        wb_px = WHEELBASE_M * ppm
        steer = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer)

    def _draw_poly(self, img, fit, colour):
        if fit is None: return
        ploty = np.linspace(0, 479, 240).astype(np.float32)
        xs    = np.polyval(fit, ploty).astype(np.float32)
        pts = np.stack([xs, ploty], axis=1).reshape(-1, 1, 2).astype(np.int32)
        pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, 639)
        cv2.polylines(img, [pts], isClosed=False, color=colour, thickness=3)

    # -------------------------------------------------------------
    # BRAND NEW SLEEK UI RENDERER (OpenCV Only)
    # -------------------------------------------------------------
    def _draw_rounded_panel(self, canvas, pt1, pt2, color, radius=15):
        """Draws a sleek, anti-aliased rounded rectangle for modern UI cards."""
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(canvas, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)

    def _draw_traffic_icon(self, canvas, label, cx, cy, size=22):
        """Draws modern, geometric traffic signs natively."""
        r = size
        lbl = label.lower()
        cv2.circle(canvas, (cx, cy), r+2, (30, 30, 30), -1, cv2.LINE_AA)
        
        if "stop" in lbl:
            pts = np.array([[cx-r//2, cy-r], [cx+r//2, cy-r], [cx+r, cy-r//2], [cx+r, cy+r//2], 
                            [cx+r//2, cy+r], [cx-r//2, cy+r], [cx-r, cy+r//2], [cx-r, cy-r//2]], np.int32)
            cv2.fillPoly(canvas, [pts], (40, 40, 220), cv2.LINE_AA)
            cv2.putText(canvas, "STOP", (cx-15, cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        elif "parking" in lbl:
            self._draw_rounded_panel(canvas, (cx-r, cy-r), (cx+r, cy+r), (220, 100, 30), radius=5)
            cv2.putText(canvas, "P", (cx-7, cy+7), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        elif "crosswalk" in lbl:
            cv2.circle(canvas, (cx, cy), r, (40, 200, 240), -1, cv2.LINE_AA)
            cv2.line(canvas, (cx-10, cy-4), (cx+10, cy-4), (20, 20, 20), 2, cv2.LINE_AA)
            cv2.line(canvas, (cx-10, cy+4), (cx+10, cy+4), (20, 20, 20), 2, cv2.LINE_AA)
        elif "priority" in lbl:
            pts = np.array([[cx, cy-r], [cx+r, cy], [cx, cy+r], [cx-r, cy]], np.int32)
            cv2.fillPoly(canvas, [pts], (40, 220, 240), cv2.LINE_AA)
            pts_inner = np.array([[cx, cy-r+6], [cx+r-6, cy], [cx, cy+r-6], [cx-r+6, cy]], np.int32)
            cv2.fillPoly(canvas, [pts_inner], (255, 255, 255), cv2.LINE_AA)
        elif "highway-entry" in lbl:
            self._draw_rounded_panel(canvas, (cx-r, cy-r), (cx+r, cy+r), (60, 180, 60), radius=5)
            cv2.putText(canvas, "HWY", (cx-15, cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        elif "highway-exit" in lbl:
            self._draw_rounded_panel(canvas, (cx-r, cy-r), (cx+r, cy+r), (60, 180, 60), radius=5)
            cv2.line(canvas, (cx-r+4, cy-r+4), (cx+r-4, cy+r-4), (40, 40, 220), 3, cv2.LINE_AA)
        elif "one-way" in lbl:
            self._draw_rounded_panel(canvas, (cx-r, cy-r), (cx+r, cy+r), (220, 100, 30), radius=5)
            cv2.arrowedLine(canvas, (cx-12, cy), (cx+12, cy), (255, 255, 255), 2, tipLength=0.4, line_type=cv2.LINE_AA)
        elif "round-about" in lbl:
            cv2.circle(canvas, (cx, cy), r, (220, 100, 30), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx, cy), (r-6, r-6), 0, 0, 270, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "^", (cx+4, cy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        elif "no-entry" in lbl:
            cv2.circle(canvas, (cx, cy), r, (40, 40, 220), -1, cv2.LINE_AA)
            cv2.line(canvas, (cx-12, cy), (cx+12, cy), (255, 255, 255), 4, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (cx, cy), r, (80, 80, 80), -1, cv2.LINE_AA)

    def _draw_telemetry_graph(self, canvas, x, y, w, h, data, title="LIVE TELEMETRY"):
        """Draws a real-time, glowing data plot overlay."""
        # Draw translucent background card
        overlay = canvas.copy()
        self._draw_rounded_panel(overlay, (x, y), (x+w, y+h), (16, 16, 18), radius=12)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        
        # Grid lines
        mid_y = y + h // 2
        cv2.line(canvas, (x+15, mid_y), (x+w-15, mid_y), (60, 60, 65), 1, cv2.LINE_AA) # Zero line
        cv2.line(canvas, (x+15, y+25), (x+w-15, y+25), (35, 35, 40), 1, cv2.LINE_AA)   # Upper bound
        cv2.line(canvas, (x+15, y+h-25), (x+w-15, y+h-25), (35, 35, 40), 1, cv2.LINE_AA) # Lower bound
        
        cv2.putText(canvas, title, (x+20, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 150), 1, cv2.LINE_AA)
        
        if not data or len(data) < 2: return
        
        # Plot data
        max_val = 45.0 # Max steering boundary
        pts = []
        step_x = (w - 30) / max(1, len(data) - 1)
        
        for i, val in enumerate(data):
            px = int(x + 15 + i * step_x)
            py = int(mid_y - (val / max_val) * (h//2 - 25)) # Invert Y so right/positive is up
            pts.append([px, py])
            
        pts = np.array(pts, np.int32)
        
        # Glowing line effect (Outer thick, inner thin)
        cv2.polylines(canvas, [pts], False, (10, 80, 200), 4, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], False, (40, 140, 255), 2, cv2.LINE_AA)
        
        # Live tracking dot
        live_x, live_y = pts[-1][0], pts[-1][1]
        cv2.circle(canvas, (live_x, live_y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"{data[-1]:+.1f}", (live_x - 15, live_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 250), 1, cv2.LINE_AA)

    def _render_dashboard(self, yolo_hd, lane_dbg, speed, steer_angle, traffic_state, traffic_reason, light_status, nav_state, anchor, batt_pct, active_labels, topology):
        # Initialize dynamic telemetry buffer if it doesn't exist
        if not hasattr(self, '_steer_history'): self._steer_history = []
        self._steer_history.append(steer_angle)
        if len(self._steer_history) > 120: self._steer_history.pop(0) # Keep last ~4 seconds

        # Base Canvas: Ultra-Deep Obsidian
        canvas = np.full((720, 1920, 3), (12, 12, 12), dtype=np.uint8)
        
        CYAN = (255, 200, 0)       
        ORANGE = (40, 140, 255)    
        RED = (60, 40, 250)        
        GREEN = (100, 220, 100)    
        PANEL_BG = (22, 22, 24)    
        MUTED = (120, 120, 125)    
        TEXT_PRIME = (245, 245, 250)

        # ---------------------------------------------------------
        # PANEL 1: DRIVE KINEMATICS (Left)
        # ---------------------------------------------------------
        self._draw_rounded_panel(canvas, (30, 30), (580, 690), PANEL_BG, radius=20)
        
        cv2.putText(canvas, "SYSTEM STATUS", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)
        t_col = RED if traffic_state == "SYS_STOP" else GREEN if traffic_state == "SYS_GO" else CYAN
        cv2.putText(canvas, traffic_state.replace("SYS_", ""), (60, 110), cv2.FONT_HERSHEY_DUPLEX, 1.2, t_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, traffic_reason.upper(), (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_PRIME, 1, cv2.LINE_AA)

        # Speedometer
        spd_center = (305, 330)
        cv2.ellipse(canvas, spd_center, (130, 130), 135, 0, 270, (40, 40, 42), 3, cv2.LINE_AA)
        spd_ratio = min(1.0, abs(speed) / 120.0)
        cv2.ellipse(canvas, spd_center, (130, 130), 135, 0, int(270 * spd_ratio), CYAN, 5, cv2.LINE_AA)
        
        spd_str = f"{int(abs(speed))}"
        tw, _ = cv2.getTextSize(spd_str, cv2.FONT_HERSHEY_DUPLEX, 3.5, 2)[0]
        cv2.putText(canvas, spd_str, (spd_center[0] - tw//2, spd_center[1] + 20), cv2.FONT_HERSHEY_DUPLEX, 3.5, TEXT_PRIME, 2, cv2.LINE_AA)
        cv2.putText(canvas, "cm/s", (spd_center[0] - 25, spd_center[1] + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1, cv2.LINE_AA)

        # Steering
        steer_center = (305, 550)
        cv2.ellipse(canvas, steer_center, (70, 70), 270, -45, 45, (40, 40, 42), 3, cv2.LINE_AA)
        cv2.ellipse(canvas, steer_center, (70, 70), 270, 0, int(steer_angle), ORANGE, 5, cv2.LINE_AA)
        cv2.putText(canvas, f"{steer_angle:+.1f} DEG", (255, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_PRIME, 1, cv2.LINE_AA)

        # ---------------------------------------------------------
        # PANEL 2: SPATIAL INTELLIGENCE & TELEMETRY (Center)
        # ---------------------------------------------------------
        self._draw_rounded_panel(canvas, (610, 30), (1340, 690), PANEL_BG, radius=20)
        
        if hasattr(self, 'map_layer_base'):
            map_mask = np.zeros((640, 710), dtype=np.uint8)
            cv2.circle(map_mask, (355, 320), 300, 255, -1, cv2.LINE_AA)
            
            map_render = self.map_layer_base.copy()
            if self.global_pose is not None:
                px = int(self.global_pose[0] * self.map_scale + self.map_offset[0])
                py = int(self.global_pose[1] * self.map_scale + self.map_offset[1])
                cv2.circle(map_render, (px, py), 8, CYAN, -1, cv2.LINE_AA)
                cv2.circle(map_render, (px, py), 16, (255, 200, 0, 50), -1, cv2.LINE_AA) 
                hx, hy = int(px + math.cos(self.global_pose[2]) * 20), int(py + math.sin(self.global_pose[2]) * 20)
                cv2.arrowedLine(map_render, (px, py), (hx, hy), TEXT_PRIME, 3, tipLength=0.3, line_type=cv2.LINE_AA)

            map_resized = cv2.resize(map_render, (710, 640))
            map_blended = cv2.bitwise_and(map_resized, map_resized, mask=map_mask)
            
            roi = canvas[40:680, 620:1330]
            canvas[40:680, 620:1330] = cv2.addWeighted(roi, 0.1, map_blended, 0.9, 0)
            
        cv2.putText(canvas, "ENVIRONMENT TOPOLOGY", (640, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, nav_state.replace("_", " "), (640, 105), cv2.FONT_HERSHEY_DUPLEX, 0.8, CYAN, 1, cv2.LINE_AA)

        # Overlay Live Graph inside Map Panel
        self._draw_telemetry_graph(canvas, 640, 520, 670, 140, self._steer_history, title="STEERING ERROR KINEMATICS")

        # ---------------------------------------------------------
        # PANEL 3: PERCEPTION SENSORS (Right)
        # ---------------------------------------------------------
        self._draw_rounded_panel(canvas, (1370, 30), (1890, 690), PANEL_BG, radius=20)
        
        # Camera Feed
        cam_h, cam_w = 281, 460
        cam_y, cam_x = 60, 1400
        canvas[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = cv2.resize(yolo_hd, (cam_w, cam_h))
        cv2.rectangle(canvas, (cam_x-1, cam_y-1), (cam_x+cam_w+1, cam_y+cam_h+1), (60, 60, 65), 1, cv2.LINE_AA)

        # Active Signals
        cv2.putText(canvas, "ACTIVE SIGNALS", (1400, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)
        icon_start_x = 1430
        icon_start_y = 430
        for idx, lbl in enumerate(active_labels):
            if idx > 7: break 
            row, col = idx // 4, idx % 4
            self._draw_traffic_icon(canvas, lbl, icon_start_x + (col * 65), icon_start_y + (row * 65), size=24)

        # BEV Radar
        bev_h, bev_w = 160, 200
        bev_y, bev_x = 510, 1400
        bev_color = cv2.cvtColor(cv2.resize(lane_dbg, (bev_w, bev_h)), cv2.COLOR_BGR2GRAY)
        bev_color = cv2.cvtColor(bev_color, cv2.COLOR_GRAY2BGR)
        canvas[bev_y:bev_y+bev_h, bev_x:bev_x+bev_w] = bev_color
        cv2.rectangle(canvas, (bev_x-1, bev_y-1), (bev_x+bev_w+1, bev_y+bev_h+1), (60, 60, 65), 1, cv2.LINE_AA)
        
        # Diagnostics
        cv2.putText(canvas, "TRACKING ANCHOR", (1630, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, anchor, (1630, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "LANE TOPOLOGY", (1630, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, topology, (1630, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1, cv2.LINE_AA)

        return canvas

    def _update_fps(self):
        now = time.time()
        dt  = now - self._fps_t
        self._fps_t = now
        # BUG 20: Change to lighter EMA:
        self._fps = 0.7 * self._fps + 0.3 * (1.0 / max(dt, 1e-3))

    def run(self):
        print("BFMC Pilot v2: STARTING MODULAR ORCHESTRATOR")
        startup_time = time.time()
        sim_batt_pct = 99.9  # Fake battery to look cool
        try:
            while self.running:
                t_frame_start = time.time()
                
                if getattr(self, '_manual_estop', False) and (time.time() - getattr(self, '_estop_timestamp', time.time()) > 3.0):
                    self._manual_estop = False
                    print("E-STOP AUTO-RECOVERY: Resuming autonomous operation")

                look_ahead    = 150
                lane_width_px = 280
                
                # BUG 8: Override manual trackbar width ONLY if estimated EMA width is reasonable
                tracker = self.lane_module.tracker
                if 150 < tracker.estimated_lane_width < 400:
                    lane_width_px = int(tracker.estimated_lane_width)
                    
                fine_offset   = 50
                base_speed    = 50

                fine_px      = (fine_offset - 50) * 2
                
                # We removed the hardcoded +140 RIGHT_LANE_OFFSET_PX because the Hybrid tracker
                # now naturally anchors off the right shoulder. We only pass the user's fine_px tweak.
                total_offset = fine_px

                # -------------------------------------------------------------
                # 1. CORE CAMERA CAPTURE (RAW FRAME BASE)
                # -------------------------------------------------------------
                if self.cam_ok:
                    raw_frame = self.picam2.capture_array()
                    # BUG 18: raw_frame is None check
                    if raw_frame is None: 
                        raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    elif raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
                        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                else:
                    raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

                # -------------------------------------------------------------
                # 2. INTELLIGENT TRAFFIC MODULE ON RAW FRAME
                # -------------------------------------------------------------
                if self.traffic_module:
                    traffic_state, traffic_mult, light_status, active_labels, yolo_dbg = self.traffic_module.process_raw_frame(raw_frame)
                else:
                    traffic_state, traffic_mult, light_status, active_labels, yolo_dbg = "SYS_GO", 1.0, "NONE", [], raw_frame.copy()

                # -------------------------------------------------------------
                # 3. ADVANCED BEV LANE MODULE ON RAW FRAME
                # -------------------------------------------------------------
                # BUG 5: Adaptive Frame Rate: Less aggressive dropping, only skip odd frames (50% preservation)
                if self._fps < 12.0 and getattr(self, "frame_skip_step", 0) % 2 != 0:
                    lane_dbg = raw_frame.copy()
                    warped = np.zeros((480, 640), dtype=np.uint8)
                    sl, sr, detect_mode = self.lane_module.tracker.sl, self.lane_module.tracker.sr, "SKIPPED"
                else:
                    warped, sl, sr, lane_dbg, detect_mode = self.lane_module.process_raw_frame(raw_frame)
                    
                self.frame_skip_step = getattr(self, "frame_skip_step", 0) + 1

                # -------------------------------------------------------------
                # 4. NAVIGATION STATE MACHINES
                # -------------------------------------------------------------
                tracker = self.lane_module.tracker
                jct_state = self.jct.update(warped, tracker.left_conf, tracker.right_conf, tracker.left_fit, tracker.right_fit, lane_width_px, active_labels)
                rbt_state = self.rbt.update(tracker.left_fit, tracker.right_fit, lane_width_px)
                nav_state = rbt_state if rbt_state == "ROUNDABOUT" else jct_state

                # Determine Road Topology
                has_l_lane = tracker.left_conf >= tracker.MIN_PIX_OK
                has_r_lane = tracker.right_conf >= tracker.MIN_PIX_OK
                
                topology = "BLIND CORNER"
                if has_l_lane and has_r_lane: topology = "DUAL LANE"
                elif has_l_lane: topology = "SINGLE LANE (LEFT)"
                elif has_r_lane: topology = "SINGLE LANE (RIGHT)"

                # -------------------------------------------------------------
                # 5. STEERING CONTROLLER
                # -------------------------------------------------------------
                curvature_pre = tracker.get_curvature(tracker.h // 2)
                if nav_state == "ROUNDABOUT": eff_la = int(look_ahead * self.rbt.LOOKAHEAD_SCALE)
                # BUG 10: nav_state.startswith instead of strict equality
                elif nav_state.startswith("JUNCTION"): eff_la = int(look_ahead * 0.75)
                elif curvature_pre > self.HIGH_CURV_THRESH: eff_la = int(look_ahead * 1.30)
                elif curvature_pre > self.MED_CURV_THRESH: eff_la = int(look_ahead * 1.10)
                else: eff_la = look_ahead

                eff_la = max(60, eff_la)
                y_eval = max(0, 480 - eff_la)

                target_x, anchor = tracker.get_target_x(y_eval, lane_width_px, total_offset, nav_state, self.lost_frames, getattr(self, '_last_speed', 0.0), getattr(self, 'prev_steer', 0.0))
                
                # -------------------------------------------------------------
                # GLOBAL PREDICTIVE LOCALIZATION & MAP FUSION
                # -------------------------------------------------------------
                dt = time.time() - getattr(self, '_last_update_t', time.time() - 0.033)
                self._last_update_t = time.time()
                
                # Forward simulation using kinematic bicycle model
                gx, gy, gyaw = self.localizer.update_dead_reckoning(getattr(self, '_last_speed', 0.0), getattr(self, 'prev_steer', 0.0), dt)
                
                if self.global_pose is not None:
                    # Query trajectory planner for the predictive waypoint target
                    map_target = self.planner.get_local_target(gx, gy, gyaw)
                    
                    if map_target is not None:
                        map_x_bev = map_target[0]
                        # Update global pose UI anchor
                        idx, _ = self.gmap.get_nearest_spline_index(gx, gy)
                        self.global_pose = [gx, gy, gyaw, idx]
                        
                        if target_x is not None and not anchor.startswith("DEAD") and nav_state != "ROUNDABOUT":
                            # VISION IS ACTIVE: We fuse it into the Map Model to correct Odometry Drift
                            lane_error_px = map_x_bev - target_x
                            lane_error_cm = lane_error_px * (35.0 / max(50.0, lane_width_px))
                            # Gently pull the car's virtual global pose towards the true detected lane
                            self.localizer.fuse_lane_correction(lane_error_cm * 0.05, 0.0)
                        else:
                            # VISION IS LOST: The global map predictive model takes absolute control!
                            target_x = map_x_bev
                            anchor = "MAP_PREDICT_OVERRIDE"

                # Lane Position Feedback Control
                # BUG 6: We keep this PID controller as the core centering mechanism 
                # instead of fighting against track curvature translation offsets.
                position_correction = self.lane_pos_controller.compute_correction(tracker.sl, tracker.sr, y_eval)
                if target_x is not None:
                    target_x += position_correction
                
                # Dynamic Autonomous Lane Swapping / Trajectory Planning
                if not hasattr(self, "trajectory_planner"):
                    self.trajectory_planner = TrajectoryPlanner()
                    
                needs_evading = (traffic_state == "SYS_LANE_CHANGE_LEFT")
                # Attempt to extract obstacle center for smarter evasion
                obs_cx = 320 # Default center
                if needs_evading and self.traffic_module:
                    for det in self.traffic_module.active_detections:
                        if det["label"] in ["car", "closed-road-stand", "no-entry-road-sign"]:
                            obs_cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
                            break
                            
                evasion_offset, is_evading = self.trajectory_planner.compute_trajectory_offset(needs_evading, lane_width_px, obs_cx)
                
                if is_evading and target_x is not None:
                    target_x += evasion_offset
                    target_x = max(50.0, min(590.0, float(target_x)))
                    anchor = "OVERTAKING_LEFT"

                lost = target_x is None
                if lost:
                    self.lost_frames += 1
                    target_x = self.last_target
                else:
                    self.lost_frames, self.last_target = 0, target_x

                raw_steer = self._pure_pursuit(target_x, eff_la, lane_width_px)

                # EMA Smoothing handles twitchiness; use SLOW mostly, FAST only on huge swings
                steer_delta_abs = abs(raw_steer - self.smooth_steer)
                if curvature_pre > self.HIGH_CURV_THRESH:
                    alpha_adaptive = 0.15
                elif curvature_pre > self.MED_CURV_THRESH:
                    alpha_adaptive = 0.25
                else:
                    alpha_adaptive = self.STEER_EMA_SLOW
                    
                alpha = (self.STEER_EMA_FAST if steer_delta_abs > 12.0 else alpha_adaptive)
                self.smooth_steer = alpha * self.smooth_steer + (1.0 - alpha) * raw_steer
                steer_angle = self.smooth_steer

                rate_delta = max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, steer_angle - self.prev_steer))
                steer_angle = self.prev_steer + rate_delta
                self.prev_steer = steer_angle

                guard_left  = tracker.sl if tracker.left_stale == 0 else None
                guard_right = tracker.sr if tracker.right_stale == 0 else None

                raw_steer_guarded, guard_spd, guard_on = self.guard.apply(steer_angle, guard_left, guard_right, y_eval=y_eval)

                if lost: self.smooth_guard, guard_on = 0.0, False
                else:
                    guard_delta = raw_steer_guarded - steer_angle
                    self.smooth_guard = (self.GUARD_EMA * guard_delta + (1.0 - self.GUARD_EMA) * self.smooth_guard)
                steer_angle = steer_angle + self.smooth_guard

                # -------------------------------------------------------------
                # 6. VELOCITY SPEED RULES
                # -------------------------------------------------------------
                curvature = tracker.get_curvature(y_eval)
                
                # Topology Memory Update
                self.topology_memory.update(getattr(self, '_last_speed', 0.0), nav_state, curvature)
                
                # Check hardware degradation
                self.safety_manager.cam_watchdog.ping()
                yolo_ok = self.traffic_module is not None
                if yolo_ok and len(self.threaded_yolo.get_detections()) >= 0:
                    self.safety_manager.yolo_watchdog.ping()
                    
                deg_level = self.safety_manager.update(yolo_ok, self.cam_ok)
                
                if deg_level >= 3:
                    traffic_state, bt_action = "FATAL DEGRADATION", "HALT"
                    traffic_mult = 0.0
                    print("CRITICAL: Camera pipeline died! E-STOP ACTIVATED.")
                elif deg_level >= 1:
                    traffic_state = "DEGRADED (NO YOLO)"
                    traffic_mult = 0.7  # Cap max speed if we are blind to signs
                
                # Behavior Tree Evaluation
                blackboard = {
                    "traffic_state": traffic_state,
                    "traffic_reason": self.traffic_module.reason if self.traffic_module else "CLEAR PATH",
                    "nav_state": nav_state,
                    "curvature": curvature
                }
                bt_action, bt_reason = self.behavior_tree.evaluate(blackboard)
                
                # SPEED DETERMINATION USING BEHAVIOR TREE & HEURISTICS
                if getattr(self, '_manual_estop', False): speed = 0.0
                elif base_speed == 0: speed = 0.0
                elif bt_action == "HALT": speed = 0.0
                elif bt_action == "SLOW": speed = base_speed * 0.4
                elif bt_action == "JUNCTION_NAV": speed = base_speed * 0.55
                elif bt_action == "LANE_CHANGE": speed = base_speed * 0.60
                elif anchor.startswith("DEAD_RECKONING"):
                    try: conf = float(anchor.split("_")[2])
                    except: conf = 0.5
                    speed = base_speed * (0.4 + 0.4 * conf)
                elif nav_state == "ROUNDABOUT": speed = base_speed * self.rbt.SPEED_SCALE
                elif curvature > self.HIGH_CURV_THRESH: speed = base_speed * 0.45
                elif curvature > self.MED_CURV_THRESH: speed = base_speed * 0.65
                else:
                    # BUG 12: Pre-emptively slow down if a curve is imminent
                    next_feature = self.topology_memory.get_next_feature(lookahead=2.0)
                    if next_feature == "CURVE":
                        speed = base_speed * 0.75
                    elif abs(steer_angle) < 8: speed = base_speed * self.DUAL_SPEED_SCALE # Straight-line boost
                    elif abs(steer_angle) > 18: speed = base_speed * 0.60
                    elif abs(steer_angle) > 10: speed = base_speed * 0.80
                    else: speed = float(base_speed)

                if 0 < self.lost_frames <= LOST_GRACE_FRAMES:
                    speed *= max(0.3, 1.0 - self.lost_frames / LOST_GRACE_FRAMES)

                speed = speed * traffic_mult * (guard_spd if guard_on else 1.0)
                self._last_speed = speed
                steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_angle))

                # -------------------------------------------------------------
                # 6.5. STARTUP CALIBRATION PHASE OVERRIDE
                # -------------------------------------------------------------
                # Freeze car for the first 5 seconds so camera auto-exposure settles
                # and user can verify lane alignments on the screen.
                elapsed_run = time.time() - startup_time
                if elapsed_run < 5.0:
                    speed = 0.0
                    steer_angle = 0.0
                    cv2.putText(lane_dbg, f"CALIBRATING... {5.0 - elapsed_run:.1f}s", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                # -------------------------------------------------------------
                # 7. CAR CONTROL
                # -------------------------------------------------------------
                if self.connected:
                    self.handler.set_speed(speed)
                    self.handler.set_steering(steer_angle)

                # -------------------------------------------------------------
                # 8. DATA LOGGING
                # -------------------------------------------------------------
                self.logger.log_telemetry(
                    fps=self._fps,
                    speed=speed,
                    steering=steer_angle,
                    nav_state=nav_state,
                    traffic_state=traffic_state,
                    curvature=curvature,
                    degrad_level=deg_level,
                    estop=getattr(self, '_manual_estop', False)
                )

                # -------------------------------------------------------------
                # 9. PRESENTATION GRAPHICS
                # -------------------------------------------------------------
                self._draw_poly(lane_dbg, sl, (255, 220, 0))
                self._draw_poly(lane_dbg, sr, (0,   200, 255))

                cv2.circle(lane_dbg, (int(target_x), y_eval), 8, (0, 255, 0), -1)
                cv2.line(lane_dbg, (int(target_x), y_eval), (320, 470), (0, 255, 0), 2)
                cv2.line(lane_dbg, (320, 450), (320, 480), (0, 0, 255), 3)

                ref_x = 320 + 120 # Reference right-lane guide dot
                for y_tick in range(0, 480, 20): cv2.line(lane_dbg, (ref_x, y_tick), (ref_x, y_tick + 10), (100, 100, 100), 1)

                if guard_on: cv2.putText(lane_dbg, "! GUARD !", (230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if lost:
                    grace_label = (f"LOST ({self.lost_frames}/{LOST_GRACE_FRAMES})" if self.lost_frames <= LOST_GRACE_FRAMES else "LOST - STOPPED")
                    cv2.putText(lane_dbg, grace_label, (130, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                self._update_fps()
                
                # Drain the fake battery slowly over time
                sim_batt_pct = max(0.0, sim_batt_pct - 0.005)

                # Render the luxurious Professional UI natively via Tkinter
                if getattr(self, "ui", None):
                    self.ui.update_state(
                        yolo_dbg, lane_dbg, speed, steer_angle, 
                        traffic_state, self.traffic_module.reason if self.traffic_module else "CLEAR", 
                        light_status, nav_state, anchor, sim_batt_pct, active_labels, topology,
                        global_pose=self.global_pose
                    )
                    # Sync any UI teleport overrides back to the control loop
                    if getattr(self.ui, "global_pose", None) is not None:
                        self.global_pose = self.ui.global_pose
                    if getattr(self.ui, "target_pose", None) is not None:
                        self.target_pose = self.ui.target_pose

                elapsed = time.time() - t_frame_start
                wait_sec = max(0.001, FRAME_PERIOD - elapsed)
                time.sleep(wait_sec)
                
                # Optional: Read key events from Tkinter if needed, but for now we loop smoothly.

        except KeyboardInterrupt: pass
        finally: 
            self.running = False
            if getattr(self, "threaded_yolo", None): self.threaded_yolo.stop()
            self.stop()

    def stop(self):
        if self.connected:
            self.handler.set_speed(0)
            self.handler.set_steering(0)
            self.handler.disconnect()
        if self.cam_ok: self.picam2.stop()
        # cv2.destroyAllWindows()
        print("BFMC Modular Pilot: STOPPED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFMC Modular Traffic & Lane Pilot")
    parser.add_argument("--sim", action="store_true", help="Simulation mode")
    args = parser.parse_args()
    
    root = tk.Tk()
    dashboard = BFMCDashboardApp(root)
    pilot = BFMC_Pilot(sim_mode=args.sim, ui=dashboard)
    
    # Run pilot loop in background
    t = threading.Thread(target=pilot.run, daemon=True)
    t.start()
    
    # Run Tkinter in main thread
    root.mainloop()
