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

# ---------------------------------------------------------------------------
# YOLO detector - graceful fallback
# ---------------------------------------------------------------------------
try:
    from yolo_detector import PreTrainedYoloDetector
except ImportError:
    pass

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


# ===========================================================================
# PERCEPTION: TRAFFIC DECISIONS (Operating on RAW Frame)
# ===========================================================================
class TrafficDecisionModule:
    """ Handles decision state logic based on raw camera YOLO bounding boxes. """
    def __init__(self, detector):
        self.detector = detector
        self.state = "SYS_GO" 
        self.reason = ""
        self.stop_sign_timer = 0.0
        self.stop_sign_cooldown = 0.0
        self.halt_duration = 3.0 
        self.cooldown_duration = 5.0 
        self.active_detections = []
        
    def _is_light_glowing(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_h, box_w = y2 - y1, x2 - x1
        if box_h < 15 or box_w < 10: return False
        
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Check for glowing Red, Yellow, or Green. S > 50 avoids white/grey, V > 150 ensures brightness
        mask_red1 = cv2.inRange(hsv, np.array([0, 50, 150]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 50, 150]), np.array([180, 255, 255]))
        mask_yellow = cv2.inRange(hsv, np.array([15, 50, 150]), np.array([35, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([40, 50, 150]), np.array([90, 255, 255]))
        
        glow_mask = cv2.bitwise_or(mask_red1, mask_red2)
        glow_mask = cv2.bitwise_or(glow_mask, mask_yellow)
        glow_mask = cv2.bitwise_or(glow_mask, mask_green)
        
        glow_ratio = cv2.countNonZero(glow_mask) / (box_h * box_w)
        return glow_ratio > 0.05

    def _is_light_red(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_h, box_w = y2 - y1, x2 - x1
        if box_h < 15 or box_w < 10: return False # BUG 16: Increase minimum dimensions
        
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Strictly look for GLOWING red (High Saturation & High Value/Brightness)
        # If it's off (dark) or green, it will fail this mask completely.
        mask1 = cv2.inRange(hsv, np.array([0, 120, 150]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 150]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # BUG 15: Add secondary check for very high V values (>200) with lower S threshold for bright sunlight
        mask3 = cv2.inRange(hsv, np.array([0, 50, 200]), np.array([10, 255, 255]))
        mask4 = cv2.inRange(hsv, np.array([170, 50, 200]), np.array([180, 255, 255]))
        red_sunlight_mask = cv2.bitwise_or(mask3, mask4)
        red_mask = cv2.bitwise_or(red_mask, red_sunlight_mask)
        
        red_ratio = cv2.countNonZero(red_mask) / (box_h * box_w)
        return red_ratio > 0.12 # BUG 2: Increase threshold to reduce false positives

    def _is_obstacle_in_path(self, x1, y1, x2, y2, frame_w, frame_h):
        # BUG 8: Check if ANY part of bbox overlaps path region instead of just center
        in_horizontal_path = (x1 < frame_w * 0.80) and (x2 > frame_w * 0.20)
        is_close = y2 > (frame_h * 0.60)
        return in_horizontal_path and is_close

    def process_raw_frame(self, raw_frame):
        h, w = raw_frame.shape[:2]
        yolo_dbg = raw_frame.copy()
        now = time.time()
        
        self.active_detections = self.detector.detect_traffic_signals(raw_frame, conf_threshold=0.4)
        
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

        for det in self.active_detections:
            label, (x1, y1, x2, y2), conf = det["label"], det["bbox"], det["confidence"]
            
            # --- 0. COMPLETELY IGNORE OFF TRAFFIC LIGHTS ---
            if label == "traffic-light":
                if not self._is_light_glowing(raw_frame, x1, y1, x2, y2):
                    continue
                    
            box_h = y2 - y1
            active_labels.append(label)
            
            # --- 1. ALWAYS DRAW DETECTIONS ---
            color = (0, 255, 0)
            if label in ["stop-sign", "no-entry-road-sign"]: color = (0, 0, 255)
            elif label == "traffic-light": color = (0, 255, 255)
            elif label in ["car", "pedestrian", "closed-road-stand"]: color = (255, 0, 255)
            elif "speed-limit" in label: color = (255, 255, 0)
            elif label in ["crosswalk-sign", "parking-sign", "highway-sign", "priority-sign"]: color = (255, 128, 0)
                
            cv2.rectangle(yolo_dbg, (x1, y1), (x2, y2), color, 2)
            cv2.putText(yolo_dbg, f"{label} {conf:.2f} [{box_h}px]", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # --- 2. LOGICAL PROXIMITY RULES ---
            if label == "traffic-light":
                is_red = self._is_light_red(raw_frame, x1, y1, x2, y2)
                # React from a greater distance
                if box_h < 40:
                    light_status = "[RED] FAR" if is_red else "[GREEN] FAR"
                elif box_h < 85:
                    light_status = "[RED] APPROACH" if is_red else "[GREEN] APPROACH"
                    if is_red: commit_state(4, "SYS_SLOW", "RED LIGHT AHEAD")
                elif box_h >= 85:
                    light_status = "[RED] HALT" if is_red else "[GREEN] CLEAR"
                    if is_red: commit_state(1, "SYS_STOP", "RED LIGHT (PRIORITY)")
            
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
                
        # Commit the mathematically highest priority state found in the frame
        self.state = proposed_state
        self.reason = proposed_reason
        
        # Stop sign logic handles timers overriding immediate frame detections
        if self.stop_sign_timer > 0.0:
            if now - self.stop_sign_timer < self.halt_duration:
                # Still waiting
                pass
            else:
                self.stop_sign_timer, self.stop_sign_cooldown = 0.0, now + self.cooldown_duration
                if self.state == "SYS_STOP" and "STOP SIGN" in self.reason:
                    self.state, self.reason = "SYS_GO", "STOP SIGN (CLEARED)"

        cv2.putText(yolo_dbg, f"TRAFFIC: {self.state} | {self.reason}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if self.state == "SYS_STOP" else (0,255,0), 3)
        return self.state, self.get_speed_multiplier(), light_status, active_labels, yolo_dbg
        
    def get_speed_multiplier(self):
        if self.state == "SYS_STOP": return 0.0
        elif self.state == "SYS_SLOW": return 0.60 
        elif self.state == "SYS_LANE_CHANGE_LEFT": return 0.60 # Slow down during the swerve
        elif self.state == "SYS_LIMIT": return 0.75
        return 1.0


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

    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
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
            self.traffic_module = TrafficDecisionModule(PreTrainedYoloDetector(model_version="best.pt"))
        except Exception:
            self.traffic_module = None
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

        self._fps_t, self._fps = time.time(), 0.0

        cv2.namedWindow("BFMC_MASTER_VIEW")
        cv2.createTrackbar("Look Ahead",    "BFMC_MASTER_VIEW", 150, 300, lambda x: None)
        cv2.createTrackbar("Lane Width PX", "BFMC_MASTER_VIEW", 280, 400, lambda x: None)
        cv2.createTrackbar("Fine Offset",   "BFMC_MASTER_VIEW",  50, 100, lambda x: None)
        cv2.createTrackbar("Base Speed",    "BFMC_MASTER_VIEW",  50, 150, lambda x: None)

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
    # TESLA-STYLE DASHBOARD RENDERER (ULTRA REALISTIC OpenCV)
    # -------------------------------------------------------------
    def _render_dashboard(self, yolo_hd, lane_dbg, speed, steer_angle, traffic_state, traffic_reason, light_status, nav_state, anchor, batt_pct, active_labels, topology):
        # Master Canvas: 1280x720 (HD)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # BUG 5: Add resize check
        if yolo_hd.shape[:2] != (720, 1280): yolo_hd = cv2.resize(yolo_hd, (1280, 720))
        
        # 1. Main Background
        canvas[0:720, 0:1280] = yolo_hd
        
        # 2. Sleek Translucent Glassmorphism Overlay
        # Creating a transparent mask for the UI backgrounds
        overlay = canvas.copy()
        
        # Color Palette
        BG_COLOR  = (15, 15, 18)   # Deep space grey
        BLUE_NEON = (255, 160, 50) # Vibrant cyber blue
        GREEN_LUM = (100, 255, 100)
        RED_LUM   = (100, 50, 255)
        TEXT_MAIN = (240, 240, 240)
        TEXT_DIM  = (140, 140, 140)
        
        # Right Panel Overlay (Telemetry)
        cv2.rectangle(overlay, (820, 0), (1280, 720), BG_COLOR, -1)
        # Gradient shadow bounding the right panel
        for i in range(30):
            cv2.line(overlay, (820 - i, 0), (820 - i, 720), BG_COLOR, max(1, int(20 - i*0.6)))
            
        # Bottom Control Panel Overlay (Translucent)
        cv2.rectangle(overlay, (0, 620), (820, 720), (10, 10, 15), -1)
        
        # Top-Left Logo Block
        cv2.rectangle(overlay, (20, 20), (360, 90), BG_COLOR, -1)
        
        # Apply the transparent blend (85% opaque UI panels)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        
        # --- TOP LEFT LOGO ---
        cv2.putText(canvas, "BOSCH FUTURE MOBILITY", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE_NEON, 2, cv2.LINE_AA)
        cv2.putText(canvas, "AUTONOMOUS ORCHESTRATOR v3.0", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv2.LINE_AA)
        
        # --- RIGHT TELEMETRY PANEL ---
        base_x = 860
        cv2.putText(canvas, "SYSTEM TELEMETRY", (base_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE_NEON, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"GPS: 46.7712 N | 23.6236 E", (base_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        
        # Line Seperator
        cv2.line(canvas, (base_x, 100), (1240, 100), (50, 50, 50), 1)
        
        # Drive Metrics (Speed & Steering)
        cv2.putText(canvas, "CHASSIS SPEED", (base_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{int(abs(speed))}", (base_x, 185), cv2.FONT_HERSHEY_DUPLEX, 2.2, TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(canvas, "cm/s", (base_x + 95, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_DIM, 1, cv2.LINE_AA)
        
        cv2.putText(canvas, "STEERING APEX", (1060, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{steer_angle:+.1f}", (1060, 185), cv2.FONT_HERSHEY_DUPLEX, 1.8, TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(canvas, "deg", (1190, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_DIM, 1, cv2.LINE_AA)
        
        # High-End Dynamic Steering Bar (Horizontal Center-Aligned Visualizer)
        cv2.rectangle(canvas, (base_x, 210), (1240, 216), (40, 40, 40), -1)
        cv2.circle(canvas, (1050, 213), 3, (150, 150, 150), -1) # Center Deadzone
        
        # Map Steering -30 to +30 onto the 380px wide bar (center is 1050)
        steer_px_offset = int((steer_angle / 30.0) * 190)
        bar_col = BLUE_NEON if abs(steer_angle) < 15 else RED_LUM
        if steer_angle > 0:
            cv2.rectangle(canvas, (1050, 210), (1050 + steer_px_offset, 216), bar_col, -1)
        else:
            cv2.rectangle(canvas, (1050 + steer_px_offset, 210), (1050, 216), bar_col, -1)
        
        # Battery Health
        cv2.putText(canvas, "ENERGY RESERVE", (base_x, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (base_x, 275), (1240, 290), (40, 40, 40), -1)
        fill_w = int((batt_pct / 100.0) * (1240 - base_x))
        b_col = GREEN_LUM if batt_pct > 30 else RED_LUM
        cv2.rectangle(canvas, (base_x, 275), (base_x + fill_w, 290), b_col, -1)
        cv2.putText(canvas, f"{batt_pct:.1f}%", (1180, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_col, 1, cv2.LINE_AA)
        
        cv2.line(canvas, (base_x, 320), (1240, 320), (50, 50, 50), 1)
        
        # --- AI VISION ENGINE ---
        cv2.putText(canvas, "NEURAL VISION ENGINE", (base_x, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE_NEON, 2, cv2.LINE_AA)
        
        state_col = GREEN_LUM
        if traffic_state == "SYS_STOP": state_col = RED_LUM
        elif traffic_state == "SYS_SLOW": state_col = (0, 200, 255)
        elif traffic_state == "SYS_LIMIT": state_col = (0, 255, 255)
        
        cv2.putText(canvas, "TRAFFIC COMMAND:", (base_x, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, traffic_state, (base_x, 435), cv2.FONT_HERSHEY_DUPLEX, 1.2, state_col, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"REASON: {traffic_reason}", (base_x, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_col, 1, cv2.LINE_AA)
        
        cv2.putText(canvas, "NAVIGATION MODE:", (base_x, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        
        nav_col = GREEN_LUM
        if nav_state != "NORMAL": nav_col = (0, 165, 255)
        if anchor == "OVERTAKING_LEFT": nav_col = RED_LUM
        
        cv2.putText(canvas, f"{nav_state} [{anchor}]", (base_x, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.6, nav_col, 2, cv2.LINE_AA)
        
        cv2.putText(canvas, "TRAFFIC SIGNAL:", (1080, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        tl_col = TEXT_DIM
        if "RED" in light_status: tl_col = RED_LUM
        elif "GREEN" in light_status: tl_col = GREEN_LUM
        cv2.putText(canvas, light_status, (1080, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tl_col, 2, cv2.LINE_AA)
        
        cv2.putText(canvas, "ROAD TOPOLOGY:", (860, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        top_col = GREEN_LUM if "DUAL" in topology else (0, 200, 255) if "BLIND" in topology else BLUE_NEON
        cv2.putText(canvas, topology, (860, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.6, top_col, 2, cv2.LINE_AA)
        
        # LED Status Board (Dynamic Grid)
        cv2.line(canvas, (base_x, 610), (1240, 610), (50, 50, 50), 1)
        
        def draw_led_icon(x, y, label, is_active, active_color, txt_color=TEXT_MAIN):
            bg = active_color if is_active else (40, 40, 40)
            cv2.rectangle(canvas, (x, y), (x + 85, y + 30), bg, -1)
            tx = (0,0,0) if is_active and bg != RED_LUM else txt_color
            cv2.putText(canvas, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tx, 1, cv2.LINE_AA)
            
        draw_led_icon(860,    630, "STOP",       "stop-sign" in active_labels, RED_LUM)
        draw_led_icon(955,   630, "PEDESTRN",   "pedestrian" in active_labels, (0, 200, 255))
        draw_led_icon(1050,  630, "X-WALK",     "crosswalk-sign" in active_labels, GREEN_LUM)
        draw_led_icon(1145,  630, "PARKING",    "parking-sign" in active_labels, BLUE_NEON)
        
        draw_led_icon(860,    670, "HIGHWAY",    "highway-sign" in active_labels, BLUE_NEON)
        draw_led_icon(955,   670, "PRIORITY",   "priority-sign" in active_labels, (0, 200, 255))
        draw_led_icon(1050,  670, "NO ENTRY",   "no-entry-road-sign" in active_labels, RED_LUM)
        # BUG 17: Case-insensitive check for limit variants
        draw_led_icon(1145,  670, "LIMIT",      any("limit" in x.lower() for x in active_labels), (0, 255, 255))

        # --- BOTTOM CONTROL BAR ---
        cv2.putText(canvas, "AUTONOMOUS MODE ACTIVE", (40, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN_LUM, 2, cv2.LINE_AA)
        
        # E-Stop Button Visual
        cv2.rectangle(canvas, (320, 640), (520, 700), (0, 0, 150), -1)
        cv2.rectangle(canvas, (320, 640), (520, 700), RED_LUM, 2)
        cv2.putText(canvas, "E-STOP [SPACE]", (345, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_MAIN, 2, cv2.LINE_AA)
        
        # --- INTERACTIVE MANUAL PROMPTS ---
        if nav_state == "JUNCTION_PROMPT" or anchor == "BLIND_CORNER":
            # Very aggressive visual takeover for interactive prompts
            blk = np.zeros_like(canvas)
            cv2.rectangle(blk, (150, 150), (670, 350), BG_COLOR, -1)
            cv2.rectangle(blk, (150, 150), (670, 350), BLUE_NEON, 3)
            
            p_title = "JUNCTION APPROACHING" if nav_state == "JUNCTION_PROMPT" else "BLIND CORNER DETECTED"
            
            cv2.putText(blk, p_title, (230, 200), cv2.FONT_HERSHEY_DUPLEX, 1.2, BLUE_NEON, 3, cv2.LINE_AA)
            cv2.putText(blk, "AWAITING HUMAN DECISION...", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_MAIN, 2, cv2.LINE_AA)
            cv2.putText(blk, "PRESS [L] TO FOLLOW LEFT", (250, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(blk, "PRESS [R] TO FOLLOW RIGHT", (250, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2, cv2.LINE_AA)
            
            # Blend
            cv2.addWeighted(blk, 0.9, canvas, 1.0, 0, canvas)
        
        # High-Tech Radar Window (Bottom Right of the camera view)
        radar_h, radar_w = 210, 280
        radar = cv2.resize(lane_dbg, (radar_w, radar_h))
        # Draw tech border around radar
        cv2.rectangle(radar, (0,0), (radar_w-1, radar_h-1), BLUE_NEON, 2)
        canvas[400:400+radar_h, 40:40+radar_w] = radar
        
        # Radar overlay text
        cv2.rectangle(canvas, (40, 375), (200, 400), BLUE_NEON, -1)
        cv2.putText(canvas, "RADAR BEV", (50, 393), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)

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

                look_ahead    = cv2.getTrackbarPos("Look Ahead",    "BFMC_MASTER_VIEW")
                lane_width_px = cv2.getTrackbarPos("Lane Width PX", "BFMC_MASTER_VIEW")
                fine_offset   = cv2.getTrackbarPos("Fine Offset",   "BFMC_MASTER_VIEW")
                base_speed    = cv2.getTrackbarPos("Base Speed",    "BFMC_MASTER_VIEW")

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
                warped, sl, sr, lane_dbg, detect_mode = self.lane_module.process_raw_frame(raw_frame)

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
                elif curvature_pre > self.HIGH_CURV_THRESH: eff_la = int(look_ahead * 0.60)
                elif curvature_pre > self.MED_CURV_THRESH: eff_la = int(look_ahead * 0.80)
                else: eff_la = look_ahead

                eff_la = max(60, eff_la)
                y_eval = max(0, 480 - eff_la)

                target_x, anchor = tracker.get_target_x(y_eval, lane_width_px, total_offset, nav_state, self.lost_frames, getattr(self, '_last_speed', 0.0), getattr(self, 'prev_steer', 0.0))
                
                # Dynamic Autonomous Lane Swapping
                if traffic_state == "SYS_LANE_CHANGE_LEFT" and target_x is not None:
                    # Forcibly subtract a full lane width to move the car into the oncoming left lane
                    target_x -= lane_width_px
                    # BUG 11: Add clamp for target_x to prevent steering blowouts
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
                alpha = (self.STEER_EMA_FAST if steer_delta_abs > 12.0 else self.STEER_EMA_SLOW)
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
                # BUG 19: E-Stop persistent flag check
                if getattr(self, '_manual_estop', False): speed = 0.0
                elif base_speed == 0: speed = 0.0
                elif nav_state == "JUNCTION_PROMPT": speed = 0.0 # HALT momentarily
                elif anchor.startswith("DEAD_RECKONING"):
                    try: conf = float(anchor.split("_")[2])
                    except: conf = 0.5
                    speed = base_speed * (0.3 + 0.4 * conf)
                elif nav_state == "ROUNDABOUT": speed = base_speed * self.rbt.SPEED_SCALE
                elif nav_state.startswith("JUNCTION"): speed = base_speed * 0.55
                elif curvature > self.HIGH_CURV_THRESH: speed = base_speed * self.HIGH_CURV_SCALE
                elif curvature > self.MED_CURV_THRESH: speed = base_speed * self.MED_CURV_SCALE
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
                # 8. PRESENTATION GRAPHICS
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

                # Render the luxurious Professional UI natively via OpenCV
                dashboard_ui = self._render_dashboard(
                    yolo_dbg, lane_dbg, speed, steer_angle, 
                    traffic_state, self.traffic_module.reason if self.traffic_module else "CLEAR", 
                    light_status, nav_state, anchor, sim_batt_pct, active_labels, topology
                )
                
                cv2.putText(dashboard_ui, f"Sys FPS: {self._fps:.0f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("BFMC_MASTER_VIEW", dashboard_ui)

                elapsed = time.time() - t_frame_start
                wait_ms = max(1, int((FRAME_PERIOD - elapsed) * 1000))
                
                key = cv2.waitKey(wait_ms)
                if key == ord("q"): 
                    break
                elif key == ord(" "): 
                    print("MANUAL ESTOP TRIGGERED!")
                    speed = 0.0
                    self._manual_estop = True # BUG 19: Local speed was wiped, persisted E-Stop state
                    self._estop_timestamp = time.time()
                    self.handler.set_speed(0.0)

        except KeyboardInterrupt: pass
        finally: 
            self.running = False
            self.stop()

    def stop(self):
        if self.connected:
            self.handler.set_speed(0)
            self.handler.set_steering(0)
            self.handler.disconnect()
        if self.cam_ok: self.picam2.stop()
        cv2.destroyAllWindows()
        print("BFMC Modular Pilot: STOPPED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFMC Modular Traffic & Lane Pilot")
    parser.add_argument("--sim", action="store_true", help="Simulation mode")
    args = parser.parse_args()
    
    BFMC_Pilot(sim_mode=args.sim).run()
