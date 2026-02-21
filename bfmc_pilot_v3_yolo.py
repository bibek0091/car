"""
BFMC Hybrid Pilot - Version 3 (YOLO Integration)
=================================================
Improvements over v2:
  - FEAT: Integrated custom BFMC YOLOv8 model for high-level decision making.
  - FEAT: Dual-window display (BEV Lane Window + Raw YOLO Camera Window).
  - FEAT: TrafficDecisionModule handles stop signs, traffic lights (red/green parsing),
          and stops for obstacles (cars, pedestrians, closed-road-stands).
  - ARCH: Speed from Lane Tracker is intercepted and safely brought to 0.0 when
          mandated by traffic rules, without corrupting lane state.

Key behaviours (unchanged):
  - Strict RIGHT-LANE driving
  - Hard divider safety margin
  - Junction / roundabout state machines
  - Hybrid sliding-window + polynomial tracking
"""

import cv2
import numpy as np
import math
import time
import logging
import argparse
import sys
import os

# --- Import YOLO Detector ---
from yolo_detector import PreTrainedYoloDetector

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
SRC_PTS = np.float32([[200, 260], [440, 260], [40,  450], [600, 450]])
DST_PTS = np.float32([[150,   0], [490,   0], [150, 480], [490, 480]])

RIGHT_LANE_OFFSET_PX  = 70
DUAL_OFFSET_PX        = 0
SINGLE_DIV_OFFSET_PX  = 40
SINGLE_EDGE_OFFSET_PX = -40

TARGET_FPS    = 30
FRAME_PERIOD  = 1.0 / TARGET_FPS
LOST_GRACE_FRAMES = 8


# ===========================================================================
# TRAFFIC DECISION MODULE (YOLO INTEGRATION)
# ===========================================================================
class TrafficDecisionModule:
    def __init__(self, detector):
        self.detector = detector
        self.state = "SYS_GO" # Driving normally
        self.reason = ""
        
        # Timers
        self.stop_sign_timer = 0.0
        self.stop_sign_cooldown = 0.0
        self.halt_duration = 3.0 # Stop for 3 seconds at a stop sign
        self.cooldown_duration = 5.0 # Ignore stop signs for 5 seconds after leaving
        
        self.active_detections = []
        
    def _is_light_red(self, frame, x1, y1, x2, y2):
        """
        Crops the traffic light bounding box and checks if the brightest
        pixels fall into the RED Hue range in HSV space.
        """
        # Ensure coordinates are within frame bounds safely
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            return False # Too small to process
            
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Red has two masks in HSV (wraps around 180). We widen it slightly
        # because the camera exposure can shift the red light towards orange/white.
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # If more than 5% of the bounding box is illuminated red
        red_ratio = cv2.countNonZero(red_mask) / (crop.shape[0] * crop.shape[1])
        return red_ratio > 0.05

    def _is_obstacle_in_path(self, x1, y1, x2, y2, frame_w, frame_h):
        """
        Determines if the bounding box of a car/pedestrian is physically
        in our driving path (bottom-center area of the screen).
        """
        center_x = (x1 + x2) / 2
        bottom_y = y2
        
        # Middle 50% of screen horizontally
        in_horizontal_path = (frame_w * 0.25) < center_x < (frame_w * 0.75)
        # Bottom 40% of screen vertically (close to us)
        is_close = bottom_y > (frame_h * 0.60)
        
        return in_horizontal_path and is_close

    def update(self, frame_bgr):
        """
        Runs YOLO inference on the unwarped frame and parses the rules.
        Returns the annotated frame for the dedicated YOLO window.
        """
        h, w = frame_bgr.shape[:2]
        dbg_frame = frame_bgr.copy()
        now = time.time()
        
        # 1. Get YOLO Detections
        # We lower the confidence slightly for traffic lights since they can 
        # get blown out by camera exposure.
        self.active_detections = self.detector.detect_traffic_signals(frame_bgr, conf_threshold=0.4)
        
        # Initialize flags for this frame
        sees_red_light = False
        sees_close_stop_sign = False
        obstacle_in_path = False
        sees_crosswalk = False
        
        # 2. Parse Detections
        for det in self.active_detections:
            label = det["label"]
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            
            box_h = y2 - y1
            
            # Draw standard bounding box
            color = (0, 255, 0)
            if label == "stop-sign": color = (0, 0, 255)
            elif label == "traffic-light": color = (0, 255, 255)
            elif label in ["car", "pedestrian", "closed-road-stand"]: color = (255, 0, 255)
                
            cv2.rectangle(dbg_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(dbg_frame, f"{label} {conf:.2f}", (x1, max(20, y1-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Rule Logic ---
            if label == "traffic-light":
                # Only check light if it's reasonably close (height > 20px)
                if box_h > 20 and self._is_light_red(frame_bgr, x1, y1, x2, y2):
                    sees_red_light = True
                    cv2.putText(dbg_frame, "RED DETECTED", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    
            elif label == "stop-sign":
                # We only stop if the sign is close enough (e.g. height > 45px)
                # and we are not currently in cooldown from just completing a stop.
                if box_h > 45 and now > self.stop_sign_cooldown:
                    sees_close_stop_sign = True
                    
            elif label in ["car", "pedestrian", "closed-road-stand"]:
                if self._is_obstacle_in_path(x1, y1, x2, y2, w, h):
                    obstacle_in_path = True
                    self.reason = f"OBSTACLE ({label})"
                    
            elif label == "crosswalk-sign":
                sees_crosswalk = True
                
        # 3. State Machine Overrides
        # Priority: Red Light > Stop Sign > Obstacle > Normal
        
        if sees_red_light:
            if self.state != "SYS_STOP": print("\n🛑 [YOLO] Red Light Detected! Halting car.")
            self.state = "SYS_STOP"
            self.reason = "RED LIGHT"
            
        elif obstacle_in_path:
            if self.state != "SYS_STOP": print(f"\n⚠️ [YOLO] Obstacle Detected in Path! Halting car ({self.reason}).")
            self.state = "SYS_STOP"
            # reason already set by the specific label
            
        elif sees_close_stop_sign:
            if self.stop_sign_timer == 0.0:
                # Start the stopping timer
                print(f"\n🛑 [YOLO] Stop Sign reached! Halting for {self.halt_duration} seconds...")
                self.stop_sign_timer = now
                self.state = "SYS_STOP"
                self.reason = "STOP SIGN (HALTING)"
            elif now - self.stop_sign_timer < self.halt_duration:
                # Still halting at the sign
                self.state = "SYS_STOP"
                self.reason = f"STOP SIGN ({self.halt_duration - (now - self.stop_sign_timer):.1f}s)"
            else:
                # Finished waiting at stop sign, proceed and initiate cooldown
                print("\n✅ [YOLO] Stop Sign complete. Proceeding (Cooldown active).")
                self.stop_sign_timer = 0.0
                self.stop_sign_cooldown = now + self.cooldown_duration
                self.state = "SYS_GO"
                self.reason = "STOP SIGN (CLEARED)"
                
        else:
            # Nothing critical triggered, return to GO state unless we are mid-stop-sign-halt
            if self.stop_sign_timer > 0.0:
                if now - self.stop_sign_timer < self.halt_duration:
                    self.state = "SYS_STOP"
                    self.reason = f"STOP SIGN ({self.halt_duration - (now - self.stop_sign_timer):.1f}s)"
                else:
                    print("\n✅ [YOLO] Stop Sign complete. Proceeding (Cooldown active).")
                    self.stop_sign_timer = 0.0
                    self.stop_sign_cooldown = now + self.cooldown_duration
                    self.state = "SYS_GO"
                    
            elif sees_crosswalk:
                self.state = "SYS_SLOW"
                self.reason = "CROSSWALK ZONE"
            else:
                self.state = "SYS_GO"
                self.reason = "CLEAR PATH"

        # Overlays for the YOLO window
        status_color = (0, 0, 255) if self.state == "SYS_STOP" else (0, 255, 0)
        cv2.putText(dbg_frame, f"TRAFFIC: {self.state} | {self.reason}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 3)
                    
        return dbg_frame
        
    def get_speed_multiplier(self):
        """ Returns the multiplier for base_speed. 0.0 for Hard Stop, 1.0 for standard. """
        if self.state == "SYS_STOP":
            return 0.0
        elif self.state == "SYS_SLOW":
            return 0.70  # Slow down 30% for crosswalks
        return 1.0


# ===========================================================================
# HYBRID LANE TRACKER (From v2)
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

    def get_target_x(self, y_eval, lane_width_px, extra_offset_px=0, nav_state="NORMAL"):
        sl = self.sl
        sr = self.sr
        hw = lane_width_px / 2.0

        def ev(fit): return float(np.polyval(fit, y_eval))

        if nav_state == "ROUNDABOUT":
            if sl is not None: return ev(sl) + hw + extra_offset_px, "RBT_INNER"
            if sr is not None: return ev(sr) - hw + extra_offset_px, "RBT_OUTER"
            return None, "RBT_LOST"

        if nav_state == "JUNCTION":
            if sr is not None: return ev(sr) - hw + extra_offset_px, "JCT_EDGE"
            if sl is not None: return ev(sl) + hw + extra_offset_px, "JCT_DIV"
            return None, "JCT_LOST"

        if sl is not None and sr is not None:
            return (ev(sl) + ev(sr)) / 2.0 + DUAL_OFFSET_PX, "DUAL"

        if sr is not None and sl is None:
            ghost_sl = sr - np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(ghost_sl) + ev(sr)) / 2.0 + SINGLE_EDGE_OFFSET_PX, "GHOST_L"

        if sl is not None and sr is None:
            ghost_sr = sl + np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(sl) + ev(ghost_sr)) / 2.0 + SINGLE_DIV_OFFSET_PX, "GHOST_R"

        return None, "LOST"

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

            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0) & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0) & (nzx < xr1)).nonzero()[0]

            li.append(gl); ri.append(gr)

            if len(gl) > self.MINPIX: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.MINPIX: rx = int(np.mean(nzx[gr]))

        li, ri = np.concatenate(li), np.concatenate(ri)
        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _poly_search(self, warped, nzx, nzy, curvature=0.0):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        m = self.POLY_MARGIN_CURV if curvature > 0.0015 else self.POLY_MARGIN_BASE
        def band(fit): return ((nzx > np.polyval(fit, nzy) - m) & (nzx < np.polyval(fit, nzy) + m)).nonzero()[0]
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
        return new.copy() if prev is None else self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev


class JunctionDetector:
    pass # Replaced with stub for brevity in YOLO iteration. Left original structure alone where we could.
    # To keep code size manageable for integration, we'll implement a minimal bypass for the classes that aren't YOLO.
    # The actual implementation of JCT and RBT is preserved exactly as original by simply keeping them.
    # We will copy the original entire classes.

# ===========================================================================
# JUNCTION DETECTOR
# ===========================================================================
class JunctionDetector:
    ENTRY_FRAMES       = 5
    EXIT_FRAMES        = 8
    CROSS_ENERGY_RATIO = 1.4
    WIDTH_RATIO_HIGH   = 1.6
    MIN_BOT_ENERGY     = 500

    def __init__(self):
        self.state         = "NORMAL"
        self.entry_count   = 0
        self.exit_count    = 0
        self.frames_in_jct = 0

    def update(self, warped_binary, left_conf, right_conf, left_fit, right_fit, lane_width_px):
        h, w = warped_binary.shape
        both_lost = (left_conf < 200) and (right_conf < 200)
        hist_top = float(np.sum(warped_binary[:h // 2, :]))
        hist_bot = float(np.sum(warped_binary[h // 2:, :]))
        cross_energy = False
        if hist_bot > self.MIN_BOT_ENERGY:
            cross_energy = (hist_top / hist_bot) > self.CROSS_ENERGY_RATIO
        wide_lane = False
        if left_fit is not None and right_fit is not None:
            if (np.polyval(right_fit, h - 50) - np.polyval(left_fit, h - 50)) > lane_width_px * self.WIDTH_RATIO_HIGH:
                wide_lane = True
        evidence = both_lost or cross_energy or wide_lane

        if self.state == "NORMAL":
            self.entry_count = self.entry_count + 1 if evidence else 0
            if self.entry_count >= self.ENTRY_FRAMES:
                self.state, self.exit_count, self.frames_in_jct = "JUNCTION", 0, 0
        elif self.state == "JUNCTION":
            self.frames_in_jct += 1
            self.exit_count = self.exit_count + 1 if not evidence else 0
            if self.exit_count >= self.EXIT_FRAMES and self.frames_in_jct > 15:
                self.state, self.entry_count = "NORMAL", 0
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
            ratio = (np.polyval(right_fit, y) - np.polyval(left_fit, y)) / max(float(lane_width_px), 1.0)
            if self.state == "NORMAL":
                if ratio < self.ENTRY_WIDTH_RATIO:
                    self.state, self.frames = "ROUNDABOUT", 0
            elif self.state == "ROUNDABOUT":
                self.frames += 1
                if (self.frames > self.MIN_CIRCLE_FRAMES and ratio > self.EXIT_WIDTH_RATIO) or self.frames > self.MAX_CIRCLE_FRAMES:
                    self.state, self.frames = "NORMAL", 0
        elif self.state == "ROUNDABOUT":
            self.frames += 1
            if self.frames > self.MAX_CIRCLE_FRAMES:
                self.state, self.frames = "NORMAL", 0
        return self.state


class DividerGuard:
    DIVIDER_SAFE_PX = 55
    EDGE_SAFE_PX    = 50
    GAIN            = 0.09
    MAX_CORR        = 8.0
    DEADBAND_PX     = 5

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        correction, speed_scale, triggered = 0.0, 1.0, False
        div_corr, edge_corr = 0.0, 0.0
        
        if left_fit is not None:
            gap = car_x - float(np.polyval(left_fit, y_eval))
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min(self.GAIN * err, self.MAX_CORR)
                speed_scale, triggered = min(speed_scale, max(0.5, 1.0 - err / 120.0)), True

        if right_fit is not None:
            gap = float(np.polyval(right_fit, y_eval)) - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR)
                speed_scale, triggered = min(speed_scale, max(0.5, 1.0 - err / 120.0)), True

        correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN) if (div_corr > 0 and edge_corr > 0) else (div_corr - edge_corr)
        return steer_angle + correction, speed_scale, triggered


# ===========================================================================
# MAIN PILOT
# ===========================================================================
class BFMC_Pilot:
    STEER_EMA_SLOW   = 0.25
    STEER_EMA_FAST   = 0.50
    GUARD_EMA        = 0.55
    MAX_STEER        = 30.0
    MAX_STEER_RATE   = 5.0
    HIGH_CURV_THRESH = 0.003
    MED_CURV_THRESH  = 0.0015
    HIGH_CURV_SCALE  = 0.60
    MED_CURV_SCALE   = 0.80
    DUAL_SPEED_SCALE = 1.15

    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        self.handler = STM32_SerialHandler()
        self.connected = False if sim_mode else self.handler.connect()

        self.cam_ok = False
        if not sim_mode and _CAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                self.picam2.configure(self.picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"}))
                self.picam2.start()
                self.cam_ok = True
            except Exception as e:
                log.warning(f"Camera init failed: {e}")

        # --- Initialize YOLO Subsystem ---
        print("\n[INIT] Booting YOLO Traffic Authority...")
        try:
            detector = PreTrainedYoloDetector(model_version="best.pt")
            self.traffic_module = TrafficDecisionModule(detector)
        except Exception as e:
            print(f"[FATAL] Could not initialize YOLO Detector. Missing best.pt? {e}")
            sys.exit(1)

        self.M = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        self.tracker = HybridLaneTracker(img_shape=(480, 640))
        self.rbt     = RoundaboutNavigator()
        self.jct     = JunctionDetector()
        self.guard   = DividerGuard()

        self.smooth_steer = 0.0
        self.smooth_guard = 0.0
        self.prev_steer   = 0.0
        self.last_target  = 320.0 + RIGHT_LANE_OFFSET_PX
        self.lost_frames  = 0

        self._fps_t, self._fps = time.time(), 0.0

        cv2.namedWindow("BFMC_v2_LANE_VIEW")
        cv2.createTrackbar("Look Ahead",    "BFMC_v2_LANE_VIEW", 150, 300, lambda x: None)
        cv2.createTrackbar("Lane Width PX", "BFMC_v2_LANE_VIEW", 280, 400, lambda x: None)
        cv2.createTrackbar("Fine Offset",   "BFMC_v2_LANE_VIEW",  50, 100, lambda x: None)
        cv2.createTrackbar("Base Speed",    "BFMC_v2_LANE_VIEW",  50, 150, lambda x: None)

    def _get_bev(self, frame):
        warped_colour = cv2.warpPerspective(frame, self.M, (640, 480))
        hls = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2HLS)
        binary = cv2.adaptiveThreshold(self.clahe.apply(hls[:, :, 1]), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    def _pure_pursuit(self, target_x, look_ahead_px, lane_width_px):
        lane_width_px = max(lane_width_px, 50)
        dx, dy = target_x - 320.0, max(float(look_ahead_px), 1.0)
        return math.degrees(math.atan2(2.0 * (WHEELBASE_M * (lane_width_px / LANE_WIDTH_M)) * math.sin(math.atan2(dx, dy)), math.sqrt(dx * dx + dy * dy)))

    def _update_fps(self):
        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - self._fps_t, 1e-6))
        self._fps_t = now

    def run(self):
        print("\nBFMC Pilot v3 (with YOLO): STARTING...")
        try:
            while True:
                t_frame_start = time.time()

                look_ahead    = max(cv2.getTrackbarPos("Look Ahead", "BFMC_v2_LANE_VIEW"), 60)
                lane_width_px = cv2.getTrackbarPos("Lane Width PX", "BFMC_v2_LANE_VIEW")
                total_offset  = RIGHT_LANE_OFFSET_PX + ((cv2.getTrackbarPos("Fine Offset", "BFMC_v2_LANE_VIEW") - 50) * 2)
                base_speed    = cv2.getTrackbarPos("Base Speed", "BFMC_v2_LANE_VIEW")

                if self.cam_ok:
                    frame = self.picam2.capture_array()
                    # FIX: Picamera2 often outputs RGB arrays even if BGR888 is requested on Pi5.
                    # OpenCV expects BGR. This flips the color channels to fix the 'bluish' filter.
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                # --- 1. RUN YOLO TRAFFIC LOGIC OVER RAW FRAME ---
                yolo_dbg_frame = self.traffic_module.update(frame)

                # --- 2. RUN LANE TRACKER OVER WARPED FRAME ---
                warped = self._get_bev(frame)
                sl, sr, dbg, detect_mode = self.tracker.update(warped)

                nav_state = "ROUNDABOUT" if self.rbt.update(self.tracker.left_fit, self.tracker.right_fit, lane_width_px) == "ROUNDABOUT" else \
                            self.jct.update(warped, self.tracker.left_conf, self.tracker.right_conf, self.tracker.left_fit, self.tracker.right_fit, lane_width_px)

                curv = self.tracker.get_curvature(480 - look_ahead)
                eff_la = int(look_ahead * {"ROUNDABOUT": self.rbt.LOOKAHEAD_SCALE, "JUNCTION": 0.75}.get(nav_state, 0.60 if curv > self.HIGH_CURV_THRESH else 0.80 if curv > self.MED_CURV_THRESH else 1.0))
                y_eval = max(0, 480 - eff_la)

                target_x, anchor = self.tracker.get_target_x(y_eval, lane_width_px, total_offset, nav_state)

                if target_x is None:
                    self.lost_frames, target_x = self.lost_frames + 1, self.last_target
                else:
                    self.lost_frames, self.last_target = 0, target_x

                raw_steer = self._pure_pursuit(target_x, eff_la, lane_width_px)
                alpha = self.STEER_EMA_FAST if abs(raw_steer - self.smooth_steer) > 8.0 else self.STEER_EMA_SLOW
                self.smooth_steer = alpha * raw_steer + (1.0 - alpha) * self.smooth_steer
                steer_angle = self.prev_steer + max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, self.smooth_steer - self.prev_steer))
                self.prev_steer = steer_angle

                guard_left  = self.tracker.sl if self.tracker.left_stale == 0 else None
                guard_right = self.tracker.sr if self.tracker.right_stale == 0 else None
                steer_guarded, guard_spd, guard_on = self.guard.apply(steer_angle, guard_left, guard_right, y_eval=y_eval)

                if target_x is None:
                    self.smooth_guard, guard_on = 0.0, False
                else:
                    self.smooth_guard = (self.GUARD_EMA * (steer_guarded - steer_angle) + (1.0 - self.GUARD_EMA) * self.smooth_guard)
                steer_angle += self.smooth_guard

                # --- 3. CALCULATE STANDARD SPEED ---
                if self.lost_frames > LOST_GRACE_FRAMES or base_speed == 0:
                    speed = 0.0
                elif nav_state == "ROUNDABOUT": speed = base_speed * self.rbt.SPEED_SCALE
                elif nav_state == "JUNCTION":   speed = base_speed * 0.55
                elif curv > self.HIGH_CURV_THRESH: speed = base_speed * self.HIGH_CURV_SCALE
                elif curv > self.MED_CURV_THRESH:  speed = base_speed * self.MED_CURV_SCALE
                elif anchor == "DUAL" and abs(steer_angle) < 10: speed = base_speed * self.DUAL_SPEED_SCALE
                elif abs(steer_angle) > 18: speed = base_speed * 0.60
                elif abs(steer_angle) > 10: speed = base_speed * 0.80
                else: speed = float(base_speed)

                if 0 < self.lost_frames <= LOST_GRACE_FRAMES: speed *= max(0.3, 1.0 - self.lost_frames / LOST_GRACE_FRAMES)
                if guard_on: speed *= guard_spd
                
                # --- 4. APPLY YOLO TRAFFIC RULE OVERRIDES ---
                # A traffic stop sign / red light WILL safely bring the car to 0 speed.
                traffic_scale_factor = self.traffic_module.get_speed_multiplier()
                speed *= traffic_scale_factor

                steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_angle))

                if self.connected:
                    self.handler.set_speed(speed)
                    self.handler.set_steering(steer_angle)

                # --- Visuals ---
                cv2.circle(dbg, (int(target_x), y_eval), 8, (0, 255, 0), -1)
                
                line1 = f"{detect_mode} | {anchor} | {nav_state} | {self._fps:.0f}fps"
                line2 = f"Steer:{steer_angle:.1f} Spd:{speed:.0f} Off:{int(total_offset)} YOLO_MUL:{traffic_scale_factor:.1f}"
                cv2.putText(dbg, line1, (10,  26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                cv2.putText(dbg, line2, (10, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 2)

                self._update_fps()
                
                # SHOW TWO WINDOWS AS REQUESTED
                cv2.imshow("BFMC_v2_LANE_VIEW", dbg)
                cv2.imshow("BFMC_v2_YOLO_VIEW", yolo_dbg_frame)

                if cv2.waitKey(max(1, int((FRAME_PERIOD - (time.time() - t_frame_start)) * 1000))) == ord("q"):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        if self.connected:
            self.handler.set_speed(0)
            self.handler.set_steering(0)
            self.handler.disconnect()
        if self.cam_ok:
            self.picam2.stop()
        cv2.destroyAllWindows()
        print("BFMC Pilot v3: STOPPED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Simulation mode")
    args = parser.parse_args()
    BFMC_Pilot(sim_mode=args.sim).run()
