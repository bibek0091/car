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
# SRC: [TL, TR, BL, BR] - Top points raised from 260 to 180 to view further ahead
SRC_PTS = np.float32([[150, 180], [490, 180], [20, 480], [620, 480]])
DST_PTS = np.float32([[150,   0], [490,   0], [150, 480], [490, 480]])

# ===========================================================================
# RIGHT-LANE OFFSET
# ===========================================================================
RIGHT_LANE_OFFSET_PX = 70
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
        
    def _is_light_red(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_h = y2 - y1
        if box_h < 10 or (x2 - x1) < 5: return False 
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        third = max(1, box_h // 3)
        top_mean = np.mean(gray[:third, :])
        bot_mean = np.mean(gray[-third:, :])
        if top_mean > bot_mean + 10: return True
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
        red_ratio = cv2.countNonZero(cv2.bitwise_or(mask1, mask2)) / (box_h * (x2 - x1))
        return red_ratio > 0.02

    def _is_obstacle_in_path(self, x1, y1, x2, y2, frame_w, frame_h):
        center_x = (x1 + x2) / 2
        in_horizontal_path = (frame_w * 0.20) < center_x < (frame_w * 0.80)
        is_close = y2 > (frame_h * 0.60)
        return in_horizontal_path and is_close

    def process_raw_frame(self, raw_frame):
        h, w = raw_frame.shape[:2]
        yolo_dbg = raw_frame.copy()
        now = time.time()
        
        self.active_detections = self.detector.detect_traffic_signals(raw_frame, conf_threshold=0.4)
        
        sees_red_light = sees_close_stop_sign = obstacle_in_path = sees_crosswalk = False
        
        for det in self.active_detections:
            label, (x1, y1, x2, y2), conf = det["label"], det["bbox"], det["confidence"]
            box_h = y2 - y1
            
            color = (0, 255, 0)
            if label == "stop-sign": color = (0, 0, 255)
            elif label == "traffic-light": color = (0, 255, 255)
            elif label in ["car", "pedestrian", "closed-road-stand"]: color = (255, 0, 255)
                
            cv2.rectangle(yolo_dbg, (x1, y1), (x2, y2), color, 2)
            cv2.putText(yolo_dbg, f"{label} {conf:.2f}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == "traffic-light" and box_h > 25 and self._is_light_red(raw_frame, x1, y1, x2, y2):
                sees_red_light = True
            elif label == "stop-sign" and box_h > 55 and now > self.stop_sign_cooldown:
                sees_close_stop_sign = True
            elif label in ["car", "pedestrian", "closed-road-stand", "no-entry-road-sign"] and box_h > 50 and self._is_obstacle_in_path(x1, y1, x2, y2, w, h):
                obstacle_in_path, self.reason = True, f"OBSTACLE ({label})"
            elif label == "crosswalk-sign" and box_h > 40:
                sees_crosswalk = True
                
        if sees_red_light:
            self.state, self.reason = "SYS_STOP", "RED LIGHT"
        elif obstacle_in_path:
            self.state = "SYS_STOP"
        elif sees_close_stop_sign:
            if self.stop_sign_timer == 0.0:
                self.stop_sign_timer, self.state, self.reason = now, "SYS_STOP", "STOP SIGN (HALTING)"
            elif now - self.stop_sign_timer < self.halt_duration:
                self.state, self.reason = "SYS_STOP", f"STOP SIGN (WAIT)"
            else:
                self.stop_sign_timer, self.stop_sign_cooldown, self.state, self.reason = 0.0, now + self.cooldown_duration, "SYS_GO", "STOP SIGN (CLEARED)"
        else:
            if self.stop_sign_timer > 0.0:
                if now - self.stop_sign_timer < self.halt_duration:
                    self.state, self.reason = "SYS_STOP", f"STOP SIGN (WAIT)"
                else:
                    self.stop_sign_timer, self.stop_sign_cooldown, self.state = 0.0, now + self.cooldown_duration, "SYS_GO"
            elif sees_crosswalk:
                self.state, self.reason = "SYS_SLOW", "CROSSWALK ZONE"
            else:
                self.state, self.reason = "SYS_GO", "CLEAR PATH"

        cv2.putText(yolo_dbg, f"TRAFFIC: {self.state} | {self.reason}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if self.state == "SYS_STOP" else (0,255,0), 3)
        return self.state, self.get_speed_multiplier(), yolo_dbg
        
    def get_speed_multiplier(self):
        if self.state == "SYS_STOP": return 0.0
        elif self.state == "SYS_SLOW": return 0.70 
        return 1.0


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

        def ev(fit):
            return float(np.polyval(fit, y_eval))

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
        warped_colour = cv2.warpPerspective(raw_frame, self.M_forward, (640, 480))
        hls = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2HLS)
        L   = self.clahe.apply(hls[:, :, 1])

        binary = cv2.adaptiveThreshold(
            L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, -8)

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
    CROSS_ENERGY_RATIO = 1.4
    WIDTH_RATIO_HIGH   = 1.6
    MIN_BOT_ENERGY     = 500

    def __init__(self):
        self.state, self.entry_count, self.exit_count, self.frames_in_jct = "NORMAL", 0, 0, 0

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
            lx = np.polyval(left_fit,  h - 50)
            rx = np.polyval(right_fit, h - 50)
            if (rx - lx) > lane_width_px * self.WIDTH_RATIO_HIGH: wide_lane = True

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
        div_corr = 0.0
        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min(self.GAIN * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered   = True

        edge_corr = 0.0
        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered   = True

        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered


# ===========================================================================
# MAIN PILOT ORCHESTRATOR
# ===========================================================================
class BFMC_Pilot:

    STEER_EMA_SLOW = 0.45   # Faster response on straights
    STEER_EMA_FAST = 0.70   # Much faster snap on tight corners
    GUARD_EMA      = 0.70   # Faster safety guard engagement
    MAX_STEER      = 30.0
    MAX_STEER_RATE = 15.0   # Lifted from 5.0 -> 15.0 to allow aggressive evasion

    HIGH_CURV_THRESH = 0.0025 # Engage high-curve speed slow down earlier
    MED_CURV_THRESH  = 0.0010

    HIGH_CURV_SCALE  = 0.60
    MED_CURV_SCALE   = 0.80
    DUAL_SPEED_SCALE = 1.15

    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        self.handler   = STM32_SerialHandler()
        self.connected = False if sim_mode else self.handler.connect()

        self.cam_ok = False
        if not sim_mode and _CAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                cfg = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
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
        self.last_target   = 320.0 + RIGHT_LANE_OFFSET_PX
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

    def _update_fps(self):
        now = time.time()
        dt  = now - self._fps_t
        self._fps_t = now
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(dt, 1e-6))

    def run(self):
        print("BFMC Pilot v2: STARTING MODULAR ORCHESTRATOR")
        try:
            while True:
                t_frame_start = time.time()

                look_ahead    = cv2.getTrackbarPos("Look Ahead",    "BFMC_MASTER_VIEW")
                lane_width_px = cv2.getTrackbarPos("Lane Width PX", "BFMC_MASTER_VIEW")
                fine_offset   = cv2.getTrackbarPos("Fine Offset",   "BFMC_MASTER_VIEW")
                base_speed    = cv2.getTrackbarPos("Base Speed",    "BFMC_MASTER_VIEW")

                fine_px      = (fine_offset - 50) * 2
                total_offset = RIGHT_LANE_OFFSET_PX + fine_px

                # -------------------------------------------------------------
                # 1. CORE CAMERA CAPTURE (RAW FRAME BASE)
                # -------------------------------------------------------------
                if self.cam_ok:
                    raw_frame = self.picam2.capture_array()
                    if raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
                        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                else:
                    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # -------------------------------------------------------------
                # 2. INTELLIGENT TRAFFIC MODULE ON RAW FRAME
                # -------------------------------------------------------------
                if self.traffic_module:
                    traffic_state, traffic_mult, yolo_dbg = self.traffic_module.process_raw_frame(raw_frame)
                else:
                    traffic_state, traffic_mult, yolo_dbg = "SYS_GO", 1.0, raw_frame.copy()

                # -------------------------------------------------------------
                # 3. ADVANCED BEV LANE MODULE ON RAW FRAME
                # -------------------------------------------------------------
                warped, sl, sr, lane_dbg, detect_mode = self.lane_module.process_raw_frame(raw_frame)

                # -------------------------------------------------------------
                # 4. NAVIGATION STATE MACHINES
                # -------------------------------------------------------------
                tracker = self.lane_module.tracker
                jct_state = self.jct.update(warped, tracker.left_conf, tracker.right_conf, tracker.left_fit, tracker.right_fit, lane_width_px)
                rbt_state = self.rbt.update(tracker.left_fit, tracker.right_fit, lane_width_px)
                nav_state = rbt_state if rbt_state == "ROUNDABOUT" else jct_state

                # -------------------------------------------------------------
                # 5. STEERING CONTROLLER
                # -------------------------------------------------------------
                curvature_pre = tracker.get_curvature(tracker.h // 2)
                if nav_state == "ROUNDABOUT": eff_la = int(look_ahead * self.rbt.LOOKAHEAD_SCALE)
                elif nav_state == "JUNCTION": eff_la = int(look_ahead * 0.75)
                elif curvature_pre > self.HIGH_CURV_THRESH: eff_la = int(look_ahead * 0.60)
                elif curvature_pre > self.MED_CURV_THRESH: eff_la = int(look_ahead * 0.80)
                else: eff_la = look_ahead

                eff_la = max(60, eff_la)
                y_eval = max(0, 480 - eff_la)

                target_x, anchor = tracker.get_target_x(y_eval, lane_width_px, total_offset, nav_state)

                lost = target_x is None
                if lost:
                    self.lost_frames += 1
                    target_x = self.last_target
                else:
                    self.lost_frames, self.last_target = 0, target_x

                raw_steer = self._pure_pursuit(target_x, eff_la, lane_width_px)

                steer_delta_abs = abs(raw_steer - self.smooth_steer)
                alpha = (self.STEER_EMA_FAST if steer_delta_abs > 8.0 else self.STEER_EMA_SLOW)
                self.smooth_steer = alpha * raw_steer + (1.0 - alpha) * self.smooth_steer
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
                if self.lost_frames > LOST_GRACE_FRAMES: speed = 0.0
                elif base_speed == 0: speed = 0.0
                elif nav_state == "ROUNDABOUT": speed = base_speed * self.rbt.SPEED_SCALE
                elif nav_state == "JUNCTION": speed = base_speed * 0.55
                elif curvature > self.HIGH_CURV_THRESH: speed = base_speed * self.HIGH_CURV_SCALE
                elif curvature > self.MED_CURV_THRESH: speed = base_speed * self.MED_CURV_SCALE
                elif anchor == "DUAL" and abs(steer_angle) < 10: speed = base_speed * self.DUAL_SPEED_SCALE
                elif abs(steer_angle) > 18: speed = base_speed * 0.60
                elif abs(steer_angle) > 10: speed = base_speed * 0.80
                else: speed = float(base_speed)

                if 0 < self.lost_frames <= LOST_GRACE_FRAMES:
                    speed *= max(0.3, 1.0 - self.lost_frames / LOST_GRACE_FRAMES)

                speed = speed * traffic_mult * (guard_spd if guard_on else 1.0)
                steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_angle))

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

                ref_x = 320 + RIGHT_LANE_OFFSET_PX
                for y_tick in range(0, 480, 20): cv2.line(lane_dbg, (ref_x, y_tick), (ref_x, y_tick + 10), (100, 100, 100), 1)

                if guard_on: cv2.putText(lane_dbg, "! GUARD !", (230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if lost:
                    grace_label = (f"LOST ({self.lost_frames}/{LOST_GRACE_FRAMES})" if self.lost_frames <= LOST_GRACE_FRAMES else "LOST - STOPPED")
                    cv2.putText(lane_dbg, grace_label, (130, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                self._update_fps()
                line1 = f"{detect_mode} | {anchor} | {nav_state} | {self._fps:.0f}fps"
                line2 = f"Steer:{steer_angle:.1f}  Speed:{speed:.0f}  Curv:{curvature:.4f}  Off:{int(total_offset)}"

                cv2.putText(lane_dbg, line1, (10,  26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                cv2.putText(lane_dbg, line2, (10, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 2)

                # Combine YOLO raw frame on top, Lane tracker on bottom
                yolo_resized = cv2.resize(yolo_dbg, (640, 480))
                stacked = np.vstack((yolo_resized, lane_dbg))
                cv2.imshow("BFMC_MASTER_VIEW", stacked)

                elapsed = time.time() - t_frame_start
                wait_ms = max(1, int((FRAME_PERIOD - elapsed) * 1000))
                if cv2.waitKey(wait_ms) == ord("q"): break

        except KeyboardInterrupt: pass
        finally: self.stop()

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
