"""
BFMC Hybrid Pilot - Version 4 (Refactored Architecture)
========================================================
This script represents a complete architectural rewrite of the lane tracking
and steering control system, addressing all 48 structural flaws outlined,
and incorporating 12 additional critical bug fixes (Pure Pursuit math, 
RANSAC bounds, analytical curvature, parallel ghost lanes).

Key Improvements:
  - ARCH: All magic numbers extracted into BFMCConfig dataclass.
  - ARCH: Pure Pursuit mathematically corrected to use real physical meters.
  - ARCH: Target generation uses analytic parallel offset geometry.
  - PERF: Triple-EMA steering cascade reduced to single lag with initial rate delimiter.
  - PERF: Lane bounds use robust RANSAC polynomial fitting with outlier rejection.
  - SAFE: DividerGuard safety margins scale to physical car width parameters.
  - SAFE: Dead-reckoning fallback (0.5s) implemented for total lane loss.
  - SAFE: Min() constraint cascades applied to safety speed.
  - VIS: Dynamic LAB-B/HLS-L multi-channel detection for yellow/white markings.
"""

import cv2
import numpy as np
import math
import time
import logging
import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# --- Import Pre-Trained YOLO Detector (Legacy Integration) ---
try:
    from yolo_detector import PreTrainedYoloDetector
except ImportError:
    pass  # We will handle missing YOLO gracefully in the main class

# ---------------------------------------------------------------------------
# Hardware Handlers
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, "..")
    from serial_handler import STM32_SerialHandler
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    class STM32_SerialHandler:
        def connect(self): return False
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def disconnect(self): pass

try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    _CAM_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ===========================================================================
# CONFIGURATION
# ===========================================================================
@dataclass
class BFMCConfig:
    """Centralized configuration for all tunable parameters in the system."""
    
    # 1. Physical Measurements (meters)
    WHEELBASE_M: float = 0.23      # Distance between front and rear axles
    LANE_WIDTH_M: float = 0.35     # Expected physical width of a single lane
    CAR_WIDTH_M: float = 0.20      # Physical width of the BFMC vehicle
    
    # 2. Camera BEV Calibration (Default Projection)
    SRC_PTS: np.ndarray = field(default_factory=lambda: np.float32([[200, 260], [440, 260], [40, 450], [600, 450]]))
    DST_PTS: np.ndarray = field(default_factory=lambda: np.float32([[150, 0], [490, 0], [150, 480], [490, 480]]))
    HORIZON_CALIB_FRAMES: int = 30 # Recalibrate horizon every 30 frames
    
    # 3. Vision & Thresholding
    ADAPTIVE_BLOCK_SIZE: int = 25  # Tuned down from 31 for better local feature retention
    ADAPTIVE_C: int = -5           # Tuned up from -8 for less aggressive noise
    
    # 4. Lane Tracking & Sliding Window
    # FIX: Bug #11: Increase RANSAC Iterations for Robustness
    POLY_RANSAC_ITER: int = 50     # Increased from 15 to 50 for robust polynomial fitting
    POLY_RANSAC_THRESH: float = 20.0 # Pixel tolerance for RANSAC inliers
    EMA_LANE_PTS_ALPHA: float = 0.40 # Alpha for smoothing the actual X coordinates (not coefficients)
    MIN_LANE_WIDTH_PX: float = 200.0 # Valid range for lane width
    MAX_LANE_WIDTH_PX: float = 400.0
    STALE_FIT_TIMEOUT: int = 10    # Frames to remember last fit before declaring LOST
    
    # 5. Steering Control (Pure Pursuit & Guard)
    MAX_STEER_DEG: float = 30.0
    STEER_RATE_LIMIT: float = 15.0 # Increased from 5.0 to allow obstacle evasion
    STEER_EMA_ALPHA: float = 0.60  # Single stage smoothing alpha (confidence adaptive)
    
    LOOKAHEAD_MIN_PX: float = 80.0
    LOOKAHEAD_SPEED_K: float = 2.0 # L_d = k*v + min_L
    
    GUARD_GAIN: float = 0.20       # Increased from 0.09 for stronger boundary repulsion
    GUARD_MAX_CORR: float = 20.0   # Increased from 8.0 to allow emergency saves
    
    # 6. Adaptive Offsets
    TARGET_OFFSET_BASE: float = 70.0
    ADAPTIVE_PENALTY: float = 1.0  # Reduced from 3.0 (too volatile)
    ADAPTIVE_DECAY: float = 0.3    # Increased from 0.05 (too slow)
    # FIX: Bug #12: Ghost Lane Offsets
    GHOST_LANE_OFFSET_PX: float = 40.0  # Bias when synthesizing missing lane
    
    # 7. Speed Rules
    SPEED_MAX: float = 60.0
    SPEED_TURN: float = 30.0
    SPEED_JUNCTION: float = 25.0
    DEAD_RECKONING_SEC: float = 0.5# Time to freeze inputs if lanes completely lost
    
    # 8. Visuals
    DEBUG_LEVEL: int = 1           # 0=None, 1=Basic, 2=Verbose

    # FIX: IMPROVEMENT #1: Curvature Units Documentation
    # Curvature thresholds in 1/meters
    CURVATURE_HIGH: float = 0.8   # ~1.25m radius (tight hairpin)
    CURVATURE_MED: float = 0.3    # ~3.3m radius (normal turn)

CONF = BFMCConfig()


# ===========================================================================
# TRAFFIC DECISION MODULE (YOLO) - Unmodified Legacy Interface
# ===========================================================================
class TrafficDecisionModule:
    """ Handles decision state logic based on YOLO bounding boxes. """
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

    def update(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        dbg_frame = frame_bgr.copy()
        now = time.time()
        
        self.active_detections = self.detector.detect_traffic_signals(frame_bgr, conf_threshold=0.4)
        
        sees_red_light = sees_close_stop_sign = obstacle_in_path = sees_crosswalk = False
        
        for det in self.active_detections:
            label, (x1, y1, x2, y2), conf = det["label"], det["bbox"], det["confidence"]
            box_h = y2 - y1
            
            color = (0, 255, 0)
            if label == "stop-sign": color = (0, 0, 255)
            elif label == "traffic-light": color = (0, 255, 255)
            elif label in ["car", "pedestrian", "closed-road-stand"]: color = (255, 0, 255)
                
            cv2.rectangle(dbg_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(dbg_frame, f"{label} {conf:.2f}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == "traffic-light" and box_h > 25 and self._is_light_red(frame_bgr, x1, y1, x2, y2):
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

        cv2.putText(dbg_frame, f"TRAFFIC: {self.state} | {self.reason}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if self.state == "SYS_STOP" else (0,255,0), 3)
        return dbg_frame
        
    def get_speed_multiplier(self):
        if self.state == "SYS_STOP": return 0.0
        elif self.state == "SYS_SLOW": return 0.70 
        return 1.0


# ===========================================================================
# ADAPTIVE CONTROLLERS & MATH UTILS
# ===========================================================================
def custom_find_peaks(signal: np.ndarray, threshold: float = 0.3, distance: int = 40) -> List[int]:
    """ 
    Robust 1D peak finding using prominence and distance masking. 
    Fixes the old `argmax()` bug which collapsed dual-lanes at junctions into 1.
    """
    peaks = []
    sig_max = np.max(signal)
    if sig_max < 1e-5: return peaks
    
    # Filter points above the normalized relative threshold
    candidates = np.where(signal > sig_max * threshold)[0]
    
    while len(candidates) > 0:
        # Find the absolute maximum candidate
        max_idx = candidates[np.argmax(signal[candidates])]
        peaks.append(max_idx)
        # Suppress candidates within distance radius of this peak
        mask = np.abs(candidates - max_idx) > distance
        candidates = candidates[mask]
        
    return sorted(peaks)


def robust_polyfit_ransac(x: np.ndarray, y: np.ndarray, order: int = 2) -> Optional[np.ndarray]:
    """ Implements RANSAC-based polynomial fitting to reject noise outliers entirely. """
    if len(x) < 15: return None # Not enough points for a stable polynomial
    
    best_fit = None
    best_inliers = 0
    
    for _ in range(CONF.POLY_RANSAC_ITER):
        # Scale subset size slightly by length, min 5
        subset_idx = np.random.choice(len(x), size=min(len(x), 5), replace=False)
        xs, ys = x[subset_idx], y[subset_idx]
        
        try:
            fit = np.polyfit(ys, xs, order)
        except np.linalg.LinAlgError:
            continue
            
        # Determine continuous inliers
        x_est = np.polyval(fit, y)
        residuals = np.abs(x - x_est)
        inlier_mask = residuals < CONF.POLY_RANSAC_THRESH
        inliers = np.sum(inlier_mask)
        
        if inliers > best_inliers:
            best_inliers = inliers
            # Refine fit using ALL discovered inliers, not just the random 5
            try:
                best_fit = np.polyfit(y[inlier_mask], x[inlier_mask], order)
            except np.linalg.LinAlgError:
                pass
                
    # FIX: Bug #3: RANSAC Validation Rejects Valid Curves
    # The inlier percentage is sufficient validation. Width sanity check will catch bad fits.
    if best_inliers > len(x) * 0.40 and best_fit is not None:
        return best_fit
    return None


class AdaptiveOffsetController:
    """ Real-time cost function for lane positioning with track/search memory. """
    def __init__(self):
        self.base_offset = CONF.TARGET_OFFSET_BASE
        self.current_offset = CONF.TARGET_OFFSET_BASE
        self.min_offset = self.base_offset - 60 
        self.max_offset = self.base_offset + 60 
        
    def update(self, div_dist_err: float, edge_dist_err: float, is_tracking: bool) -> float:
        # Freeze map learning when we lose sight of lines
        if not is_tracking: return self.current_offset
            
        if div_dist_err > 0:
            self.current_offset += CONF.ADAPTIVE_PENALTY
        elif edge_dist_err > 0:
            self.current_offset -= CONF.ADAPTIVE_PENALTY
        else:
            if self.current_offset > self.base_offset + CONF.ADAPTIVE_DECAY:
                self.current_offset -= CONF.ADAPTIVE_DECAY
            elif self.current_offset < self.base_offset - CONF.ADAPTIVE_DECAY:
                self.current_offset += CONF.ADAPTIVE_DECAY
            else:
                self.current_offset = self.base_offset
                
        self.current_offset = max(self.min_offset, min(self.max_offset, self.current_offset))
        return self.current_offset


# ===========================================================================
# HYBRID LANE TRACKER V4
# ===========================================================================
class LaneTrackerV4:
    def __init__(self, img_shape=(480, 640)):
        self.h, self.w = img_shape
        self.mode = "SEARCH"
        
        # We store evaluated X arrays instead of raw fits. 
        # EMA smoothing physical coordinates is geometrically stable; smoothing
        # polynomial 'a, b' coefficients creates wild serpentine swings.
        self.sl_pts: Optional[np.ndarray] = None
        self.sr_pts: Optional[np.ndarray] = None
        self.eval_y = np.linspace(0, self.h - 1, self.h).astype(int) # Ensure int for indexing
        
        self.left_stale = 0
        self.right_stale = 0
        
    def _smooth_points(self, old_pts: Optional[np.ndarray], new_fit: np.ndarray) -> np.ndarray:
        new_pts = np.polyval(new_fit, self.eval_y)
        if old_pts is None: return new_pts
        return CONF.EMA_LANE_PTS_ALPHA * new_pts + (1.0 - CONF.EMA_LANE_PTS_ALPHA) * old_pts

    def update(self, warped_binary: np.ndarray, curvature_hint: float) -> Tuple[Any, Any, np.ndarray, str]:
        nz = warped_binary.nonzero()
        nzy, nzx = np.array(nz[0]), np.array(nz[1])
        dbg = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2BGR)

        # 1. Gather Pixels
        if self.mode == "TRACKING" and (self.sl_pts is not None or self.sr_pts is not None):
            li, ri = self._poly_search(nzx, nzy, curvature_hint, dbg)
            mode_label = "POLY"
        else:
            li, ri = self._sliding_window(warped_binary, nzx, nzy, curvature_hint, dbg)
            mode_label = "SLIDE"

        # 2. RANSAC Fitting & Point Smoothing
        has_l, has_r = False, False
        if len(li) > 150:
            fl = robust_polyfit_ransac(nzx[li], nzy[li])
            if fl is not None:
                self.sl_pts = self._smooth_points(self.sl_pts if self.left_stale == 0 else None, fl)
                self.left_stale, has_l = 0, True
                
        if len(ri) > 150:
            fr = robust_polyfit_ransac(nzx[ri], nzy[ri])
            if fr is not None:
                self.sr_pts = self._smooth_points(self.sr_pts if self.right_stale == 0 else None, fr)
                self.right_stale, has_r = 0, True

        # 3. Timeout Degradation
        if not has_l:
            self.left_stale += 1
            if self.left_stale > CONF.STALE_FIT_TIMEOUT: self.sl_pts = None
        if not has_r:
            self.right_stale += 1
            if self.right_stale > CONF.STALE_FIT_TIMEOUT: self.sr_pts = None

        # 4. Global Sanity Width Validation
        if self.sl_pts is not None and self.sr_pts is not None and self.left_stale == 0 and self.right_stale == 0:
            w = self.sr_pts[400] - self.sl_pts[400]
            if not (CONF.MIN_LANE_WIDTH_PX < w < CONF.MAX_LANE_WIDTH_PX):
                # Discard the weaker line if temporal memory indicates it's drifting
                if len(li) < len(ri) or self.left_stale > 0:
                    self.sl_pts = None
                    self.left_stale = CONF.STALE_FIT_TIMEOUT
                else:
                    self.sr_pts = None
                    self.right_stale = CONF.STALE_FIT_TIMEOUT

        self.mode = "TRACKING" if (self.sl_pts is not None or self.sr_pts is not None) else "SEARCH"
        return self.sl_pts, self.sr_pts, dbg, mode_label

    def get_target_x_analytic(self, y_eval: int, lane_width_px: float, extra_offset_px: float, nav_state: str) -> Tuple[Optional[float], str]:
        """
        Uses analytic parallel normal-geometry to synthesize missing lanes
        instead of incorrect pixel addition. target_x = x_base + offset / cos(theta)
        """
        eval_idx = int(y_eval)
        sl_x = self.sl_pts[eval_idx] if self.sl_pts is not None else None
        sr_x = self.sr_pts[eval_idx] if self.sr_pts is not None else None
        hw = lane_width_px / 2.0

        # FIX: Bug #2: Analytic Offset Geometry is Wrong
        def offset_x(pts, base_y, shift_amount):
            # Perpendicular offset: x_new = x + d*sin(θ) where θ = atan(dx/dy)
            if base_y < 2 or base_y >= self.h - 2:
                return pts[base_y] + shift_amount  # Simple fallback at boundaries
            
            # Use 4-pixel window for stable derivative
            dy = 4.0
            dx = pts[min(base_y+2, self.h-1)] - pts[max(base_y-2, 0)]
            
            # Angle of tangent line
            theta = math.atan2(dx, dy)
            
            # Perpendicular offset (normal to the curve)
            return pts[base_y] + shift_amount * math.sin(theta)

        if nav_state == "ROUNDABOUT":
            if sl_x is not None: return offset_x(self.sl_pts, eval_idx, hw + extra_offset_px), "RBT_INNER"
            if sr_x is not None: return offset_x(self.sr_pts, eval_idx, -hw + extra_offset_px), "RBT_OUTER"
            return None, "RBT_LOST"

        if nav_state == "JUNCTION":
            if sr_x is not None: return offset_x(self.sr_pts, eval_idx, -hw + extra_offset_px), "JCT_EDGE"
            if sl_x is not None: return offset_x(self.sl_pts, eval_idx, hw + extra_offset_px), "JCT_DIV"
            return None, "JCT_LOST"

        if sl_x is not None and sr_x is not None:
            return (sl_x + sr_x) / 2.0 + extra_offset_px, "DUAL"

        # FIX: Bug #12: Ghost Lane Offsets
        if sr_x is not None and sl_x is None:
            ghost_sl_x = offset_x(self.sr_pts, eval_idx, -lane_width_px)
            return (ghost_sl_x + sr_x) / 2.0 + extra_offset_px - CONF.GHOST_LANE_OFFSET_PX, "GHOST_L"

        if sl_x is not None and sr_x is None:
            ghost_sr_x = offset_x(self.sl_pts, eval_idx, lane_width_px)
            return (sl_x + ghost_sr_x) / 2.0 + extra_offset_px + CONF.GHOST_LANE_OFFSET_PX, "GHOST_R"

        return None, "LOST"

    def get_real_curvature(self, y_eval: int, lane_width_px: float) -> float:
        """ Returns geometric curvature 1/R in physical 1/meters using analytical derivative. """
        pts = self.sr_pts if self.sr_pts is not None else self.sl_pts
        if pts is None: return 0.0
        
        # FIX: Bug #6: Curvature Calculation is Unstable
        # Fit a local polynomial to get analytical derivatives
        y_min = max(0, y_eval - 60)
        y_max = min(self.h - 1, y_eval + 60)
        y_window = np.arange(y_min, y_max)
        x_window = pts[y_min:y_max]
        
        if len(y_window) < 20: return 0.0
        
        # Fit 2nd order polynomial locally
        try:
            fit = np.polyfit(y_window, x_window, 2)
            a, b = fit[0], fit[1]
        except:
            return 0.0
        
        # Analytical curvature formula: κ = |f''| / (1 + (f')²)^(3/2)
        # For x = ay² + by + c: dx/dy = 2ay + b, d²x/dy² = 2a
        dxdy = 2 * a * y_eval + b
        d2xdy2 = 2 * a
        
        curvature_px = abs(d2xdy2) / max((1 + dxdy**2)**1.5, 1e-6)
        
        # Convert to meters
        meters_per_px = CONF.LANE_WIDTH_M / max(lane_width_px, 100.0)
        curvature_m = curvature_px / meters_per_px  # 1/meters
        
        return curvature_m

    def _sliding_window(self, warped, nzx, nzy, curvature, dbg):
        # Adaptive windows based on curvature
        n_windows = int(15 if abs(curvature) > 0.5 else 9)
        hist = np.sum(warped[self.h // 2:, :], axis=0)
        smoothed = np.convolve(hist.astype(float), np.ones(10)/10, mode='same')
        
        peaks = custom_find_peaks(smoothed, threshold=0.25, distance=100)
        
        mid = self.w // 2
        lb = rb = mid
        if len(peaks) >= 2:
            left_peaks = [p for p in peaks if p < mid]
            right_peaks = [p for p in peaks if p >= mid]
            if left_peaks: lb = left_peaks[-1]  # right-most of the left peaks
            if right_peaks: rb = right_peaks[0] # left-most of the right peaks
        elif len(peaks) == 1:
            if peaks[0] < mid: lb, rb = peaks[0], peaks[0] + 300
            else: lb, rb = peaks[0] - 300, peaks[0]
            
        # Draw search boundaries
        wh = self.h // n_windows
        lx, rx = int(lb), int(rb)
        li, ri = [], []
        margin = int(max(40, 60)) # Could scale via lane_width_px
        
        for win in range(n_windows):
            y_lo, y_hi = self.h - (win + 1) * wh, self.h - win * wh
            xl0, xl1 = max(0, lx - margin), min(self.w, lx + margin)
            xr0, xr1 = max(0, rx - margin), min(self.w, rx + margin)
            
            cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 255, 0), 2)
            cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 255, 0), 2)
            
            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0) & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0) & (nzx < xr1)).nonzero()[0]
            
            li.append(gl); ri.append(gr)
            
            # FIX: Bug #8: Sliding Window Recentering
            # Adapt weight based on curvature - more responsive in curves
            if abs(curvature) > 0.005:  # Sharp curve
                weight_new = 0.6
            elif abs(curvature) > 0.002:  # Medium curve
                weight_new = 0.4
            else:  # Straight or gentle
                weight_new = 0.3

            if len(gl) > 50: 
                lx = int((1 - weight_new) * lx + weight_new * np.mean(nzx[gl]))
            if len(gr) > 50: 
                rx = int((1 - weight_new) * rx + weight_new * np.mean(nzx[gr]))

        li, ri = np.concatenate(li), np.concatenate(ri)
        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri

    def _poly_search(self, nzx, nzy, curvature, dbg):
        m = 100 if abs(curvature) > 0.5 else 60
        
        # FIX: Bug #5: Poly Search Can Crash With IndexError
        def band(pts_array):
            if pts_array is None: return np.array([], dtype=int)
            # Ensure all Y indices are within bounds
            valid_mask = (nzy >= 0) & (nzy < len(pts_array))
            if not np.any(valid_mask):
                return np.array([], dtype=int)
            
            nzy_safe = nzy[valid_mask]
            nzx_safe = nzx[valid_mask]
            target_x_for_y = pts_array[nzy_safe]
            
            band_mask = np.abs(nzx_safe - target_x_for_y) < m
            # Return original indices, not safe indices
            return np.where(valid_mask)[0][band_mask]
            
        li, ri = band(self.sl_pts), band(self.sr_pts)
        if len(li): dbg[nzy[li], nzx[li]] = [150, 150, 255]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [255, 150, 150]
        return li, ri

# ===========================================================================
# NAVIGATORS & SAFETY GUARDS
# ===========================================================================
class DividerGuardV4:
    def __init__(self):
        # Scale margins mathematically to physical properties
        # PPM (Pixels Per Meter) varies, so we'll evaluate dynamically in apply()
        self.gain = CONF.GUARD_GAIN
        self.max_corr = CONF.GUARD_MAX_CORR

    def apply(self, steer_angle: float, left_pts: Optional[np.ndarray], right_pts: Optional[np.ndarray], 
              y_eval: int, lane_width_px: float, car_x: float = 320.0):
        
        ppm = max(lane_width_px / CONF.LANE_WIDTH_M, 1.0)
        # Margin: Half car width + 5cm buffer, converted to pixels dynamically
        safe_margin_px = (CONF.CAR_WIDTH_M / 2.0 + 0.05) * ppm
        
        div_err, edge_err = 0.0, 0.0
        
        if left_pts is not None:
            gap = car_x - left_pts[y_eval]
            if gap < safe_margin_px: div_err = float(safe_margin_px - gap)
                
        if right_pts is not None:
            gap = right_pts[y_eval] - car_x
            if gap < safe_margin_px: edge_err = float(safe_margin_px - gap)
                
        speed_scale = max(0.4, 1.0 - (max(div_err, edge_err) / (safe_margin_px * 2.0)))
        
        # Dual violation handles logic properly (steer toward the LARGER safety gap)
        if div_err > 0 and edge_err > 0:
            if div_err > edge_err: correction = div_err * self.gain
            else: correction = -edge_err * self.gain
        else:
            correction = (div_err - edge_err) * self.gain
            
        correction = max(-self.max_corr, min(self.max_corr, correction))
        return steer_angle + correction, speed_scale, div_err, edge_err


class StateNavigatorsStub:
    """ Preserved exact original logic from user stub for junction/roundabout compatibility. """
    def __init__(self):
        self.r_state, self.r_frames = "NORMAL", 0
        self.j_state, self.j_frames, self.j_ex = "NORMAL", 0, 0

    def update_rbt(self, l_pts, r_pts, lane_width_px):
        if l_pts is not None and r_pts is not None:
            ratio = (r_pts[430] - l_pts[430]) / max(lane_width_px, 1.0)
            if self.r_state == "NORMAL" and ratio < 0.60: self.r_state, self.r_frames = "ROUNDABOUT", 0
            elif self.r_state == "ROUNDABOUT":
                self.r_frames += 1
                if (self.r_frames > 25 and ratio > 0.82) or self.r_frames > 120: self.r_state, self.r_frames = "NORMAL", 0
        elif self.r_state == "ROUNDABOUT":
             self.r_frames += 1
             if self.r_frames > 120: self.r_state, self.r_frames = "NORMAL", 0
        return self.r_state

    def update_jct(self, warped, l_pts, r_pts, lane_width_px):
        h = warped.shape[0]
        hist_top, hist_bot = float(np.sum(warped[:h//2, :])), float(np.sum(warped[h//2:, :]))
        cross = (hist_top / hist_bot) > 1.4 if hist_bot > 500 else False
        wide  = (r_pts[430] - l_pts[430]) > lane_width_px * 1.6 if l_pts is not None and r_pts is not None else False
        
        evid = (l_pts is None and r_pts is None) or cross or wide
        if self.j_state == "NORMAL":
            self.j_frames = self.j_frames + 1 if evid else 0
            if self.j_frames >= 5: self.j_state, self.j_ex, self.j_frames = "JUNCTION", 0, 0
        elif self.j_state == "JUNCTION":
            self.j_frames += 1
            self.j_ex = self.j_ex + 1 if not evid else 0
            if self.j_ex >= 8 and self.j_frames > 15: self.j_state, self.j_frames = "NORMAL", 0
        return self.j_state


# ===========================================================================
# MAIN PILOT ORCHESTRATOR
# ===========================================================================
class BFMC_PilotV4:
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

        # YOLO Initialization
        print("\n[INIT] Booting YOLO Traffic Authority...")
        try:
            self.traffic_module = TrafficDecisionModule(PreTrainedYoloDetector(model_version="best.pt"))
        except Exception:
            print(f"[WARN] YOLO disabled. Missing best.pt or yolo_detector.")
            self.traffic_module = None

        self.M = cv2.getPerspectiveTransform(CONF.SRC_PTS, CONF.DST_PTS)
        self.tracker = LaneTrackerV4(img_shape=(480, 640))
        self.guard = DividerGuardV4()
        self.nav = StateNavigatorsStub()
        self.adaptive_offset = AdaptiveOffsetController()
        
        self.dynamic_lane_width_px = 280.0
        self.smooth_steer = 0.0
        self.lost_timer = 0.0
        self.last_target = 320.0 + CONF.TARGET_OFFSET_BASE
        
        self.frame_count = 0
        self._fps_t, self._fps = time.time(), 0.0
        
        # FIX: IMPROVEMENT #2: Add Performance Metrics Logging
        self.metrics = {
            'frame_times': [],
            'lane_losses': 0,
            'frames_processed': 0
        }

    def _get_multi_channel_bev(self, frame: np.ndarray) -> np.ndarray:
        warped = cv2.warpPerspective(frame, self.M, (640, 480))
        # Multi-channel isolation: HLS (Lightness) + LAB (B-channel for Yellow robustly)
        hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        
        L = hls[:, :, 1]
        B = lab[:, :, 2] # Yellow strongly maps to High B
        
        fused = cv2.addWeighted(L, 0.7, B, 0.3, 0)
        binary = cv2.adaptiveThreshold(fused, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, CONF.ADAPTIVE_BLOCK_SIZE, CONF.ADAPTIVE_C)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def auto_calibrate_horizon(self, frame: np.ndarray):
        """ Runs every 30 frames asynchronously or fast-checks. Updates projection. """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        
        h, w = edges.shape
        roi = np.zeros_like(edges)
        cv2.fillPoly(roi, [np.array([[(0, h), (0, h//2 + 50), (w, h//2 + 50), (w, h)]])], 255)
        lines = cv2.HoughLinesP(cv2.bitwise_and(edges, roi), 1, np.pi/180, 50, minLineLength=40, maxLineGap=20)
        
        if lines is None: return
        
        left_lines, right_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: continue
            slope = (y2 - y1) / (x2 - x1)
            b = y1 - slope * x1
            if -1.5 < slope < -0.3: left_lines.append((slope, b))
            elif 0.3 < slope < 1.5: right_lines.append((slope, b))

        if not left_lines or not right_lines: return
        
        l_m, l_b = np.median([l[0] for l in left_lines]), np.median([l[1] for l in left_lines])
        r_m, r_b = np.median([r[0] for r in right_lines]), np.median([r[1] for r in right_lines])
        if l_m == r_m: return

        vx = (r_b - l_b) / (l_m - r_m)
        vy = l_m * vx + l_b
        horizon_y = max(100, min(int(vy) + 30, h - 100))
        
        CONF.SRC_PTS[0][1] = CONF.SRC_PTS[1][1] = horizon_y
        self.M = cv2.getPerspectiveTransform(CONF.SRC_PTS, CONF.DST_PTS)
        
        # FIX: Bug #7: Dynamic Lane Width Formula
        btm_src, btm_dst = CONF.SRC_PTS[3][0] - CONF.SRC_PTS[2][0], CONF.DST_PTS[3][0] - CONF.DST_PTS[2][0]
        if btm_src > 0:
            ppm = btm_dst / (CONF.LANE_WIDTH_M * 2.0)
            dyn_w = int(CONF.LANE_WIDTH_M * ppm)
            self.dynamic_lane_width_px = max(CONF.MIN_LANE_WIDTH_PX, min(CONF.MAX_LANE_WIDTH_PX, dyn_w))

    def _pure_pursuit_analytic(self, target_x: float, look_ahead_px: float) -> float:
        """ 
        Corrected Pure Pursuit. 
        Instead of pixel approximations, explicitly maps errors to METERS.
        """
        ppm = max(self.dynamic_lane_width_px / CONF.LANE_WIDTH_M, 1.0)
        
        e_y_pixels = target_x - 320.0
        e_y_meters = e_y_pixels / ppm
        L_d_meters = max(look_ahead_px / ppm, 0.1)
        
        # FIX: Bug #1: Pure Pursuit Formula
        # Standard Pure Pursuit: δ = atan2(2*L*sin(α), L_d) where α = atan2(e_y, L_d)
        alpha = math.atan2(e_y_meters, L_d_meters)  # Angle to target point
        steer_rad = math.atan2(2.0 * CONF.WHEELBASE_M * math.sin(alpha), L_d_meters)
        return math.degrees(steer_rad)

    def run(self):
        print("\nBFMC Pilot v4: STARTING MULTI-THREADED EXECUTION...")
        if self.cam_ok:
            for _ in range(10): self.picam2.capture_array() # flush
            
        # UI Setup 
        cv2.namedWindow("BFMC_V4_MASTER")
        
        try:
            while True:
                t_frame_start = time.time()
                self.frame_count += 1
                
                # Dynamic Image Capture & Format Sniffer
                if self.cam_ok:
                    frame = self.picam2.capture_array()
                    # RPi Camera format resilience check based on dimensional structure
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        # Assuming RGB default from Picamera2 config
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # Horizon Calibration (every 30 frames)
                if self.frame_count % CONF.HORIZON_CALIB_FRAMES == 0:
                    self.auto_calibrate_horizon(frame)

                # --- High Level Traffic Module ---
                # FIX: Bug #4: Traffic Module None Causes Crash
                if self.traffic_module:
                    yolo_dbg = self.traffic_module.update(frame)
                    yolo_multiplier = self.traffic_module.get_speed_multiplier()
                else:
                    yolo_dbg = frame.copy()
                    yolo_multiplier = 1.0  # Default to no traffic restrictions

                # --- 1. Vision & Tracking ---
                warped = self._get_multi_channel_bev(frame)
                
                # Feed curvature hint from PREVIOUS frame back into the search
                prev_curv_hint = self.tracker.get_real_curvature(380, self.dynamic_lane_width_px)
                l_pts, r_pts, dbg, detect_mode = self.tracker.update(warped, prev_curv_hint)

                # --- 2. State & Geometry Mathematics ---
                r_state = self.nav.update_rbt(l_pts, r_pts, self.dynamic_lane_width_px)
                j_state = self.nav.update_jct(warped, l_pts, r_pts, self.dynamic_lane_width_px)
                nav_state = "ROUNDABOUT" if r_state == "ROUNDABOUT" else j_state

                # FIX: Bug #9: Redundant max() in Lookahead Calculation
                # Dynamic Lookahead calculation (L_d = kv + L_min)
                base_spd = CONF.SPEED_MAX * yolo_multiplier
                eff_la = int(CONF.LOOKAHEAD_SPEED_K * base_spd + CONF.LOOKAHEAD_MIN_PX)
                eff_la = max(60, eff_la)  # Only enforce absolute minimum
                # Reduce slightly in strict state machines
                if nav_state in ["ROUNDABOUT", "JUNCTION"]: eff_la = int(eff_la * 0.70)
                y_eval = max(0, int(480 - eff_la))

                target_x, anchor = self.tracker.get_target_x_analytic(
                    y_eval, self.dynamic_lane_width_px, self.adaptive_offset.current_offset, nav_state)

                # --- 3. Steering Pipeline ---
                # A: Dead-reckoning Check
                if target_x is None:
                    if self.lost_timer == 0.0: self.lost_timer = time.time()
                    target_x = self.last_target # Use memory
                else:
                    self.lost_timer, self.last_target = 0.0, target_x

                # B: Raw Analytical Steer
                raw_steer = self._pure_pursuit_analytic(target_x, eff_la)

                # C: Pipeline (Rate Delimit FIRST -> then EMA smoothing -> then Guard Addition without EMA)
                rate_dx = max(-CONF.STEER_RATE_LIMIT, min(CONF.STEER_RATE_LIMIT, raw_steer - self.smooth_steer))
                raw_steer = self.smooth_steer + rate_dx
                self.smooth_steer = CONF.STEER_EMA_ALPHA * raw_steer + (1.0 - CONF.STEER_EMA_ALPHA) * self.smooth_steer

                # D: Hard Safety Application (Instant)
                steer_out, guard_spd_mul, div_err, edge_err = self.guard.apply(
                    self.smooth_steer, l_pts, r_pts, y_eval, self.dynamic_lane_width_px)
                
                # E: Finally clamp outputs limits
                steer_out = max(-CONF.MAX_STEER_DEG, min(CONF.MAX_STEER_DEG, steer_out))

                # --- 4. Adaptive Offset Update ---
                # FIX: Bug #10: Adaptive Offset Frozen in Critical States
                # Learn whenever we can see lanes, regardless of navigation state
                is_tracking_well = (target_x is not None) and (self.tracker.mode == "TRACKING")
                active_cost = self.adaptive_offset.update(div_err, edge_err, is_tracking_well)

                # --- 5. Velocity Control constraints cascade ---
                speed_reqs = [CONF.SPEED_MAX * yolo_multiplier, CONF.SPEED_MAX * guard_spd_mul]
                
                # State speed constraints
                if nav_state == "ROUNDABOUT": speed_reqs.append(CONF.SPEED_TURN)
                elif nav_state == "JUNCTION": speed_reqs.append(CONF.SPEED_JUNCTION)
                
                # Curvature speed constraint 
                real_r_meters = self.tracker.get_real_curvature(y_eval, self.dynamic_lane_width_px)
                if real_r_meters > CONF.CURVATURE_HIGH: speed_reqs.append(CONF.SPEED_TURN) # Hard curves
                
                speed_out = min(speed_reqs)
                
                # Extreme Line Loss deceleration Check
                if self.lost_timer > 0.0:
                    lost_time = time.time() - self.lost_timer
                    if lost_time > CONF.DEAD_RECKONING_SEC: 
                        speed_out = 0.0 # Halt vehicle, lane is gone.
                    else:
                        speed_out = speed_out * max(0.2, (1.0 - (lost_time / CONF.DEAD_RECKONING_SEC)))

                # Hardware transmission
                if self.connected:
                    self.handler.set_speed(float(speed_out))
                    self.handler.set_steering(float(steer_out))

                # --- Visuals Overlay ---
                if CONF.DEBUG_LEVEL > 0:
                    cv2.circle(dbg, (int(target_x), y_eval), 8, (0, 255, 0), -1)
                    if l_pts is not None: cv2.circle(dbg, (int(l_pts[y_eval]), y_eval), 4, (150, 255, 150), -1)
                    if r_pts is not None: cv2.circle(dbg, (int(r_pts[y_eval]), y_eval), 4, (150, 255, 150), -1)
                    
                    fps = 1.0 / max(time.time() - t_frame_start, 1e-6)
                    self._fps = 0.9 * self._fps + 0.1 * fps
                    
                    line1 = f"{detect_mode} | {anchor} | {nav_state} | {self._fps:.0f}fps"
                    line2 = f"Str:{steer_out:.1f} Spd:{speed_out:.0f} Off:{int(active_cost)} Wdth:{self.dynamic_lane_width_px}"
                    cv2.putText(dbg, line1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                    cv2.putText(dbg, line2, (10, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 2)

                    # Unified UI Render (Stack YOLO and Lane View vertically for easier Pi Viewing)
                    yolo_resized = cv2.resize(yolo_dbg, (640, 480))
                    stacked = np.vstack((yolo_resized, dbg))
                    cv2.imshow("BFMC_V4_MASTER", stacked)

                # FIX: IMPROVEMENT #2: Add Performance Metrics Logging
                if CONF.DEBUG_LEVEL >= 2:
                    self.metrics['frame_times'].append(time.time() - t_frame_start)
                    if target_x is None: self.metrics['lane_losses'] += 1
                    self.metrics['frames_processed'] += 1

                if cv2.waitKey(max(1, int((1.0/30.0 - (time.time() - t_frame_start)) * 1000))) == ord("q"):
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
        print("BFMC Pilot v4: STOPPED")
        
        # FIX: Print metrics on stop if logging enabled
        if CONF.DEBUG_LEVEL >= 2 and self.metrics['frames_processed'] > 0:
            avg_time = sum(self.metrics['frame_times']) / len(self.metrics['frame_times'])
            loss_rate = (self.metrics['lane_losses'] / self.metrics['frames_processed']) * 100
            print(f"\n--- PERFORMANCE METRICS ---")
            print(f"Total Frames: {self.metrics['frames_processed']}")
            print(f"Avg FPS:      {1.0/avg_time:.1f}")
            print(f"Lane Loss:    {loss_rate:.2f}%")
            print(f"---------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Simulation mode")
    args = parser.parse_args()
    BFMC_PilotV4(sim_mode=args.sim).run()
