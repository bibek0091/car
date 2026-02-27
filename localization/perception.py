"""
perception.py — BFMC BEV Lane Tracker (Upgraded to Hybrid Tracker with Dead Reckoning)
======================================================================================
"""

import cv2
import numpy as np
import math
import os
from dataclasses import dataclass
from collections import deque

@dataclass
class PerceptionResult:
    warped_binary:     np.ndarray
    lane_dbg:          np.ndarray
    sl:                object
    sr:                object
    target_x:          float
    lateral_error_px:  float
    anchor:            str
    confidence:        float
    lane_width_px:     float
    curvature:         float
    heading_rad:       float = 0.0
    heading_conf:      float = 0.0
    y_eval:            float = 400.0
    optical_yaw_rate:  float = 0.0
    optical_vel:       float = 0.0


class VisualOdometry:
    """
    Lucas-Kanade optical flow on the bottom 40% of the frame (ground plane).
    Produces optical_yaw_rate (rad/s) and optical_vel (m/s) as fallback signals
    when lane lines are absent.
    """

    def __init__(self):
        self.feature_params = dict(
            maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.p0       = None
        self.old_gray = None

    def update(self, frame_bgr, dt: float):
        """Returns (optical_yaw_rate rad/s, optical_vel m/s)."""
        if dt <= 0:
            return 0.0, 0.0

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi  = gray[int(h * 0.6):, :]   # bottom 40% — ground plane

        if self.p0 is None or len(self.p0) < 10:
            p0_roi = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
            if p0_roi is not None:
                p0_roi[:, 0, 1] += int(h * 0.6)   # lift coords to full frame
                self.p0       = p0_roi
                self.old_gray = gray.copy()
            return 0.0, 0.0

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self.old_gray, gray, self.p0, None, **self.lk_params)

        if p1 is None or st is None:
            self.p0 = None
            return 0.0, 0.0

        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        yaw_rate = vel = 0.0
        if len(good_new) > 3:
            dx = good_new[:, 0] - good_old[:, 0]
            dy = good_new[:, 1] - good_old[:, 1]
            # Calibration: ~0.015 rad/s per px/frame lateral; 0.008 m/s per px/frame forward
            yaw_rate = float(-np.median(dx) * 0.015 / dt)
            vel      = float( np.median(dy) * 0.008 / dt)

        self.old_gray = gray.copy()
        self.p0       = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
        return yaw_rate, vel


class DeadReckoningNavigator:
    def __init__(self):
        self.last_valid_target = 320.0
        self.last_valid_curvature = 0.0

    def predict_target(self, frames_lost, last_speed, last_steering):
        # Fallback target generator when lines are completely lost
        time_lost = frames_lost / max(30, 1) # Assuming 30 FPS
        lateral_drift = last_steering * 2.0 * time_lost
        predicted_target = self.last_valid_target + lateral_drift
        if abs(self.last_valid_curvature) > 0.001:
            predicted_target += self.last_valid_curvature * 5000 * time_lost
        predicted_target = np.clip(predicted_target, 150, 490)
        confidence = max(0.0, 1.0 - frames_lost / 30.0)
        return predicted_target, confidence


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
                self.left_fit, self.sl = None, None

        if has_r:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            self.right_fit  = fr
            self.sr         = self._ema(self.sr, fr)
            self.right_stale = 0
        else:
            self.right_stale += 1
            if self.right_stale > self.STALE_FIT_FRAMES:
                self.right_fit, self.sr = None, None

        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                if self.left_conf < self.right_conf:
                    self.left_fit, self.sl, self.left_stale, has_l = None, None, self.STALE_FIT_FRAMES, False
                else:
                    self.right_fit, self.sr, self.right_stale, has_r = None, None, self.STALE_FIT_FRAMES, False
            else:
                y_positions = [100, 200, 300, 400]
                widths = [np.polyval(self.sr, y) - np.polyval(self.sl, y) for y in y_positions]
                weighted_avg_width = np.average(widths, weights=[4, 3, 2, 1])
                self.estimated_lane_width = 0.8 * self.estimated_lane_width + 0.2 * weighted_avg_width

        self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"
        return self.sl, self.sr, dbg, mode_label

    def get_target_x(self, y_eval, lane_width_px, extra_offset_px=0, nav_state="NORMAL", frames_lost=0, last_speed=0.0, last_steering=0.0):
        sl, sr = self.sl, self.sr
        hw = lane_width_px / 2.0

        def ev(fit): return float(np.polyval(fit, y_eval))

        if nav_state == "ROUNDABOUT":
            if sl is not None: return ev(sl) + hw + extra_offset_px, "RBT_INNER"
            if sr is not None: return ev(sr) - hw + extra_offset_px, "RBT_OUTER"
            return None, "RBT_LOST"

        if nav_state.startswith("JUNCTION"):
            if nav_state == "JUNCTION_RIGHT":
                if sr is not None: return ev(sr) - (lane_width_px * 0.40) + extra_offset_px, "JCT_RIGHT_EDGE"
                elif sl is not None: return ev(sl) + (lane_width_px * 1.5) + extra_offset_px, "JCT_RIGHT_GHOST"
                else: return 320.0 + (lane_width_px * 0.8) + extra_offset_px, "JCT_RIGHT_BLIND"
            elif nav_state == "JUNCTION_LEFT":
                if sl is not None: return ev(sl) + (lane_width_px * 0.40) + extra_offset_px, "JCT_LEFT_EDGE"
                elif sr is not None: return ev(sr) - (lane_width_px * 1.5) + extra_offset_px, "JCT_LEFT_GHOST"
                else: return 320.0 - (lane_width_px * 0.8) + extra_offset_px, "JCT_LEFT_BLIND"
            return 320.0 + extra_offset_px, "JCT_WAITING_CHOICE"

        # NORMAL DRIVING (Middle-lane priority)
        if sl is None and sr is None:
            predicted_x, conf = self.dead_reckoner.predict_target(frames_lost, last_speed, last_steering)
            return predicted_x + extra_offset_px, f"DEAD_RECKONING_{conf:.2f}"
        
        if sl is not None and sr is not None: base_x, anchor = (ev(sl) + ev(sr)) / 2.0, "CENTERED_DUAL"
        elif sr is not None: base_x, anchor = ev(sr) - hw, "CENTERED_FROM_RIGHT"
        elif sl is not None: base_x, anchor = ev(sl) + hw, "CENTERED_FROM_LEFT"
            
        self.dead_reckoner.last_valid_target = base_x
        self.dead_reckoner.last_valid_curvature = self.get_curvature(y_eval)
        return base_x + extra_offset_px, anchor

    def get_curvature(self, y_eval):
        fit = self.sr if self.sr is not None else self.sl
        if fit is None: return 0.0
        a, b = fit[0], fit[1]
        denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return abs(2.0 * a) / max(denom, 1e-6)

    def _sliding_window(self, warped, nzx, nzy):
        dbg  = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        hist = np.sum(warped[self.h // 2:, :], axis=0)
        mid, margin = int(self.w * 0.40), self.SW_MARGIN
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
        if prev is None: return new.copy()
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev


class VisionPipeline:
    def __init__(self):
        self.SRC_PTS = np.float32([[200, 260], [440, 260], [40, 450], [600, 450]])
        self.DST_PTS = np.float32([[150, 0], [490, 0], [150, 480], [490, 480]])
        self.M_forward = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.tracker = HybridLaneTracker(img_shape=(480, 640))
        self.vo = VisualOdometry()
        self.lost_frames = 0
        self.last_target_x = 320.0

    def process(self, raw_frame, dt: float = 0.033, extra_offset_px=0.0,
                nav_state="NORMAL", velocity_ms=0.0, last_steering=0.0) -> PerceptionResult:
        if raw_frame.shape[:2] != (480, 640):
            process_frame = cv2.resize(raw_frame, (640, 480))
        else:
            process_frame = raw_frame

        # Run Visual Odometry on raw frame (ground-plane features)
        opt_yaw_rate, opt_vel = self.vo.update(process_frame, dt)

        warped_colour = cv2.warpPerspective(process_frame, self.M_forward, (640, 480))
        lab = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2LAB)
        L = self.clahe.apply(lab[:, :, 0])
        
        # Adaptive Lighting Compensation
        mean_l = np.mean(L)
        if mean_l < 100:
            L = cv2.convertScaleAbs(L, alpha=1.0 + (100 - mean_l)/200, beta=int((100 - mean_l)*0.6))
        elif mean_l > 180:
            L = cv2.convertScaleAbs(L, alpha=1.0 - (mean_l - 180)/350, beta=int(-(mean_l - 180)*0.4))

        binary = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
        warped_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        
        sl, sr, line_dbg, mode_label = self.tracker.update(warped_binary)
        
        y_eval = 400.0
        lw = self.tracker.estimated_lane_width
        
        # Determine Target
        target_x, anchor = self.tracker.get_target_x(
            y_eval, lw, extra_offset_px, nav_state, self.lost_frames, velocity_ms, last_steering
        )
        
        if target_x is None:
            self.lost_frames += 1
            target_x = self.last_target_x
        else:
            self.lost_frames = 0
            self.last_target_x = target_x

        # Confidence & Curvature Formatting
        curv = self.tracker.get_curvature(y_eval)
        conf = 1.0 if (sl is not None and sr is not None) else 0.5 if (sl is not None or sr is not None) else 0.0
        
        # Calculate pseudo-heading for Localizer mapping
        heading_rad = 0.0
        if sl is not None and sr is not None:
            heading_rad = math.atan2(np.polyval(sl, y_eval-50) - np.polyval(sl, y_eval), 50)

        return PerceptionResult(
            warped_binary=warped_binary,
            lane_dbg=line_dbg,
            sl=sl, sr=sr,
            target_x=target_x,
            lateral_error_px=target_x - 320.0,
            anchor=anchor,
            confidence=conf,
            lane_width_px=lw,
            curvature=curv,
            heading_rad=heading_rad,
            heading_conf=conf,
            y_eval=y_eval,
            optical_yaw_rate=opt_yaw_rate,
            optical_vel=opt_vel,
        )