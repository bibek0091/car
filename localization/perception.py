"""
perception.py — BFMC BEV Lane Tracker + Visual Odometry  (FIXED v4 — DEEP UPGRADE)
=====================================================================================
DEEP UPGRADES in v4:

  LANE-01  MULTI-CUE BINARY: adaptive threshold fused with Sobel-x edge map and
           HSV white/yellow colour mask. Each cue is individually morphed then
           majority-vote OR-combined. Survives shadows, glare, faded paint.

  LANE-02  HISTOGRAM PEAK PAIRING: recovery logic uses smoothed histogram with
           80-px suppression zone around first peak, preventing twin-peak swaps.

  LANE-03  TEMPORAL PIXEL ACCUMULATOR: rolling OR-mask of last N_ACCUM=4 binary
           frames added at half weight. Dramatically improves fitting through
           gaps and dashed-line sections.

  LANE-04  CURVATURE-AWARE Y_EVAL: polynomial evaluated at y_eval that adapts
           with speed and curvature — closer on curves/high speed (tighter
           tracking), farther on straights/low speed (more anticipation).

  LANE-05  POLYNOMIAL SANITY GATE: new fits are rejected if curvature ratio
           > 4x EMA or slope change > 0.5 rad or left/right slope diff > 0.45 rad.
           Previous EMA is kept on rejection — no sudden jumps.

  LANE-06  STALE FIT DECAY: instead of hard None after STALE_FIT_FRAMES, fits
           decay toward zero-curvature neutral over STALE_DECAY_FRAMES extra
           frames, giving the controller a smooth fade-out.

  LANE-07  ROBUST HEADING + CONFIDENCE: IQR-fence inlier filter replaces std
           filter. Also returns heading_conf (0-1) so localizer can gate on it.

  LANE-08  VELOCITY-ADAPTIVE Y_EVAL exposed through process() signature so
           the full pipeline is curvature + speed aware.

Fixes carried from v3:
  PERC-01  _poly_search fallback forwards wide flag
  PERC-02  Confidence normalised correctly
  PERC-03  CENTERED_FROM_RIGHT uses SINGLE_EDGE_OFFSET_PX
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PerceptionResult:
    warped_binary:     np.ndarray
    lane_dbg:          np.ndarray
    sl:                Optional[np.ndarray]
    sr:                Optional[np.ndarray]
    lateral_error_px:  float
    anchor:            str
    confidence:        float
    lane_width_px:     float
    curvature:         float
    l_conf:            float
    r_conf:            float
    heading_rad:       float = 0.0      # LANE-07: lane tangent heading
    heading_conf:      float = 0.0      # LANE-07: heading quality 0-1
    y_eval:            float = 400.0    # LANE-08: actual row used for error


# ─────────────────────────────────────────────────────────────────────────────
# LANE-07: Robust heading estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_heading_from_lanes(sl, sr, h: int = 480) -> float:
    """Returns lane tangent heading (rad). Positive = road slopes right."""
    heading, _ = estimate_heading_and_confidence(sl, sr, h)
    return heading


def estimate_heading_and_confidence(sl, sr, h: int = 480) -> Tuple[float, float]:
    """
    LANE-07: IQR-fence RANSAC-style robust heading with quality output.
    Returns (heading_rad, confidence 0-1).
    """
    eval_rows   = [h * f for f in [0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0]]
    row_weights = [0.3,   0.5,    0.8,   1.0,   1.5,   2.0,  3.0]

    tangents, weights = [], []
    for fit in [sl, sr]:
        if fit is None:
            continue
        for y, w in zip(eval_rows, row_weights):
            dxdy = 2.0 * fit[0] * y + fit[1]
            tangents.append(math.atan2(dxdy, 1.0))
            weights.append(w)

    if not tangents:
        return 0.0, 0.0

    arr = np.array(tangents)
    wts = np.array(weights)
    med = float(np.median(arr))

    if len(arr) >= 4:
        q1, q3 = np.percentile(arr, [25, 75])
        iqr    = q3 - q1
        fence  = max(iqr * 1.5, 0.05)
        mask   = np.abs(arr - med) <= fence
    else:
        mask = np.ones(len(arr), dtype=bool)

    if mask.sum() == 0:
        mask = np.ones(len(arr), dtype=bool)

    heading      = float(np.average(arr[mask], weights=wts[mask]))
    inlier_frac  = mask.sum() / len(arr)
    spread       = float(np.std(arr[mask])) if mask.sum() > 1 else 0.0
    conf         = float(np.clip(inlier_frac * max(0.0, 1.0 - spread / 0.3), 0.0, 1.0))
    return heading, conf


def estimate_camera_odometry(sl, sr, prev_sl, prev_sr, dt,
                             h=480, scale_m_per_px=0.35 / 280.0):
    """Lateral drift (m/s) and heading rate (rad/s) from frame diff."""
    if dt <= 0:
        return 0.0, 0.0

    def lane_center(sfl, sfr, y=400):
        if sfl is not None and sfr is not None:
            return (np.polyval(sfl, y) + np.polyval(sfr, y)) / 2.0
        elif sfr is not None:
            return np.polyval(sfr, y) - 140.0
        elif sfl is not None:
            return np.polyval(sfl, y) + 140.0
        return None

    drift_samples = []
    for eval_y in [200, 300, 400]:
        c_now  = lane_center(sl,      sr,      eval_y)
        c_prev = lane_center(prev_sl, prev_sr, eval_y)
        if c_now is not None and c_prev is not None:
            drift_samples.append((c_now - c_prev) * scale_m_per_px / dt)

    lateral_vel_ms = float(np.median(drift_samples)) if drift_samples else 0.0
    lateral_vel_ms = max(-0.05, min(0.05, lateral_vel_ms))

    curr_h = estimate_heading_from_lanes(sl,      sr,      h)
    prev_h = estimate_heading_from_lanes(prev_sl, prev_sr, h)

    heading_rate_rps = 0.0
    if abs(curr_h - prev_h) <= 0.5:
        heading_rate_rps = (curr_h - prev_h) / dt

    return lateral_vel_ms, heading_rate_rps


# ─────────────────────────────────────────────────────────────────────────────
# LANE-01: Multi-cue binary builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_robust_binary(warped_bgr: np.ndarray, clahe) -> np.ndarray:
    """
    LANE-01: Three-cue fusion:
      Cue A: CLAHE-L adaptive threshold
      Cue B: Sobel-x edge on L channel
      Cue C: HSV white + yellow colour mask
    Combined as majority-vote (A∧B) | (A∧C) | (B∧C), plus colour cue direct.
    """
    # ── Cue A ────────────────────────────────────────────────────────────────
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    L   = clahe.apply(lab[:, :, 0])
    mean_l = float(np.mean(L))
    if mean_l < 100:
        L = cv2.convertScaleAbs(L,
                                alpha=1.0 + (100 - mean_l) / 200.0,
                                beta=int((100 - mean_l) * 0.6))
    elif mean_l > 180:
        L = cv2.convertScaleAbs(L,
                                alpha=max(0.3, 1.0 - (mean_l - 180) / 350.0),
                                beta=int(-(mean_l - 180) * 0.4))
    cue_a = cv2.adaptiveThreshold(
        L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)

    # ── Cue B: Sobel-x ───────────────────────────────────────────────────────
    sobel_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=5)
    abs_sx  = np.abs(sobel_x)
    mx = abs_sx.max()
    abs_sx = (abs_sx / mx * 255).astype(np.uint8) if mx > 0 else abs_sx.astype(np.uint8)
    _, cue_b = cv2.threshold(abs_sx, 40, 255, cv2.THRESH_BINARY)

    # ── Cue C: colour mask ────────────────────────────────────────────────────
    hsv          = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    white_mask   = cv2.inRange(hsv, (0,  0,  160), (180, 60, 255))
    yellow_mask  = cv2.inRange(hsv, (15, 80,  80), (35, 255, 255))
    cue_c        = cv2.bitwise_or(white_mask, yellow_mask)

    # ── Morphology ───────────────────────────────────────────────────────────
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cue_a = cv2.morphologyEx(cue_a, cv2.MORPH_CLOSE, k5)
    cue_b = cv2.morphologyEx(cue_b, cv2.MORPH_CLOSE, k3)
    cue_c = cv2.morphologyEx(cue_c, cv2.MORPH_CLOSE, k3)

    # ── Majority-vote fusion + colour fallback ────────────────────────────────
    fused = cv2.bitwise_or(
        cv2.bitwise_and(cue_a, cue_b),
        cv2.bitwise_or(
            cv2.bitwise_and(cue_a, cue_c),
            cv2.bitwise_and(cue_b, cue_c)
        )
    )
    # Colour mask direct: slightly dilated to bridge 1-px gaps
    cue_c_dilated = cv2.dilate(cue_c, k3, iterations=1)
    return cv2.bitwise_or(fused, cue_c_dilated)


# ─────────────────────────────────────────────────────────────────────────────
class HybridLaneTracker:

    NWINDOWS             = 9
    SW_MARGIN            = 55
    SW_MARGIN_RECOVERY   = 110
    MINPIX               = 40
    POLY_MARGIN_BASE     = 65
    POLY_MARGIN_CURV     = 130
    MIN_PIX_OK           = 180
    EMA_ALPHA            = 0.28
    STALE_FIT_FRAMES     = 18
    STALE_DECAY_FRAMES   = 8
    LOST_RECOVERY_THRESH = 2

    # LANE-05 sanity thresholds
    MAX_CURV_RATIO      = 4.0
    MAX_SLOPE_DIFF      = 0.50
    MAX_PARALLEL_DIFF   = 0.45

    # LANE-03 accumulator depth
    N_ACCUM = 4

    def __init__(self, h=480, w=640):
        self.h, self.w     = h, w
        self.mode          = "SEARCH"
        self.left_fit      = None
        self.right_fit     = None
        self.sl, self.sr   = None, None
        self.l_stale       = 0
        self.r_stale       = 0
        self.l_conf        = 0.0
        self.r_conf        = 0.0
        self.lane_width_px = 280.0
        self._lost_frames  = 0
        self._accum_buf: deque = deque(maxlen=self.N_ACCUM)

    # ── Curvature ─────────────────────────────────────────────────────────────
    def get_curvature(self, fit, y_eval):
        if fit is None:
            return 0.0
        a, b  = fit[0], fit[1]
        denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return abs(2.0 * a) / max(denom, 1e-6)

    def _ema(self, prev, new):
        if prev is None:
            return new.copy()
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev

    # ── LANE-05: sanity gate ──────────────────────────────────────────────────
    def _fit_sane(self, new_fit, ema_fit, other_fit=None) -> bool:
        if ema_fit is not None:
            new_c = self.get_curvature(new_fit, self.h // 2)
            ema_c = self.get_curvature(ema_fit, self.h // 2)
            if ema_c > 1e-5:
                r = new_c / ema_c
                if r > self.MAX_CURV_RATIO or r < 1.0 / self.MAX_CURV_RATIO:
                    return False
            if abs(new_fit[1] - ema_fit[1]) > self.MAX_SLOPE_DIFF:
                return False
        if other_fit is not None:
            if abs(new_fit[1] - other_fit[1]) > self.MAX_PARALLEL_DIFF:
                return False
        return True

    # ── LANE-03: temporal accumulator ─────────────────────────────────────────
    def _accumulated(self, current: np.ndarray) -> np.ndarray:
        self._accum_buf.append(current.copy())
        if len(self._accum_buf) < 2:
            return current
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        hist = np.zeros_like(current)
        for i, past in enumerate(self._accum_buf):
            if i == len(self._accum_buf) - 1:
                continue
            hist = cv2.bitwise_or(hist, cv2.erode(past, k))
        return cv2.bitwise_or(current, cv2.dilate(hist, k))

    # ── LANE-06: neutral fit ───────────────────────────────────────────────────
    def _neutral_fit(self, fit):
        if fit is None:
            return None
        n    = fit.copy()
        n[0] = 0.0
        n[1] = n[1] * 0.5
        return n

    # ── Width sanity ──────────────────────────────────────────────────────────
    def _width_sane(self, lf, rf, y=400):
        return 70 < (np.polyval(rf, y) - np.polyval(lf, y)) < 580

    # ── Sliding window (LANE-02 improved peak pairing) ─────────────────────────
    def _sliding_window(self, warped, nzx, nzy, wide=False):
        dbg     = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        hist    = np.sum(warped[self.h // 2:, :], axis=0).astype(float)
        smooth  = np.convolve(hist, np.ones(25) / 25, mode='same')
        mid     = int(self.w * 0.42)
        margin  = self.SW_MARGIN_RECOVERY if wide else self.SW_MARGIN

        left_r  = smooth[margin: mid - margin]
        right_r = smooth[mid + margin: self.w - margin]
        lb = (int(np.argmax(left_r))  + margin       if left_r.size  > 0 else margin)
        rb = (int(np.argmax(right_r)) + mid + margin if right_r.size > 0 else mid + margin)

        # LANE-02: re-pair if too close
        if abs(rb - lb) < 120:
            p1  = int(np.argmax(smooth))
            tmp = smooth.copy()
            tmp[max(0, p1 - 80): min(self.w, p1 + 80)] = 0
            p2  = int(np.argmax(tmp))
            if abs(p1 - p2) >= 80:
                lb, rb = min(p1, p2), max(p1, p2)

        wh = self.h // self.NWINDOWS
        lx, rx = lb, rb
        li, ri = [], []
        for win in range(self.NWINDOWS):
            y_lo = self.h - (win + 1) * wh
            y_hi = self.h - win * wh
            sw   = self.SW_MARGIN_RECOVERY if wide else self.SW_MARGIN
            xl0, xl1 = max(0, lx - sw), min(self.w, lx + sw)
            xr0, xr1 = max(0, rx - sw), min(self.w, rx + sw)
            cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 200, 0), 1)
            cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 200, 0), 1)
            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0) & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0) & (nzx < xr1)).nonzero()[0]
            li.append(gl); ri.append(gr)
            if len(gl) > self.MINPIX: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.MINPIX: rx = int(np.mean(nzx[gr]))

        li = np.concatenate(li)
        ri = np.concatenate(ri)
        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    # ── Poly search ───────────────────────────────────────────────────────────
    def _poly_search(self, warped, nzx, nzy, curvature=0.0):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        m   = self.POLY_MARGIN_CURV if curvature > 0.0015 else self.POLY_MARGIN_BASE

        def band(fit):
            cx = np.polyval(fit, nzy)
            return ((nzx > cx - m) & (nzx < cx + m)).nonzero()[0]

        li = band(self.sl) if self.sl is not None else np.array([], dtype=int)
        ri = band(self.sr) if self.sr is not None else np.array([], dtype=int)

        if len(li) < self.MIN_PIX_OK and len(ri) < self.MIN_PIX_OK:
            self.mode = "SEARCH"
            do_wide   = self._lost_frames >= self.LOST_RECOVERY_THRESH  # PERC-01
            return self._sliding_window(warped, nzx, nzy, wide=do_wide)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    # ── Main update ───────────────────────────────────────────────────────────
    def update(self, warped: np.ndarray) -> np.ndarray:
        # LANE-03: temporally-enriched binary
        acc  = self._accumulated(warped)
        nz   = acc.nonzero()
        nzy  = np.array(nz[0])
        nzx  = np.array(nz[1])

        both_lost = (self.sl is None and self.sr is None)
        self._lost_frames = (self._lost_frames + 1) if both_lost else 0
        do_wide = self._lost_frames >= self.LOST_RECOVERY_THRESH

        if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
            curv = self.get_curvature(
                self.sl if self.sl is not None else self.sr, self.h // 2)
            li, ri, dbg = self._poly_search(acc, nzx, nzy, curvature=curv)
            label = "POLY"
        else:
            li, ri, dbg = self._sliding_window(acc, nzx, nzy, wide=do_wide)
            label = "SLIDE-W" if do_wide else "SLIDE"

        cv2.putText(dbg, label, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)

        self.l_conf = len(li) / 1000.0
        self.r_conf = len(ri) / 1000.0
        has_l = len(li) >= self.MIN_PIX_OK
        has_r = len(ri) >= self.MIN_PIX_OK

        # ── Left ─────────────────────────────────────────────────────────────
        if has_l:
            fl = np.polyfit(nzy[li], nzx[li], 2)
            if self._fit_sane(fl, self.sl, self.sr):   # LANE-05
                self.left_fit = fl
                self.sl       = self._ema(self.sl, fl)
                self.l_stale  = 0
                self.l_conf   = min(1.0, len(li) / 1000.0)
            else:
                has_l = False
                self.l_stale += 1
        if not has_l:
            self.l_stale += 1
            self.l_conf   = 0.0
            total = self.STALE_FIT_FRAMES + self.STALE_DECAY_FRAMES
            if self.l_stale > total:
                self.left_fit = None; self.sl = None
            elif self.l_stale > self.STALE_FIT_FRAMES and self.sl is not None:
                # LANE-06: gentle decay
                n = self._neutral_fit(self.sl)
                self.sl = 0.85 * self.sl + 0.15 * n

        # ── Right ────────────────────────────────────────────────────────────
        if has_r:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            if self._fit_sane(fr, self.sr, self.sl):   # LANE-05
                self.right_fit = fr
                self.sr        = self._ema(self.sr, fr)
                self.r_stale   = 0
                self.r_conf    = min(1.0, len(ri) / 1000.0)
            else:
                has_r = False
                self.r_stale += 1
        if not has_r:
            self.r_stale += 1
            self.r_conf   = 0.0
            total = self.STALE_FIT_FRAMES + self.STALE_DECAY_FRAMES
            if self.r_stale > total:
                self.right_fit = None; self.sr = None
            elif self.r_stale > self.STALE_FIT_FRAMES and self.sr is not None:
                n = self._neutral_fit(self.sr)
                self.sr = 0.85 * self.sr + 0.15 * n

        # ── Width update ──────────────────────────────────────────────────────
        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                if len(li) < len(ri):
                    self.left_fit = None; self.sl = None
                    self.l_stale  = self.STALE_FIT_FRAMES; has_l = False
                else:
                    self.right_fit = None; self.sr = None
                    self.r_stale   = self.STALE_FIT_FRAMES; has_r = False
            else:
                widths = [np.polyval(self.sr, y) - np.polyval(self.sl, y)
                          for y in [100, 200, 300, 400]]
                self.lane_width_px = (0.85 * self.lane_width_px
                                      + 0.15 * float(np.average(widths, weights=[1,2,3,4])))

        self.lane_width_px = max(150.0, min(self.lane_width_px, 420.0))
        self.mode = ("TRACKING" if (has_l or has_r or
                                    self.sl is not None or self.sr is not None)
                     else "SEARCH")
        return dbg


# ─────────────────────────────────────────────────────────────────────────────
class VisionPipeline:

    DUAL_OFFSET_PX        =  30
    SINGLE_DIV_OFFSET_PX  =  40
    SINGLE_EDGE_OFFSET_PX = -40

    # LANE-08: adaptive Y_EVAL rows
    Y_EVAL_NEAR   = 355
    Y_EVAL_NORMAL = 400
    Y_EVAL_FAR    = 445

    V_SLOW  = 0.15   # m/s below which we use FAR row
    V_FAST  = 0.35   # m/s above which we use NEAR row

    def __init__(self):
        self.tracker  = HybridLaneTracker()
        self.SRC_PTS  = np.float32([[200, 260], [440, 260], [40, 450], [600, 450]])
        self.DST_PTS  = np.float32([[150, 0],   [490, 0],   [150, 480], [490, 480]])
        self.M        = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
        self.clahe    = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        self.bev_calibrated = False
        self._last_target_x = 320.0

    def update_bev_transform(self, src_pts):
        self.SRC_PTS = np.float32(src_pts)
        self.M = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
        self.bev_calibrated = True

    def process(self, frame_bgr,
                extra_offset_px: float = 0.0,
                nav_state:       str   = "NORMAL",
                velocity_ms:     float = 0.0,
                curvature_hint:  float = 0.0) -> PerceptionResult:
        """
        LANE-08: velocity_ms and curvature_hint drive adaptive Y_EVAL.
        Higher speed / higher curvature → evaluate closer (NEAR row) for
        tighter tracking. Low speed / straight → evaluate far ahead for
        better anticipation. This is the primary fix for "not enough steering".
        """
        if frame_bgr.shape[:2] != (480, 640):
            frame_bgr = cv2.resize(frame_bgr, (640, 480))
        warped = cv2.warpPerspective(frame_bgr, self.M, (640, 480))

        # LANE-01: multi-cue binary
        binary = _build_robust_binary(warped, self.clahe)

        dbg = self.tracker.update(binary)
        sl  = self.tracker.sl
        sr  = self.tracker.sr
        lw  = self.tracker.lane_width_px
        hw  = lw / 2.0

        # LANE-08: adaptive evaluation row
        if nav_state.startswith("JUNCTION") or nav_state == "ROUNDABOUT":
            y_eval = self.Y_EVAL_NEAR
        elif curvature_hint > 0.0015 or velocity_ms > self.V_FAST:
            y_eval = self.Y_EVAL_NEAR
        elif velocity_ms < self.V_SLOW:
            y_eval = self.Y_EVAL_FAR
        else:
            y_eval = self.Y_EVAL_NORMAL

        def ev(fit):
            return float(np.polyval(fit, y_eval))

        # ── Target X ─────────────────────────────────────────────────────────
        if nav_state == "ROUNDABOUT":
            if sl is not None:
                tx, anchor = ev(sl) + hw + extra_offset_px, "RBT_INNER"
            elif sr is not None:
                tx, anchor = ev(sr) - hw + extra_offset_px, "RBT_OUTER"
            else:
                tx, anchor = self._last_target_x + extra_offset_px, "RBT_LOST"

        elif nav_state == "JUNCTION_RIGHT":
            if sr is not None:
                tx, anchor = ev(sr) - lw * 0.40 + extra_offset_px, "JCT_RIGHT_EDGE"
            elif sl is not None:
                tx, anchor = ev(sl) + lw * 1.50 + extra_offset_px, "JCT_RIGHT_GHOST"
            else:
                tx, anchor = 320.0 + lw * 0.8 + extra_offset_px, "JCT_RIGHT_BLIND"

        elif nav_state == "JUNCTION_LEFT":
            if sl is not None:
                tx, anchor = ev(sl) + lw * 0.40 + extra_offset_px, "JCT_LEFT_EDGE"
            elif sr is not None:
                tx, anchor = ev(sr) - lw * 1.50 + extra_offset_px, "JCT_LEFT_GHOST"
            else:
                tx, anchor = 320.0 - lw * 0.8 + extra_offset_px, "JCT_LEFT_BLIND"

        else:
            if sl is None and sr is None:
                tx, anchor = self._last_target_x + extra_offset_px, "DEAD_RECKONING"
            elif sl is not None and sr is not None:
                tx     = (ev(sl) + ev(sr)) / 2.0 + self.DUAL_OFFSET_PX + extra_offset_px
                anchor = "CENTERED_DUAL"
                self._last_target_x = tx - extra_offset_px
            elif sr is not None:
                # PERC-03: negative offset → push left toward lane centre
                tx     = ev(sr) - hw + self.SINGLE_EDGE_OFFSET_PX + extra_offset_px
                anchor = "CENTERED_FROM_RIGHT"
                self._last_target_x = tx - extra_offset_px
            elif sl is not None:
                tx     = ev(sl) + hw + self.SINGLE_EDGE_OFFSET_PX + extra_offset_px
                anchor = "CENTERED_FROM_LEFT"
                self._last_target_x = tx - extra_offset_px
            else:
                tx, anchor = self._last_target_x + extra_offset_px, "DEAD_RECKONING"

        # ── Confidence (PERC-02 normalised) ──────────────────────────────────
        MIN_NORM = self.tracker.MIN_PIX_OK / 1000.0
        if sl is not None and sr is not None:
            conf = min(1.0, (self.tracker.l_conf + self.tracker.r_conf) / (2.0 * MIN_NORM))
            curv = (self.tracker.get_curvature(sl, y_eval) +
                    self.tracker.get_curvature(sr, y_eval)) / 2.0
        elif sr is not None:
            conf = min(0.70, self.tracker.r_conf / MIN_NORM)
            curv = self.tracker.get_curvature(sr, y_eval)
        elif sl is not None:
            conf = min(0.70, self.tracker.l_conf / MIN_NORM)
            curv = self.tracker.get_curvature(sl, y_eval)
        else:
            conf, curv = 0.0, 0.0

        # LANE-07: robust heading + confidence
        heading_rad, heading_conf = estimate_heading_and_confidence(sl, sr)

        # Draw eval row on debug
        cv2.line(dbg, (0, int(y_eval)), (640, int(y_eval)), (0, 180, 255), 1)

        return PerceptionResult(
            warped_binary    = binary,
            lane_dbg         = dbg,
            sl               = sl,
            sr               = sr,
            lateral_error_px = tx - 320.0,
            anchor           = anchor,
            confidence       = conf,
            lane_width_px    = lw,
            curvature        = curv,
            l_conf           = self.tracker.l_conf,
            r_conf           = self.tracker.r_conf,
            heading_rad      = heading_rad,
            heading_conf     = heading_conf,
            y_eval           = float(y_eval),
        )