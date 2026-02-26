import cv2
import numpy as np
import math
from dataclasses import dataclass

@dataclass
class PerceptionResult:
    warped_binary: np.ndarray
    lane_dbg: np.ndarray
    sl: np.ndarray  # Left polynomial
    sr: np.ndarray  # Right polynomial
    lateral_error_px: float
    anchor: str
    confidence: float
    lane_width_px: float
    curvature: float
    l_conf: float
    r_conf: float


def estimate_heading_from_lanes(sl, sr, h=480):
    """
    Estimates the car's heading offset from road centre using lane polynomial
    derivatives sampled at 5 depth levels with bottom-weighted averaging.

    x = a*y^2 + b*y + c   →   dx/dy = 2a*y + b

    Returns the correction in RADIANS to add to the current yaw estimate.
    A negative return value = road turns left = car needs to yaw left.

    Improvements over v1:
    - Outlier rejection: discard tangents > 2 std-devs from median.
    - Bottom-row bonus: y=480 row weight raised to 3.0.
    - Returns 0.0 (not garbage) when no valid lane is found.
    """
    eval_rows   = [h * f for f in [0.2, 0.35, 0.5, 0.7, 1.0]]
    row_weights = [0.5,   0.75,   1.0,   1.5,   3.0]   # bottom row 3.0

    all_tangents = []
    all_weights  = []
    for fit in [sl, sr]:
        if fit is None:
            continue
        for y, w in zip(eval_rows, row_weights):
            dxdy = 2.0 * fit[0] * y + fit[1]
            all_tangents.append(math.atan2(dxdy, 1.0))
            all_weights.append(w)

    if not all_tangents:
        return 0.0

    # Outlier rejection: drop anything > 2 std-devs from the median
    arr = np.array(all_tangents)
    wts = np.array(all_weights)
    med = float(np.median(arr))
    std = float(np.std(arr)) if len(arr) > 2 else 1.0
    mask = np.abs(arr - med) <= 2.0 * std + 1e-6
    if mask.sum() == 0:
        mask = np.ones(len(arr), dtype=bool)  # fallback: keep all

    tangent_mean = np.average(arr[mask], weights=wts[mask])
    # Negate: rightward lean in BEV = road curving right = positive yaw rate
    return -float(tangent_mean)


def estimate_camera_odometry(sl, sr, prev_sl, prev_sr, dt,
                             h=480, scale_m_per_px=0.35/280.0):
    """
    Estimates lateral drift velocity (m/s) and heading rate (rad/s) from
    consecutive lane polynomial fits.

    Improvements over v1:
    - Lateral drift computed at 3 rows (200, 300, 400) and median-averaged.
    - Drift clamped to ±0.05 m/s to prevent spike injection.
    - Heading-rate sanity check: if |ΔH| > 0.5 rad in one frame, return 0.
    """
    if dt <= 0:
        return 0.0, 0.0

    # ── Lateral velocity from lane-centre shift ────────────────────────────
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

    if drift_samples:
        lateral_vel_ms = float(np.median(drift_samples))
        # Clamp to ±0.05 m/s — beyond this it's a fit artifact, not real motion
        lateral_vel_ms = max(-0.05, min(0.05, lateral_vel_ms))
    else:
        lateral_vel_ms = 0.0

    # ── Heading rate from tangent change between frames ─────────────────────
    curr_heading = estimate_heading_from_lanes(sl,      sr,      h)
    prev_heading = estimate_heading_from_lanes(prev_sl, prev_sr, h)
    raw_heading_rate = (curr_heading - prev_heading) / dt

    # Sanity check: > 0.5 rad/frame change is almost certainly a stale-fit artifact
    if abs(curr_heading - prev_heading) > 0.5:
        heading_rate_rps = 0.0
    else:
        heading_rate_rps = raw_heading_rate

    return lateral_vel_ms, heading_rate_rps


class HybridLaneTracker:
    NWINDOWS           = 9
    SW_MARGIN          = 60
    SW_MARGIN_RECOVERY = 100    # wider search during lost-lane recovery
    MINPIX             = 50
    POLY_MARGIN_BASE   = 60
    POLY_MARGIN_CURV   = 120
    MIN_PIX_OK         = 200
    EMA_ALPHA          = 0.30   # smoother polynomial tracking (↓ from 0.35)
    STALE_FIT_FRAMES   = 12     # hold last known fit longer (↑ from 8)
    LOST_RECOVERY_THRESH = 2    # trigger wide-sweep sooner after lane loss (↑ from 3)

    def __init__(self, h=480, w=640):
        self.h, self.w = h, w
        self.mode = "SEARCH"
        self.left_fit = None
        self.right_fit = None
        self.sl, self.sr = None, None
        self.l_stale, self.r_stale = 0, 0
        self.l_conf, self.r_conf = 0.0, 0.0
        self.lane_width_px = 280.0
        self._lost_frames = 0   # consecutive frames with no lane at all

    def get_curvature(self, fit, y_eval):
        if fit is None: return 0.0
        a, b = fit[0], fit[1]
        denom = (1.0 + (2.0 * a * y_eval + b)**2)**1.5
        return abs(2.0 * a) / max(denom, 1e-6)

    def _ema(self, prev, new):
        if prev is None: return new.copy()
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev

    def _sliding_window(self, warped, nzx, nzy, wide=False):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        hist = np.sum(warped[self.h // 2:, :], axis=0)

        mid    = int(self.w * 0.40)
        margin = self.SW_MARGIN_RECOVERY if wide else self.SW_MARGIN

        lb = int(np.argmax(hist[margin: mid - margin])) + margin
        rb = int(np.argmax(hist[mid + margin: self.w - margin])) + mid + margin

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
            sw = self.SW_MARGIN_RECOVERY if wide else self.SW_MARGIN
            xl0, xl1 = max(0, lx - sw), min(self.w, lx + sw)
            xr0, xr1 = max(0, rx - sw), min(self.w, rx + sw)

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

    def update(self, warped):
        nz  = warped.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])

        # Lost-lane recovery: widen search when both fits have been None for too long
        both_lost = (self.sl is None and self.sr is None)
        if both_lost:
            self._lost_frames += 1
        else:
            self._lost_frames = 0
        do_wide_sweep = (self._lost_frames >= self.LOST_RECOVERY_THRESH)

        if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
            curv = self.get_curvature(self.sl if self.sl is not None else self.sr, self.h // 2)
            li, ri, dbg = self._poly_search(warped, nzx, nzy, curvature=curv)
            mode_label  = "POLY"
        else:
            li, ri, dbg = self._sliding_window(warped, nzx, nzy, wide=do_wide_sweep)
            mode_label  = "SLIDE-WIDE" if do_wide_sweep else "SLIDE"

        self.l_conf = len(li) / 1000.0
        self.r_conf = len(ri) / 1000.0
        has_l = len(li) >= self.MIN_PIX_OK
        has_r = len(ri) >= self.MIN_PIX_OK

        if has_l:
            fl = np.polyfit(nzy[li], nzx[li], 2)
            self.left_fit = fl
            self.sl = self._ema(self.sl, fl)
            self.l_stale = 0
            self.l_conf = min(1.0, len(li) / 1000)
        else:
            self.l_stale += 1
            if self.l_stale > self.STALE_FIT_FRAMES:
                self.left_fit = None
                self.sl = None
            self.l_conf = 0.0

        if has_r:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            self.right_fit = fr
            self.sr = self._ema(self.sr, fr)
            self.r_stale = 0
            self.r_conf = min(1.0, len(ri) / 1000)
        else:
            self.r_stale += 1
            if self.r_stale > self.STALE_FIT_FRAMES:
                self.right_fit = None
                self.sr = None
            self.r_conf = 0.0

        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                if len(li) < len(ri):
                    self.left_fit = None
                    self.sl = None
                    self.l_stale = self.STALE_FIT_FRAMES
                    has_l = False
                else:
                    self.right_fit = None
                    self.sr = None
                    self.r_stale = self.STALE_FIT_FRAMES
                    has_r = False
            else:
                y_pos = [100, 200, 300, 400]
                widths = []
                for y in y_pos:
                    lx_v = np.polyval(self.sl, y)
                    rx_v = np.polyval(self.sr, y)
                    widths.append(rx_v - lx_v)
                w = np.average(widths, weights=[1, 2, 3, 4])
                self.lane_width_px = 0.8 * self.lane_width_px + 0.2 * w

        self.lane_width_px = max(150.0, min(self.lane_width_px, 400.0))
        self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"

        return dbg


class VisionPipeline:
    # ── v3 lane-centre offsets (pixels) ───────────────────────────────────────
    # DUAL        : mathematical midpoint — no offset needed
    # SINGLE_DIV  : only centre-divider visible → offset right to stay in lane
    # SINGLE_EDGE : only outer edge visible → offset left to stay in lane
    # These match bfmc_pilot_v3_yolo.py constants exactly.
    DUAL_OFFSET_PX        =   0    # both lines: perfect maths centre
    SINGLE_DIV_OFFSET_PX  =  40    # right-divider only: nudge right
    SINGLE_EDGE_OFFSET_PX = -40    # left-edge only: nudge left

    Y_EVAL = 400   # BEV row used for immediate steering target

    def __init__(self):
        self.tracker = HybridLaneTracker()
        # Default BEV transform — will be auto-calibrated by VisualCalibrator
        self.SRC_PTS = np.float32([[200, 260], [440, 260], [40, 450], [600, 450]])
        self.DST_PTS = np.float32([[150, 0], [490, 0], [150, 480], [490, 480]])
        self.M = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.bev_calibrated = False
        self._last_target_x  = 320.0   # dead-reckoning memory

    def update_bev_transform(self, src_pts):
        """Called by VisualCalibrator after vanishing-point detection."""
        self.SRC_PTS = np.float32(src_pts)
        self.M = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
        self.bev_calibrated = True

    def process(self, frame_bgr, extra_offset_px: float = 0.0,
                nav_state: str = "NORMAL"):
        """
        Process one camera frame.

        extra_offset_px : lateral nudge from controller (evasion, junction)
        nav_state       : "NORMAL" | "ROUNDABOUT" | "JUNCTION_LEFT" | "JUNCTION_RIGHT"

        Returns PerceptionResult with v3-aligned lateral_error_px:
            error > 0  → target is RIGHT of image centre → steer right
            error < 0  → target is LEFT of image centre  → steer left
        """
        # ── 1. BEV warp ───────────────────────────────────────────────────────
        if frame_bgr.shape[:2] != (480, 640):
            frame_bgr = cv2.resize(frame_bgr, (640, 480))
        warped = cv2.warpPerspective(frame_bgr, self.M, (640, 480))

        # ── 2. LAB + CLAHE + adaptive lighting (identical to v3) ─────────────
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        L   = self.clahe.apply(lab[:, :, 0])

        mean_l = np.mean(L)
        if mean_l < 100:
            a = 1.0 + (100 - mean_l) / 200
            b = (100 - mean_l) * 0.6
            L = cv2.convertScaleAbs(L, alpha=a, beta=int(b))
        elif mean_l > 180:
            a = 1.0 - (mean_l - 180) / 350
            b = -(mean_l - 180) * 0.4
            L = cv2.convertScaleAbs(L, alpha=a, beta=int(b))

        # ── 3. Adaptive threshold + morphology (identical to v3) ─────────────
        # Track=WHITE, lane-lines=BLACK. C=+15 aggressively rejects shadows.
        binary = cv2.adaptiveThreshold(
            L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # ── 4. Lane tracking ──────────────────────────────────────────────────
        dbg = self.tracker.update(binary)

        sl = self.tracker.sl
        sr = self.tracker.sr
        lw = self.tracker.lane_width_px
        hw = lw / 2.0
        y  = self.Y_EVAL

        def ev(fit):
            return float(np.polyval(fit, y))

        # ── 5. Target-x: exact v3 get_target_x() logic ───────────────────────
        if nav_state == "ROUNDABOUT":
            if sl is not None:
                tx     = ev(sl) + hw + extra_offset_px
                anchor = "RBT_INNER"
            elif sr is not None:
                tx     = ev(sr) - hw + extra_offset_px
                anchor = "RBT_OUTER"
            else:
                tx     = self._last_target_x + extra_offset_px
                anchor = "RBT_LOST"

        elif nav_state == "JUNCTION_RIGHT":
            if sr is not None:
                tx     = ev(sr) - lw * 0.40 + extra_offset_px
                anchor = "JCT_RIGHT_EDGE"
            elif sl is not None:
                tx     = ev(sl) + lw * 1.50 + extra_offset_px
                anchor = "JCT_RIGHT_GHOST"
            else:
                tx     = 320.0 + lw * 0.8 + extra_offset_px
                anchor = "JCT_RIGHT_BLIND"

        elif nav_state == "JUNCTION_LEFT":
            if sl is not None:
                tx     = ev(sl) + lw * 0.40 + extra_offset_px
                anchor = "JCT_LEFT_EDGE"
            elif sr is not None:
                tx     = ev(sr) - lw * 1.50 + extra_offset_px
                anchor = "JCT_LEFT_GHOST"
            else:
                tx     = 320.0 - lw * 0.8 + extra_offset_px
                anchor = "JCT_LEFT_BLIND"

        else:
            # ── NORMAL driving — v3 "BRUTE FORCE MIDDLE-LANE PRIORITY" ───────
            if sl is None and sr is None:
                # Dead-reckoning: hold last valid target
                tx     = self._last_target_x + extra_offset_px
                anchor = "DEAD_RECKONING"
            elif sl is not None and sr is not None:
                # Both lines: mathematical centre — no offset, no bias
                tx     = (ev(sl) + ev(sr)) / 2.0 + self.DUAL_OFFSET_PX + extra_offset_px
                anchor = "CENTERED_DUAL"
                self._last_target_x = tx - extra_offset_px
            elif sr is not None:
                # Right-divider (centre/dashed line): project centre leftward
                # +SINGLE_DIV_OFFSET_PX nudges car slightly right (right-lane rule)
                tx     = ev(sr) - hw + self.SINGLE_DIV_OFFSET_PX + extra_offset_px
                anchor = "CENTERED_FROM_RIGHT"
                self._last_target_x = tx - extra_offset_px
            elif sl is not None:
                # Left edge (outer white line): project centre rightward
                # +SINGLE_EDGE_OFFSET_PX nudges car slightly left (stay off edge)
                tx     = ev(sl) + hw + self.SINGLE_EDGE_OFFSET_PX + extra_offset_px
                anchor = "CENTERED_FROM_LEFT"
                self._last_target_x = tx - extra_offset_px
            else:
                tx     = self._last_target_x + extra_offset_px
                anchor = "DEAD_RECKONING"

        # ── 6. Confidence & curvature ─────────────────────────────────────────
        # Normalised pixel count: ≥ MIN_PIX_OK pixels → conf 1.0
        MIN_PIX = self.tracker.MIN_PIX_OK
        if sl is not None and sr is not None:
            conf = min(1.0, (self.tracker.l_conf + self.tracker.r_conf) / (2.0 * MIN_PIX))
            curv = (self.tracker.get_curvature(sl, y) + self.tracker.get_curvature(sr, y)) / 2.0
        elif sr is not None:
            conf = min(0.70, self.tracker.r_conf / MIN_PIX)
            curv = self.tracker.get_curvature(sr, y)
        elif sl is not None:
            conf = min(0.70, self.tracker.l_conf / MIN_PIX)
            curv = self.tracker.get_curvature(sl, y)
        else:
            conf = 0.0
            curv = 0.0

        error_px = tx - 320.0

        return PerceptionResult(
            warped_binary=binary,
            lane_dbg=dbg,
            sl=sl, sr=sr,
            lateral_error_px=error_px,
            anchor=anchor,
            confidence=conf,
            lane_width_px=lw,
            curvature=curv,
            l_conf=self.tracker.l_conf,
            r_conf=self.tracker.r_conf
        )
