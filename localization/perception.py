import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class PerceptionResult:
    warped_binary: np.ndarray
    lane_dbg: np.ndarray
    sl: np.ndarray # Left polynomial
    sr: np.ndarray # Right polynomial
    lateral_error_px: float
    anchor: str
    confidence: float
    lane_width_px: float
    curvature: float
    l_conf: float
    r_conf: float

class HybridLaneTracker:
    NWINDOWS         = 9
    SW_MARGIN        = 60
    MINPIX           = 50
    POLY_MARGIN_BASE = 60
    POLY_MARGIN_CURV = 120
    MIN_PIX_OK       = 200
    EMA_ALPHA        = 0.50
    STALE_FIT_FRAMES = 5

    def __init__(self, h=480, w=640):
        self.h, self.w = h, w
        self.mode = "SEARCH"
        self.left_fit = None
        self.right_fit = None
        self.sl, self.sr = None, None
        self.l_stale, self.r_stale = 0, 0
        self.l_conf, self.r_conf = 0.0, 0.0
        self.lane_width_px = 280.0

    def get_curvature(self, fit, y_eval):
        if fit is None: return 0.0
        a, b = fit[0], fit[1]
        denom = (1.0 + (2.0 * a * y_eval + b)**2)**1.5
        return abs(2.0 * a) / max(denom, 1e-6)

    def _ema(self, prev, new):
        if prev is None: return new.copy()
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev

    def _sliding_window(self, warped, nzx, nzy):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
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

    def update(self, warped):
        nz  = warped.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])

        if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
            curv = self.get_curvature(self.sl if self.sl is not None else self.sr, self.h // 2)
            li, ri, dbg = self._poly_search(warped, nzx, nzy, curvature=curv)
            mode_label  = "POLY"
        else:
            li, ri, dbg = self._sliding_window(warped, nzx, nzy)
            mode_label  = "SLIDE"

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
                    lx = np.polyval(self.sl, y)
                    rx = np.polyval(self.sr, y)
                    widths.append(rx - lx)
                w = np.average(widths, weights=[4, 3, 2, 1])
                self.lane_width_px = 0.8 * self.lane_width_px + 0.2 * w

        self.lane_width_px = max(150.0, min(self.lane_width_px, 400.0))
        self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"
        
        return dbg

class VisionPipeline:
    def __init__(self):
        self.tracker = HybridLaneTracker()
        self.SRC_PTS = np.float32([[200,260],[440,260],[40,450],[600,450]])
        self.DST_PTS = np.float32([[150,0],[490,0],[150,480],[490,480]])
        self.M = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
    def process(self, frame_bgr):
        warped = cv2.warpPerspective(frame_bgr, self.M, (640, 480))
        
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        L = self.clahe.apply(lab[:, :, 0])
        
        # Adaptive Lighting Compensation
        # BUG 11: Gradual interpolation prevents harsh contrast snapping (from bfmc_pilot_v3_yolo)
        mean_l = np.mean(L)
        if mean_l < 100:
            a = 1.0 + (100 - mean_l) / 200
            b = (100 - mean_l) * 0.6
            L = cv2.convertScaleAbs(L, alpha=a, beta=int(b))
        elif mean_l > 180:
            a = 1.0 - (mean_l - 180) / 350
            b = -(mean_l - 180) * 0.4
            L = cv2.convertScaleAbs(L, alpha=a, beta=int(b))

        # Track is WHITE, lines are BLACK (dark spots). Filtering out shadows with a +15 constant adjustment.
        binary = cv2.adaptiveThreshold(
            L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 15)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        dbg = self.tracker.update(binary)
        
        sl, sr = self.tracker.sl, self.tracker.sr
        lw = self.tracker.lane_width_px
        anchor = "DEAD_RECKONING"
        
        # Coarse 3-level confidence (1.0, 0.7, 0.0) is sufficient for current controller architecture
        conf = 0.0
        curv = 0.0
        
        y_eval = 400 # look lower down for immediate steering
        if sl is not None and sr is not None:
            tx = (np.polyval(sl, y_eval) + np.polyval(sr, y_eval))/2.0
            anchor, conf, curv = "DUAL", (self.tracker.l_conf + self.tracker.r_conf) / 2.0, (self.tracker.get_curvature(sl, y_eval)+self.tracker.get_curvature(sr, y_eval))/2.0
        elif sr is not None:
            tx = np.polyval(sr, y_eval) - lw/2.0
            anchor, conf, curv = "RIGHT", self.tracker.r_conf * 0.7, self.tracker.get_curvature(sr, y_eval)
        elif sl is not None:
            tx = np.polyval(sl, y_eval) + lw/2.0
            anchor, conf, curv = "LEFT", self.tracker.l_conf * 0.7, self.tracker.get_curvature(sl, y_eval)
        else:
            tx = 320.0
            anchor, conf, curv = "DEAD_RECKONING", 0.0, 0.0
            
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
