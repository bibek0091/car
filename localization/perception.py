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
    NWINDOWS = 9
    SW_MARGIN = 60
    MINPIX = 50
    STALE_FIT_FRAMES = 5

    def __init__(self, h=480, w=640):
        self.h, self.w = h, w
        self.mode = "SEARCH"
        self.sl, self.sr = None, None
        self.l_stale, self.r_stale = 0, 0
        self.l_conf, self.r_conf = 0.0, 0.0
        self.lane_width_px = 280.0

    def get_curvature(self, fit, y_eval):
        if fit is None: return 0.0
        a, b = fit[0], fit[1]
        denom = (1.0 + (2.0 * a * y_eval + b)**2)**1.5
        return abs(2.0 * a) / max(denom, 1e-6)

    def _sliding_window(self, warped):
        nz = warped.nonzero()
        nzy, nzx = np.array(nz[0]), np.array(nz[1])
        hist = np.sum(warped[self.h//2:, :], axis=0)
        mid = int(self.w * 0.40)  # 256 instead of 320 to handle right driving lanes
        
        lb = int(np.argmax(hist[:mid]))
        rb = int(np.argmax(hist[mid:])) + mid
        
        wh = self.h // self.NWINDOWS
        lx, rx = lb, rb
        li, ri = [], []
        
        for w in range(self.NWINDOWS):
            y_lo, y_hi = self.h - (w+1)*wh, self.h - w*wh
            xl0, xl1 = lx - self.SW_MARGIN, lx + self.SW_MARGIN
            xr0, xr1 = rx - self.SW_MARGIN, rx + self.SW_MARGIN
            
            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0) & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0) & (nzx < xr1)).nonzero()[0]
            
            li.append(gl); ri.append(gr)
            if len(gl) > self.MINPIX: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.MINPIX: rx = int(np.mean(nzx[gr]))
            
        return np.concatenate(li), np.concatenate(ri), nzx, nzy

    def _poly_search(self, warped):
        nz = warped.nonzero()
        nzy, nzx = np.array(nz[0]), np.array(nz[1])
        m = 60
        
        def band(fit):
            if fit is None: return np.array([], dtype=int)
            cx = np.polyval(fit, nzy)
            return ((nzx > cx - m) & (nzx < cx + m)).nonzero()[0]
            
        li = band(self.sl)
        ri = band(self.sr)
        
        if len(li) < 200 and len(ri) < 200:
            return self._sliding_window(warped)
        return li, ri, nzx, nzy

    def update(self, warped):
        if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
            li, ri, nzx, nzy = self._poly_search(warped)
        else:
            li, ri, nzx, nzy = self._sliding_window(warped)
            
        has_l = len(li) > 200
        has_r = len(ri) > 200
        
        # Extract points for polyfit
        ly, lx = nzy[li], nzx[li]
        ry, rx = nzy[ri], nzx[ri]

        if has_l:
            self.sl = np.polyfit(ly, lx, 2)
            self.l_stale = 0
            self.l_conf = min(1.0, len(li) / 1000)
        else:
            self.l_stale += 1
            if self.l_stale > self.STALE_FIT_FRAMES: self.sl = None
            self.l_conf = 0.0
            
        if has_r:
            self.sr = np.polyfit(ry, rx, 2)
            self.r_stale = 0
            self.r_conf = min(1.0, len(ri) / 1000)
        else:
            self.r_stale += 1
            if self.r_stale > self.STALE_FIT_FRAMES: self.sr = None
            self.r_conf = 0.0
            
        # Decision
        self.mode = "TRACKING" if (self.sl is not None or self.sr is not None) else "SEARCH"
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80, 80, 255]
        
        if has_l and has_r:
            lx = np.polyval(self.sl, self.h//2)
            rx = np.polyval(self.sr, self.h//2)
            w = rx - lx
            if 150 < w < 400:
                self.lane_width_px = 0.9 * self.lane_width_px + 0.1 * w
        
        self.lane_width_px = max(150, min(self.lane_width_px, 400))
        self.mode = "TRACKING" if (has_l or has_r) else "SEARCH"
        
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
        L = lab[:,:,0]
        L_clahe = self.clahe.apply(L)
        mean_L = np.mean(L_clahe)
        
        if mean_L < 100: alpha = 1.2
        elif mean_L > 180: alpha = 0.8
        else: alpha = 1.0
        
        L_adj = cv2.convertScaleAbs(L_clahe, alpha=alpha, beta=0)
        binary = cv2.adaptiveThreshold(L_adj, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 15)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
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
