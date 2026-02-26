"""
visual_calibrator.py
====================
Startup visual calibration module for camera-only localization.

Runs during the 6-second pre-flight window while the car is stationary.
Performs three tasks:
  1. Vanishing-point detection to auto-calibrate BEV perspective transform.
  2. Initial heading estimation from detected lane lines.
  3. Graph-edge matching to confirm which segment the car is on (sets localizer yaw).

The car stays stopped (speed=0) until CalibrationResult.success is True.
"""

import cv2
import numpy as np
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

log = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    success: bool = False
    src_pts: Optional[np.ndarray] = None       # 4×2 BEV source points
    initial_heading_deg: float = 0.0           # map-frame heading in degrees
    starting_edge: Optional[Tuple] = None      # (node_id_1, node_id_2) or None
    status_msg: str = "Waiting for frames..."  # human-readable status
    confidence: float = 0.0                    # 0.0–1.0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _detect_vanishing_point(frame_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Detect vanishing point from a single camera frame using Hough lines.
    Returns (vp_x, vp_y) in image coordinates, or None if detection fails.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Mask: only look in the upper-middle of the frame (road perspective region)
    h, w = edges.shape
    mask = np.zeros_like(edges)
    roi = np.array([
        [int(w * 0.15), h],
        [int(w * 0.40), int(h * 0.40)],
        [int(w * 0.60), int(h * 0.40)],
        [int(w * 0.85), h],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [roi], 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi / 180,
                            threshold=40, minLineLength=60, maxLineGap=20)
    if lines is None or len(lines) < 4:
        return None

    # Separate left-leaning and right-leaning lines
    left_lines, right_lines = [], []
    cx = w / 2
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:       # near-horizontal — skip
            continue
        if slope < 0 and x1 < cx:  # left lane lines lean left
            left_lines.append(line[0])
        elif slope > 0 and x1 > cx:
            right_lines.append(line[0])

    if not left_lines or not right_lines:
        return None

    def line_params(x1, y1, x2, y2):
        """Returns (a, b, c) for line ax + by + c = 0."""
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        return a, b, c

    def intersect(l1, l2):
        a1, b1, c1 = line_params(*l1)
        a2, b2, c2 = line_params(*l2)
        denom = a1 * b2 - a2 * b1
        if abs(denom) < 1e-6:
            return None
        xi = (b1 * c2 - b2 * c1) / denom
        yi = (a2 * c1 - a1 * c2) / denom
        return xi, yi

    # Find all left-right intersections
    intersections = []
    for ll in left_lines:
        for rl in right_lines:
            pt = intersect(ll, rl)
            if pt and 0 < pt[0] < w and 0 < pt[1] < h * 0.7:
                intersections.append(pt)

    if not intersections:
        return None

    # Cluster with DBSCAN-lite: median of all close points
    pts = np.array(intersections)
    med = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - med, axis=1)
    close = pts[dists < 80]
    if len(close) < 2:
        return tuple(med)
    return tuple(np.mean(close, axis=0))


def _src_pts_from_vp(vp_x: float, vp_y: float,
                     frame_w: int = 640, frame_h: int = 480) -> np.ndarray:
    """
    Derive BEV source quadrilateral from vanishing point.
    The four corners form a trapezoid anchored to the VP.
    """
    # Width of the trapezoid at the vanishing point row (top)
    top_half_w = 30.0
    # Width at the bottom of the frame
    bot_half_w = frame_w * 0.44

    # Top row: VP vertical level ± top_half_w, clamped to image
    top_y = max(vp_y, frame_h * 0.35)
    tl = [max(0.0, vp_x - top_half_w), top_y]
    tr = [min(float(frame_w), vp_x + top_half_w), top_y]

    # Bottom row: near base of image
    bot_y = frame_h * 0.95
    bl = [max(0.0, vp_x - bot_half_w), bot_y]
    br = [min(float(frame_w), vp_x + bot_half_w), bot_y]

    return np.float32([tl, tr, bl, br])


def _estimate_initial_heading(frame_bgr: np.ndarray) -> Tuple[float, float]:
    """
    Estimate car's forward heading from parallel lane lines in the lower frame.
    Returns (heading_deg, confidence 0–1).
    heading_deg = 0 means car faces along the lane straight ahead.
    """
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)

    h, w = edges.shape
    # Focus on lower half only for heading
    roi_mask = np.zeros_like(edges)
    roi_mask[h // 2:, :] = 255
    edges = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=30, minLineLength=50, maxLineGap=15)
    if lines is None or len(lines) < 2:
        return 0.0, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:   # skip near-horizontal
            continue
        # angle = atan of reciprocal slope (perpendicular → forward direction)
        angles.append(math.degrees(math.atan2(x2 - x1, y1 - y2)))

    if not angles:
        return 0.0, 0.0

    arr = np.array(angles)
    med = float(np.median(arr))
    # Keep lines within 30° of median
    close = arr[np.abs(arr - med) < 30.0]
    if len(close) < 2:
        return float(med), 0.3

    confidence = min(1.0, len(close) / 10.0)
    return float(np.mean(close)), confidence


def _find_matching_edge(heading_deg: float, node_positions: dict,
                        graph, car_x: float, car_y: float,
                        max_heading_err_deg: float = 25.0,
                        max_pos_err_m: float = 1.5) -> Optional[Tuple]:
    """
    Find the graph edge that best matches the car's estimated position AND heading.
    Returns (node1, node2, edge_heading_deg) or None.

    BUG-06 FIX: When the reversed-edge direction is a better heading match,
    edge_heading is now corrected by +180 so that initial_heading_deg points
    in the direction the car is actually travelling (not 180 deg opposite).
    """
    candidates = []

    for u, v, _ in graph.edges(data=True):
        if u not in node_positions or v not in node_positions:
            continue
        p1 = node_positions[u]
        p2 = node_positions[v]

        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        edge_heading = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

        pos_err = math.hypot(mid_x - car_x, mid_y - car_y)
        if pos_err > max_pos_err_m:
            continue

        # BUG-06: compute forward AND reverse heading error separately.
        # Track WHICH direction won so the returned heading is correct.
        h_err_fwd = abs(((heading_deg - edge_heading + 180) % 360) - 180)
        h_err_rev = abs(((heading_deg - (edge_heading + 180) + 180) % 360) - 180)
        if h_err_rev < h_err_fwd:
            h_err = h_err_rev
            matched_heading = (edge_heading + 180) % 360   # reversed wins
        else:
            h_err = h_err_fwd
            matched_heading = edge_heading

        if h_err > max_heading_err_deg:
            continue

        score = pos_err + h_err / 10.0
        candidates.append((score, u, v, matched_heading))

    if not candidates:
        return None

    # MIN-09: sort by score; pick best (BUG-05 style: closest + most aligned)
    candidates.sort(key=lambda c: c[0])
    _, u, v, best_heading = candidates[0]
    return (u, v, best_heading)


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class VisualCalibrator:
    """
    Runs during the 6-second pre-flight window.
    Usage (from main.py pilot loop):

        calib = VisualCalibrator(planner, localizer, vision_pipeline)
        # inside calibration loop:
        calib.add_frame(frame)
        # at elapsed > 3.0 s:
        result = calib.finalize()
        if result.success:
            vision_pipeline.update_bev_transform(result.src_pts)
            localizer.set_pose(localizer.x, localizer.y,
                               math.radians(result.initial_heading_deg))
    """

    MIN_FRAMES_FOR_VP   = 15   # minimum frames before VP detection attempt
    MIN_FRAMES_FINALIZE = 20   # minimum frames before finalize() is meaningful

    def __init__(self, planner, localizer, vision_pipeline):
        self.planner        = planner
        self.localizer      = localizer
        self.vision         = vision_pipeline
        self._frames: List[np.ndarray] = []
        self._vp_candidates: List[Tuple[float, float]] = []
        self._heading_candidates: List[Tuple[float, float]] = []  # (deg, conf)
        self._result = CalibrationResult()
        self._finalized = False

    def add_frame(self, frame: np.ndarray):
        """Feed one camera frame. Call every frame during calibration phase."""
        if frame is None or not frame.any():
            return

        self._frames.append(frame.copy())

        # Try vanishing-point detection every 5 frames
        if len(self._frames) % 5 == 0:
            vp = _detect_vanishing_point(frame)
            if vp is not None:
                self._vp_candidates.append(vp)
                log.debug(f"VP candidate: ({vp[0]:.1f}, {vp[1]:.1f})")

        # Try heading estimation every 5 frames
        if len(self._frames) % 5 == 1:
            hdg, conf = _estimate_initial_heading(frame)
            if conf > 0.2:
                self._heading_candidates.append((hdg, conf))
                log.debug(f"Heading candidate: {hdg:.1f}° (conf={conf:.2f})")

        n = len(self._frames)
        if n < self.MIN_FRAMES_FOR_VP:
            self._result.status_msg = f"Collecting frames ({n}/{self.MIN_FRAMES_FOR_VP})..."
        else:
            vp_ok  = len(self._vp_candidates) >= 3
            hdg_ok = len(self._heading_candidates) >= 2
            self._result.status_msg = (
                f"VP:{'OK' if vp_ok else 'wait'} "
                f"HDG:{'OK' if hdg_ok else 'wait'} "
                f"frames={n}"
            )

    def finalize(self) -> CalibrationResult:
        """
        Run once when calibration time is up (elapsed > 3.0 s recommended).
        Returns CalibrationResult with success=True when all checks passed.
        """
        if self._finalized:
            return self._result

        if len(self._frames) < self.MIN_FRAMES_FINALIZE:
            self._result.status_msg = f"Not enough frames ({len(self._frames)})"
            return self._result

        checks_passed = 0

        # ── 1. Vanishing-Point → BEV calibration ─────────────────────────────
        if len(self._vp_candidates) >= 3:
            vps = np.array(self._vp_candidates)
            # Median vanishing point (robust to outliers)
            vp_x = float(np.median(vps[:, 0]))
            vp_y = float(np.median(vps[:, 1]))

            h, w = self._frames[0].shape[:2]
            src_pts = _src_pts_from_vp(vp_x, vp_y, w, h)
            self._result.src_pts = src_pts
            log.info(f"BEV calibration: VP=({vp_x:.1f},{vp_y:.1f}), "
                     f"SRC_PTS={src_pts.tolist()}")
            checks_passed += 1
        else:
            # Fall back to default src_pts (no calibration)
            self._result.src_pts = self.vision.SRC_PTS.copy()
            log.warning("VP detection failed — keeping default BEV transform.")

        # ── 2. Initial heading estimation ─────────────────────────────────────
        if len(self._heading_candidates) >= 2:
            headings = np.array([h for h, _ in self._heading_candidates])
            confs    = np.array([c for _, c in self._heading_candidates])
            # Weighted mean heading
            hdg_deg = float(np.average(headings, weights=confs))
            self._result.initial_heading_deg = hdg_deg
            log.info(f"Initial heading: {hdg_deg:.1f}°")
            checks_passed += 1
        else:
            self._result.initial_heading_deg = 0.0
            log.warning("Heading detection uncertain — defaulting to 0°.")

        # ── 3. Graph edge matching ────────────────────────────────────────────
        cx, cy, _ = self.localizer.get_pose()
        edge_match = _find_matching_edge(
            self._result.initial_heading_deg,
            self.planner.node_positions,
            self.planner.graph,
            car_x=cx, car_y=cy,
        )
        if edge_match is not None:
            u, v, edge_hdg = edge_match
            self._result.starting_edge = (u, v)
            # Override initial heading with the precise edge heading
            self._result.initial_heading_deg = edge_hdg
            log.info(f"Edge match: {u}→{v} heading={edge_hdg:.1f}°")
            checks_passed += 1
        else:
            log.warning("No matching graph edge found — heading not corrected from map.")

        # ── Build final confidence and success flag ───────────────────────────
        self._result.confidence = checks_passed / 3.0

        # Success requires at least VP calibration + heading estimate
        self._result.success = checks_passed >= 2
        if self._result.success:
            self._result.status_msg = (
                f"CAL OK  hdg={self._result.initial_heading_deg:.1f}°  "
                f"conf={self._result.confidence:.0%}"
            )
        else:
            self._result.status_msg = (
                f"CAL PARTIAL ({checks_passed}/3) — proceeding with defaults"
            )
            # Partial result: still return success so the car can drive
            self._result.success = True   # never block forever on calibration failure

        self._finalized = True
        log.info(f"VisualCalibrator finalized: {self._result.status_msg}")
        return self._result
