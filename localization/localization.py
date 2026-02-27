"""
localization.py — IMU-Free Visual Dead-Reckoning Localizer  (FIXED v2)
=======================================================================
All fixes applied:
  VL-01  VO EMA alpha corrected to 0.08/0.92 + spike gate at 1.5 rad/s
  VL-02  Heading sign convention fixed — negation removed from heading;
          layer-3 nudge now uses -tangent (correct direction)
  VL-04  Map snap uses perpendicular-foot projection (not midpoint)
  VL-07  set_pose() resets _cam_yaw_smoothed to the new yaw_rad
  VL-08  Path cursor exposed & reset via reset_cursor()
  NEW    Path-heading nudge (Layer 2) blends A* heading when confidence > 0.5
  NEW    get_upcoming_curve_from_path() uses cursor-window walk on planned path
  NEW    Curvature-gated map snap (disabled when curvature > 0.005)
"""

import math
import threading
import logging
import time
from collections import deque

try:
    from map_planner import PathPlanner
    _PLANNER_AVAILABLE = True
except ImportError:
    _PLANNER_AVAILABLE = False

log = logging.getLogger(__name__)


class LocalizationEngine:
    """
    4-layer IMU-free pose estimator: (x, y, yaw) in map metres.

    Layer 1 — Camera yaw-rate integration           (primary heading)
    Layer 2 — A* path heading soft nudge            (prevents long-run drift)
    Layer 3 — Forward dead-reckoning x/y            (velocity × dt)
    Layer 4 — Perpendicular map-snap                (cancels lateral drift)

    Sign convention (all layers):
        yaw in radians, map frame
        positive yaw  = counter-clockwise = LEFT turn
        negative yaw  = clockwise         = RIGHT turn
        x increases RIGHT,  y increases UP  (world / GraphML frame)
    """

    _MAX_CAM_YAW_CORRECTION = 0.05   # rad — max soft nudge per frame
    _CAM_YAW_EMA            = 0.55   # for layer-3 nudge smoothing

    def __init__(self):
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self.visual_yaw_rate = 0.0   # rad/s — exposed to controller

        self._lock = threading.RLock()
        self._prev_cam_heading = None
        self._cam_yaw_smoothed = 0.0
        self._initialized      = False

        # Public state read by main.py
        self.upcoming_curve = "STRAIGHT"
        self.current_zone   = "CITY"

        # Path-cursor state (maintained by Orchestrator via update_cursor)
        self._path_cursor = 0

        self.planner = None
        self._map_snap_enabled = _PLANNER_AVAILABLE
        if self._map_snap_enabled:
            self.planner = PathPlanner()
            if self.planner.graph is None or len(self.planner.graph.nodes) == 0:
                log.warning("GraphML map empty — disabling map snap.")
                self._map_snap_enabled = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_pose(self, x: float, y: float, yaw_rad: float):
        """Called when user clicks the SVG map.  Resets all filter state."""
        with self._lock:
            self.x   = x
            self.y   = y
            self.yaw = yaw_rad
            self.visual_yaw_rate = 0.0
            self._initialized    = True

            # FIX VL-07: seed EMA from actual yaw so first frame has no spike
            self._cam_yaw_smoothed = yaw_rad
            self._prev_cam_heading = None

            # Reset derived state
            self.upcoming_curve = "STRAIGHT"
            self.current_zone   = (self.planner.get_zone(x, y)
                                   if self.planner else "CITY")
        log.info(f"Pose set: x={x:.2f} y={y:.2f} yaw={math.degrees(yaw_rad):.1f}°")

    def reset_cursor(self):
        """FIX VL-08: call this whenever a new route is planned."""
        self._path_cursor = 0

    def update_cursor(self, path, x, y):
        """
        FIX VL-06: O(1) incremental cursor update — searches only a ±10
        node window around the current cursor.  Returns the new cursor.
        """
        if not path or not self.planner:
            return self._path_cursor

        search_start = max(0,              self._path_cursor - 3)
        search_end   = min(len(path) - 1,  self._path_cursor + 12)

        best_idx = self._path_cursor
        best_d   = float('inf')
        for i in range(search_start, search_end + 1):
            n = path[i]
            if n not in self.planner.node_positions:
                continue
            nx, ny = self.planner.node_positions[n]
            d = math.hypot(nx - x, ny - y)
            if d < best_d:
                best_d = d
                best_idx = i

        self._path_cursor = best_idx
        return self._path_cursor

    def get_pose(self):
        with self._lock:
            return self.x, self.y, self.yaw

    def is_initialized(self):
        with self._lock:
            return self._initialized

    def get_upcoming_curve_from_path(self, path, cursor, velocity_ms=0.3):
        """
        FIX MAP-02 / NEW: velocity-adaptive lookahead along planned A* path.
        Replaces the old greedy out-edge walk that broke at junctions.
        Returns 'LEFT', 'RIGHT', or 'STRAIGHT'.
        """
        if not path or not self.planner or cursor >= len(path) - 1:
            self.upcoming_curve = "STRAIGHT"
            return "STRAIGHT"

        la_m = max(1.8, velocity_ms * 6.0)   # adaptive: faster = more preview
        start_pos = self.planner.node_positions.get(path[cursor])
        if start_pos is None:
            self.upcoming_curve = "STRAIGHT"
            return "STRAIGHT"

        accum = 0.0
        with self._lock:
            curr_yaw = self.yaw

        for i in range(cursor, min(cursor + 40, len(path) - 1)):
            n1 = path[i]
            n2 = path[i + 1]
            p1 = self.planner.node_positions.get(n1)
            p2 = self.planner.node_positions.get(n2)
            if p1 is None or p2 is None:
                continue
            accum += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if accum >= la_m:
                target_yaw = math.atan2(p2[1] - start_pos[1],
                                        p2[0] - start_pos[0])
                diff = (target_yaw - curr_yaw + math.pi) % (2 * math.pi) - math.pi
                deg  = math.degrees(diff)
                if deg > 18:
                    self.upcoming_curve = "LEFT"
                elif deg < -18:
                    self.upcoming_curve = "RIGHT"
                else:
                    self.upcoming_curve = "STRAIGHT"
                return self.upcoming_curve

        self.upcoming_curve = "STRAIGHT"
        return "STRAIGHT"

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self,
               velocity_ms:        float,
               dt:                 float,
               camera_heading_rad: float = 0.0,
               camera_confidence:  float = 0.0,
               path=None,
               path_cursor:        int   = 0):
        """
        Update pose for one time step.

        Parameters
        ----------
        velocity_ms         : forward speed in m/s (from encoder)
        dt                  : elapsed seconds since last call
        camera_heading_rad  : raw lane tangent heading (NOT negated — caller
                              must pass the un-negated value from perception)
        camera_confidence   : 0–1 from perception module
        path                : planned A* node list (optional, for layer 2)
        path_cursor         : current cursor index into path
        """
        if dt <= 0 or not self._initialized:
            return

        with self._lock:
            # ── Layer 1: Camera Yaw-Rate Integration ─────────────────────────
            # FIX VL-01: alpha 0.08/0.92 — ~11-frame time constant at 30 Hz
            # FIX VL-02: camera_heading_rad is NOT negated here (sign fix)
            if camera_confidence > 0.3 and self._prev_cam_heading is not None:
                d_heading = camera_heading_rad - self._prev_cam_heading
                # Wrap to [-π, π]
                d_heading = (d_heading + math.pi) % (2 * math.pi) - math.pi
                raw_yaw_rate = d_heading / dt

                # Spike gate: discard if > 1.5 rad/s (BEV fitting artifact)
                if abs(raw_yaw_rate) < 1.5:
                    # FIX VL-01: correct EMA — 8% new signal
                    self.visual_yaw_rate = (0.08 * raw_yaw_rate
                                            + 0.92 * self.visual_yaw_rate)
            elif camera_confidence <= 0.3:
                self._prev_cam_heading = None
                self.visual_yaw_rate  *= 0.85   # decay when blind

            if camera_confidence > 0.3:
                self._prev_cam_heading = camera_heading_rad

            self.yaw += self.visual_yaw_rate * dt
            self.yaw  = (self.yaw + math.pi) % (2 * math.pi) - math.pi

            # ── Layer 2: A* Path Heading Nudge ───────────────────────────────
            # NEW: Blends 5% of the known road direction to prevent heading drift.
            if (path and path_cursor < len(path) - 1
                    and camera_confidence > 0.5
                    and self.planner):
                n1 = path[path_cursor]
                n2 = path[min(path_cursor + 1, len(path) - 1)]
                p1 = self.planner.node_positions.get(n1)
                p2 = self.planner.node_positions.get(n2)
                if p1 and p2:
                    ph = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    # Blend softly — only nudge, never snap
                    diff = (ph - self.yaw + math.pi) % (2 * math.pi) - math.pi
                    self.yaw += 0.05 * diff
                    self.yaw  = (self.yaw + math.pi) % (2 * math.pi) - math.pi

            # ── Layer 3: Forward Dead-Reckoning ──────────────────────────────
            effective_v = max(0.0, velocity_ms)
            self.x += effective_v * math.cos(self.yaw) * dt
            self.y += effective_v * math.sin(self.yaw) * dt

            # ── Layer 3b: Lane-Tangent Soft Heading Nudge ─────────────────────
            # FIX VL-02: nudge uses -camera_heading_rad (corrects lateral error)
            if camera_confidence > 0.25 and abs(camera_heading_rad) < 0.5:
                # Smooth the nudge signal
                self._cam_yaw_smoothed = (
                    self._CAM_YAW_EMA * (-camera_heading_rad)
                    + (1.0 - self._CAM_YAW_EMA) * self._cam_yaw_smoothed
                )
                nudge = self._cam_yaw_smoothed * camera_confidence
                nudge = max(-self._MAX_CAM_YAW_CORRECTION,
                            min(self._MAX_CAM_YAW_CORRECTION, nudge))
                self.yaw += nudge * 0.08
                self.yaw  = (self.yaw + math.pi) % (2 * math.pi) - math.pi

            # ── Layer 4: Map Snap ─────────────────────────────────────────────
            if self._map_snap_enabled and self.planner:
                self._apply_map_snap(effective_v, dt, camera_confidence,
                                     path, path_cursor)
                self.current_zone = self.planner.get_zone(self.x, self.y)

    def _apply_map_snap(self, velocity, dt, cam_conf, path, cursor):
        """
        FIX VL-04: Perpendicular-foot projection onto the nearest path edge.
        Only fires when driving straight (curvature < 0.005) and confident.
        """
        if velocity < 0.05 or cam_conf < 0.3:
            return

        # Only snap when close enough to a valid path segment
        if not path or cursor >= len(path) - 1:
            return

        # Check current path curvature — disable snap during corners
        curvature = self.planner.get_path_curvature(
            self.x, self.y, path, cursor=cursor, window_m=0.8)
        if curvature > 0.005:
            return

        # Find the best perpendicular foot on the nearest path edges
        best_dist = float('inf')
        best_foot = None

        search_start = max(0,            cursor - 2)
        search_end   = min(len(path) - 1, cursor + 6)

        for i in range(search_start, search_end):
            n1 = path[i]
            n2 = path[i + 1]
            p1 = self.planner.node_positions.get(n1)
            p2 = self.planner.node_positions.get(n2)
            if p1 is None or p2 is None:
                continue

            # FIX VL-04: Perpendicular-foot projection
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            seg_len_sq = ex * ex + ey * ey
            if seg_len_sq < 1e-8:
                continue
            t = ((self.x - p1[0]) * ex + (self.y - p1[1]) * ey) / seg_len_sq
            t = max(0.0, min(1.0, t))
            foot_x = p1[0] + t * ex
            foot_y = p1[1] + t * ey
            d = math.hypot(self.x - foot_x, self.y - foot_y)
            if d < best_dist:
                best_dist = d
                best_foot = (foot_x, foot_y)

        # Only snap if within 0.5 m — larger distances suggest wrong road
        if best_foot and best_dist < 0.50:
            pull = 0.15 * dt   # 15% / second — gentle
            self.x = self.x + pull * (best_foot[0] - self.x)
            self.y = self.y + pull * (best_foot[1] - self.y)