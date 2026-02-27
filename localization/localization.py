"""
localization.py — IMU-Free Visual Dead-Reckoning Localizer  (FIXED v4)
=======================================================================
FIXES in v4 (on top of v3):

  LOC-01  upcoming_curve written under self._lock to prevent race condition
          with pilot thread. Result computed first, then assigned atomically.

  LOC-02  Map-snap pull dt clamped to 100 ms max. A stall spike (dt > 100 ms)
          can no longer jump the car position by more than 1.5% per frame.

  LOC-03  _update_cursor_internal enforces monotonic advance: cursor only
          increases, preventing oscillation on looping track sections.

Unchanged from v3:
  VL-FIX-A  Cursor self-managed inside update()
  VL-FIX-B  Map-snap 1.0 m primary + 2.0 m recovery radius
  VL-FIX-C  Yaw-rate EMA alpha 0.18
  VL-FIX-D  Uninitialized guard (silent ignore)
  VL-FIX-E  POI arrival detection
  VL-FIX-F  get_pose_for_dashboard() rich dict
  VL-01/02/04/07/08  (see v2 notes)
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

    Dashboard integration:
        Call get_pose_for_dashboard() each frame to get a dict with
        x, y, yaw_deg, zone, upcoming_curve, cursor, speed_ms.

    POI stopping:
        Call check_poi_arrival(target_node_id, threshold_m=0.40) each frame.
        Returns True when the car is within threshold_m of the target.
    """

    # ── Tunable constants ─────────────────────────────────────────────────────
    _MAX_CAM_YAW_CORRECTION = 0.05   # rad — max soft nudge per frame
    _CAM_YAW_EMA            = 0.55   # for Layer-3b nudge smoothing

    # FIX VL-FIX-C: 0.18 (was 0.08) → ~5.5-frame TC at 30 Hz
    _YAW_RATE_EMA_ALPHA     = 0.18   # new-signal weight for yaw-rate IIR

    _MAP_SNAP_RADIUS_M      = 1.00   # FIX VL-FIX-B: was 0.50 m
    _MAP_SNAP_RECOVERY_M    = 2.00   # wider radius after _SNAP_LOST_LIMIT frames
    _SNAP_LOST_LIMIT        = 60     # ~2 s at 30 Hz before recovery mode kicks in
    _MAP_SNAP_PULL          = 0.15   # fraction per second toward foot
    _POI_DEFAULT_THRESH_M   = 0.40   # default stop radius around target

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self):
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self.visual_yaw_rate = 0.0   # rad/s — exposed to controller

        self._lock = threading.RLock()
        self._prev_cam_heading = None
        self._cam_yaw_smoothed = 0.0
        self._initialized      = False

        # Public state read by main loop / dashboard
        self.upcoming_curve = "STRAIGHT"
        self.current_zone   = "CITY"
        self._last_speed_ms = 0.0    # cached for dashboard dict

        # FIX VL-FIX-A: cursor is fully self-managed
        self._path_cursor = 0
        self._snap_miss_frames = 0   # for recovery snap

        self.planner = None
        self._map_snap_enabled = _PLANNER_AVAILABLE
        if self._map_snap_enabled:
            self.planner = PathPlanner()
            if self.planner.graph is None or len(self.planner.graph.nodes) == 0:
                log.warning("GraphML map empty — disabling map snap.")
                self._map_snap_enabled = False

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def path_cursor(self) -> int:
        """Read-only access to the self-managed path cursor."""
        return self._path_cursor

    def set_pose(self, x: float, y: float, yaw_rad: float = None):
        """
        Called when user clicks the SVG map (or on startup).
        Resets all filter state.  yaw_rad is optional — if omitted, the
        current yaw is preserved (useful for position-only corrections).
        """
        with self._lock:
            self.x = x
            self.y = y
            if yaw_rad is not None:
                self.yaw = yaw_rad
            self.visual_yaw_rate = 0.0
            self._initialized    = True
            self._snap_miss_frames = 0

            # FIX VL-07: seed EMA from actual yaw so first frame has no spike
            self._cam_yaw_smoothed = self.yaw
            self._prev_cam_heading = None

            # Reset derived state
            self.upcoming_curve = "STRAIGHT"
            self.current_zone   = (self.planner.get_zone(x, y)
                                   if self.planner else "CITY")
        log.info(f"Pose set: x={x:.2f} y={y:.2f} "
                 f"yaw={math.degrees(self.yaw):.1f}°")

    def reset_cursor(self):
        """Call whenever a new route is planned (resets path cursor to 0)."""
        with self._lock:
            self._path_cursor = 0
            self._snap_miss_frames = 0
        log.info("Path cursor reset.")

    def get_pose(self):
        with self._lock:
            return self.x, self.y, self.yaw

    def is_initialized(self) -> bool:
        with self._lock:
            return self._initialized

    def get_pose_for_dashboard(self) -> dict:
        """
        FIX VL-FIX-F: Returns a dict suitable for direct dashboard display.
        Call this every frame from the main loop.
        """
        with self._lock:
            return {
                "x":             self.x,
                "y":             self.y,
                "yaw_deg":       math.degrees(self.yaw),
                "zone":          self.current_zone,
                "upcoming_curve": self.upcoming_curve,
                "cursor":        self._path_cursor,
                "speed_ms":      self._last_speed_ms,
                "initialized":   self._initialized,
            }

    def check_poi_arrival(self, target_node_id: str,
                          threshold_m: float = None) -> bool:
        """
        FIX VL-FIX-E: Returns True when the car is within threshold_m of
        target_node_id's map position.  Call each frame; when True, the
        orchestrator should command speed = 0 and hold.
        """
        if threshold_m is None:
            threshold_m = self._POI_DEFAULT_THRESH_M
        if not self.planner or target_node_id not in self.planner.node_positions:
            return False
        tx, ty = self.planner.node_positions[target_node_id]
        with self._lock:
            dist = math.hypot(self.x - tx, self.y - ty)
        return dist <= threshold_m

    def get_distance_to_node(self, node_id: str) -> float:
        """Returns metres to a named map node (inf if unknown)."""
        if not self.planner or node_id not in self.planner.node_positions:
            return float('inf')
        tx, ty = self.planner.node_positions[node_id]
        with self._lock:
            return math.hypot(self.x - tx, self.y - ty)

    def get_upcoming_curve_from_path(self, path, cursor=None,
                                     velocity_ms: float = 0.3) -> str:
        """
        Velocity-adaptive lookahead along planned A* path.
        cursor defaults to the internal self-managed cursor.
        Returns 'LEFT', 'RIGHT', or 'STRAIGHT'.
        """
        if cursor is None:
            cursor = self._path_cursor

        if not path or not self.planner or cursor >= len(path) - 1:
            self.upcoming_curve = "STRAIGHT"
            return "STRAIGHT"

        la_m = max(1.8, velocity_ms * 6.0)
        start_pos = self.planner.node_positions.get(path[cursor])
        if start_pos is None:
            self.upcoming_curve = "STRAIGHT"
            return "STRAIGHT"

        accum = 0.0
        with self._lock:
            curr_yaw = self.yaw

        result = "STRAIGHT"  # FIX LOC-01: compute result before writing shared attr
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
                diff = ((target_yaw - curr_yaw + math.pi)
                        % (2 * math.pi) - math.pi)
                deg = math.degrees(diff)
                if deg > 18:
                    result = "LEFT"
                elif deg < -18:
                    result = "RIGHT"
                else:
                    result = "STRAIGHT"
                break

        # FIX LOC-01: write shared attribute under lock
        with self._lock:
            self.upcoming_curve = result
        return result

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self,
               velocity_ms:        float,
               dt:                 float,
               camera_heading_rad: float = 0.0,
               camera_confidence:  float = 0.0,
               path=None):
        """
        Update pose for one time step.

        FIX VL-FIX-A: 'path_cursor' parameter removed — cursor is managed
        internally.  Callers no longer need to track or pass it.

        Parameters
        ----------
        velocity_ms         : forward speed in m/s (from encoder)
        dt                  : elapsed seconds since last call
        camera_heading_rad  : raw lane tangent heading (not negated)
        camera_confidence   : 0–1 from perception module
        path                : planned A* node list (optional, for layers 2 & 4)
        """
        if dt <= 0 or not self._initialized:
            return

        with self._lock:
            self._last_speed_ms = velocity_ms

            # FIX VL-FIX-A: advance cursor internally before any layer uses it
            if path:
                self._update_cursor_internal(path)
            cursor = self._path_cursor

            # ── Layer 1: Camera Yaw-Rate Integration ─────────────────────────
            # FIX VL-FIX-C: alpha = 0.18 (was 0.08); ~5.5-frame TC at 30 Hz
            if camera_confidence > 0.3 and self._prev_cam_heading is not None:
                d_heading = camera_heading_rad - self._prev_cam_heading
                d_heading = (d_heading + math.pi) % (2 * math.pi) - math.pi
                raw_yaw_rate = d_heading / dt

                # Spike gate: discard if > 1.5 rad/s (BEV fitting artifact)
                if abs(raw_yaw_rate) < 1.5:
                    self.visual_yaw_rate = (
                        self._YAW_RATE_EMA_ALPHA * raw_yaw_rate
                        + (1.0 - self._YAW_RATE_EMA_ALPHA) * self.visual_yaw_rate
                    )
            elif camera_confidence <= 0.3:
                self._prev_cam_heading = None
                self.visual_yaw_rate  *= 0.85

            if camera_confidence > 0.3:
                self._prev_cam_heading = camera_heading_rad

            self.yaw += self.visual_yaw_rate * dt
            self.yaw  = (self.yaw + math.pi) % (2 * math.pi) - math.pi

            # ── Layer 2: A* Path Heading Nudge ───────────────────────────────
            if (path and cursor < len(path) - 1
                    and camera_confidence > 0.5
                    and self.planner):
                n1 = path[cursor]
                n2 = path[min(cursor + 1, len(path) - 1)]
                p1 = self.planner.node_positions.get(n1)
                p2 = self.planner.node_positions.get(n2)
                if p1 and p2:
                    ph   = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    diff = (ph - self.yaw + math.pi) % (2 * math.pi) - math.pi
                    self.yaw += 0.05 * diff
                    self.yaw  = (self.yaw + math.pi) % (2 * math.pi) - math.pi

            # ── Layer 3: Forward Dead-Reckoning ──────────────────────────────
            effective_v = max(0.0, velocity_ms)
            self.x += effective_v * math.cos(self.yaw) * dt
            self.y += effective_v * math.sin(self.yaw) * dt

            # ── Layer 3b: Lane-Tangent Soft Heading Nudge ─────────────────────
            if camera_confidence > 0.25 and abs(camera_heading_rad) < 0.5:
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
                                     path, cursor)
                self.current_zone = self.planner.get_zone(self.x, self.y)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_cursor_internal(self, path):
        """
        FIX VL-FIX-A: O(1) incremental cursor advance (±12 window).
        FIX LOC-03: cursor only advances — never regresses — to prevent
        oscillation on looping/revisited track sections.
        Mutates self._path_cursor directly (called under self._lock).
        """
        if not path or not self.planner:
            return

        search_start = max(0,             self._path_cursor - 3)
        search_end   = min(len(path) - 1, self._path_cursor + 12)

        best_idx = self._path_cursor
        best_d   = float('inf')
        for i in range(search_start, search_end + 1):
            n = path[i]
            if n not in self.planner.node_positions:
                continue
            nx_, ny_ = self.planner.node_positions[n]
            d = math.hypot(nx_ - self.x, ny_ - self.y)
            if d < best_d:
                best_d  = d
                best_idx = i

        # FIX LOC-03: only advance cursor, never go backwards
        self._path_cursor = max(self._path_cursor, best_idx)

    def update_cursor(self, path, x, y):
        """
        Public cursor update — still available for callers that need it,
        but update() now calls this internally, so external calls are
        optional.  Returns the new cursor index.
        """
        if not path or not self.planner:
            return self._path_cursor

        search_start = max(0,             self._path_cursor - 3)
        search_end   = min(len(path) - 1, self._path_cursor + 12)

        best_idx = self._path_cursor
        best_d   = float('inf')
        for i in range(search_start, search_end + 1):
            n = path[i]
            if n not in self.planner.node_positions:
                continue
            nx_, ny_ = self.planner.node_positions[n]
            d = math.hypot(nx_ - x, ny_ - y)
            if d < best_d:
                best_d  = d
                best_idx = i

        self._path_cursor = best_idx
        return self._path_cursor

    def _apply_map_snap(self, velocity, dt, cam_conf, path, cursor):
        """
        FIX VL-FIX-B: Two-tier snap radius:
          Normal: 1.0 m  (was 0.5 m — too tight, failed after any drift)
          Recovery: 2.0 m after _SNAP_LOST_LIMIT consecutive misses

        Also: curvature gate still disables snap during corners > 0.005.
        """
        if velocity < 0.05 or cam_conf < 0.3:
            return
        if not path or cursor >= len(path) - 1:
            return

        curvature = self.planner.get_path_curvature(
            self.x, self.y, path, cursor=cursor, window_m=0.8)
        if curvature > 0.005:
            return

        # Decide which radius to use
        snap_radius = (self._MAP_SNAP_RECOVERY_M
                       if self._snap_miss_frames >= self._SNAP_LOST_LIMIT
                       else self._MAP_SNAP_RADIUS_M)

        best_dist = float('inf')
        best_foot = None

        search_start = max(0,             cursor - 2)
        search_end   = min(len(path) - 1, cursor + 6)

        for i in range(search_start, search_end):
            n1 = path[i]
            n2 = path[i + 1]
            p1 = self.planner.node_positions.get(n1)
            p2 = self.planner.node_positions.get(n2)
            if p1 is None or p2 is None:
                continue

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

        if best_foot and best_dist < snap_radius:
            # FIX LOC-02: clamp dt to 100 ms max so a stall spike can't jump the car
            dt_clamped = min(dt, 0.10)
            pull = self._MAP_SNAP_PULL * dt_clamped
            self.x = self.x + pull * (best_foot[0] - self.x)
            self.y = self.y + pull * (best_foot[1] - self.y)
            self._snap_miss_frames = 0   # reset recovery counter
        else:
            self._snap_miss_frames += 1
            if self._snap_miss_frames >= self._SNAP_LOST_LIMIT:
                log.warning(
                    f"Map snap lost for {self._snap_miss_frames} frames "
                    f"(dist={best_dist:.2f}m). Recovery radius active.")