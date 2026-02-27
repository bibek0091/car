"""
localization.py — IMU + Camera Dead-Reckoning Localizer
=========================================================
No A* path planning.  Pose is updated from three sources:
  1. IMU gyro integration (yaw) — primary heading sensor @ 100 Hz
  2. IMU/encoder velocity dead-reckoning (x, y)
  3. Camera lane tangent — soft heading nudge per frame

User clicks on SVG map to set the starting (x, y, yaw) pose.
After that the pose drifts via dead-reckoning; no map snap is needed.
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
    Lean 3-DOF pose estimator: (x, y, yaw) in map metres.

    Layer 1 — Visual Odometry integration for yaw (from camera_heading rate)
    Layer 2 — Forward dead-reckoning: x += v*cos(yaw)*dt, y += v*sin(yaw)*dt
    Layer 3 — Lane tangent soft absolute heading nudge

    set_pose() is called once when the user clicks the SVG map.
    update() is called every pilot loop iteration (~30 Hz).
    get_pose() returns thread-safe (x, y, yaw_rad).
    """

    # Camera absolute heading nudge max per frame
    _MAX_CAM_YAW_CORRECTION = 0.05   # radians
    # EMA weight for smoothing the camera heading correction
    _CAM_YAW_EMA = 0.55

    def __init__(self):
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0    # radians, map frame, wrapped [-pi, pi]
        self.visual_yaw_rate = 0.0 # radians / second (extracted from camera)
        self._lock = threading.RLock()

        self._prev_cam_heading = None

        self._cam_yaw_smoothed = 0.0
        self._initialized      = False   # True once user clicks start position

        self.upcoming_curve = "STRAIGHT"
        self.current_zone   = "CITY"

        # Load GraphML map for Snapping and Look-ahead
        self.planner = None
        self._map_snap_enabled = _PLANNER_AVAILABLE
        if self._map_snap_enabled:
            self.planner = PathPlanner()
            if self.planner.graph is None or len(self.planner.graph.nodes) == 0:
                log.warning("GraphML map empty! Disabling map snapping.")
                self._map_snap_enabled = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_pose(self, x: float, y: float, yaw_rad: float):
        """
        Called when user clicks the SVG map.
        Sets absolute position and resets heading.
        """
        with self._lock:
            self.x             = x
            self.y             = y
            self.yaw           = yaw_rad
            self.visual_yaw_rate = 0.0
            self._initialized  = True
            self._cam_yaw_smoothed = 0.0
            self._prev_cam_heading = None
        log.info(f"Pose set: x={x:.2f} y={y:.2f} yaw={math.degrees(yaw_rad):.1f}°")

    def get_pose(self):
        """Returns (x, y, yaw_rad) thread-safely."""
        with self._lock:
            return self.x, self.y, self.yaw

    def is_initialized(self):
        with self._lock:
            return self._initialized

    def update(self,
               velocity_ms: float,
               dt: float,
               camera_heading_rad: float = 0.0,
               camera_confidence: float  = 0.0):
        """
        Update pose estimate for one time step.

        Parameters
        ----------
        velocity_ms          : forward speed estimate in m/s
        dt                   : elapsed seconds since last call
        camera_heading_rad   : heading correction from lane tangent (radians)
        camera_confidence    : 0–1 confidence from perception (gates nudge)
        """
        if dt <= 0 or not self._initialized:
            return

        with self._lock:
            # ── Layer 1: Visual Odometry (Camera Yaw Rate) ───────────────────
            if camera_confidence > 0.3:
                if self._prev_cam_heading is not None:
                    d_heading = camera_heading_rad - self._prev_cam_heading
                    d_heading = (d_heading + math.pi) % (2 * math.pi) - math.pi
                    
                    raw_yaw_rate = d_heading / dt
                    self.visual_yaw_rate = (0.40 * raw_yaw_rate) + (0.60 * self.visual_yaw_rate)
                
                self._prev_cam_heading = camera_heading_rad
            else:
                self._prev_cam_heading = None
                self.visual_yaw_rate *= 0.90  # Decay to 0 when blind
            
            self.yaw += self.visual_yaw_rate * dt
            self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

            # ── Layer 2: Forward dead-reckoning ──────────────────────────────
            effective_v = max(0.0, velocity_ms)
            self.x += effective_v * math.cos(self.yaw) * dt
            self.y += effective_v * math.sin(self.yaw) * dt

            # ── Layer 3: Camera lane-tangent soft heading nudge ───────────────
            if camera_confidence > 0.25 and abs(camera_heading_rad) < 0.5:
                self._cam_yaw_smoothed = (
                    self._CAM_YAW_EMA * camera_heading_rad
                    + (1.0 - self._CAM_YAW_EMA) * self._cam_yaw_smoothed
                )
                nudge = self._cam_yaw_smoothed * camera_confidence
                nudge = max(-self._MAX_CAM_YAW_CORRECTION,
                            min(self._MAX_CAM_YAW_CORRECTION, nudge))
                self.yaw += nudge * 0.10   # Even more gentle; primary is VO rate
                self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

            # ── Layer 4: GraphML Map Snapping & Lookahead ─────────────────────
            if self._map_snap_enabled and self.planner:
                self._apply_map_snap(effective_v, dt, camera_confidence)
                self._update_upcoming_curve()

        return self.get_pose()

    def _apply_map_snap(self, velocity, dt, cam_conf):
        """Soft snap to the nearest edge if driving normally (to cancel lateral drift)."""
        if velocity < 0.05 or cam_conf < 0.2:
            return

        nearest_node = self.planner.get_nearest_node(self.x, self.y)
        if not nearest_node:
            return

        min_dist = float('inf')
        best_mid = None
        for u, v in self.planner.graph.edges(nearest_node):
            if v in self.planner.node_positions:
                p1 = self.planner.node_positions[u]
                p2 = self.planner.node_positions[v]
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                dist = math.hypot(mx - self.x, my - self.y)
                if dist < min_dist:
                    min_dist = dist
                    best_mid = (mx, my)

        if best_mid and min_dist < 0.6:  # snap within 60 cm
            pull_alpha = 0.20 * dt  # 20% / second
            self.x = self.x * (1 - pull_alpha) + best_mid[0] * pull_alpha
            self.y = self.y * (1 - pull_alpha) + best_mid[1] * pull_alpha

        self.current_zone = self.planner.get_zone(self.x, self.y)

    def _update_upcoming_curve(self, lookahead_m=0.8):
        """Walks the directed graph forward by ~0.8m to anticipate heading change."""
        nearest_node = self.planner.get_nearest_node(self.x, self.y)
        if not nearest_node:
            self.upcoming_curve = "STRAIGHT"
            return
        
        curr_node = nearest_node
        accum_dist = 0.0
        
        for _ in range(5):
            out_edges = list(self.planner.graph.out_edges(curr_node))
            if not out_edges:
                break
            best_edge = None
            best_cos = -2.0
            p1 = self.planner.node_positions[curr_node]
            for u, v in out_edges:
                p2 = self.planner.node_positions[v]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                mag = math.hypot(dx, dy)
                if mag == 0: continue
                align = (dx/mag)*math.cos(self.yaw) + (dy/mag)*math.sin(self.yaw)
                if align > best_cos:
                    best_cos = align
                    best_edge = v
                    
            if not best_edge:
                break
                
            p2 = self.planner.node_positions[best_edge]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            accum_dist += dist
            curr_node = best_edge
            
            if accum_dist >= lookahead_m:
                break
                
        p_start = self.planner.node_positions[nearest_node]
        p_end   = self.planner.node_positions[curr_node]
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        
        if math.hypot(dx, dy) < 0.1:
            self.upcoming_curve = "STRAIGHT"
            return
            
        target_yaw = math.atan2(dy, dx)
        diff = (target_yaw - self.yaw + math.pi) % (2*math.pi) - math.pi
        deg_diff = math.degrees(diff)
        
        if deg_diff > 18.0:
            self.upcoming_curve = "LEFT"
        elif deg_diff < -18.0:
            self.upcoming_curve = "RIGHT"
        else:
            self.upcoming_curve = "STRAIGHT"