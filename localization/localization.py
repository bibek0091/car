import math
import numpy as np
import logging
import threading

log = logging.getLogger(__name__)

class LocalizationEngine:
    """
    3-Layer Sensor Fusion: (x, y, yaw)
      Layer 1 : IMU yaw  (or camera-lane-tangent fallback when IMU dead)
      Layer 2 : Bicycle-model dead reckoning
      Layer 3a: Vision lateral correction
      Layer 3b: Map path snap (called externally every 6 frames)
    """

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0          # radians, map frame
        self.wheelbase = 0.23   # metres
        self.pose_lock = threading.Lock()

    def set_pose(self, x, y, yaw):
        with self.pose_lock:
            self.x = x
            self.y = y
            self.yaw = yaw

    def node_soft_snap(self, node_x, node_y, alpha=0.15):
        """
        Gentle blend toward a known node position (replaces the old hard teleport).
        alpha=0.15 → at most 15 cm correction per call, prevents steering spikes.
        """
        with self.pose_lock:
            self.x = (1.0 - alpha) * self.x + alpha * node_x
            self.y = (1.0 - alpha) * self.y + alpha * node_y

    # Keep old name as alias so nothing else breaks
    def node_reset(self, node_x, node_y):
        self.node_soft_snap(node_x, node_y)

    def get_pose(self):
        with self.pose_lock:
            return self.x, self.y, self.yaw

    def update(self, velocity_ms, steer_angle_deg, imu_yaw_deg,
               lane_error_px, lane_width_px, conf, dt,
               imu_available=True, camera_yaw_correction=0.0):
        """
        imu_available       : True when BNO055 is present and calibrated
        camera_yaw_correction: radians offset from estimate_heading_from_lanes()
                               Used ONLY when imu_available=False
        """
        with self.pose_lock:

            # ── Layer 1: Heading ─────────────────────────────────────────────
            if imu_available:
                # Hard IMU yaw with wrap-safe delta
                new_yaw = math.radians(imu_yaw_deg)
                delta = (new_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
                self.yaw += delta
            else:
                # Kinematic model blended with lane-tangent vision heading
                steer_rad = math.radians(max(-45.0, min(45.0, steer_angle_deg)))
                yaw_rate_km = 0.0
                if velocity_ms > 0.05:
                    yaw_rate_km = (velocity_ms / self.wheelbase) * math.tan(steer_rad)

                # Weight visual correction by lane confidence (0→pure kinematic)
                vis_weight = min(0.75, conf)
                if abs(camera_yaw_correction) > 0.0 and dt > 0:
                    yaw_rate_vis = camera_yaw_correction / dt
                    yaw_rate = yaw_rate_km * (1.0 - vis_weight) + yaw_rate_vis * vis_weight
                else:
                    yaw_rate = yaw_rate_km
                self.yaw += yaw_rate * dt

            # ── Layer 2: Dead reckoning ──────────────────────────────────────
            self.x += velocity_ms * dt * math.cos(self.yaw)
            self.y += velocity_ms * dt * math.sin(self.yaw)

            # ── Layer 3a: Visual lateral correction ──────────────────────────
            # lateral_error_px = lane_center_x - 320
            # Positive  → lane centre is right of image centre
            #           → car is LEFT of lane centre in camera frame
            #           → push estimated pose rightward in world frame
            # "rightward" from the car = perpendicular-right = (sin yaw, -cos yaw)
            if conf > 0.4 and lane_width_px > 50:
                lane_error_m = lane_error_px * (0.35 / lane_width_px)
                gain = 0.10 * min(conf, 1.0)
                perp_x =  math.sin(self.yaw)
                perp_y = -math.cos(self.yaw)
                self.x += gain * lane_error_m * perp_x
                self.y += gain * lane_error_m * perp_y

            return self.x, self.y, self.yaw

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _nearest_point_on_segment(self, p, a, b):
        ab = (b[0]-a[0], b[1]-a[1])
        t = max(0, min(1, ((p[0]-a[0])*ab[0] + (p[1]-a[1])*ab[1]) /
                          max(ab[0]**2 + ab[1]**2, 1e-9)))
        return (a[0] + t*ab[0], a[1] + t*ab[1])

    def fuse_map_correction(self, planned_path, node_positions,
                            max_snap_m=0.60, lane_conf=1.0):
        """Soft-snap estimated position toward the nearest point on the A* path."""
        if not planned_path or len(planned_path) < 2:
            return
        if lane_conf < 0.3:
            return

        min_dist = float('inf')
        best_pt  = None

        for i in range(len(planned_path) - 1):
            n1, n2 = planned_path[i], planned_path[i+1]
            if n1 not in node_positions or n2 not in node_positions:
                continue
            pt = self._nearest_point_on_segment(
                (self.x, self.y),
                node_positions[n1],
                node_positions[n2]
            )
            d = math.hypot(pt[0]-self.x, pt[1]-self.y)
            if d < min_dist:
                min_dist = d
                best_pt  = pt

        if best_pt and min_dist < max_snap_m:
            with self.pose_lock:
                self.x = self.x * 0.70 + best_pt[0] * 0.30
                self.y = self.y * 0.70 + best_pt[1] * 0.30

    def detect_slip(self, imu_accel_ms2, velocity_ms, dt):
        return abs(imu_accel_ms2) > 3.0 and velocity_ms < 0.05