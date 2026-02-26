import math
import numpy as np
import logging
import threading
from collections import deque

log = logging.getLogger(__name__)

class LocalizationEngine:
    """
    Camera-Only 3-Layer Pose Estimator: (x, y, yaw)

      Layer 1 : Bicycle-model kinematic yaw + camera lane-tangent heading correction
                (confidence-weighted, EMA-smoothed heading rate)
      Layer 2 : Forward dead reckoning (x, y from velocity + yaw)
      Layer 3a: Visual lateral correction from lane-centre offset
                (dynamic gain that scales with rolling confidence)
      Layer 3b: Map path snap — called externally every 3 frames
    """

    YAW_EMA_ALPHA    = 0.30   # smoothing on yaw_rate estimate (lower = smoother)
    LATERAL_GAIN_MIN = 0.25   # base lateral correction gain
    LATERAL_GAIN_MAX = 0.50   # max gain when camera confidence is high
    CONF_HISTORY_LEN = 10     # frames of confidence history for dynamic gain

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0          # radians, map frame
        self.wheelbase = 0.23   # metres
        self.pose_lock = threading.Lock()

        self._yaw_rate_smoothed = 0.0           # EMA-filtered yaw rate
        self._conf_history = deque(maxlen=self.CONF_HISTORY_LEN)

    def set_pose(self, x, y, yaw):
        with self.pose_lock:
            self.x = x
            self.y = y
            self.yaw = yaw

    def node_soft_snap(self, node_x, node_y, alpha=0.15):
        """
        Gentle blend toward a known node position.
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

    def update(self, velocity_ms, steer_angle_deg,
               lane_error_px, lane_width_px, conf, dt,
               camera_yaw_correction=0.0,
               camera_lateral_vel_ms=0.0,
               camera_heading_rate_rps=0.0):
        """
        Camera-only fused pose update.

        Layer 1 — Heading (camera-heavy):
          Source A: Bicycle kinematic model (always available).
          Source B: Camera lane-tangent heading correction.
                    Confidence-weighted. Applied when conf > 0.25.
          Source C: Camera heading rate from consecutive frame tangent change.
                    Applied when conf > 0.50.
          All combined into a single yaw_rate, then EMA-smoothed.

        Layer 2 — Dead reckoning x, y from velocity × yaw.

        Layer 3a — Visual lateral correction from lane-centre offset.
                   Dynamic gain from rolling confidence history.

        Parameters
        ----------
        camera_yaw_correction     : rad, signed lane-tangent heading offset.
        camera_lateral_vel_ms     : m/s, signed lateral drift from lane-centre shift.
        camera_heading_rate_rps   : rad/s, signed heading rate from consecutive fits.
        """
        with self.pose_lock:
            self._conf_history.append(conf)
            mean_conf = float(np.mean(self._conf_history)) if self._conf_history else conf

            # ── Layer 1: Heading ─────────────────────────────────────────────
            steer_rad = math.radians(max(-45.0, min(45.0, steer_angle_deg)))

            # Source A: bicycle kinematic yaw rate
            yaw_rate_km = 0.0
            if velocity_ms > 0.05:
                yaw_rate_km = (velocity_ms / self.wheelbase) * math.tan(steer_rad)

            # Source B: lane-tangent heading correction
            vis_weight = min(0.80, conf * 1.2) if conf > 0.25 else 0.0
            if abs(camera_yaw_correction) > 0.0 and dt > 0 and vis_weight > 0:
                yaw_rate_vis = camera_yaw_correction / dt
                yaw_rate = yaw_rate_km * (1.0 - vis_weight) + yaw_rate_vis * vis_weight
            else:
                yaw_rate = yaw_rate_km

            # Source C: consecutive-frame tangent heading rate
            if abs(camera_heading_rate_rps) > 0.001 and conf > 0.5:
                cam_rate_weight = min(0.40, conf - 0.1)
                yaw_rate = yaw_rate * (1.0 - cam_rate_weight) + camera_heading_rate_rps * cam_rate_weight

            # EMA smoothing — prevents sudden yaw spikes from noisy tangent estimates
            self._yaw_rate_smoothed = (
                self.YAW_EMA_ALPHA * yaw_rate
                + (1.0 - self.YAW_EMA_ALPHA) * self._yaw_rate_smoothed
            )
            self.yaw += self._yaw_rate_smoothed * dt

            # ── Layer 2: Dead reckoning ──────────────────────────────────────
            self.x += velocity_ms * dt * math.cos(self.yaw)
            self.y += velocity_ms * dt * math.sin(self.yaw)

            # ── Layer 3a: Visual lateral correction ──────────────────────────
            # lateral_error_px = lane_centre_x - 320
            # Positive  → lane centre is right of image centre
            #           → car is LEFT of lane centre → push pose rightward
            if conf > 0.20 and lane_width_px > 50:
                lane_error_m = lane_error_px * (0.35 / max(lane_width_px, 50))

                # Dynamic gain: scales from LATERAL_GAIN_MIN to LATERAL_GAIN_MAX
                # using rolling mean confidence so transient detections don't spike
                gain = self.LATERAL_GAIN_MIN + (self.LATERAL_GAIN_MAX - self.LATERAL_GAIN_MIN) * min(mean_conf, 1.0)

                perp_x =  math.sin(self.yaw)
                perp_y = -math.cos(self.yaw)
                self.x += gain * lane_error_m * perp_x
                self.y += gain * lane_error_m * perp_y

                # Direct lateral velocity correction
                if abs(camera_lateral_vel_ms) > 0.001 and conf > 0.4:
                    lat_gain = min(0.6, conf)
                    self.x += lat_gain * camera_lateral_vel_ms * dt * perp_x
                    self.y += lat_gain * camera_lateral_vel_ms * dt * perp_y

            return self.x, self.y, self.yaw

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _nearest_point_on_segment(self, p, a, b):
        ab = (b[0]-a[0], b[1]-a[1])
        t = max(0, min(1, ((p[0]-a[0])*ab[0] + (p[1]-a[1])*ab[1]) /
                          max(ab[0]**2 + ab[1]**2, 1e-9)))
        return (a[0] + t*ab[0], a[1] + t*ab[1])

    def fuse_map_correction(self, planned_path, node_positions,
                            max_snap_m=0.90, lane_conf=1.0):
        """
        Soft-snap estimated position toward the nearest point on the A* path.
        Also corrects heading toward the path tangent at the snap point.
        Higher snap alpha (0.45) than before — camera-only needs tighter map anchoring.
        """
        if not planned_path or len(planned_path) < 2:
            return
        if lane_conf < 0.20:
            return

        min_dist = float('inf')
        best_pt  = None
        best_tangent = None

        for i in range(len(planned_path) - 1):
            n1, n2 = planned_path[i], planned_path[i+1]
            if n1 not in node_positions or n2 not in node_positions:
                continue
            p1 = node_positions[n1]
            p2 = node_positions[n2]
            pt = self._nearest_point_on_segment(
                (self.x, self.y), p1, p2
            )
            d = math.hypot(pt[0]-self.x, pt[1]-self.y)
            if d < min_dist:
                min_dist  = d
                best_pt   = pt
                # Tangent direction of this segment in map frame
                seg_dx = p2[0] - p1[0]
                seg_dy = p2[1] - p1[1]
                seg_len = math.hypot(seg_dx, seg_dy)
                if seg_len > 1e-4:
                    best_tangent = math.atan2(seg_dy, seg_dx)

        if best_pt and min_dist < max_snap_m:
            # Position snap — stronger when camera is confident
            snap_alpha = 0.45 if lane_conf > 0.7 else 0.30
            with self.pose_lock:
                self.x = self.x * (1.0 - snap_alpha) + best_pt[0] * snap_alpha
                self.y = self.y * (1.0 - snap_alpha) + best_pt[1] * snap_alpha

                # Heading correction toward path tangent (weight 0.20)
                if best_tangent is not None:
                    yaw_delta = (best_tangent - self.yaw + math.pi) % (2*math.pi) - math.pi
                    self.yaw += 0.20 * yaw_delta