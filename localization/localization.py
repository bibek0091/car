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
      Layer 3b: Map path snap — called externally each frame (cursor-windowed)

    Fixes applied vs original:
      FIX-1  : camera_yaw_correction is applied as a direct heading nudge, NOT
                divided by dt. Dividing an absolute tangent angle by dt converted a
                ~0.1 rad offset into a 3+ rad/s rate spike at 30 Hz.
      FIX-3  : node_soft_snap now accepts an optional yaw_target so node passage
                also corrects heading, not only position.
      FIX-4  : yaw is wrapped to [-pi, pi] after every integration step.
      FIX-6  : fuse_map_correction reads self.x/y inside pose_lock (race fix).
      FIX-7  : YAW_EMA_ALPHA raised from 0.30 to 0.55 for faster curve response.
      FIX-8  : update() accepts optional bev_scale_mpp for calibrated px/m scale.
      FIX-10 : IMU absolute yaw is fused via a weighted per-frame nudge once the
                IMU->map offset is established in set_pose().
      FIX-12 : fuse_map_correction restricts search to a cursor-centred window
                (O(1)) and can safely be called every frame.
      FIX-13 : fuse_map_correction applies only the LATERAL component of the
                nearest-point error; longitudinal position follows dead-reckoning.
    """

    # FIX-7: raised from 0.30 to 0.55 — faster response on curves, still smooth
    YAW_EMA_ALPHA    = 0.55
    LATERAL_GAIN_MIN = 0.25
    LATERAL_GAIN_MAX = 0.50
    CONF_HISTORY_LEN = 10

    # FIX-1: maximum per-frame heading correction from camera tangent (~2.9 deg)
    _MAX_CAM_YAW_CORRECTION = 0.05   # radians

    # FIX-10: gentle per-frame pull toward IMU absolute yaw (8 %)
    _IMU_ABS_YAW_WEIGHT = 0.08

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0          # radians, map frame, wrapped to [-pi, pi]
        self.wheelbase = 0.23   # metres (1:10 scale car)
        self.pose_lock = threading.RLock()  # BUG-10: RLock prevents deadlock if
        # fuse_map_correction and update() are ever called from different threads.

        self._yaw_rate_smoothed = 0.0
        self._conf_history = deque(maxlen=self.CONF_HISTORY_LEN)

        # FIX-10: offset between IMU frame (zeroed at startup) and map frame.
        # Remains None until set_pose() is called with a valid imu_yaw_rad.
        self._imu_yaw_offset = None

        # Current navigation mode — set by Orchestrator each frame.
        # "BASIC" | "INTERSECTION" | "AEB" | "HIGHWAY"
        self._nav_mode = "BASIC"

    # ── Public API ────────────────────────────────────────────────────────────

    def set_nav_mode(self, mode: str):
        """
        Called by Orchestrator each frame to inform the localizer of the
        current driving mode.  Adjusts map-correction aggressiveness:
          BASIC        — default gains
          INTERSECTION — stronger snap (0.60), wider search window
          HIGHWAY      — weaker snap (0.15), trust dead-reckoning at speed
          AEB          — frozen pose (car is stopped, no correction needed)
        """
        self._nav_mode = mode

    def set_pose(self, x, y, yaw, imu_yaw_rad=None):
        """
        Hard-set pose (startup / re-localisation).
        FIX-10: if imu_yaw_rad is provided the IMU->map offset is computed
        here so absolute IMU yaw can be fused every frame thereafter.
        """
        with self.pose_lock:
            self.x   = x
            self.y   = y
            self.yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
            if imu_yaw_rad is not None:
                self._imu_yaw_offset = self.yaw - imu_yaw_rad
                log.info(
                    f"IMU->map yaw offset set: "
                    f"{math.degrees(self._imu_yaw_offset):.1f} deg"
                )
            # Reset EMA so old drift does not bleed into the new pose
            self._yaw_rate_smoothed = 0.0

    def node_soft_snap(self, node_x, node_y, alpha=0.15,
                       yaw_target=None, yaw_alpha=0.10):
        """
        Gentle blend toward a known node position and, optionally, a heading.
        FIX-3: yaw_target enables heading correction on node passage.

        alpha     = 0.15 -> at most 15 % position correction per call.
        yaw_alpha = 0.10 -> at most 10 % heading correction per call.
        """
        with self.pose_lock:
            self.x = (1.0 - alpha) * self.x + alpha * node_x
            self.y = (1.0 - alpha) * self.y + alpha * node_y
            if yaw_target is not None:
                yaw_delta = (
                    (yaw_target - self.yaw + math.pi) % (2 * math.pi) - math.pi
                )
                self.yaw += yaw_alpha * yaw_delta
                # FIX-4: keep wrapped
                self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

    def node_reset(self, node_x, node_y, yaw_target=None):
        """Alias kept for backward compatibility. Forwards yaw_target (FIX-3)."""
        self.node_soft_snap(node_x, node_y, yaw_target=yaw_target)

    def get_pose(self):
        with self.pose_lock:
            return self.x, self.y, self.yaw

    def update(self, velocity_ms, steer_angle_deg,
               lane_error_px, lane_width_px, conf, dt,
               camera_yaw_correction=0.0,
               camera_lateral_vel_ms=0.0,
               camera_heading_rate_rps=0.0,
               imu_yaw_rate_rps=None,
               imu_yaw_rad=None,
               bev_scale_mpp=None):
        """
        Fused pose update.  Call once per frame from the pilot loop.

        Parameters
        ----------
        camera_yaw_correction   : rad  — instantaneous lane-tangent heading OFFSET
                                  (from estimate_heading_from_lanes). NOT a rate.
        camera_lateral_vel_ms   : m/s  — lateral drift from consecutive lane fits.
        camera_heading_rate_rps : rad/s — heading rate from consecutive tangents.
        imu_yaw_rate_rps        : rad/s — gyro yaw rate (replaces bicycle model).
        imu_yaw_rad             : rad  — absolute IMU yaw in IMU frame (FIX-10).
        bev_scale_mpp           : m/px — calibrated BEV pixel scale (FIX-8).
                                  Falls back to 0.35/lane_width_px when None.
        """
        with self.pose_lock:
            self._conf_history.append(conf)
            mean_conf = (
                float(np.mean(self._conf_history)) if self._conf_history else conf
            )

            # ── Layer 1: Heading ─────────────────────────────────────────────

            steer_rad = math.radians(max(-45.0, min(45.0, steer_angle_deg)))

            # Source A: bicycle kinematic yaw rate OR IMU gyro rate
            yaw_rate_km = 0.0
            if imu_yaw_rate_rps is not None:
                yaw_rate_km = imu_yaw_rate_rps
            elif velocity_ms > 0.05:
                yaw_rate_km = (velocity_ms / self.wheelbase) * math.tan(steer_rad)

            yaw_rate = yaw_rate_km

            # Source C: consecutive-frame tangent heading rate
            # (blended before EMA so it participates in smoothing)
            # Guard raised from 0.5 to 0.6: single-lane (LEFT/RIGHT) anchors
            # are noisier; only blend in camera rate when we have solid dual detection.
            if abs(camera_heading_rate_rps) > 0.001 and conf > 0.60:
                cam_rate_weight = min(0.35, conf - 0.15)
                yaw_rate = (
                    yaw_rate * (1.0 - cam_rate_weight)
                    + camera_heading_rate_rps * cam_rate_weight
                )

            # EMA smoothing — damps noisy tangent-rate spikes
            self._yaw_rate_smoothed = (
                self.YAW_EMA_ALPHA * yaw_rate
                + (1.0 - self.YAW_EMA_ALPHA) * self._yaw_rate_smoothed
            )
            self.yaw += self._yaw_rate_smoothed * dt

            # Source B: lane-tangent direct heading correction.
            # FIX-1: camera_yaw_correction is an ABSOLUTE angle offset in radians,
            # not a per-frame delta.  The original code divided it by dt which turned
            # a 0.1 rad tangent offset into a 3 rad/s rate spike at 30 Hz.
            # We apply it as a clamped, confidence-weighted direct nudge to self.yaw.
            vis_weight = min(0.80, conf * 1.2) if conf > 0.25 else 0.0
            if abs(camera_yaw_correction) > 0.0 and vis_weight > 0.0:
                correction = float(np.clip(
                    camera_yaw_correction * vis_weight,
                    -self._MAX_CAM_YAW_CORRECTION,
                     self._MAX_CAM_YAW_CORRECTION
                ))
                self.yaw += correction

            # Source D: IMU absolute yaw fusion (FIX-10).
            # A gentle per-frame pull toward the IMU-derived map heading prevents
            # long-term heading drift without causing sharp steering spikes.
            if imu_yaw_rad is not None and self._imu_yaw_offset is not None:
                map_yaw_from_imu = imu_yaw_rad + self._imu_yaw_offset
                yaw_delta = (
                    (map_yaw_from_imu - self.yaw + math.pi) % (2 * math.pi) - math.pi
                )
                self.yaw += self._IMU_ABS_YAW_WEIGHT * yaw_delta

            # FIX-4: wrap to [-pi, pi] every frame — prevents unbounded growth
            self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

            # ── Layer 2: Dead reckoning ──────────────────────────────────────
            self.x += velocity_ms * dt * math.cos(self.yaw)
            self.y += velocity_ms * dt * math.sin(self.yaw)

            # ── Layer 3a: Visual lateral correction ──────────────────────────
            if conf > 0.20 and lane_width_px > 50:
                # FIX-8: use calibrated BEV pixel/metre scale when available.
                # Fall back to geometric estimate from measured lane_width_px.
                if bev_scale_mpp is not None:
                    lane_error_m = lane_error_px * bev_scale_mpp
                else:
                    lane_error_m = lane_error_px * (0.35 / max(lane_width_px, 50))

                gain = (
                    self.LATERAL_GAIN_MIN
                    + (self.LATERAL_GAIN_MAX - self.LATERAL_GAIN_MIN)
                    * min(mean_conf, 1.0)
                )

                # BUG-04: SIGN CONVENTION NOTE
                # Heading vector: (cos yaw, sin yaw)
                # Left  perpendicular: (-sin yaw,  cos yaw)
                # Right perpendicular: ( sin yaw, -cos yaw)  ← used here
                # lane_error_px = target_x - 320:
                #   positive = car must move RIGHT → right perpendicular is correct.
                # camera_lateral_vel_ms uses the same perp; rightward velocity
                # increments x by +sin(yaw), which is correct for all headings.
                perp_x =  math.sin(self.yaw)
                perp_y = -math.cos(self.yaw)
                self.x += gain * lane_error_m * perp_x
                self.y += gain * lane_error_m * perp_y

                if abs(camera_lateral_vel_ms) > 0.001 and conf > 0.4:
                    lat_gain = min(0.6, conf)
                    self.x += lat_gain * camera_lateral_vel_ms * dt * perp_x
                    self.y += lat_gain * camera_lateral_vel_ms * dt * perp_y

            return self.x, self.y, self.yaw

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _nearest_point_on_segment(self, p, a, b):
        ab = (b[0] - a[0], b[1] - a[1])
        t  = max(0.0, min(1.0,
              ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1])
              / max(ab[0] ** 2 + ab[1] ** 2, 1e-9)))
        return (a[0] + t * ab[0], a[1] + t * ab[1])

    def fuse_map_correction(self, planned_path, node_positions,
                            max_snap_m=0.38, lane_conf=1.0, cursor=0):
        """
        Laterally soft-snap the estimated position toward the nearest point on
        the A* path segment closest to the current cursor position.

        FIX-6:  self.x/y are read inside pose_lock (eliminates read race).
        FIX-12: Search is restricted to a cursor-centred window -> O(1).
        FIX-13: Only the LATERAL component of the error is applied.
        Nav-mode tuning:
          INTERSECTION — wider window (±8 nodes), snap_alpha up to 0.60
          HIGHWAY      — narrower window (±4), snap_alpha max 0.15
          AEB          — skip entirely (car is stopped)
        """
        if not planned_path or len(planned_path) < 2:
            return

        # AEB: car is stopped, skip map correction entirely
        if self._nav_mode == "AEB":
            return

        low_conf = lane_conf < 0.20

        # Search window tuned per nav_mode
        if self._nav_mode == "INTERSECTION":
            window_back, window_fwd = 3, 8
        elif self._nav_mode == "HIGHWAY":
            window_back, window_fwd = 2, 4
        else:  # BASIC
            window_back, window_fwd = 2, 6

        search_start = max(0, cursor - window_back)
        search_end   = min(len(planned_path) - 1, cursor + window_fwd)

        # FIX-6: read position inside the lock
        with self.pose_lock:
            car_x, car_y = self.x, self.y

        min_dist     = float('inf')
        best_pt      = None
        best_tangent = None

        for i in range(search_start, search_end):
            n1, n2 = planned_path[i], planned_path[i + 1]
            if n1 not in node_positions or n2 not in node_positions:
                continue
            p1 = node_positions[n1]
            p2 = node_positions[n2]
            pt = self._nearest_point_on_segment((car_x, car_y), p1, p2)
            d  = math.hypot(pt[0] - car_x, pt[1] - car_y)
            if d < min_dist:
                min_dist     = d
                best_pt      = pt
                seg_dx = p2[0] - p1[0]
                seg_dy = p2[1] - p1[1]
                seg_len = math.hypot(seg_dx, seg_dy)
                if seg_len > 1e-4:
                    best_tangent = math.atan2(seg_dy, seg_dx)

        if best_pt is None or min_dist >= max_snap_m:
            return

        # snap_alpha tuned per nav_mode and confidence
        if low_conf:
            snap_alpha = 0.10
        elif self._nav_mode == "INTERSECTION":
            snap_alpha = 0.60  # strong correction at junctions
        elif self._nav_mode == "HIGHWAY":
            snap_alpha = 0.15  # gentle — trust dead-reckoning at speed
        else:
            snap_alpha = 0.45 if lane_conf > 0.7 else 0.30

        with self.pose_lock:
            if best_tangent is not None:
                # FIX-13: decompose the nearest-point error into lateral and
                # longitudinal components relative to the path tangent.
                # Apply ONLY the lateral correction; longitudinal follows
                # dead-reckoning so the car does not jump forward/backward.
                tx     =  math.cos(best_tangent)   # unit tangent vector
                ty     =  math.sin(best_tangent)
                px_dir = -ty                        # unit perpendicular (lateral)
                py_dir =  tx

                dx = best_pt[0] - self.x
                dy = best_pt[1] - self.y
                lateral_err = dx * px_dir + dy * py_dir

                self.x += snap_alpha * lateral_err * px_dir
                self.y += snap_alpha * lateral_err * py_dir

                # Heading correction toward path tangent
                yaw_delta = (
                    (best_tangent - self.yaw + math.pi) % (2 * math.pi) - math.pi
                )
                self.yaw += 0.20 * yaw_delta
                # FIX-4: re-wrap after correction
                self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
            else:
                # Fallback (segment too short to compute tangent): full position snap
                self.x = self.x * (1.0 - snap_alpha) + best_pt[0] * snap_alpha
                self.y = self.y * (1.0 - snap_alpha) + best_pt[1] * snap_alpha