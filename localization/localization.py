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
from collections import deque

log = logging.getLogger(__name__)


class LocalizationEngine:
    """
    Lean 3-DOF pose estimator: (x, y, yaw) in map metres.

    Layer 1 — IMU gyro integration for yaw (primary, ~100 Hz via get_yaw_data())
    Layer 2 — Forward dead-reckoning: x += v*cos(yaw)*dt, y += v*sin(yaw)*dt
    Layer 3 — Lane tangent soft nudge (camera heading correction)

    set_pose() is called once when the user clicks the SVG map.
    update() is called every pilot loop iteration (~30 Hz).
    get_pose() returns thread-safe (x, y, yaw_rad).
    """

    # Camera lane-tangent heading nudge max per frame (≈2.9°)
    _MAX_CAM_YAW_CORRECTION = 0.05   # radians
    # EMA weight for smoothing the camera heading correction
    _CAM_YAW_EMA = 0.55

    def __init__(self):
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0    # radians, map frame, wrapped [-pi, pi]
        self._lock = threading.RLock()

        self._cam_yaw_smoothed = 0.0
        self._initialized      = False   # True once user clicks start position

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
            self._initialized  = True
            self._cam_yaw_smoothed = 0.0
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
               imu_yaw_rate_rps: float,
               dt: float,
               camera_heading_rad: float = 0.0,
               camera_confidence: float  = 0.0):
        """
        Update pose estimate for one time step.

        Parameters
        ----------
        velocity_ms          : forward speed estimate in m/s
                               (from hardware_io.get_velocity_ms or IMU accel)
        imu_yaw_rate_rps     : gz from MPU-9250 in rad/s (bias-corrected)
        dt                   : elapsed seconds since last call
        camera_heading_rad   : heading correction from lane tangent (radians)
                               0.0 if no lane visible
        camera_confidence    : 0–1 confidence from perception (gates nudge)
        """
        if dt <= 0 or not self._initialized:
            return

        with self._lock:
            # ── Layer 1: IMU gyro yaw integration ────────────────────────────
            # This is the primary heading update — integrates gz at ~30 Hz
            # (the IMU thread runs at 100 Hz but we only call update() at 30 Hz,
            #  so dt is ~0.033 s and the product is the correct yaw increment).
            self.yaw += imu_yaw_rate_rps * dt
            self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

            # ── Layer 2: Forward dead-reckoning ──────────────────────────────
            effective_v = max(0.0, velocity_ms)
            self.x += effective_v * math.cos(self.yaw) * dt
            self.y += effective_v * math.sin(self.yaw) * dt

            # ── Layer 3: Camera lane-tangent soft heading nudge ───────────────
            # Only nudge when confidence is reasonable and correction is small
            if camera_confidence > 0.25 and abs(camera_heading_rad) < 0.5:
                # EMA-smooth the camera correction to avoid abrupt yaw jumps
                self._cam_yaw_smoothed = (
                    self._CAM_YAW_EMA * camera_heading_rad
                    + (1.0 - self._CAM_YAW_EMA) * self._cam_yaw_smoothed
                )
                # Scale nudge by confidence and clamp to max per-frame correction
                nudge = self._cam_yaw_smoothed * camera_confidence
                nudge = max(-self._MAX_CAM_YAW_CORRECTION,
                            min(self._MAX_CAM_YAW_CORRECTION, nudge))
                self.yaw += nudge * 0.15   # very gentle — IMU is primary
                self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi