"""
hardware_io.py — Hardware Abstraction Layer  (FIXED v2)
=======================================================
Fixes applied:
  HW-01  SPEED_CALIB corrected to 0.00568 m/s/PWM (was 0.014 — 2.5× too large)
         Derived: MAX_SPEED_MS / (100 - DEADBAND_PWM) = 0.50 / 88 = 0.00568
  HW-02  Duplicate class/instance attribute removed — constants defined once
         in __init__ only
"""

import sys
import math
import time
import numpy as np
import logging

log = logging.getLogger(__name__)

# ── STM32 Serial Handler ──────────────────────────────────────────────────────
try:
    from STM32_SerialHandler import STM32_SerialHandler
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    log.warning("STM32_SerialHandler not found. Using simulation mode for STM32.")

    class STM32_SerialHandler:
        def connect(self):     return False
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def disconnect(self):  pass

# ── Camera ────────────────────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    _CAM_AVAILABLE = False
    log.warning("picamera2 not found. Using simulation mode for Camera.")

# ── OpenCV ────────────────────────────────────────────────────────────────────
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    log.warning("OpenCV (cv2) not found.")


class HardwareIO:

    def __init__(self, sim_mode=False, sim_video=None):
        _no_hw = (not _SERIAL_AVAILABLE and not _CAM_AVAILABLE)
        if _no_hw and not sim_mode:
            log.warning("No hardware drivers found — entering simulation mode.")
            sim_mode = True

        self.sim_mode  = sim_mode
        self.sim_video = sim_video
        self.camera    = None
        self.video_cap = None
        self.serial    = STM32_SerialHandler()

        # FIX HW-01: corrected calibration constant
        #   MAX_SPEED_MS = 0.50 m/s at PWM=100
        #   SPEED_CALIB  = 0.50 / (100 - 12) = 0.00568  (was 0.014)
        # FIX HW-02: constants defined ONLY here (no class-level duplicates)
        self.DEADBAND_PWM  = 12.0
        self.SPEED_CALIB   = 0.00568   # m/s per PWM unit above deadband
        self.MAX_SPEED_MS  = 0.50      # physical top speed

        self._vel_filtered    = 0.0
        self._sim_yaw         = 0.0
        self._last_sim_time   = time.time()
        self._last_cmd_speed  = 0.0
        self._last_cmd_steer  = 0.0

        # Initialize STM32
        if not self.sim_mode and _SERIAL_AVAILABLE:
            connected = self.serial.connect()
            if not connected:
                log.error("Failed to connect to STM32. Motor commands will be ignored.")

        # Initialize Camera or Video
        if self.sim_video and _CV2_AVAILABLE:
            self.video_cap = cv2.VideoCapture(self.sim_video)
            log.info(f"Loaded simulation video: {self.sim_video}")
        elif not self.sim_mode and _CAM_AVAILABLE:
            try:
                self.camera = Picamera2()
                cfg = self.camera.create_video_configuration(
                    main={"size": (1280, 720), "format": "XRGB8888"},
                    controls={
                        "AwbEnable":   False,
                        "ColourGains": (3.5, 1.2),
                        "AeEnable":    True,
                        "Saturation":  1.4,
                        "Sharpness":   1.2,
                    }
                )
                self.camera.configure(cfg)
                self.camera.start()
                log.info("PiCamera2 initialized with manual ColourGains.")
            except Exception as e:
                log.error(f"PiCamera2 init error: {e}")
                self.camera = None

    # ── Input ─────────────────────────────────────────────────────────────────

    def read_camera(self):
        """Returns a 640×480 BGR image."""
        if self.video_cap and _CV2_AVAILABLE:
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_cap.read()
            if ret:
                return cv2.resize(frame, (640, 480))

        if self.camera and _CV2_AVAILABLE:
            frame = self.camera.capture_array()
            if frame is not None:
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return cv2.resize(frame, (640, 480))

        return np.zeros((480, 640, 3), dtype=np.uint8)

    def capture_frame(self):
        return self.read_camera()

    def get_sim_heading_deg(self):
        if self.sim_mode:
            v = max(0.0, (self._last_cmd_speed - self.DEADBAND_PWM) * self.SPEED_CALIB)
            yaw_rate = 0.0
            if v > 0.05:
                steer_rad = math.radians(max(-45.0, min(45.0, self._last_cmd_steer)))
                yaw_rate  = (v / 0.23) * math.tan(steer_rad)
            self._sim_yaw += yaw_rate * 0.033
            return math.degrees(self._sim_yaw)
        return 0.0

    # ── Output ────────────────────────────────────────────────────────────────

    def set_steering(self, steer_angle_deg):
        steer_angle_deg = max(-45.0, min(45.0, steer_angle_deg))
        self._last_cmd_steer = steer_angle_deg
        if self.sim_mode:
            return
        self.serial.set_steering(steer_angle_deg)

    def set_speed(self, speed_pwm):
        speed_pwm = max(0.0, min(100.0, speed_pwm))
        self._last_cmd_speed = speed_pwm
        if self.sim_mode:
            return
        if speed_pwm == 0.0:
            speed_mm_s = 0.0
        else:
            speed_ms   = max(0.0, (speed_pwm - self.DEADBAND_PWM) * self.SPEED_CALIB)
            speed_mm_s = min(500.0, speed_ms * 1000.0)
        self.serial.set_speed(speed_mm_s)

    def get_velocity_ms(self):
        """IIR-filtered encoder speed in m/s."""
        if self.sim_mode:
            cmd = self._last_cmd_speed
            raw = max(0.0, (cmd - self.DEADBAND_PWM) * self.SPEED_CALIB)
        else:
            try:
                if hasattr(self.serial, 'get_feedback'):
                    raw_mms = self.serial.get_feedback()[0]
                else:
                    import contextlib
                    with getattr(self.serial, 'feedback_lock',
                                 contextlib.nullcontext()):
                        raw_mms = getattr(self.serial, '_feedback_speed', 0.0)
                raw = max(0.0, raw_mms / 1000.0)
            except Exception as e:
                log.warning(f"get_velocity_ms error: {e}")
                raw = 0.0

        self._vel_filtered = 0.80 * self._vel_filtered + 0.20 * raw
        return self._vel_filtered

    def get_encoder_steer_deg(self):
        if self.sim_mode:
            return self._last_cmd_steer
        try:
            if hasattr(self.serial, 'get_feedback'):
                return self.serial.get_feedback()[1]
            return getattr(self.serial, '_feedback_steer', 0.0)
        except Exception as e:
            log.warning(f"get_encoder_steer_deg error: {e}")
            return 0.0

    def shutdown(self):
        self.set_speed(0)
        time.sleep(0.1)
        self.serial.disconnect()
        if self.camera:
            self.camera.stop()
        if self.video_cap:
            self.video_cap.release()