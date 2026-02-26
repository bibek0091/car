import sys
import time
import math
import numpy as np
import logging

log = logging.getLogger(__name__)

# ===========================================================================
# STM32 Serial Handler
# ===========================================================================
try:
    sys.path.insert(0, "..")
    from serial_handler import STM32_SerialHandler
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    log.warning("serial_handler not found. Using simulation mode for STM32.")

    class STM32_SerialHandler:
        def connect(self): return False
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def disconnect(self): pass


# ===========================================================================
# Camera Interface
# ===========================================================================
try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    _CAM_AVAILABLE = False
    log.warning("picamera2 not found. Using simulation mode for Camera.")


# ===========================================================================
# OpenCV
# ===========================================================================
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    log.warning("OpenCV (cv2) not found. Some features may fail.")


class HardwareIO:
    def __init__(self, sim_mode=False, sim_video=None):
        # Auto-sim ONLY when no hardware drivers at all (e.g. Windows dev machine).
        _no_hw = (not _SERIAL_AVAILABLE and not _CAM_AVAILABLE)
        if _no_hw and not sim_mode:
            log.warning("No hardware drivers found — automatically entering simulation mode.")
            sim_mode = True

        self.sim_mode = sim_mode
        self.sim_video = sim_video
        self.camera = None
        self.video_cap = None
        self.serial = STM32_SerialHandler()
        self.SPEED_CALIB = 0.014

        # Simulator Kinematic Model State
        self._sim_yaw = 0.0          # integrated heading for sim
        self._last_sim_time = time.time()
        self._last_cmd_speed = 0.0
        self._last_cmd_steer = 0.0

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
                log.info("PiCamera2 initialized successfully with manual ColourGains.")
            except Exception as e:
                log.error(f"Error initializing PiCamera2: {e}")
                self.camera = None

    # ── Input Data Access ─────────────────────────────────────────────────────
    def read_camera(self):
        """Returns a 640x480 BGR image"""
        if self.video_cap and _CV2_AVAILABLE:
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
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

        # Fallback simulation blank frame
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # Alias used in calibration phase
    def capture_frame(self):
        return self.read_camera()

    def get_sim_heading_deg(self):
        """
        Simulation-only: returns kinematic-integrated heading.
        Used only when sim_mode=True — replaces what the IMU would give.
        In camera-only real-hardware mode this is not called.
        """
        if self.sim_mode:
            yaw_rate = 0.0
            v = max(0.0, (self._last_cmd_speed - 12.0) * self.SPEED_CALIB)
            if v > 0.05:
                steer_rad = math.radians(
                    max(-45.0, min(45.0, self._last_cmd_steer))
                )
                yaw_rate = (v / 0.23) * math.tan(steer_rad)
            self._sim_yaw += yaw_rate * 0.033  # assume 30 Hz
            return math.degrees(self._sim_yaw)
        return 0.0

    def set_steering(self, steer_angle_deg):
        """steer_angle_deg: -45 to +45."""
        steer_angle_deg = max(-45.0, min(45.0, steer_angle_deg))
        self._last_cmd_steer = steer_angle_deg
        if self.sim_mode:
            return
        self.serial.set_steering(steer_angle_deg)

    def set_speed(self, speed_pwm):
        """speed_pwm: 0-100."""
        speed_pwm = max(0.0, min(100.0, speed_pwm))
        if self.sim_mode:
            self._last_cmd_speed = speed_pwm
            return
        self.serial.set_speed(speed_pwm)

    def get_velocity_ms(self):
        """
        Return IIR-filtered encoder speed in m/s.
        FIX-11: raw encoder/sim readings are noisy and feed directly into
        dead-reckoning x/y.  A one-pole IIR (alpha=0.20) cuts high-frequency
        noise while adding only ~1 frame of lag at 30 Hz.
        """
        if self.sim_mode:
            cmd = getattr(self, "_last_cmd_speed", 0.0)
            raw = max(0.0, (cmd - 12.0) * self.SPEED_CALIB)
        else:
            try:
                if hasattr(self.serial, 'get_feedback'):
                    raw_mms = self.serial.get_feedback()[0]
                else:
                    import contextlib
                    with getattr(self.serial, 'feedback_lock', contextlib.nullcontext()):
                        raw_mms = getattr(self.serial, '_feedback_speed', 0.0)
                raw = max(0.0, raw_mms / 1000.0)   # mm/s -> m/s, clamp negative
            except Exception as e:
                log.warning(f"get_velocity_ms error: {e}")
                raw = 0.0

        # IIR low-pass: y[n] = 0.80*y[n-1] + 0.20*x[n]
        # alpha=0.20 balances noise rejection vs. response lag
        self._vel_filtered = 0.80 * getattr(self, '_vel_filtered', 0.0) + 0.20 * raw
        return self._vel_filtered

    def get_encoder_steer_deg(self):
        """Return encoder steering angle in degrees."""
        if self.sim_mode:
            return getattr(self, "_last_cmd_steer", 0.0)
        try:
            if hasattr(self.serial, 'get_feedback'):
                return self.serial.get_feedback()[1]
            else:
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