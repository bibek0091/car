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
# IMU (BNO055) Interface
# ===========================================================================
try:
    import board
    import busio
    import adafruit_bno055
    _BNO_AVAILABLE = True
except ImportError:
    _BNO_AVAILABLE = False
    log.warning("adafruit_bno055 not found. Using simulation mode for IMU.")

# ===========================================================================
# Camera Interface
# ===========================================================================
# We will use OpenCV with a GStreamer pipeline for the Raspberry Pi camera.


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
        self.sim_mode = sim_mode
        self.sim_video = sim_video
        self.camera = None
        self.video_cap = None
        self.imu = None
        self.serial = STM32_SerialHandler()
        self.yaw_offset = 0.0

        # Simulator Kinematic Model State
        self._sim_yaw = 0.0
        self._last_sim_time = time.time()
        self._last_cmd_speed = 0.0
        self._last_cmd_steer = 0.0
        self.SPEED_CALIB = 0.014

        # Initialize STM32
        if not self.sim_mode and _SERIAL_AVAILABLE:
            connected = self.serial.connect()
            if not connected:
                log.error("Failed to connect to STM32. Motor commands will be ignored.")
        
        # Initialize IMU
        if not self.sim_mode and _BNO_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.imu = adafruit_bno055.BNO055_I2C(i2c)
                log.info("BNO055 initialized successfully.")
            except Exception as e:
                log.error(f"Error initializing BNO055: {e}")
                self.imu = None
        
        # Initialize Camera or Video
        if self.sim_video and _CV2_AVAILABLE:
            self.video_cap = cv2.VideoCapture(self.sim_video)
            log.info(f"Loaded simulation video: {self.sim_video}")
        elif not self.sim_mode and _CV2_AVAILABLE:
            try:
                pipeline = "libcamerasrc ! video/x-raw, width=1280, height=720, framerate=30/1 ! videoconvert ! appsink drop=true max-buffers=1"
                self.camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self.camera.isOpened():
                    log.info("GStreamer PiCamera initialized successfully.")
                else:
                    log.error("Failed to open GStreamer camera pipeline.")
                    self.camera = None
            except Exception as e:
                log.error(f"Error initializing GStreamer camera: {e}")
                self.camera = None

    def capture_frame(self):
        """Returns a 640x480 BGR image"""
        if self.video_cap and _CV2_AVAILABLE:
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                ret, frame = self.video_cap.read()
            if ret:
                return cv2.resize(frame, (640, 480))
            
        if self.camera and _CV2_AVAILABLE:
            ret, frame = self.camera.read()
            if ret:
                # Resize from 1280x720 to 640x480
                return cv2.resize(frame, (640, 480))
            
        # Fallback simulation blank frame
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def read_imu(self):
        """Returns (yaw_deg, calibration_tuple)"""
        if self.imu and not self.sim_mode:
            try:
                yaw = self.imu.euler[0]
                if yaw is None:
                    yaw = 0.0
                calib = self.imu.calibration_status
                return yaw, calib
            except Exception as e:
                log.error(f"IMU read error: {e}")
        return 0.0, (3, 3, 3, 3)
    
    def zero_imu_yaw(self, current_raw_yaw):
        self.yaw_offset = current_raw_yaw

    def get_fused_imu_yaw(self):
        if self.sim_mode:
            yaw_rate = (self._last_cmd_speed * self.SPEED_CALIB / 0.23) * math.tan(math.radians(self._last_cmd_steer))
            self._sim_yaw += yaw_rate * 0.033  # assume 30Hz
            return math.degrees(self._sim_yaw), (3,3,3,3)
            
        raw_yaw, calib = self.read_imu()
        yaw = ((raw_yaw - self.yaw_offset + 540) % 360) - 180
        return yaw, calib

    def set_steering(self, steer_angle_deg):
        """steer_angle_deg: -45 to +45."""
        # Clamp to -45 / +45
        steer_angle_deg = max(-45.0, min(45.0, steer_angle_deg))
        self._last_cmd_steer = steer_angle_deg
        if self.sim_mode:
            return
        self.serial.set_steering(steer_angle_deg)

    def set_speed(self, speed_pwm):
        """speed_pwm: 0-100."""
        speed_pwm = max(0.0, min(100.0, speed_pwm))
        if self.sim_mode:
            self._sim_speed_pwm = speed_pwm
            self._last_cmd_speed = speed_pwm
            return
        self.serial.set_speed(speed_pwm)

    def get_velocity_ms(self):
        if self.sim_mode:
            cmd = getattr(self, "_last_cmd_speed", 0.0)
            return max(0.0, (cmd - 12.0) * self.SPEED_CALIB)
        return self.serial.get_feedback()[0]

    def get_encoder_steer_deg(self):
        if self.sim_mode:
            return getattr(self, "_last_cmd_steer", 0.0)
        return self.serial.get_feedback()[1]

    def get_imu_accel(self):
        if self.imu and not self.sim_mode:
            try:
                accel = self.imu.linear_acceleration
                if accel[0] is not None:
                    return math.hypot(accel[0], accel[1])
            except Exception:
                pass
        return 0.0

    def shutdown(self):
        self.set_speed(0)
        time.sleep(0.1)
        self.serial.disconnect()
        if self.camera:
            self.camera.release()
        if self.video_cap:
            self.video_cap.release()
