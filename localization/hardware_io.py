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
# Requires on Raspberry Pi:
#   pip install adafruit-blinka               ← provides board + busio
#   pip install adafruit-circuitpython-bno055 ← provides adafruit_bno055
#
# I2C must be enabled: sudo raspi-config → Interface Options → I2C → Enable
# User must be in i2c group: sudo usermod -aG i2c $USER  (then re-login)
# ===========================================================================
_BNO_AVAILABLE   = False
_BLINKA_AVAILABLE = False

try:
    import board
    import busio
    _BLINKA_AVAILABLE = True
except ImportError:
    log.warning("adafruit-blinka not found (provides 'board' and 'busio'). "
                "Run: pip install adafruit-blinka --break-system-packages")

if _BLINKA_AVAILABLE:
    try:
        import adafruit_bno055
        _BNO_AVAILABLE = True
    except ImportError:
        log.warning("adafruit_bno055 not found. "
                    "Run: pip install adafruit-circuitpython-bno055 --break-system-packages")

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
        # A missing IMU library alone does NOT trigger sim — the system runs in
        # hardware mode and the localizer falls back to camera-only heading.
        _no_hw = (not _SERIAL_AVAILABLE and not _BLINKA_AVAILABLE and not _CAM_AVAILABLE)
        if _no_hw and not sim_mode:
            log.warning("No hardware drivers found — automatically entering simulation mode.")
            sim_mode = True

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
            self.imu = self._init_bno055()
        elif not self.sim_mode and _BLINKA_AVAILABLE and not _BNO_AVAILABLE:
            log.error("adafruit_bno055 library missing — IMU disabled.")
        elif not self.sim_mode and not _BLINKA_AVAILABLE:
            log.error("adafruit-blinka missing — IMU disabled. "
                      "Install with: pip install adafruit-blinka --break-system-packages")
        
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

    # ── BNO055 Initialization ─────────────────────────────────────────────────
    def _init_bno055(self, retries=3):
        """
        Robust BNO055 init with:
          - I2C bus scan to confirm device is visible before init
          - Address fallback: tries 0x28 then 0x29
          - Retry loop (sensor needs up to 650ms boot time)
          - Detailed per-failure logging so you know exactly what went wrong
        Returns the BNO055 object, or None on total failure.
        """
        BNO_ADDRESSES = [0x28, 0x29]   # 0x28 = PS1 low (default), 0x29 = PS1 high

        for attempt in range(1, retries + 1):
            try:
                # busio.I2C uses 100kHz by default; pass frequency= for fast mode
                # BNO055 works at both 100kHz and 400kHz; 400k is more reliable
                # on Pi because it reduces the window where clock-stretching stalls.
                i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)

                # ── I2C Bus Scan ──────────────────────────────────────────────
                # Lock the bus, scan for visible addresses, release.
                while not i2c.try_lock():
                    pass
                try:
                    found = i2c.scan()
                    log.info(f"I2C scan (attempt {attempt}): found addresses "
                             f"{[hex(a) for a in found]}")
                finally:
                    i2c.unlock()

                if not found:
                    log.error(f"I2C scan empty (attempt {attempt}/{retries}). "
                              "Check: SDA=GPIO2(pin3), SCL=GPIO3(pin5), "
                              "3.3V power, GND, I2C enabled in raspi-config.")
                    if attempt < retries:
                        time.sleep(0.7)
                    continue

                # ── Try BNO055 at each known address ─────────────────────────
                imu_obj = None
                for addr in BNO_ADDRESSES:
                    if addr not in found:
                        log.debug(f"BNO055 address {hex(addr)} not on bus — skipping")
                        continue
                    try:
                        imu_obj = adafruit_bno055.BNO055_I2C(i2c, address=addr)
                        log.info(f"BNO055 initialised at address {hex(addr)} "
                                 f"(attempt {attempt}/{retries}) ✓")
                        return imu_obj
                    except Exception as addr_err:
                        log.warning(f"BNO055 at {hex(addr)} failed: {addr_err}")

                # Neither address worked even though something was on the bus
                log.error(f"BNO055 not responding at {[hex(a) for a in BNO_ADDRESSES]} "
                          f"(attempt {attempt}/{retries}). "
                          f"Found on bus: {[hex(a) for a in found]}")

            except Exception as e:
                log.error(f"BNO055 init error (attempt {attempt}/{retries}): "
                          f"{type(e).__name__}: {e}")
                if "Permission" in str(e) or "Access" in str(e):
                    log.error("Permission denied — add user to i2c group: "
                              "sudo usermod -aG i2c $USER  then log out and back in")
                if "No module" in str(e):
                    log.error("Missing dependency — run: "
                              "pip install adafruit-blinka adafruit-circuitpython-bno055 "
                              "--break-system-packages")

            if attempt < retries:
                log.info(f"Retrying BNO055 init in 0.7s...")
                time.sleep(0.7)   # BNO055 boot time is up to 650ms

        log.error("BNO055 failed to initialise after all attempts. "
                  "Localization will use camera-only heading estimation.")
        return None

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
            frame = self.camera.capture_array()
            if frame is not None:
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # XRGB8888 can be RGB depending on version
                
                # Resize from 1280x720 to 640x480
                return cv2.resize(frame, (640, 480))
            
        # Fallback simulation blank frame
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def read_imu(self):
        """Returns (yaw_deg, calibration_tuple).
        calibration_tuple = (sys, gyro, accel, mag), each 0-3.
        Returns (0.0, (0,0,0,0)) when IMU is not connected — NOT (3,3,3,3)
        so the dashboard LEDs correctly show red instead of green.
        """
        if self.imu and not self.sim_mode:
            try:
                yaw = self.imu.euler[0]
                if yaw is None:
                    yaw = 0.0
                calib = self.imu.calibration_status
                return yaw, calib
            except Exception as e:
                log.error(f"IMU read error: {e}")
        # IMU absent or failed — return zero calib so LEDs show red
        return 0.0, (0, 0, 0, 0)
    
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
            self.camera.stop()
        if self.video_cap:
            self.video_cap.release()