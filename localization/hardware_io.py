"""
hardware_io.py — Hardware Abstraction Layer (Thread-Safe Daemon)
================================================================
Camera I/O is isolated in a daemon thread so picamera2 latency
never stalls the main control loop.
"""

import sys
import math
import time
import numpy as np
import logging
import threading
import queue

log = logging.getLogger(__name__)

# ── STM32 Serial Handler ──────────────────────────────────────────────────────
try:
    from STM32_SerialHandler import STM32_SerialHandler
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    log.warning("STM32_SerialHandler not found. Using simulation mode for STM32.")

    class STM32_SerialHandler:
        def connect(self):      return False
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def disconnect(self):   pass

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

        self.DEADBAND_PWM  = 12.0
        # SPEED_CALIB: m/s per PWM unit above deadband.
        # User requested even slower behavior.
        self.SPEED_CALIB   = 0.008     # m/s per PWM unit above deadband
        self.MAX_SPEED_MS  = 0.15      # absolute fallback cap (m/s)

        # Hard mm/s limit sent to STM32 regardless of PWM or SPEED_CALIB.
        # EXTREME SLOW: City: 100 mm/s = 10 cm/s. Highway: 120 mm/s.
        # Change ONLY this value to tune actual physical cap.
        self.MAX_SPEED_MM_S_CITY    = 100
        self.MAX_SPEED_MM_S_HIGHWAY = 120

        self._vel_filtered    = 0.0
        self._sim_yaw         = 0.0
        self._last_sim_time   = time.time()
        self._last_cmd_speed  = 0.0
        self._last_cmd_steer  = 0.0
        self._encoder_fail_count = 0
        self._ENCODER_FAIL_LIMIT = 30
        
        # Hardware Failsafes
        self._last_cmd_time      = time.time()
        self._last_serial_rx_time= time.time()

        # ── Thread-safe frame queue (maxsize=1 → always freshest frame) ───────
        self._frame_queue = queue.Queue(maxsize=1)
        self._running     = True

        # ── STM32 init ────────────────────────────────────────────────────────
        if not self.sim_mode and _SERIAL_AVAILABLE:
            connected = self.serial.connect()
            if not connected:
                log.error("Failed to connect to STM32. Motor commands will be ignored.")

        # ── Camera / video init + daemon thread ───────────────────────────────
        if self.sim_video and _CV2_AVAILABLE:
            self.video_cap = cv2.VideoCapture(self.sim_video)
            log.info(f"Loaded simulation video: {self.sim_video}")
            threading.Thread(target=self._video_worker, daemon=True,
                             name="video_worker").start()
        elif not self.sim_mode and _CAM_AVAILABLE:
            try:
                self.camera = Picamera2()
                cfg = self.camera.create_video_configuration(
                    main={"size": (1280, 720), "format": "XRGB8888"},
                    controls={
                        "AwbEnable":   True,   # let camera balance colors naturally
                        "AeEnable":    True,
                        "Saturation":  1.2,
                        "Sharpness":   1.2,
                    }
                )
                self.camera.configure(cfg)
                self.camera.start()
                log.info("PiCamera2 initialized with Auto White Balance enabled.")
                threading.Thread(target=self._camera_worker, daemon=True,
                                 name="camera_worker").start()
            except Exception as e:
                log.error(f"PiCamera2 init error: {e}")
                self.camera = None

        # ── Hardware Watchdog Thread ──────────────────────────────────────────
        threading.Thread(target=self._watchdog_worker, daemon=True,
                         name="hw_watchdog").start()

    # ── Frame queue helper ────────────────────────────────────────────────────

    def _push_frame(self, frame):
        """Drop the oldest frame and push the newest — always keep latest."""
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        self._frame_queue.put(frame)

    # ── Daemon workers ────────────────────────────────────────────────────────

    def _camera_worker(self):
        """Runs in background thread: continually captures and enqueues frames."""
        while self._running:
            try:
                frame = self.camera.capture_array()   # blocks until new frame ready
                if frame is not None and _CV2_AVAILABLE:
                    # PiCamera2 natively outputs RGB when format="XRGB8888" or "RGB888" is requested.
                    # We just need to ensure we map it to BGR for OpenCV.
                    if frame.ndim == 3:
                        if frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self._push_frame(cv2.resize(frame, (640, 480)))
            except Exception as e:
                log.warning(f"Camera worker error: {e}")
                time.sleep(0.033)   # brief pause only on error, then retry

    def _video_worker(self):
        """Runs in background thread: reads sim video at 30 Hz and enqueues."""
        while self._running:
            if not _CV2_AVAILABLE or self.video_cap is None:
                time.sleep(0.033)
                continue
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_cap.read()
            if ret:
                self._push_frame(cv2.resize(frame, (640, 480)))
            time.sleep(0.033)   # 30 Hz

    def _watchdog_worker(self):
        """Monitors for command starvation (100ms) or serial dropout (500ms)."""
        while self._running:
            now = time.time()
            
            # 1. Command Starvation Timeout (100 ms)
            if now - self._last_cmd_time > 0.1 and self._last_cmd_speed > 0:
                log.error("WATCHDOG: Command starvation (>100ms). Halting vehicle.")
                self._last_cmd_speed = 0.0
                if not self.sim_mode and hasattr(self.serial, 'set_speed'):
                    self.serial.set_speed(0.0)

            # 2. Serial Heartbeat Timeout (500 ms)
            if not self.sim_mode and now - self._last_serial_rx_time > 0.5:
                # If we've missed serial for 0.5s, force stop.
                if self._last_cmd_speed > 0:
                    log.error("WATCHDOG: STM32 Serial Heartbeat lost (>500ms). Halting vehicle.")
                    self._last_cmd_speed = 0.0
                    if hasattr(self.serial, 'set_speed'):
                        self.serial.set_speed(0.0)

            time.sleep(0.05)

    # ── Public camera read ────────────────────────────────────────────────────

    def read_camera(self):
        """Returns the latest 640×480 BGR frame. Never blocks — returns black if queue empty."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def capture_frame(self):
        return self.read_camera()

    # ── Sim heading ───────────────────────────────────────────────────────────

    def get_sim_heading_deg(self):
        if self.sim_mode:
            now = time.time()
            dt_actual = max(0.001, min(now - self._last_sim_time, 0.10))
            self._last_sim_time = now
            v = max(0.0, (self._last_cmd_speed - self.DEADBAND_PWM) * self.SPEED_CALIB)
            yaw_rate = 0.0
            if v > 0.05:
                steer_rad = math.radians(max(-45.0, min(45.0, self._last_cmd_steer)))
                yaw_rate  = (v / 0.23) * math.tan(steer_rad)
            self._sim_yaw += yaw_rate * dt_actual
            return math.degrees(self._sim_yaw)
        return 0.0

    # ── Motor commands ────────────────────────────────────────────────────────

    def set_steering(self, steer_angle_deg):
        self._last_cmd_time = time.time()
        self._last_cmd_steer = max(-45.0, min(45.0, steer_angle_deg))
        if self.sim_mode:
            return
        self.serial.set_steering(self._last_cmd_steer)

    def set_speed(self, speed_pwm: float, highway_mode: bool = False):
        self._last_cmd_time = time.time()
        speed_pwm = max(0.0, min(100.0, speed_pwm))
        self._last_cmd_speed = speed_pwm
        if self.sim_mode:
            return
        if speed_pwm == 0.0:
            speed_mm_s = 0.0
        else:
            speed_ms = max(0.0, (speed_pwm - self.DEADBAND_PWM) * self.SPEED_CALIB)
            raw_mm_s = speed_ms * 1000.0
            # Hard physical cap — prevents any code path from over-speeding the car.
            # Highway gets a 20% higher ceiling (set at init, user-tunable).
            hard_cap = (self.MAX_SPEED_MM_S_HIGHWAY if highway_mode
                        else self.MAX_SPEED_MM_S_CITY)
            speed_mm_s = min(hard_cap, raw_mm_s)
            if raw_mm_s > hard_cap:
                log.debug(
                    "set_speed: clipped %.0f → %.0f mm/s (cap=%d, pwm=%.1f)",
                    raw_mm_s, speed_mm_s, hard_cap, speed_pwm)
        self.serial.set_speed(speed_mm_s)

    # ── Encoder velocity ──────────────────────────────────────────────────────

    def get_velocity_ms(self):
        """IIR-filtered encoder speed in m/s."""
        if self.sim_mode:
            raw = max(0.0, (self._last_cmd_speed - self.DEADBAND_PWM) * self.SPEED_CALIB)
            self._encoder_fail_count = 0
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
                self._last_serial_rx_time = time.time()
                self._encoder_fail_count = 0
            except Exception as e:
                self._encoder_fail_count += 1
                if self._encoder_fail_count == 1:
                    log.warning(f"get_velocity_ms error: {e}")
                elif self._encoder_fail_count >= self._ENCODER_FAIL_LIMIT:
                    log.error(
                        f"Encoder read failed {self._encoder_fail_count} consecutive "
                        f"times — velocity locked at 0. Check STM32 connection.")
                    self._encoder_fail_count = 0
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

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self):
        self._running = False
        self.set_speed(0)
        time.sleep(0.1)
        self.serial.disconnect()
        if self.camera:
            self.camera.stop()
        if self.video_cap:
            self.video_cap.release()