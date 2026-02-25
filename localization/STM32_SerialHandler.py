import serial
import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum

logger = logging.getLogger(__name__)

# ===================== ENUMS & DATA =====================

class VehicleState(Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    SHUTDOWN = "SHUTDOWN"

@dataclass
class SerialConfig:
    port: str = None
    baudrate: int = 115200
    timeout: float = 1.0
    write_timeout: float = 1.0
    reconnect_attempts: int = 3
    reconnect_delay: float = 1.0
    heartbeat_interval: float = 0.2  # < 500ms BFMC watchdog

@dataclass
class VehicleStatus:
    state: VehicleState = VehicleState.IDLE
    speed_mm_s: float = 0.0
    steering_angle: float = 0.0
    battery_voltage: float = 0.0
    instant_current: float = 0.0
    imu_data: Optional[Dict] = None
    last_heartbeat: float = 0.0

# ===================== MAIN CLASS =====================

class STM32_SerialHandler:

    def __init__(self, config: SerialConfig = None):
        self.config = config or SerialConfig()
        self.serial_port = None
        self.running = False

        self.read_thread = None
        self.heartbeat_thread = None
        self.command_lock = threading.Lock()

        self.status = VehicleStatus()
        self.callbacks = {
            'emergency_stop': [],
            'connection_lost': [],
        }

        self._feedback_speed = 0.0
        self._feedback_steer = 0.0
        self.feedback_lock = threading.Lock()

        self.read_buffer = ""
        logger.info("STM32 Serial Handler initialized")

    # ===================== CONNECTION =====================

    def auto_detect_port(self) -> Optional[str]:
        import serial.tools.list_ports
        for port in serial.tools.list_ports.comports():
            if hasattr(port, "vid") and port.vid == 0x0483:
                logger.info(f"Detected STM32 on {port.device}")
                return port.device
        return None

    def connect(self, port: str = None) -> bool:
        self.config.port = port or self.config.port or self.auto_detect_port()
        if not self.config.port:
            logger.error("No STM32 port found")
            return False

        for _ in range(self.config.reconnect_attempts):
            try:
                logger.info(f"Connecting to STM32 on {self.config.port}")
                self.serial_port = serial.Serial(
                    self.config.port,
                    self.config.baudrate,
                    timeout=self.config.timeout,
                    write_timeout=self.config.write_timeout
                )

                self.serial_port.reset_input_buffer()
                self.serial_port.reset_output_buffer()
                time.sleep(2)

                self.running = True

                self.read_thread = threading.Thread(
                    target=self._read_loop,
                    daemon=True
                )
                self.read_thread.start()

                self.heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    daemon=True
                )
                self.heartbeat_thread.start()

                # Ensure heartbeat is active before ignition
                time.sleep(self.config.heartbeat_interval * 1.2)

                self._initialize_vehicle()
                logger.info("STM32 connected and initialized")
                return True

            except Exception as e:
                logger.error(f"Connection failed: {e}")
                time.sleep(self.config.reconnect_delay)

        return False

    # ===================== INITIALIZATION =====================

    def _initialize_vehicle(self):
        self.status.state = VehicleState.INITIALIZING

        self.send_command("alive", "1")
        time.sleep(0.1)

        # BFMC-CORRECT IGNITION (KL30)
        self.enable_ignition()
        time.sleep(0.4)

        self.send_command("steer", "0")
        self.send_command("speed", "0")
        
        # ACTIVATE IMU STREAM
        self.send_command("imu", "1")

        self.status.state = VehicleState.READY
        logger.info("Vehicle READY (KL30 enabled)")

    # ===================== HEARTBEAT =====================

    def _heartbeat_loop(self):
        while self.running:
            try:
                self.send_command("alive", "1")
                self.status.last_heartbeat = time.time()
                time.sleep(self.config.heartbeat_interval)
            except:
                time.sleep(1)

    # ===================== SERIAL IO =====================

    def send_command(self, cmd: str, value: str = "") -> bool:
        """
        BFMC protocol:
        #command:value;;\r\n
        """
        if not self.serial_port or not self.serial_port.is_open:
            return False

        # BFMC FRAME WRAPPER
        msg = f"#{cmd}:{value};;\r\n" if value else f"#{cmd}:;;\r\n"

        try:
            with self.command_lock:
                self.serial_port.write(msg.encode())
                self.serial_port.flush()
            return True
        except Exception as e:
            logger.error(f"Serial write failed: {e}")
            return False

    def _read_loop(self):
        try:
            while self.running and self.serial_port and self.serial_port.is_open:
                if self.serial_port.in_waiting:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    self.read_buffer += data.decode(errors="ignore")

                    while "\r\n" in self.read_buffer:
                        line, self.read_buffer = self.read_buffer.split("\r\n", 1)
                        self._process_line(line.strip())

                time.sleep(0.001)

        except Exception as e:
            logger.error(f"Serial read error: {e}")
            self.running = False
            for cb in self.callbacks['connection_lost']:
                cb(str(e))

    def _process_line(self, line: str):
        if line.startswith("@battery:"):
            try:
                # e.g. @battery:8200;;
                val = line.split(":")[1].split(";")[0]
                self.status.battery_voltage = float(val) / 1000.0  # mV to V
            except: pass
        elif line.startswith("@imu:"):
            try:
                # e.g. @imu:roll;pitch;yaw;vx;vy;vz;;
                parts = line.split(":")[1].split(";")[0:6]
                if len(parts) >= 6:
                    self.status.imu_data = {
                        'roll': float(parts[0]),
                        'pitch': float(parts[1]),
                        'yaw': float(parts[2]),
                        'vx': float(parts[3]),
                        'vy': float(parts[4]),
                        'vz': float(parts[5]),
                    }
            except Exception as e:
                logger.debug(f"IMU Parse Error: {e}")
        elif line.startswith("{") and "feedback" in line:
            import json
            try:
                pkt = json.loads(line)
                if pkt.get("action") == "feedback":
                    with self.feedback_lock:
                        self._feedback_speed = float(pkt.get("speed", self._feedback_speed))
                        self._feedback_steer = float(pkt.get("steer", self._feedback_steer))
            except Exception as e:
                logger.debug(f"JSON Parse Error on feedback: {e}")

    def get_feedback(self):
        with self.feedback_lock:
            return self._feedback_speed, self._feedback_steer

    # ===================== VEHICLE COMMANDS =====================

    def set_speed(self, speed_mm_s: float) -> bool:
        speed_mm_s = max(-500, min(500, speed_mm_s))
        self.status.speed_mm_s = speed_mm_s
        self.status.state = VehicleState.RUNNING if speed_mm_s != 0 else VehicleState.READY
        return self.send_command("speed", str(int(speed_mm_s)))

    def set_steering(self, angle_deg: float) -> bool:
        angle_deg = max(-25.0, min(25.0, angle_deg))
        self.status.steering_angle = angle_deg
        return self.send_command("steer", str(int(angle_deg * 10)))

    def emergency_brake(self):
        logger.warning("EMERGENCY STOP")
        self.status.state = VehicleState.EMERGENCY_STOP

        self.send_command("speed", "0")
        self.send_command("steer", "0")
        self.send_command("brake", "1")

        # DROP IGNITION
        self.send_command("kl", "0")

        for cb in self.callbacks['emergency_stop']:
            cb("Emergency brake activated")

    # ===================== IGNITION =====================

    def enable_ignition(self) -> bool:
        logger.info("KL30 ignition ENABLED automatically")
        return self.send_command("kl", "30")

    # ===================== SHUTDOWN =====================

    def disconnect(self):
        logger.info("Disconnecting STM32")
        self.running = False

        try:
            self.send_command("speed", "0")
            self.send_command("steer", "0")
            self.send_command("kl", "0")
        except:
            pass

        if self.serial_port:
            self.serial_port.close()

        self.status.state = VehicleState.IDLE

    def __del__(self):
        self.disconnect()
