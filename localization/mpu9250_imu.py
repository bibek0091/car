import smbus2
import math
import time
import threading
import logging

log = logging.getLogger(__name__)

class ComplementaryFilter:
    def __init__(self, alpha=0.96):
        self.alpha = alpha
        self.roll = self.pitch = self.yaw = 0.0

    def update(self, ax, ay, az, gx, gy, gz, dt):
        # Convert accel to roll/pitch in degrees
        roll_acc  = math.degrees(math.atan2(ay, az))
        pitch_acc = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
        
        # Complementary filter for roll/pitch
        self.roll  = self.alpha * (self.roll  + gx * dt) + (1 - self.alpha) * roll_acc
        self.pitch = self.alpha * (self.pitch + gy * dt) + (1 - self.alpha) * pitch_acc
        
        # Integrate gyro for yaw (relative to start)
        self.yaw  += gz * dt
        
        # Wrap to proper bounds? For yaw, we usually let it continuously grow or wrap to [-180, 180]
        # In the context of localization, radians are mostly used, but keeping it in degrees here is fine
        if self.yaw >  180: self.yaw -= 360
        if self.yaw < -180: self.yaw += 360
        
        return self.roll, self.pitch, self.yaw

class MPU9250_Thread:
    """
    Background thread that continuously polls the MPU-9250 over I2C at ~100Hz.
    Computes absolute roll, pitch, yaw using a complementary filter.
    Exposes thread-safe method to get the latest yaw and yaw-rate.
    """
    def __init__(self, bus=1, address=0x68):
        self.bus_num = bus
        self.addr = address
        self.running = False
        self.thread = None
        self.bus = None
        
        self.cf = ComplementaryFilter(alpha=0.96)
        
        # Calibration bias
        self.gx_b = self.gy_b = self.gz_b = 0.0
        
        # State locks
        self.state_lock = threading.Lock()
        self.yaw_deg = 0.0
        self.yaw_rate_deg_s = 0.0  # (gz after bias)
        self.is_connected = False
        
        self._init_sensor()

    def _init_sensor(self):
        try:
            self.bus = smbus2.SMBus(self.bus_num)
            self.bus.write_byte_data(self.addr, 0x6B, 0x00)  # wake up
            time.sleep(0.1)
            self.bus.write_byte_data(self.addr, 0x1B, 0x10)  # gyro ±1000 deg/s
            self.bus.write_byte_data(self.addr, 0x1C, 0x10)  # accel ±8g
            time.sleep(0.1)
            
            # Simple online calibration — take 50 samples
            log.info(f"MPU-9250 -> Found at I2C bus {self.bus_num}, address 0x{self.addr:02X}. Calibrating Gyro...")
            gx_sum = gy_sum = gz_sum = 0.0
            N = 50
            for _ in range(N):
                _, _, _, gx, gy, gz = self._read_raw()
                gx_sum += gx; gy_sum += gy; gz_sum += gz
                time.sleep(0.01)
            self.gx_b = gx_sum / N
            self.gy_b = gy_sum / N
            self.gz_b = gz_sum / N
            log.info(f"MPU-9250 -> Gyro Bias: gx={self.gx_b:.2f}, gy={self.gy_b:.2f}, gz={self.gz_b:.2f}")

            self.is_connected = True
        except Exception as e:
            log.warning(f"MPU-9250 -> Initialization Failed (Running in SIM mode or disconnected): {e}")
            self.is_connected = False

    def _signed(self, v):
        return v - 65536 if v > 32767 else v

    def _read_raw(self):
        # Read 14 bytes from 0x3B (ACCEL_XOUT_H)
        d = self.bus.read_i2c_block_data(self.addr, 0x3B, 14)
        ax = self._signed(d[0]  << 8 | d[1])  / 4096.0
        ay = self._signed(d[2]  << 8 | d[3])  / 4096.0
        az = self._signed(d[4]  << 8 | d[5])  / 4096.0
        gx = self._signed(d[8]  << 8 | d[9])  / 32.8
        gy = self._signed(d[10] << 8 | d[11]) / 32.8
        gz = self._signed(d[12] << 8 | d[13]) / 32.8
        return ax, ay, az, gx, gy, gz

    def start(self):
        if not self.is_connected:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.bus:
            try:
                self.bus.close()
            except Exception:
                pass

    def _loop(self):
        prev_time = time.time()
        while self.running:
            now = time.time()
            dt = max(now - prev_time, 0.001)
            prev_time = now
            
            try:
                ax, ay, az, gx, gy, gz = self._read_raw()
                # Apply bias
                gx -= self.gx_b
                gy -= self.gy_b
                gz -= self.gz_b
                
                # Update complementary filter
                roll, pitch, yaw = self.cf.update(ax, ay, az, gx, gy, gz, dt)
                
                with self.state_lock:
                    self.yaw_deg = yaw
                    self.yaw_rate_deg_s = gz
                    
            except Exception as e:
                # Occasional I2C failures can happen, just ignore step
                pass
                
            # Sleep to maintain roughly ~100Hz loop
            elapsed = time.time() - now
            time.sleep(max(0.001, 0.01 - elapsed))

    def get_yaw_data(self):
        """
        Returns (yaw_rad, yaw_rate_rad_s) based on the latest IMU readings.
        If IMU is not connected, returns (None, None).
        """
        if not self.is_connected:
            return None, None
            
        with self.state_lock:
            # We return the yaw RATE in rad/s, as this is immensely robust for sensor fusion.
            # We could also return absolute yaw if the filter is stable, but RATE is best for EKF/kinematics.
            y_rad = math.radians(self.yaw_deg)
            yr_rad_s = math.radians(self.yaw_rate_deg_s)
            
            # The BNO055 used East=0, CCW=positive. 
            # Depending on MPU9250 physical mount, we might need to invert gz.
            # For now, we assume standard right-hand rule Z-axis up (CCW = positive).
            return y_rad, yr_rad_s
