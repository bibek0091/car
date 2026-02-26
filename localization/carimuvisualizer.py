"""
MPU-9250 Live IMU Visualizer — Matplotlib
==========================================
Shows Roll, Pitch, Yaw + raw Accel & Gyro live plots.

Install:
    pip install smbus2 numpy matplotlib

Run:
    python imu_plot.py
"""

import smbus2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time

# ─────────────────────────────────────────────
#  MPU-9250
# ─────────────────────────────────────────────
class MPU9250:
    def __init__(self, bus=1, address=0x68):
        self.bus  = smbus2.SMBus(bus)
        self.addr = address
        self.bus.write_byte_data(self.addr, 0x6B, 0x00)  # wake up
        time.sleep(0.1)
        self.bus.write_byte_data(self.addr, 0x1B, 0x10)  # gyro ±1000 deg/s
        self.bus.write_byte_data(self.addr, 0x1C, 0x10)  # accel ±8g
        time.sleep(0.1)
        print(f"MPU-9250 connected at 0x{self.addr:02X}")

    def _signed(self, v):
        return v - 65536 if v > 32767 else v

    def read(self):
        d  = self.bus.read_i2c_block_data(self.addr, 0x3B, 14)
        ax = self._signed(d[0]  << 8 | d[1])  / 4096.0
        ay = self._signed(d[2]  << 8 | d[3])  / 4096.0
        az = self._signed(d[4]  << 8 | d[5])  / 4096.0
        gx = self._signed(d[8]  << 8 | d[9])  / 32.8
        gy = self._signed(d[10] << 8 | d[11]) / 32.8
        gz = self._signed(d[12] << 8 | d[13]) / 32.8
        return ax, ay, az, gx, gy, gz


# ─────────────────────────────────────────────
#  COMPLEMENTARY FILTER
# ─────────────────────────────────────────────
class ComplementaryFilter:
    def __init__(self, alpha=0.96):
        self.alpha = alpha
        self.roll = self.pitch = self.yaw = 0.0

    def update(self, ax, ay, az, gx, gy, gz, dt):
        roll_acc  = math.degrees(math.atan2(ay, az))
        pitch_acc = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
        self.roll  = self.alpha * (self.roll  + gx * dt) + (1 - self.alpha) * roll_acc
        self.pitch = self.alpha * (self.pitch + gy * dt) + (1 - self.alpha) * pitch_acc
        self.yaw  += gz * dt
        if self.yaw >  180: self.yaw -= 360
        if self.yaw < -180: self.yaw += 360
        return self.roll, self.pitch, self.yaw


# ─────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────
mpu = MPU9250(bus=1, address=0x68)
cf  = ComplementaryFilter(alpha=0.96)

# Calibrate gyro bias
print("Calibrating gyro (keep still 2 sec)...")
gx_b = gy_b = gz_b = 0.0
N = 200
for _ in range(N):
    _, _, _, gx, gy, gz = mpu.read()
    gx_b += gx; gy_b += gy; gz_b += gz
    time.sleep(0.01)
gx_b /= N; gy_b /= N; gz_b /= N
print(f"Bias: gx={gx_b:.3f}  gy={gy_b:.3f}  gz={gz_b:.3f}")
print("Starting live plot... (close window to stop)")

# Data buffers (last 200 samples)
MAXLEN = 200
t_buf                          = list(range(MAXLEN))
roll_buf  = [0.0] * MAXLEN
pitch_buf = [0.0] * MAXLEN
yaw_buf   = [0.0] * MAXLEN
ax_buf    = [0.0] * MAXLEN
ay_buf    = [0.0] * MAXLEN
az_buf    = [0.0] * MAXLEN
gx_buf    = [0.0] * MAXLEN
gy_buf    = [0.0] * MAXLEN
gz_buf    = [0.0] * MAXLEN

prev_time = time.time()

# ─────────────────────────────────────────────
#  MATPLOTLIB SETUP
# ─────────────────────────────────────────────
plt.style.use("dark_background")
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
fig.suptitle("MPU-9250 Live IMU Data  |  Bosch Future Mobility Challenge",
             fontsize=14, fontweight="bold", color="white")
fig.tight_layout(pad=3.0)

# ── Plot 1: Roll Pitch Yaw ──
ax1 = axes[0]
ax1.set_title("Orientation (deg)", color="white")
ax1.set_ylim(-180, 180)
ax1.set_xlim(0, MAXLEN)
ax1.set_ylabel("Degrees")
ax1.grid(True, alpha=0.3)
line_roll,  = ax1.plot(roll_buf,  color="#3498db", label="Roll",  linewidth=2)
line_pitch, = ax1.plot(pitch_buf, color="#2ecc71", label="Pitch", linewidth=2)
line_yaw,   = ax1.plot(yaw_buf,   color="#f1c40f", label="Yaw",   linewidth=2)
ax1.legend(loc="upper right")
ax1.axhline(0, color="white", linewidth=0.5, linestyle="--")

# ── Plot 2: Accelerometer ──
ax2 = axes[1]
ax2.set_title("Accelerometer (g)", color="white")
ax2.set_ylim(-4, 4)
ax2.set_xlim(0, MAXLEN)
ax2.set_ylabel("g force")
ax2.grid(True, alpha=0.3)
line_ax, = ax2.plot(ax_buf, color="#e74c3c", label="Ax", linewidth=1.5)
line_ay, = ax2.plot(ay_buf, color="#e67e22", label="Ay", linewidth=1.5)
line_az, = ax2.plot(az_buf, color="#9b59b6", label="Az", linewidth=1.5)
ax2.legend(loc="upper right")
ax2.axhline(0, color="white", linewidth=0.5, linestyle="--")

# ── Plot 3: Gyroscope ──
ax3 = axes[2]
ax3.set_title("Gyroscope (deg/s)", color="white")
ax3.set_ylim(-200, 200)
ax3.set_xlim(0, MAXLEN)
ax3.set_ylabel("deg/s")
ax3.set_xlabel("Samples")
ax3.grid(True, alpha=0.3)
line_gx, = ax3.plot(gx_buf, color="#1abc9c", label="Gx", linewidth=1.5)
line_gy, = ax3.plot(gy_buf, color="#e91e63", label="Gy", linewidth=1.5)
line_gz, = ax3.plot(gz_buf, color="#00bcd4", label="Gz", linewidth=1.5)
ax3.legend(loc="upper right")
ax3.axhline(0, color="white", linewidth=0.5, linestyle="--")

# Live text display
info_text = fig.text(0.01, 0.01,
    "Roll: 0.00  Pitch: 0.00  Yaw: 0.00",
    fontsize=11, color="white",
    bbox=dict(facecolor="#1a1a2e", edgecolor="gray", boxstyle="round"))


# ─────────────────────────────────────────────
#  ANIMATION UPDATE
# ─────────────────────────────────────────────
def update(_):
    global prev_time

    now = time.time()
    dt  = max(now - prev_time, 0.001)
    prev_time = now

    try:
        ax, ay, az, gx, gy, gz = mpu.read()
        gx -= gx_b; gy -= gy_b; gz -= gz_b
        roll, pitch, yaw = cf.update(ax, ay, az, gx, gy, gz, dt)
    except Exception as e:
        print(f"Read error: {e}")
        return

    # Append and trim buffers
    def push(buf, val):
        buf.append(val)
        if len(buf) > MAXLEN: buf.pop(0)

    push(roll_buf,  roll)
    push(pitch_buf, pitch)
    push(yaw_buf,   yaw)
    push(ax_buf, ax); push(ay_buf, ay); push(az_buf, az)
    push(gx_buf, gx); push(gy_buf, gy); push(gz_buf, gz)

    # Update lines
    line_roll.set_ydata(roll_buf)
    line_pitch.set_ydata(pitch_buf)
    line_yaw.set_ydata(yaw_buf)
    line_ax.set_ydata(ax_buf)
    line_ay.set_ydata(ay_buf)
    line_az.set_ydata(az_buf)
    line_gx.set_ydata(gx_buf)
    line_gy.set_ydata(gy_buf)
    line_gz.set_ydata(gz_buf)

    # Update info text
    info_text.set_text(
        f"Roll: {roll:+7.2f}°   Pitch: {pitch:+7.2f}°   Yaw: {yaw:+7.2f}°  |  "
        f"Ax: {ax:+.3f}g  Ay: {ay:+.3f}g  Az: {az:+.3f}g"
    )

    print(f"Roll:{roll:+7.2f}  Pitch:{pitch:+7.2f}  Yaw:{yaw:+7.2f}  |  "
          f"Ax:{ax:+.3f}  Ay:{ay:+.3f}  Az:{az:+.3f}  "
          f"Gx:{gx:+.2f}  Gy:{gy:+.2f}  Gz:{gz:+.2f}")


ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()