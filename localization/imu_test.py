#!/usr/bin/env python3
"""
imu_test.py — MPU-9250 / ICM-20689 / MPU-6500 Auto-Detection & Live Test
=========================================================================
Run on Raspberry Pi:
    python3 imu_test.py

What it does
------------
1. Scans ALL common I2C buses (0, 1, 3, 4) and ALL common IMU addresses
   (0x68, 0x69) automatically — no config needed.
2. Reads the WHO_AM_I register (0x75) to identify the exact chip variant.
3. Wakes up the chip with the correct register sequence for each variant.
4. Calibrates gyro bias on 100 still samples.
5. Streams live Roll / Pitch / Yaw + raw accel/gyro at ~50 Hz with a
   live ASCII bar graph — easy to see if the sensor is responding.
6. Prints a PASS / FAIL verdict at the end (Ctrl-C to stop early).
"""

import time
import math
import sys

# ── Try smbus2 first, fall back to smbus ────────────────────────────────────
try:
    import smbus2 as smbus_mod
    SMBus = smbus_mod.SMBus
    print("[OK] Using smbus2 library")
except ImportError:
    try:
        import smbus as smbus_mod
        SMBus = smbus_mod.SMBus
        print("[OK] Using smbus library")
    except ImportError:
        print("[FATAL] Neither smbus2 nor smbus is installed.")
        print("        Run: pip3 install smbus2")
        sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────
SCAN_BUSES     = [1, 0, 3, 4]          # RPi buses to try (bus 1 is default)
SCAN_ADDRESSES = [0x68, 0x69]          # AD0=LOW → 0x68, AD0=HIGH → 0x69

REG_WHO_AM_I  = 0x75                   # Read this to identify chip
REG_PWR_MGMT1 = 0x6B                   # Reset / wake
REG_GYRO_CFG  = 0x1B                   # ±1000 dps → 0x10
REG_ACCEL_CFG = 0x1C                   # ±8g     → 0x10
REG_DATA      = 0x3B                   # Start of 14-byte accel+temp+gyro block

# WHO_AM_I → chip name + gyro divisor (LSB per deg/s)
WHO_AM_I_MAP = {
    0x71: ("MPU-9250",   32.8),
    0x73: ("MPU-9255",   32.8),
    0x70: ("MPU-6500",   32.8),
    0x68: ("MPU-6050",   131.0),  # default range ±250 dps
    0x69: ("ICM-20689",  32.8),
    0x98: ("ICM-20689",  32.8),
}

GYRO_DIV_OVERRIDE = 32.8   # used when WHO_AM_I is unknown but chip woke up
ACCEL_DIV         = 4096.0 # ±8g range → 4096 LSB/g

CALIB_SAMPLES = 100
LOOP_HZ       = 50


# ── Helpers ──────────────────────────────────────────────────────────────────

def signed16(val):
    return val - 65536 if val > 32767 else val


def read_raw(bus, addr):
    """Read 14 bytes starting at 0x3B → ax,ay,az,temp,gx,gy,gz in raw LSB."""
    d  = bus.read_i2c_block_data(addr, REG_DATA, 14)
    ax = signed16(d[0]  << 8 | d[1])
    ay = signed16(d[2]  << 8 | d[3])
    az = signed16(d[4]  << 8 | d[5])
    # skip temp bytes d[6], d[7]
    gx = signed16(d[8]  << 8 | d[9])
    gy = signed16(d[10] << 8 | d[11])
    gz = signed16(d[12] << 8 | d[13])
    return ax, ay, az, gx, gy, gz


def bar(val, lo=-5.0, hi=5.0, width=30):
    """ASCII bar between lo and hi."""
    pct  = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    fill = int(pct * width)
    return "[" + "#" * fill + "-" * (width - fill) + f"] {val:+7.2f}"


# ── Scan & detect ─────────────────────────────────────────────────────────────

def scan_and_open():
    """Returns (bus_obj, bus_num, addr, chip_name, gyro_div) or None."""
    print("\n=== IMU Auto-Detection Scan ===")
    for bus_num in SCAN_BUSES:
        try:
            bus = SMBus(bus_num)
        except (FileNotFoundError, OSError):
            print(f"  Bus {bus_num}: not available (skip)")
            continue

        for addr in SCAN_ADDRESSES:
            try:
                who = bus.read_byte_data(addr, REG_WHO_AM_I)
                chip_name, gyro_div = WHO_AM_I_MAP.get(who, (f"UNKNOWN(0x{who:02X})", GYRO_DIV_OVERRIDE))
                print(f"  Bus {bus_num}, addr 0x{addr:02X}: WHO_AM_I=0x{who:02X}  →  {chip_name}  ✅ FOUND")
                return bus, bus_num, addr, chip_name, gyro_div
            except OSError:
                print(f"  Bus {bus_num}, addr 0x{addr:02X}: no response")

        bus.close()

    return None


# ── Wake-up / configure ───────────────────────────────────────────────────────

def wake_up(bus, addr, chip_name):
    print(f"\n=== Initialising {chip_name} ===")
    # 1. Reset + wake from sleep
    bus.write_byte_data(addr, REG_PWR_MGMT1, 0x00)
    time.sleep(0.1)
    # 2. Gyro range ±1000 dps
    bus.write_byte_data(addr, REG_GYRO_CFG, 0x10)
    time.sleep(0.05)
    # 3. Accel range ±8g
    bus.write_byte_data(addr, REG_ACCEL_CFG, 0x10)
    time.sleep(0.05)
    print("  PWR_MGMT1 = 0x00 (awake)")
    print("  GYRO_CFG  = 0x10 (±1000 dps)")
    print("  ACCEL_CFG = 0x10 (±8g)")
    # Verify WHO_AM_I again post-wake
    who = bus.read_byte_data(addr, REG_WHO_AM_I)
    print(f"  WHO_AM_I post-wake: 0x{who:02X}  {'✅' if who in WHO_AM_I_MAP else '⚠️ unrecognised'}")


# ── Gyro calibration ─────────────────────────────────────────────────────────

def calibrate(bus, addr, gyro_div, n=CALIB_SAMPLES):
    print(f"\n=== Gyro Calibration ({n} samples — KEEP IMU STILL) ===")
    gx_s = gy_s = gz_s = 0.0
    for i in range(n):
        _, _, _, gx, gy, gz = read_raw(bus, addr)
        gx_s += gx / gyro_div
        gy_s += gy / gyro_div
        gz_s += gz / gyro_div
        time.sleep(0.01)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n}...")
    bx, by, bz = gx_s / n, gy_s / n, gz_s / n
    print(f"  Bias → gx:{bx:.4f}  gy:{by:.4f}  gz:{bz:.4f} deg/s")
    return bx, by, bz


# ── Live data stream ──────────────────────────────────────────────────────────

def live_stream(bus, addr, gyro_div, bx, by, bz):
    print(f"\n=== Live Data Stream at {LOOP_HZ} Hz (Ctrl-C to stop) ===\n")

    roll = pitch = yaw = 0.0
    prev_t = time.time()
    alpha  = 0.96          # complementary filter coefficient
    samples_ok = 0
    samples_err = 0

    try:
        while True:
            now = time.time()
            dt  = max(now - prev_t, 0.001)
            prev_t = now

            try:
                ax, ay, az, gx_r, gy_r, gz_r = read_raw(bus, addr)
                samples_ok += 1
            except OSError as e:
                samples_err += 1
                print(f"  [READ ERROR #{samples_err}] {e}")
                time.sleep(0.02)
                continue

            # Convert to physical units
            ax_g = ax / ACCEL_DIV;  ay_g = ay / ACCEL_DIV;  az_g = az / ACCEL_DIV
            gx_d = gx_r / gyro_div - bx
            gy_d = gy_r / gyro_div - by
            gz_d = gz_r / gyro_div - bz

            # Accel-derived roll/pitch (degrees)
            roll_acc  = math.degrees(math.atan2(ay_g, az_g))
            pitch_acc = math.degrees(math.atan2(-ax_g, math.sqrt(ay_g**2 + az_g**2)))

            # Complementary filter
            roll  = alpha * (roll  + gx_d * dt) + (1 - alpha) * roll_acc
            pitch = alpha * (pitch + gy_d * dt) + (1 - alpha) * pitch_acc
            yaw  += gz_d * dt
            if yaw >  180: yaw -= 360
            if yaw < -180: yaw += 360

            # Print live telemetry (overwrite single block)
            print(
                f"\r  Roll {bar(roll,-30,30,20)} "
                f"  Pitch {bar(pitch,-30,30,20)} "
                f"  Yaw {bar(yaw,-180,180,20)} "
                f"  gz={gz_d:+6.2f}°/s   ",
                end="", flush=True
            )

            elapsed = time.time() - now
            time.sleep(max(0.0, 1.0 / LOOP_HZ - elapsed))

    except KeyboardInterrupt:
        err_rate = 100.0 * samples_err / max(1, samples_ok + samples_err)
        print(f"\n\n=== RESULT ===")
        print(f"  Samples OK  : {samples_ok}")
        print(f"  Errors      : {samples_err}  ({err_rate:.1f}%)")
        if err_rate < 5.0:
            print("  STATUS      : ✅ PASS — IMU is working correctly")
        elif err_rate < 20.0:
            print("  STATUS      : ⚠️  MARGINAL — check wiring / pull-up resistors")
        else:
            print("  STATUS      : ❌ FAIL — IMU is not communicating reliably")
        print(f"\n  Final pose  : Roll={roll:.1f}°  Pitch={pitch:.1f}°  Yaw={yaw:.1f}°")
        print(f"\n  >>> Use these values in mpu9250_imu.py <<<")
        print(f"  MPU9250_Thread(bus=<BUS_SHOWN_ABOVE>, address=<ADDR_SHOWN_ABOVE>)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BFMC IMU Auto-Detection & Live Test")
    print("=" * 60)

    result = scan_and_open()
    if result is None:
        print("\n[FATAL] No IMU found on any bus or address.")
        print("  Checklist:")
        print("  1. Enable I2C: sudo raspi-config → Interface Options → I2C")
        print("  2. Check wiring: VCC=3.3V  GND=GND  SDA=Pin3  SCL=Pin5")
        print("  3. Verify device: sudo i2cdetect -y 1")
        print("  4. Reboot after enabling I2C: sudo reboot")
        sys.exit(1)

    bus, bus_num, addr, chip_name, gyro_div = result
    print(f"\n  → Will connect using  bus={bus_num}  address=0x{addr:02X}")

    try:
        wake_up(bus, addr, chip_name)
        bx, by, bz = calibrate(bus, addr, gyro_div)
        live_stream(bus, addr, gyro_div, bx, by, bz)
    finally:
        bus.close()
        print("\n  I2C bus closed. Done.")


if __name__ == "__main__":
    main()
