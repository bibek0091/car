import time
import board
import busio
import adafruit_bno055

def main():
    print("==================================================")
    print("    Direct Raspberry Pi BNO055 IMU Reader    ")
    print("==================================================")
    print("Attempting to connect to BNO055 via Pi Hardware I2C...\n")

    try:
        # Initialize Software I2C bus (bus 3) to bypass Pi hardware bugs
        import smbus2
        
        try:
            # We use smbus2 to open bus 3, but Adafruit library expects a busio.I2C object. 
            # We can force board.I2C() or use an alternative library, but standard busio doesn't easily let us pick bus 3.
            # INSTEAD: We will just tell the adafruit library to use bus 3 by mocking the I2C interface.
            pass
        except Exception:
            pass

        # The easiest way to select an arbitrary I2C bus via Adfruit Blinka is:
        import busio
        import microcontroller
        
        # In Blinka, to force bus 3 (which we create via config.txt overlay), we use the extended board pins if they exist,
        # OR just use the specific smbus/adafruit_extended_bus library. Since this is a simple script, we'll
        # just use `board.I2C()` and assume the default maps to 1. 
        # Actually, Blinka allows configuring the default I2C bus number via an environment variable, 
        # but let's just use `adafruit_extended_bus` if available, otherwise fallback to standard.
        try:
            from adafruit_extended_bus import ExtendedI2C as I2C
            i2c = I2C(3)
        except ImportError:
            print("Installing adafruit-extended-bus is recommended.")
            print("Falling back to default bus.")
            i2c = busio.I2C(board.SCL, board.SDA)
        
        # Try default Adafruit address (0x28) first, then alternative (0x29)
        sensor = None
        for addr in [0x28, 0x29]:
            try:
                sensor = adafruit_bno055.BNO055_I2C(i2c, address=addr)
                print(f"✅ SUCCESS: BNO055 found at I2C Address: {hex(addr)}")
                break
            except ValueError:
                continue

        if sensor is None:
            print("❌ ERROR: Could not find BNO055 on I2C bus.")
            print("Please check your wiring, and ensure it is connected to the Pi (not the STM32).")
            return

        print("\nStreaming Data (Press Ctrl+C to quit)...\n")
        
        while True:
            # Read Euler angles
            yaw, roll, pitch = sensor.euler
            if yaw is None:
                continue
                
            # Read calibration status (sys, gyro, accel, mag)
            sys_cal, gyro_cal, accel_cal, mag_cal = sensor.calibration_status
            
            # Print live data
            print(f"Yaw: {yaw:6.1f} | Roll: {roll:6.1f} | Pitch: {pitch:6.1f}    [Calib: S{sys_cal} G{gyro_cal} A{accel_cal} M{mag_cal}]", end="\r")
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ FATAL I2C ERROR: {e}")
        print("Tip: Have you run 'sudo raspi-config' to Interfacing Options -> Enable I2C?")

if __name__ == "__main__":
    main()
