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
        # Initialize I2C bus on Raspberry Pi Pins 3 (SDA) and 5 (SCL)
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
