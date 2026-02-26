import time
import serial
import adafruit_bno055

def main():
    print("==================================================")
    print("    Direct Raspberry Pi BNO055 IMU Reader (UART)  ")
    print("==================================================")
    print("Attempting to connect to BNO055 via Pi Hardware UART...\n")

    try:
        # Connect to Pi's hardware serial port
        uart = serial.Serial("/dev/serial0", 115200, timeout=1.0)
        sensor = adafruit_bno055.BNO055_UART(uart)

        print("✅ SUCCESS: BNO055 found on UART!")
        print("\nStreaming Data (Press Ctrl+C to quit)...\n")
        
        while True:
            try:
                # Read Euler angles
                yaw, roll, pitch = sensor.euler
                if yaw is None:
                    continue
                    
                # Read calibration status (sys, gyro, accel, mag)
                sys_cal, gyro_cal, accel_cal, mag_cal = sensor.calibration_status
                
                # Print live data
                print(f"Yaw: {yaw:6.1f} | Roll: {roll:6.1f} | Pitch: {pitch:6.1f}    [Calib: S{sys_cal} G{gyro_cal} A{accel_cal} M{mag_cal}]", end="\r")
                
                time.sleep(0.05)
            except RuntimeError:
                # UART drops partial packets sometimes, just catch and retry
                pass

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ FATAL UART ERROR: {e}")
        print("Tip: Have you run 'sudo raspi-config' to Interfacing Options -> Serial?")
        print("     - Login shell over serial: NO")
        print("     - Serial port hardware: YES")

if __name__ == "__main__":
    main()
