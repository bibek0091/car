import time
import board
import busio
import digitalio
import adafruit_bno055

def main():
    print("==================================================")
    print("    Direct Raspberry Pi BNO055 IMU Reader (SPI)   ")
    print("==================================================")
    print("Attempting to connect to BNO055 via Pi Hardware SPI...\n")

    try:
        # Initialize SPI bus
        spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
        
        # Initialize Chip Select Pin (CE0)
        cs = digitalio.DigitalInOut(board.CE0)
        
        # Connect to BNO055 via SPI
        sensor = adafruit_bno055.BNO055_SPI(spi, cs)

        print("✅ SUCCESS: BNO055 found on SPI!")
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
                # SPI can occasionally drop a frame 
                pass

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ FATAL SPI ERROR: {e}")
        print("Tip: Have you run 'sudo raspi-config' to Interfacing Options -> Enable SPI?")
        print("     Make sure PS0 is pulled HIGH (3.3V) and PS1 is pulled LOW (GND) on the IMU!")

if __name__ == "__main__":
    main()
