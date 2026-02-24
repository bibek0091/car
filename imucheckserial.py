import time
import sys
import os

# Ensure the 'common' directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.serialhandler.SerialHandler import SerialHandler

def main():
    # 1. Initialization
    # On the BFMC RPi, the serial port is usually /dev/ttyAMA0
    # The baud rate for the Nucleo is typically 115200
    serial_port = "/dev/ttyAMA0"
    baud_rate = 115200
    
    print(f"--- Initializing Serial Connection on {serial_port} ---")
    s_handler = SerialHandler(serial_port, baud_rate)
    
    # 2. Start the Serial Thread
    s_handler.start()
    print("Listening for IMU data... (Press Ctrl+C to stop)\n")

    try:
        while True:
            # 3. Access the IMU dictionary
            # The SerialHandler typically updates 'imu_data' automatically
            imu = s_handler.get_imu_data()
            
            if imu and all(key in imu for key in ['roll', 'pitch', 'yaw']):
                # Formatting for a clean display
                output = (
                    f"ORIENTATION | "
                    f"Yaw: {imu['yaw']:7.2f}° | "
                    f"Pitch: {imu['pitch']:7.2f}° | "
                    f"Roll: {imu['roll']:7.2f}°"
                )
                # '\r' allows the line to overwrite itself in the console
                print(output, end='\r', flush=True)
            else:
                print("Waiting for IMU packets...", end='\r', flush=True)
            
            # High frequency reading (100Hz)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopping Serial Handler...")
    finally:
        s_handler.stop()
        print("Done.")

if __name__ == "__main__":
    main()