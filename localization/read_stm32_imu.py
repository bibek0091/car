import serial
import serial.tools.list_ports
import time
import sys

def auto_detect_port():
    print("Scanning for STM32...")
    for port in serial.tools.list_ports.comports():
        if hasattr(port, "vid") and port.vid == 0x0483:
            print(f"Found STM32 on {port.device}")
            return port.device
    return None

def main():
    port_name = auto_detect_port()
    if not port_name:
        print("ERROR: Could not find STM32. Is it plugged in?")
        sys.exit(1)

    try:
        # 1. Open Serial Port
        ser = serial.Serial(port_name, 115200, timeout=1.0)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"Connected to {port_name} at 115200 baud.")
        time.sleep(2) # Wait for STM32 to reboot if DTR triggered
        
        # 2. Send Start Sequence
        print("Sending Initialization Sequence...")
        
        # Step A: Keep Alive
        ser.write(b"#alive:1;;\r\n")
        time.sleep(0.1)
        
        # Step B: KL15 (Required by mbed-os-empty to enable IMU tasks)
        print(" -> Requesting KL:15 (Accessories)")
        ser.write(b"#kl:15;;\r\n")
        time.sleep(0.5)
        
        # Step C: Turn on IMU Stream
        print(" -> Requesting IMU stream (#imu:1;;)")
        ser.write(b"#imu:1;;\r\n")
        time.sleep(0.5)
        
        # Step D: KL30 (Ignition / Motor Enable)
        print(" -> Requesting KL:30 (Ignition)")
        ser.write(b"#kl:30;;\r\n")
        time.sleep(0.5)

        print("\n==========================================")
        print("Listening for @imu: messages...")
        print("==========================================")
        
        # 3. Read loop
        read_buffer = ""
        while True:
            # Send keep-alive every loop to prevent STM32 timeout
            ser.write(b"#alive:1;;\r\n")
            
            if ser.in_waiting:
                data = ser.read(ser.in_waiting).decode(errors='ignore')
                read_buffer += data
                
                while "\r\n" in read_buffer:
                    line, read_buffer = read_buffer.split("\r\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Print everything the STM32 says for debugging!
                    if line.startswith("@imu:"):
                        # We got IMU data!
                        print(f"✅ IMU DATA RECEIVED: {line}")
                    else:
                        # Print other telemetry
                        print(f"   Serial msg: {line}")
            
            time.sleep(0.05)

    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.write(b"#kl:0;;\r\n") # Turn off the car
            ser.close()
            print("Port closed.")

if __name__ == '__main__':
    main()
