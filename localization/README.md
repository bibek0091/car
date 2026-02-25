# BFMC Autonomous Pilot System with Localization

## Overview
This is a comprehensive, production-ready system for the Bosch Future Mobility Challenge (BFMC) autonomous 1:10 scale vehicle. It incorporates map-based A* path planning, YOLO traffic decision logic, BEV lane tracking, sensor fusion (IMU + Dead Reckoning + Map matching), and a custom Tkinter Dashboard.

## Hardware Requirements
- **Compute:** Raspberry Pi 4 (or Pi 5)
- **Camera:** PiCamera2 connected via CSI
- **IMU:** BNO055 connected via I2C (SDA=GPIO2, SCL=GPIO3, VIN=3.3V, GND)
- **Motor & Steering:** STM32 controller handling ESC PWM and Servo PWM via serial connection.

## Key Modules
1. `main.py`: The Main Orchestrator and Tkinter GUI. Handles the 30Hz pilot loop and renders the dashboard.
2. `hardware_io.py`: Interfaces with the BNO055, PiCamera2, and `STM32_SerialHandler`.
3. `map_planner.py`: Parses the `.graphml` map to calculate A* routes and target lookahead points.
4. `perception.py`: Runs BEV transformation, image binarization, and polynomial lane fitting.
5. `traffic_module.py`: Runs YOLO object detection asynchronously and manages a state machine for traffic lights and stops.
6. `localization.py`: Multi-layer pose estimator correcting kinematics with IMU and map checkpoints.
7. `control.py`: Dynamic speed controller and Pure Pursuit steering algorithm logic.

## Setup and Run
1. Ensure the Python environment matches `requirements.txt`.
2. Connect your hardware. Start I2C capabilities via `raspi-config`. 
3. Verify your map file `Competition_track_graph.graphml` and model `best.pt` exist alongside `main.py`.

### Execution
For standard hardware execution:
```bash
python main.py --start 1 --target 84
```

For Simulation mode without hardware logic errors:
```bash
python main.py --sim --sim-video test_footage.mp4 --start 1 --target 84
```

## Dashboard Operations
- The Dashboard operates entirely off the main thread to prevent blocking steering loops.
- **E-STOP:** Pressing the button immediately halts the vehicle.
- **STARTUP:** The first 6.0 seconds require the vehicle to be stationary to calibrate camera auto-exposure and zero the IMU.
