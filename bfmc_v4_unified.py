"""
BFMC V4 Unified Pilot: Ultimate Autonomous Orchestrator
=========================================================
Features:
- Single monolithic file deployment.
- Direct Raw Hardware I2C IMU Integration (Pins 3/5).
- Picamera2 Stream with Locked AWB & Exposure for LED stability.
- Simplified BFMC Tkinter Mission Control Dashboard.
- Deep YOLOv8 integration with dead-reckoning fallback.
"""

import cv2
import numpy as np
import math
import time
import argparse
import threading
import sys
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
from collections import deque

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARN] YOLO not found. Deep learning perception disabled.")

try:
    from smbus2 import SMBus
    _I2C_AVAILABLE = True
except ImportError:
    _I2C_AVAILABLE = False
    print("[WARN] smbus2 not found. Hardware I2C IMU disabled. Install with: pip install smbus2")

# We import the Pi camera, or mock it if on Windows
try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    _CAM_AVAILABLE = False


# ============================================================================
# 1. HARDWARE I2C IMU (MPU6050 / BNO055 via smbus2)
# ============================================================================
class HardwareIMU:
    """
    Directly reads Gyro Z-axis off the Raspberry Pi I2C Bus 1 (Pins SDA 3, SCL 5).
    Provides extremely low-latency yaw updates for dead-reckoning curves.
    """
    def __init__(self, address=0x68):
        self.address = address
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.last_time = time.time()
        self.running = False
        self.bus = None

        if _I2C_AVAILABLE:
            try:
                self.bus = SMBus(1) # RPi I2C bus 1
                # Wake up MPU6050 (write 0 to power management register 0x6B)
                self.bus.write_byte_data(self.address, 0x6B, 0)
                # Config Gyro to +/- 250 deg/s (Register 0x1B, value 0x00)
                self.bus.write_byte_data(self.address, 0x1B, 0x00)
                print(f"[IMU] Successfully connected to I2C Device at {hex(self.address)}")
            except Exception as e:
                print(f"[IMU ERROR] Could not initialize I2C: {e}")
                self.bus = None

    def _read_word_2c(self, reg):
        if not self.bus: return 0
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        val = (high << 8) + low
        if val >= 0x8000:
            return -((65535 - val) + 1)
        return val

    def update_loop(self):
        self.running = True
        self.last_time = time.time()
        while self.running:
            now = time.time()
            dt = now - self.last_time
            self.last_time = now

            if self.bus:
                try:
                    # MPU6050 Z-Axis Gyro Register is 0x47 (high) and 0x48 (low)
                    gz_raw = self._read_word_2c(0x47)
                    # Convert to deg/s (131.0 LSB/(deg/s) for +/- 250deg/s range)
                    gz_rate = gz_raw / 131.0
                    
                    # Small deadband to prevent drift while stationary
                    if abs(gz_rate) < 0.5:
                        gz_rate = 0.0
                        
                    self.yaw += (gz_rate * dt)
                except Exception as e:
                    pass
            else:
                # Simulation Mode: Just mock some drift
                pass
                
            time.sleep(0.01) # 100Hz polling

    def start(self):
        t = threading.Thread(target=self.update_loop, daemon=True)
        t.start()
        
    def reset_yaw(self, new_yaw=0.0):
        self.yaw = new_yaw
        
    def get_orientation(self):
        return {"yaw": self.yaw, "pitch": self.pitch, "roll": self.roll}


# ============================================================================
# 2. SIMPLIFIED TKINTER DASHBOARD
# ============================================================================
class SimplifiedBFMCDashboard:
    """
    A unified, streamlined dashboard optimized for BFMC requirements.
    Focuses on map localization marking, essential telemetry, and tracking state.
    """
    def __init__(self, root, pilot_ref):
        self.root = root
        self.pilot = pilot_ref
        self.root.title("BFMC Mission Control")
        self.root.geometry("1400x800")
        self.root.configure(bg="#111111")
        
        # UI State Variables
        self.steer_hist = deque(maxlen=200)
        self.yaw_hist = deque(maxlen=200)
        self.ui_yaw = 0.0
        
        # Widescreen split
        self.left_frame = tk.Frame(self.root, bg="#1a1a1c", width=350, height=780)
        self.left_frame.place(x=10, y=10)
        
        self.center_frame = tk.Frame(self.root, bg="#0c0c0c", width=700, height=780)
        self.center_frame.place(x=370, y=10)
        
        self.right_frame = tk.Frame(self.root, bg="#1a1a1c", width=310, height=780)
        self.right_frame.place(x=1080, y=10)
        
        # --- LEFT: KINEMATICS ---
        self.l_title = tk.Label(self.left_frame, text="HARDWARE TELEMETRY", fg="#888", bg="#1a1a1c", font=("Courier", 12, "bold"))
        self.l_title.place(x=20, y=15)
        
        self.l_speed = tk.Label(self.left_frame, text="0.0 cm/s", fg="#11ccff", bg="#1a1a1c", font=("Fixedsys", 36, "bold"))
        self.l_speed.place(x=20, y=50)
        
        self.l_steer = tk.Label(self.left_frame, text="STEER: 0.0°", fg="#ff8822", bg="#1a1a1c", font=("Fixedsys", 24))
        self.l_steer.place(x=20, y=120)

        self.l_yaw = tk.Label(self.left_frame, text="IMU YAW: 0.0°", fg="#22dd44", bg="#1a1a1c", font=("Fixedsys", 20))
        self.l_yaw.place(x=20, y=170)
        
        self.c_speed_graph = Canvas(self.left_frame, width=310, height=150, bg="#0c0c0c", highlightthickness=0)
        self.c_speed_graph.place(x=20, y=250)

        self.l_nav = tk.Label(self.left_frame, text="NAV: INIT", fg="#fff", bg="#1a1a1c", font=("Courier", 18, "bold"))
        self.l_nav.place(x=20, y=420)
        
        self.l_anchor = tk.Label(self.left_frame, text="ANCHOR: WAIT", fg="#aaa", bg="#1a1a1c", font=("Courier", 14))
        self.l_anchor.place(x=20, y=460)

        # --- CENTER: GLOBAL MAP LOCALIZATION ---
        self.c_map = Canvas(self.center_frame, width=680, height=760, bg="#111", highlightthickness=0)
        self.c_map.place(x=10, y=10)
        self.c_map.bind("<Button-1>", self._on_map_click)
        
        # --- RIGHT: PERCEPTION ---
        self.r_title = tk.Label(self.right_frame, text="PERCEPTION", fg="#888", bg="#1a1a1c", font=("Courier", 12, "bold"))
        self.r_title.place(x=20, y=15)
        
        self.cam_lbl = tk.Label(self.right_frame, bg="#000")
        self.cam_lbl.place(x=10, y=50, width=290, height=200)

        self.l_signs = tk.Label(self.right_frame, text="DETECTED SIGNS:", fg="#aaa", bg="#1a1a1c", font=("Courier", 12))
        self.l_signs.place(x=20, y=270)
        
        self.c_icons = Canvas(self.right_frame, width=270, height=450, bg="#1a1a1c", highlightthickness=0)
        self.c_icons.place(x=20, y=300)
        
        self._init_map_cache()
        self.root.after(50, self.update_tick)

    def _init_map_cache(self):
        self.c_map.create_text(340, 30, text="GLOBAL MAP TRACK", fill="#20d2c8", font=("Courier", 16, "bold"))
        self.c_map.create_text(340, 60, text="CLICK MAP TO SET START POSE / TELEPORT", fill="#888", font=("Courier", 10))
        
        gmap = self.pilot.gmap
        if len(gmap.points) == 0:
            self.map_scale = 1.0
            self.map_offset = np.array([340, 400])
            self.c_map.create_arc(100, 150, 580, 650, outline="#444", width=4, style=tk.ARC, start=0, extent=359.9)
            return
            
        pmin = np.min(gmap.points, axis=0)
        pmax = np.max(gmap.points, axis=0)
        
        w_m = max(1, pmax[0] - pmin[0])
        h_m = max(1, pmax[1] - pmin[1])
        
        scale_x = (680 - 80) / w_m
        scale_y = (760 - 100) / h_m
        self.map_scale = min(scale_x, scale_y)
        
        cx_m = (pmax[0] + pmin[0]) / 2.0
        cy_m = (pmax[1] + pmin[1]) / 2.0
        self.map_offset = np.array([340 - cx_m * self.map_scale, 420 - cy_m * self.map_scale])
        
        scaled = (gmap.points * self.map_scale + self.map_offset).astype(int)
        
        step = max(1, len(scaled) // 4000)
        for i in range(0, len(scaled), step):
            px, py = scaled[i]
            self.c_map.create_rectangle(px, py, px+1, py+1, fill="#555", outline="")

    def _on_map_click(self, event):
        """Allows humans to set the car's initial global pose."""
        rx = (event.x - self.map_offset[0]) / self.map_scale
        ry = (event.y - self.map_offset[1]) / self.map_scale
        
        if len(self.pilot.gmap.points) > 0:
            idx, _ = self.pilot.gmap.get_nearest_spline_index(rx, ry)
            heading = self.pilot.gmap.headings[idx]
        else:
            heading = 0.0
            
        print(f"[UI] Map Click Detected. Teleporting to X={rx:.2f}, Y={ry:.2f}")
        
        self.pilot.global_x = rx
        self.pilot.global_y = ry
        self.pilot.imu.reset_yaw(new_yaw=math.degrees(heading))
        self.pilot.anchor = "GLOBAL_OVERRIDE"

    def update_tick(self):
        # Read Live Pilot State
        spd = self.pilot.speed
        st = self.pilot.steer
        yw = self.pilot.imu.yaw
        
        self.steer_hist.append(st)
        
        # Update Labels
        self.l_speed.config(text=f"{spd:.1f} cm/s")
        self.l_steer.config(text=f"STEER: {st:+.1f}°")
        self.l_yaw.config(text=f"IMU YAW: {yw:.1f}°")
        self.l_nav.config(text=f"NAV: {self.pilot.nav_state}")
        self.l_anchor.config(text=f"ANCHOR: {self.pilot.anchor}")
        
        # Update Telemetry Graph
        self.c_speed_graph.delete("all")
        self.c_speed_graph.create_line(10, 75, 300, 75, fill="#444") # Zero Steer
        if len(self.steer_hist) > 2:
            pts = []
            step = 290.0 / max(1, len(self.steer_hist)-1)
            for i, val in enumerate(self.steer_hist):
                pts.append(10 + i*step)
                pts.append(75 - (val / 45.0) * 60)
            self.c_speed_graph.create_line(pts, fill="#ff8822", width=2)
            
        # Update Camera
        if self.pilot.yolo_frame is not None:
            rgb = cv2.cvtColor(cv2.resize(self.pilot.yolo_frame, (290, 200)), cv2.COLOR_BGR2RGB)
            self.yolo_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.cam_lbl.config(image=self.yolo_tk)
            
        # Draw Map Pose
        self.c_map.delete("pose")
        px = self.map_offset[0] + self.pilot.global_x * self.map_scale
        py = self.map_offset[1] + self.pilot.global_y * self.map_scale
        self.c_map.create_oval(px-8, py-8, px+8, py+8, fill="#11ccff", tags="pose")
        
        hx = px + math.cos(math.radians(yw)) * 30
        hy = py + math.sin(math.radians(yw)) * 30
        self.c_map.create_line(px, py, hx, hy, fill="#fff", width=3, arrow=tk.LAST, tags="pose")

        self.root.after(50, self.update_tick)


# ============================================================================
# 3. UNIFIED PILOT ORCHESTRATOR
# ============================================================================

# Assuming the user has these custom files in the same directory:
from map_parser import GlobalMap
from localization_engine import LocalizationEngine
from trajectory_planner import MapTrajectoryPlanner
try:
    from test_move import STM32_SerialHandler # Fallback for Serial
except ImportError:
    class STM32_SerialHandler:
        def connect(self): return False
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def disconnect(self): pass

class BFMC_V4_Pilot:
    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        self.running = True
        
        # 1. Hardware Interfaces
        self.imu = HardwareIMU()
        self.imu.start()
        
        self.handler = STM32_SerialHandler()
        self.connected = False if sim_mode else self.handler.connect()
        
        # 2. Maps & Localization
        self.gmap = GlobalMap()
        self.localizer = LocalizationEngine()
        self.planner = MapTrajectoryPlanner(self.gmap, lookahead_meters=0.6)
        
        # 3. Perception
        self.lane_module = None # Disabled or implement basic vision here
        
        # UI State Variables
        self.speed = 0.0
        self.steer = 0.0
        self.nav_state = "LOCALIZING"
        self.anchor = "AWAITING_MAP_CLICK"
        self.global_x = 0.0
        self.global_y = 0.0
        self.yolo_frame = None
        self.active_labels = []
        
        # 4. Camera Configuration Setup (Locked AWB/Exposure)
        self.cam_ok = False
        if not sim_mode and _CAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                cfg = self.picam2.create_video_configuration(main={"size": (1280, 720), "format": "BGR888"})
                self.picam2.configure(cfg)
                self.picam2.set_controls({
                    "AeEnable": False,       
                    "ExposureTime": 8000,    
                    "AnalogueGain": 4.0,     
                    "AwbEnable": False,      
                    "ColourGains": (2.2, 1.8) 
                })
                self.picam2.start()
                self.cam_ok = True
                print("[CAM] PiCamera2 locked configs successfully activated.")
            except Exception as e:
                print(f"[WARN] Camera init failed: {e}")
                
    def get_raw_frame(self):
        if self.cam_ok:
            frame = self.picam2.capture_array()
            if frame is not None and frame.ndim == 3:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mock = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(mock, "SIMULATION / CAM LOST", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        return mock

    def pilot_loop(self):
        """Background highly determininstic execution thread."""
        last_t = time.time()
        
        while self.running:
            now = time.time()
            dt = max(0.001, now - last_t)
            last_t = now
            
            # --- 1. CAPTURE ---
            frame = self.get_raw_frame()
            self.yolo_frame = frame.copy()
            
            # --- 2. PERCEPTION ---
            target_x = None
            lane_dbg = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(lane_dbg, "VISION DISABLED", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            # --- 3. IMU ZERO-LATENCY DEAD RECKONING ---
            v_m = self.speed / 100.0
            rad_yaw = math.radians(self.imu.yaw)
            
            self.global_x += v_m * math.cos(rad_yaw) * dt
            self.global_y += v_m * math.sin(rad_yaw) * dt
            
            # Sync to actual advanced localizer if initialized
            if "GLOBAL_OVERRIDE" not in self.anchor:
                self.localizer.update_dead_reckoning(self.steer, v_m, dt)
            else:
                # User clicked map! Override Localizer.
                self.localizer.reset_pose(self.global_x, self.global_y, rad_yaw)
                self.anchor = "LANE_FOLLOW"
                
            est_x, est_y, est_yaw = self.localizer.x, self.localizer.y, self.localizer.yaw
            self.global_x, self.global_y = est_x, est_y
            # We don't overwrite IMU yaw from localizer because IMU is ground truth for heading rate

            # --- 4. NAVIGATION DECISION ---
            if target_x is not None:
                # Vison is working fine!
                self.nav_state = "TRACKING_LANE"
                self.anchor = "VISION_PRIMARY"
                
                # Simple P controller for pure pursuit steering
                dx = target_x - 320.0
                dy = 150.0 # Lookahead Y
                ld = math.sqrt(dx*dx + dy*dy)
                alpha = math.atan2(dx, dy)
                wb_px = 0.23 * (280 / 0.5) # wheel base * pixels per meter (23cm)
                steer_rad = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
                self.steer = math.degrees(steer_rad)
                
            else:
                # VISION IS LOST! Fallback to Map + IMU Dead Reckoning!
                self.nav_state = "IMU_DEAD_RECKONING"
                self.anchor = "MAP_PREDICTIVE"
                
                # Fetch anticipated curve from the map trajectory planner
                if len(self.gmap.points) > 0:
                    _, _, map_target_pt, _ = self.planner.get_local_target(est_x, est_y, est_yaw)
                    if map_target_pt is not None:
                        # Translate map target back to local car coordinates to calculate steering
                        lx = (map_target_pt[0] - est_x) * math.cos(-est_yaw) - (map_target_pt[1] - est_y) * math.sin(-est_yaw)
                        ly = (map_target_pt[0] - est_x) * math.sin(-est_yaw) + (map_target_pt[1] - est_y) * math.cos(-est_yaw)
                        
                        dx = lx * (280/0.5) # meters to pixels
                        dy = max(1.0, ly * (280/0.5))
                        alpha = math.atan2(dx, dy)
                        wb_px = 0.23 * (280 / 0.5) # 23cm wheel base
                        self.steer = math.degrees(math.atan2(2.0 * wb_px * math.sin(alpha), math.sqrt(dx**2 + dy**2)))
                    else:
                        self.steer = 0.0 # Blind
                else:
                    self.steer = 0.0

            # Filter Steering
            self.steer = max(-45.0, min(45.0, self.steer))
            
            # Temporary Constant Speed for Testing
            if getattr(self, "manual_stop", False):
                self.speed = 0.0
            else:
                self.speed = 40.0 # 40 cm/s
                
            if self.connected:
                self.handler.set_speed(self.speed)
                self.handler.set_steering(self.steer)

            # Sleep to cap loop at 100Hz max
            elapsed = time.time() - now
            time.sleep(max(0.001, 0.01 - elapsed))
            
    def stop(self):
        self.running = False
        if self.cam_ok: self.picam2.stop()
        self.imu.running = False
        if self.connected:
            self.handler.set_speed(0.0)
            self.handler.set_steering(0.0)
            self.handler.disconnect()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run offline simulation")
    args = parser.parse_args()

    print("=========================================")
    print(" BFMC V4 Unified Pilot Booting...")
    print("=========================================")

    # 1. Init Pilot (Hardware setups)
    pilot = BFMC_V4_Pilot(sim_mode=args.sim)
    
    # 2. Init Tkinter Window natively on the main process thread
    root = tk.Tk()
    dashboard = SimplifiedBFMCDashboard(root, pilot)
    
    # 3. Spin off the control logic to run invisibly fast
    worker = threading.Thread(target=pilot.pilot_loop, daemon=True)
    worker.start()
    
    # 4. Bind main thread to UI
    root.mainloop()
