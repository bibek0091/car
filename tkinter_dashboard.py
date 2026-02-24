import tkinter as tk
from tkinter import Canvas
import cv2
import numpy as np
import math
from PIL import Image, ImageTk
import time

class BFMCDashboardApp:
    def __init__(self, root, gmap=None):
        self.root = root
        self.gmap = gmap
        self.root.title("BFMC Mission Control")
        # 1920x720 window, adjust standard for 1366 screens if needed, sticking to 1920 to retain full cluster look
        self.root.geometry("1920x720")
        self.root.configure(bg="#0c0c0c")
        
        # We process UI updates without stalling by pulling from these internal buffers
        self.yolo_img = None
        self.radar_img = None
        self.speed = 0.0
        self.steer_angle = 0.0
        self.batt_pct = 99.0
        self.traffic_state = "SYS_GO"
        self.traffic_reason = "CLEAR PATH"
        self.topology = "DUAL LANE"
        self.nav_state = "FWD"
        self.anchor = "LANE_FOLLOW"
        self.active_labels = []
        self.global_pose = None # [x, y, yaw]
        
        # Telemetry history for the graph
        self.steer_history = []
        
        # Destination pointer
        self.target_pose = None
        
        self._init_ui()
        self._update_loop() # Start the non-blocking Tkinter refresh loop

    def _init_ui(self):
        # Three Widescreen Frames
        self.left_frame = tk.Frame(self.root, bg="#0c0c0c", width=580, height=700)
        self.left_frame.place(x=20, y=10)
        
        self.center_frame = tk.Frame(self.root, bg="#0c0c0c", width=720, height=700)
        self.center_frame.place(x=610, y=10)
        
        self.right_frame = tk.Frame(self.root, bg="#0c0c0c", width=580, height=700)
        self.right_frame.place(x=1340, y=10)
        
        # Left Panel Canvases
        self.c_speed = Canvas(self.left_frame, width=540, height=300, bg="#121214", highlightthickness=0)
        self.c_speed.place(x=20, y=10)
        
        self.c_steer = Canvas(self.left_frame, width=540, height=260, bg="#121214", highlightthickness=0)
        self.c_steer.place(x=20, y=320)
        
        self.c_status = Canvas(self.left_frame, width=540, height=100, bg="#121214", highlightthickness=0)
        self.c_status.place(x=20, y=590)
        
        # Center Panel
        self.c_map = Canvas(self.center_frame, width=680, height=520, bg="#121214", highlightthickness=0)
        self.c_map.place(x=20, y=10)
        self.c_map.bind("<Button-1>", self._on_map_lclick)
        self.c_map.bind("<Button-3>", self._on_map_rclick)
        
        self.c_telemetry = Canvas(self.center_frame, width=680, height=150, bg="#121214", highlightthickness=0)
        self.c_telemetry.place(x=20, y=540)
        
        # Right Panel
        self.yolo_lbl = tk.Label(self.right_frame, bg="#0c0c0c")
        self.yolo_lbl.place(x=20, y=10)
        
        self.c_icons = Canvas(self.right_frame, width=540, height=200, bg="#121214", highlightthickness=0)
        self.c_icons.place(x=20, y=340)
        
        self.radar_lbl = tk.Label(self.right_frame, bg="#0c0c0c")
        self.radar_lbl.place(x=20, y=550)
        
        # Initial draw of static map geometry
        self.map_scale = 1.0
        self.map_offset = np.array([0, 0])
        self._draw_static_map_cache()

    def update_state(self, yolo_hd, lane_dbg, speed, steer_angle, traffic_state, traffic_reason, light_status, nav_state, anchor, batt_pct, active_labels, topology, global_pose=None):
        """Thread-safe state ingestion called by the main BFMC pilot thread."""
        self.speed = speed
        self.steer_angle = steer_angle
        self.traffic_state = traffic_state
        self.traffic_reason = traffic_reason
        self.nav_state = nav_state
        self.anchor = anchor
        self.batt_pct = batt_pct
        self.active_labels = active_labels
        self.topology = topology
        self.global_pose = global_pose
        
        self.steer_history.append(steer_angle)
        if len(self.steer_history) > 150: self.steer_history.pop(0)
        
        # Image conversion for Tkinter
        if yolo_hd is not None:
            rgb = cv2.cvtColor(cv2.resize(yolo_hd, (540, 320)), cv2.COLOR_BGR2RGB)
            self.yolo_img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            
        if lane_dbg is not None:
            gray = cv2.cvtColor(cv2.resize(lane_dbg, (260, 140)), cv2.COLOR_BGR2GRAY)
            self.radar_img = ImageTk.PhotoImage(image=Image.fromarray(gray))

    def _draw_static_map_cache(self):
        """Pre-computes map scaling and draws the background track."""
        if self.gmap is None or len(self.gmap.points) == 0: return
        pmin = np.min(self.gmap.points, axis=0)
        pmax = np.max(self.gmap.points, axis=0)
        w_m = max(1, pmax[0] - pmin[0])
        h_m = max(1, pmax[1] - pmin[1])
        
        scale_x = (680 - 40) / w_m
        scale_y = (520 - 40) / h_m
        self.map_scale = min(scale_x, scale_y)
        self.map_offset = np.array([20 - pmin[0]*self.map_scale, 20 - pmin[1]*self.map_scale])
        
        self.scaled_pts = (self.gmap.points * self.map_scale + self.map_offset).astype(int)

    def _on_map_lclick(self, event):
        """Teleport global pose."""
        if self.gmap is None: return
        real_x = (event.x - self.map_offset[0]) / self.map_scale
        real_y = (event.y - self.map_offset[1]) / self.map_scale
        idx, _ = self.gmap.get_nearest_spline_index(real_x, real_y)
        heading = self.gmap.headings[idx]
        # We signal back to the pilot logic by mutating global_pose (if they share the ref)
        # OR we just fire a callback. For now, we update local state. The pilot reads `self.global_pose` occasionally,
        # but to be thread-safe we should just update it.
        self.global_pose = [real_x, real_y, heading, idx]
        # In a real system, you'd trigger a callback to update `localizer` 
        print(f"[UI] Map Left-Click Pose Set: X={real_x:.1f}, Y={real_y:.1f}")

    def _on_map_rclick(self, event):
        """Set Destination."""
        real_x = (event.x - self.map_offset[0]) / self.map_scale
        real_y = (event.y - self.map_offset[1]) / self.map_scale
        self.target_pose = (real_x, real_y)
        print(f"[UI] Map Right-Click Target Set: X={real_x:.1f}, Y={real_y:.1f}")

    def _draw_glowing_arc(self, canvas, cx, cy, r, start, extent, color_hex, width=6, glow_layers=3):
        """Draws Tkinter arcs with anti-aliasing via translucent layering."""
        # Tkinter lacks native opacity, so we draw thicker, darker lines underneath
        # Actually in vanilla Tkinter, creating translucent colors is hard natively without PIL composite.
        # We will draw a dark shadow, then a solid bright inner core.
        canvas.create_arc(cx-r-width*2, cy-r-width*2, cx+r+width*2, cy+r+width*2, start=start, extent=extent, style=tk.ARC, outline="#2c2c30", width=width*2.5)
        canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=start, extent=extent, style=tk.ARC, outline=color_hex, width=width)

    def _draw_traffic_label(self, c, label, cx, cy, r=22):
        lbl = label.lower()
        # Draw background shadow
        c.create_oval(cx-r-2, cy-r-2, cx+r+2, cy+r+2, fill="#1c1c1e", outline="")
        
        if "stop" in lbl:
            c.create_polygon([cx-r//2, cy-r, cx+r//2, cy-r, cx+r, cy-r//2, cx+r, cy+r//2, cx+r//2, cy+r, cx-r//2, cy+r, cx-r, cy+r//2, cx-r, cy-r//2], fill="#dd2222", outline="")
            c.create_text(cx, cy, text="STOP", fill="#fff", font=("Arial", 9, "bold"))
        elif "parking" in lbl:
            c.create_rectangle(cx-r, cy-r, cx+r, cy+r, fill="#2266dd", outline="")
            c.create_text(cx, cy, text="P", fill="#fff", font=("Arial", 16, "bold"))
        elif "crosswalk" in lbl:
            c.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#22ddcc", outline="")
            c.create_line(cx-10, cy-4, cx+10, cy-4, fill="#111", width=3)
            c.create_line(cx-10, cy+4, cx+10, cy+4, fill="#111", width=3)
        elif "priority" in lbl:
            c.create_polygon([cx, cy-r, cx+r, cy, cx, cy+r, cx-r, cy], fill="#ddcc22", outline="")
            c.create_polygon([cx, cy-r+6, cx+r-6, cy, cx, cy+r-6, cx-r+6, cy], fill="#fff", outline="")
        elif "entry" in lbl and "highway" in lbl:
            c.create_rectangle(cx-r, cy-r, cx+r, cy+r, fill="#22dd44", outline="")
            c.create_text(cx, cy, text="HWY", fill="#fff", font=("Arial", 9, "bold"))
        elif "exit" in lbl and "highway" in lbl:
            c.create_rectangle(cx-r, cy-r, cx+r, cy+r, fill="#22dd44", outline="")
            c.create_line(cx-r, cy-r, cx+r, cy+r, fill="#dd2222", width=4)
        elif "one-way" in lbl:
            c.create_rectangle(cx-r, cy-r, cx+r, cy+r, fill="#2266dd", outline="")
            c.create_line(cx-12, cy, cx+12, cy, fill="#fff", width=3, arrow=tk.LAST)
        elif "round" in lbl:
            c.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#2266dd", outline="")
            c.create_arc(cx-r+6, cy-r+6, cx+r-6, cy+r-6, start=0, extent=270, style=tk.ARC, outline="#fff", width=3)
            c.create_text(cx, cy, text="^", fill="#fff", font=("Arial", 10))
        elif "no-entry" in lbl:
            c.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#dd2222", outline="")
            c.create_line(cx-12, cy, cx+12, cy, fill="#fff", width=5)
        else:
            c.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#555555", outline="")
            
    def _update_loop(self):
        # 1. Update Left Panel (Kinematics)
        self.c_speed.delete("all")
        self.c_speed.create_text(20, 20, text="VELOCITY KINEMATICS", fill="#888", font=("Arial", 10), anchor=tk.NW)
        cx, cy, r = 270, 180, 100
        # Background arc
        self._draw_glowing_arc(self.c_speed, cx, cy, r, start=-45, extent=270, color_hex="#222222")
        # Active arc
        ratio = min(1.0, abs(self.speed) / 120.0)
        self._draw_glowing_arc(self.c_speed, cx, cy, r, start=-45, extent=int(270 * ratio), color_hex="#11ccff")
        self.c_speed.create_text(cx, cy, text=f"{int(abs(self.speed))}", fill="#fff", font=("Arial", 48, "bold"))
        self.c_speed.create_text(cx, cy+40, text="cm/s", fill="#888", font=("Arial", 14))

        self.c_steer.delete("all")
        self.c_steer.create_text(20, 20, text="STEERING GEOMETRY", fill="#888", font=("Arial", 10), anchor=tk.NW)
        cx, cy, r = 270, 160, 80
        self._draw_glowing_arc(self.c_steer, cx, cy, r, start=45, extent=90, color_hex="#222222")
        # Adjust extent for steering (-45 to +45 mapped to 45 to 135 tk angles)
        # Tkinter arc start=0 is 3 o'clock, growing CCW.
        # We want top center to be 90. Steering right (+deg) means pulling start towards 90 and extent negative.
        # Let's just draw an indicator line for simplicity if exact arc mapping is complex
        self.c_steer.create_arc(cx-r, cy-r, cx+r, cy+r, start=45, extent=90, style=tk.ARC, outline="#333", width=8)
        c_ang = 90 - self.steer_angle
        sa_rad = math.radians(c_ang)
        ex = cx + r * math.cos(sa_rad)
        ey = cy - r * math.sin(sa_rad) # Y is down in tk
        self.c_steer.create_line(cx, cy, ex, ey, fill="#ff8822", width=6)
        self.c_steer.create_text(cx, cy+30, text=f"{self.steer_angle:+.1f}°", fill="#fff", font=("Arial", 24, "bold"))

        self.c_status.delete("all")
        tcol = "#ff3333" if self.traffic_state == "SYS_STOP" else "#33ff33" if self.traffic_state == "SYS_GO" else "#ffcc22"
        self.c_status.create_text(20, 20, text=self.traffic_state, fill=tcol, font=("Arial", 24, "bold"), anchor=tk.NW)
        self.c_status.create_text(20, 55, text=self.traffic_reason.upper(), fill="#fff", font=("Arial", 12), anchor=tk.NW)
        # Battery bar
        self.c_status.create_rectangle(300, 25, 500, 45, fill="#333", outline="")
        b_w = int(200 * (self.batt_pct / 100.0))
        self.c_status.create_rectangle(300, 25, 300+b_w, 45, fill="#33ff33", outline="")
        self.c_status.create_text(400, 35, text=f"{self.batt_pct:.1f}%", fill="#000", font=("Arial", 10, "bold"))

        # 2. Update Center Panel (Map & Telemetry)
        self.c_map.delete("all")
        self.c_map.create_text(20, 20, text="GLOBAL LOCALIZATION & HD MAP", fill="#888", font=("Arial", 10), anchor=tk.NW)
        
        # Draw base map tracking scatter
        if hasattr(self, 'scaled_pts'):
            for p in self.scaled_pts[::3]: # Subsample for Tkinter draw speed
                self.c_map.create_rectangle(p[0]-1, p[1]-1, p[0]+1, p[1]+1, fill="#444", outline="")
                
        # Draw vehicle pose
        if self.global_pose is not None:
            px = int(self.global_pose[0] * self.map_scale + self.map_offset[0])
            py = int(self.global_pose[1] * self.map_scale + self.map_offset[1])
            self.c_map.create_oval(px-6, py-6, px+6, py+6, fill="#11ccff", outline="")
            
            hx = px + math.cos(self.global_pose[2]) * 20
            hy = py + math.sin(self.global_pose[2]) * 20
            self.c_map.create_line(px, py, hx, hy, fill="#fff", width=3, arrow=tk.LAST)
            
        if self.target_pose is not None:
            tx = int(self.target_pose[0] * self.map_scale + self.map_offset[0])
            ty = int(self.target_pose[1] * self.map_scale + self.map_offset[1])
            self.c_map.create_line(tx-10, ty-10, tx+10, ty+10, fill="#ff3333", width=2)
            self.c_map.create_line(tx+10, ty-10, tx-10, ty+10, fill="#ff3333", width=2)
            
            if self.global_pose is not None:
                self.c_map.create_line(px, py, tx, ty, fill="#ff8822", dash=(4, 4), width=2)

        # Telemetry Graph
        self.c_telemetry.delete("all")
        self.c_telemetry.create_text(20, 20, text="STEERING ERROR KINEMATICS", fill="#888", font=("Arial", 10), anchor=tk.NW)
        self.c_telemetry.create_line(20, 75, 660, 75, fill="#444") # Zero line
        
        if len(self.steer_history) > 2:
            pts = []
            step = 640.0 / float(max(1, len(self.steer_history) - 1))
            for i, val in enumerate(self.steer_history):
                # mapped from -45..45 to 20..130 (75 is center)
                py = 75 - (val / 45.0) * 55
                pts.append(20 + i*step)
                pts.append(py)
            self.c_telemetry.create_line(*pts, fill="#11ccff", width=2, smooth=True)

        # 3. Update Right Panel (Perception)
        if self.yolo_img is not None:
            self.yolo_lbl.config(image=self.yolo_img)
        if self.radar_img is not None:
            self.radar_lbl.config(image=self.radar_img)
            
        self.c_icons.delete("all")
        self.c_icons.create_text(20, 20, text="ACTIVE MACHINE VISION SIGNALS", fill="#888", font=("Arial", 10), anchor=tk.NW)
        
        ix, iy = 60, 80
        for idx, lbl in enumerate(self.active_labels):
            if idx > 7: break
            row, col = idx // 4, idx % 4
            self._draw_traffic_label(self.c_icons, lbl, ix + col*80, iy + row*80, r=26)

        self.root.after(30, self._update_loop)

