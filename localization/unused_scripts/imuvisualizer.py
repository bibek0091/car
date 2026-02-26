#!/usr/bin/env python3
"""
==================================================================
           BNO055 IMU  -  3D Dead-Reckoning Visualiser            
                                                                  
  Standalone script.  No project files needed.                    
  Works on Raspberry Pi with BNO055 wired to I2C, or on any      
  machine in --sim mode (kinematic simulation).                   
                                                                  
  Usage:                                                          
    python imu_visualizer.py              # real IMU              
    python imu_visualizer.py --trail 600  # longer trail          
    python imu_visualizer.py --rate 20    # 20 Hz update rate     
                                                                  
  Dependencies (Pi):                                              
    pip install adafruit-blinka adafruit-circuitpython-bno055     
    pip install matplotlib numpy                                  
==================================================================
"""

import sys
import time
import math
import argparse
import threading
import collections
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch

# ==================================================================
# Colour palette  (dark cockpit theme)
# ==================================================================
BG      = "#0A0A12"
PANEL   = "#0F0F1A"
CYAN    = "#00E5FF"
AMBER   = "#FFB300"
GREEN   = "#00E676"
RED     = "#FF1744"
MUTED   = "#555570"
WHITE   = "#E8E8F0"
YELLOW  = "#FFD600"
BLUE_LT = "#2878C8"

# ==================================================================
# IMU Driver  -  reading from STM32_SerialHandler
# ==================================================================
class IMUReader:
    """
    Reads IMU data parsed by STM32_SerialHandler.
    """
    def __init__(self):
        self.serial = None
        self.calib = (3, 3, 3, 3)  # NDOF handled on STM32 side
        self._connect_serial()

    def _connect_serial(self):
        try:
            from STM32_SerialHandler import STM32_SerialHandler, SerialConfig
            self.serial = STM32_SerialHandler(SerialConfig(baudrate=115200))
            if self.serial.connect():
                print("[IMUReader] Connected to STM32")
            else:
                print("[IMUReader] Failed to connect to STM32")
        except ImportError as e:
            print(f"[IMUReader] Could not import STM32_SerialHandler: {e}")

    # ── Read (thread-safe) ───────────────────────────────────────
    def read(self):
        """
        Returns (yaw_deg, pitch_deg, roll_deg, calib_tuple, lin_vel_xyz).
        calib_tuple = (sys, gyro, accel, mag)  each 0-3.
        Returns None if IMU is not connected.
        """
        if self.serial is None or not self.serial.running:
            return None

        # Give the serial handler a moment to receive the very first IMU frame
        data = self.serial.status.imu_data
        if data is None:
            return None

        try:
            yaw = data.get('yaw', 0.0)
            pitch = data.get('pitch', 0.0)
            roll = data.get('roll', 0.0)
            vx = data.get('vx', 0.0)
            vy = data.get('vy', 0.0)
            vz = data.get('vz', 0.0)
            return yaw, pitch, roll, self.calib, (vx, vy, vz)
        except Exception as e:
            print(f"[IMU] Read error from serial status: {e}")
            return None


# ==================================================================
# Dead-Reckoning integrator
# ==================================================================
class DeadReckoning:
    """
    Integrates linear velocity to produce position.
    """
    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self._prev_t = time.time()

    def update(self, vx, vy, vz, yaw_deg):
        now = time.time()
        dt  = min(now - self._prev_t, 0.1)
        self._prev_t = now
        if dt <= 0:
            return self.x, self.y, self.z

        self.vx = vx
        self.vy = vy
        self.vz = vz

        # Rotate body-frame velocity into world frame using yaw (2-D approx)
        r  = math.radians(yaw_deg)
        c, s = math.cos(r), math.sin(r)
        
        # Convert local velocity to world XYZ displacements
        wx = c * vx - s * vy
        wy = s * vx + c * vy

        self.x += wx * dt
        self.y += wy * dt
        self.z += max(0.0, self.z + vz * dt) - self.z   # clamp to z≥0

        return self.x, self.y, self.z

    def reset(self):
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0


# ==================================================================
# Build car geometry  (3D Race Car in local frame)
# ==================================================================
def make_car_verts():
    """
    Returns (vertices, face_indices, face_colors) for a 3D car model.
    The car points in the +X direction (forward).
    """
    # 1. Chassis (Main body)
    L_c, W_c, H_c = 0.40, 0.20, 0.08  # Length, Width, Height
    x1, x2 = -L_c * 0.4, L_c * 0.6    # Chassis is shifted slightly forward
    y1, y2 = -W_c / 2, W_c / 2
    z1, z2 = 0.02, 0.02 + H_c

    chassis_verts = [
        [x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],  # Bottom
        [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]   # Top
    ]
    
    CHASSIS_COLOR = "#D32F2F" # Red

    chassis_faces = [
        ([0, 1, 2, 3], "#111111"), # Bottom (dark)
        ([4, 5, 6, 7], CHASSIS_COLOR), # Top
        ([0, 1, 5, 4], CHASSIS_COLOR), # Right
        ([2, 3, 7, 6], CHASSIS_COLOR), # Left
        ([1, 2, 6, 5], "#FFCDD2"), # Front
        ([3, 0, 4, 7], CHASSIS_COLOR)  # Back
    ]

    # 2. Cabin (Cockpit)
    L_cab, W_cab, H_cab = 0.15, 0.14, 0.06
    xc1, xc2 = -L_c * 0.1, -L_c * 0.1 + L_cab
    yc1, yc2 = -W_cab / 2, W_cab / 2
    zc1, zc2 = z2, z2 + H_cab

    cabin_verts = [
        [xc1, yc1, zc1], [xc2, yc1, zc1], [xc2, yc2, zc1], [xc1, yc2, zc1],
        [xc1+0.02, yc1+0.01, zc2], [xc2-0.03, yc1+0.01, zc2], 
        [xc2-0.03, yc2-0.01, zc2], [xc1+0.02, yc2-0.01, zc2] # Tapered top
    ]
    
    CABIN_COLOR = "#1E88E5" # Blue/Glass
    
    cabin_faces = [
        ([4, 5, 6, 7], "#424242"), # Roof
        ([0, 1, 5, 4], CABIN_COLOR), # Right window
        ([2, 3, 7, 6], CABIN_COLOR), # Left window
        ([1, 2, 6, 5], "#90CAF9"), # Windshield (lighter)
        ([3, 0, 4, 7], CABIN_COLOR)  # Rear window
    ]

    # 3. Wheels
    WHEEL_R = 0.05
    WHEEL_W = 0.04
    wheel_base = 0.25
    track_width = 0.24
    
    wx_f, wx_r = x2 - 0.08, x1 + 0.08
    wy_l, wy_r = track_width/2, -track_width/2
    wz = WHEEL_R
    
    wheel_centers = [
        (wx_f, wy_l, wz), (wx_f, wy_r, wz), # Front-Left, Front-Right
        (wx_r, wy_l, wz), (wx_r, wy_r, wz)  # Rear-Left, Rear-Right
    ]
    
    WHEEL_COLOR = "#212121"
    
    wheel_verts_list = []
    wheel_faces = []
    v_idx = len(chassis_verts) + len(cabin_verts)
    
    # Very simple wheels (just vertical rectangles for low polygon count in matplotlib)
    for cx, cy, cz in wheel_centers:
        w_verts = [
            [cx-WHEEL_R, cy-WHEEL_W/2, cz-WHEEL_R], [cx+WHEEL_R, cy-WHEEL_W/2, cz-WHEEL_R],
            [cx+WHEEL_R, cy+WHEEL_W/2, cz-WHEEL_R], [cx-WHEEL_R, cy+WHEEL_W/2, cz-WHEEL_R],
            [cx-WHEEL_R, cy-WHEEL_W/2, cz+WHEEL_R], [cx+WHEEL_R, cy-WHEEL_W/2, cz+WHEEL_R],
            [cx+WHEEL_R, cy+WHEEL_W/2, cz+WHEEL_R], [cx-WHEEL_R, cy+WHEEL_W/2, cz+WHEEL_R]
        ]
        wheel_verts_list.extend(w_verts)
        
        # Add faces for this wheel (sides, front, back, top, bottom)
        wheel_faces.extend([
            ([v_idx+0, v_idx+1, v_idx+2, v_idx+3], "#111111"), # Bottom
            ([v_idx+4, v_idx+5, v_idx+6, v_idx+7], WHEEL_COLOR), # Top
            ([v_idx+0, v_idx+1, v_idx+5, v_idx+4], WHEEL_COLOR), # Right
            ([v_idx+2, v_idx+3, v_idx+7, v_idx+6], WHEEL_COLOR), # Left
            ([v_idx+1, v_idx+2, v_idx+6, v_idx+5], "#424242"), # Front
            ([v_idx+3, v_idx+0, v_idx+4, v_idx+7], WHEEL_COLOR)  # Back
        ])
        v_idx += 8

    # Combine everything
    all_verts = np.array(chassis_verts + cabin_verts + wheel_verts_list)
    
    # Adjust cabin indices
    chassis_len = len(chassis_verts)
    adjusted_cabin_faces = [([idx + chassis_len for idx in face], color) for face, color in cabin_faces]

    all_faces_and_colors = chassis_faces + adjusted_cabin_faces + wheel_faces
    
    face_indices = [item[0] for item in all_faces_and_colors]
    face_colors = [item[1] for item in all_faces_and_colors]

    return all_verts, face_indices, face_colors



def rotate_verts(verts, yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    """Full 3-axis rotation: yaw (Z) → pitch (X) → roll (Y)."""
    def Rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    def Rx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def Ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])

    R = Rz(math.radians(yaw_deg)) @ Rx(math.radians(pitch_deg)) @ Ry(math.radians(roll_deg))
    return verts @ R.T


# ==================================================================
# Main Visualiser
# ==================================================================
class IMUVisualiser:

    GRID_R = 1.8    # metres, half-width of visible floor grid

    def __init__(self, imu: IMUReader, dr: DeadReckoning,
                 trail_len=400, update_hz=30):
        self.imu        = imu
        self.dr         = dr
        self.trail_len  = trail_len
        self.interval   = int(1000 / update_hz)

        # History buffers
        self.trail_x = collections.deque(maxlen=trail_len)
        self.trail_y = collections.deque(maxlen=trail_len)
        self.trail_z = collections.deque(maxlen=trail_len)
        self.yaw_hist   = collections.deque(maxlen=180)
        self.pitch_hist = collections.deque(maxlen=180)
        self.roll_hist  = collections.deque(maxlen=180)
        self.spd_hist   = collections.deque(maxlen=180)
        self.t_hist     = collections.deque(maxlen=180)

        self._t0        = time.time()
        self._frame     = 0

        self._build_ui()

    # ── Build figure ─────────────────────────────────────────────
    def _build_ui(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(15, 8), facecolor=BG)
        self.fig.canvas.manager.set_window_title("BNO055 IMU — 3D Dead-Reckoning Visualiser")

        gs = gridspec.GridSpec(
            3, 3,
            figure=self.fig,
            left=0.04, right=0.98,
            top=0.93,  bottom=0.07,
            wspace=0.32, hspace=0.55
        )

        # ── 3D scene (spans left 2 columns, all 3 rows) ───────────
        self.ax3d = self.fig.add_subplot(gs[:, :2], projection="3d")
        self._setup_3d()

        # ── Right column: 3 strip charts ─────────────────────────
        self.ax_yaw   = self.fig.add_subplot(gs[0, 2])
        self.ax_pitch = self.fig.add_subplot(gs[1, 2])
        self.ax_spd   = self.fig.add_subplot(gs[2, 2])
        self._setup_strips()

        # ── Title bar ────────────────────────────────────────────
        self.fig.text(0.50, 0.97, "BNO055  IMU  3D  DEAD-RECKONING  VISUALISER",
                      ha="center", va="top",
                      fontfamily="monospace", fontsize=11,
                      color=CYAN, fontweight="bold")
        self.fig.text(0.50, 0.93, "SOURCE: SERIAL STM32",
                      ha="center", va="top",
                      fontfamily="monospace", fontsize=7, color=MUTED)

        # Reset button
        self.ax_btn = self.fig.add_axes([0.86, 0.01, 0.07, 0.035])
        from matplotlib.widgets import Button
        self._btn = Button(self.ax_btn, "RESET", color="#1A1A28", hovercolor="#2A2A40")
        self._btn.label.set_color(AMBER)
        self._btn.label.set_fontfamily("monospace")
        self._btn.label.set_fontsize(8)
        self._btn.on_clicked(self._on_reset)

    # ── 3D axes setup ─────────────────────────────────────────────
    def _setup_3d(self):
        ax = self.ax3d
        ax.set_facecolor("#06060F")
        self.fig.patch.set_facecolor(BG)

        # Pane colours
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#141428")

        # ── Static floor grid ─────────────────────────────────────
        gr = self.GRID_R
        ticks = np.linspace(-gr, gr, 9)
        for v in ticks:
            ax.plot([v, v],   [-gr, gr], [0, 0], color="#14142A", lw=0.6, zorder=1)
            ax.plot([-gr, gr], [v, v],  [0, 0], color="#14142A", lw=0.6, zorder=1)

        # Floor fill
        fl = Poly3DCollection(
            [[(-gr,-gr,0),(gr,-gr,0),(gr,gr,0),(-gr,gr,0)]],
            facecolors=["#07070E"], edgecolors="none", zorder=0, alpha=1.0)
        ax.add_collection3d(fl)

        # Origin cross
        ax.plot([-0.12, 0.12], [0, 0], [0, 0], color=MUTED, lw=1.0, zorder=2)
        ax.plot([0, 0], [-0.12, 0.12], [0, 0], color=MUTED, lw=1.0, zorder=2)

        # ── Car body (Poly3DCollection) ───────────────────────────
        self._car_verts, self._car_face_idx, self._car_face_cols = make_car_verts()
        dummy = [np.zeros((4, 3))] * len(self._car_face_idx)
        self._car_col = Poly3DCollection(
            dummy,
            facecolors=self._car_face_cols,
            edgecolors="#3AAFFF",
            linewidths=0.7,
            zorder=5, zsort="average"
        )
        ax.add_collection3d(self._car_col)

        # The dashboard elements for the windshield and arrow will just be empty
        self._ws_col = None

        # ── Trail line ────────────────────────────────────────────
        self._trail_line, = ax.plot([], [], [], color=AMBER,
                                    lw=2.0, alpha=0.9, zorder=4)

        # ── Heading arrow (nose → forward) ────────────────────────
        self._arrow_line, = ax.plot([], [], [], color=CYAN,
                                    lw=2.5, solid_capstyle="round", zorder=7)

        # ── Shadow dot on floor ───────────────────────────────────
        self._shadow = ax.scatter([0], [0], [0], s=80, c=[MUTED],
                                   edgecolors="none", zorder=3, alpha=0.5)

        # ── HUD text overlays (axes transform = 0..1 = stable position) ──
        _tf = dict(transform=ax.transAxes, fontfamily="monospace",
                   fontsize=8.5, zorder=10)
        self._hud_yaw   = ax.text2D(0.02, 0.97, "YAW      0.0°",   color=CYAN,   **_tf)
        self._hud_pitch = ax.text2D(0.02, 0.91, "PITCH    0.0°",   color=GREEN,  **_tf)
        self._hud_roll  = ax.text2D(0.02, 0.85, "ROLL     0.0°",   color=AMBER,  **_tf)
        self._hud_x     = ax.text2D(0.02, 0.76, "X   +0.000 m",    color=WHITE,  **_tf)
        self._hud_y     = ax.text2D(0.02, 0.70, "Y   +0.000 m",    color=WHITE,  **_tf)
        self._hud_z     = ax.text2D(0.02, 0.64, "Z   +0.000 m",    color=WHITE,  **_tf)
        self._hud_spd   = ax.text2D(0.02, 0.55, "V   0.000 m/s",   color=AMBER,  **_tf)
        # Calib LEDs label
        self._hud_calib = ax.text2D(0.02, 0.46, "SYS○ GYR○ ACC○ MAG○",
                                     color=MUTED, **_tf)
        # Right side: frame counter / FPS
        self._hud_fps   = ax.text2D(0.98, 0.97, "FPS  —", color=MUTED,
                                     ha="right", **_tf)
        self._hud_mode  = ax.text2D(0.98, 0.91,
                                     "HW",
                                     color=GREEN,
                                     ha="right", **_tf)

        # ── Axis cosmetics ────────────────────────────────────────
        ax.tick_params(colors=MUTED, labelsize=6, pad=1)
        ax.set_xlabel("X (m)", color=MUTED, fontsize=7, labelpad=2)
        ax.set_ylabel("Y (m)", color=MUTED, fontsize=7, labelpad=2)
        ax.set_zlabel("Z (m)", color=MUTED, fontsize=7, labelpad=2)
        ax.set_zlim(0.0, 0.25)
        ax.view_init(elev=26, azim=-50)
        ax.set_title("DEAD-RECKONING  TRAJECTORY",
                     color=MUTED, fontsize=8, fontfamily="monospace", pad=4)

        self._last_frame_t = time.time()

    # ── Strip chart setup ─────────────────────────────────────────
    def _setup_strips(self):
        strips = [
            (self.ax_yaw,   "YAW (°)",    CYAN,  -180, 180),
            (self.ax_pitch, "PITCH (°)",  GREEN,  -30,  30),
            (self.ax_spd,   "SPEED (m/s)",AMBER,    0,  1.0),
        ]
        self._strip_lines = []
        for ax, title, col, ylo, yhi in strips:
            ax.set_facecolor("#09090F")
            ax.set_xlim(0, 180)
            ax.set_ylim(ylo, yhi)
            ax.axhline(0, color="#202030", lw=0.8)
            ax.tick_params(colors=MUTED, labelsize=6)
            for sp in ax.spines.values():
                sp.set_color("#1A1A2A")
            ax.set_title(title, color=col, fontsize=7,
                         fontfamily="monospace", pad=2)
            line, = ax.plot([], [], color=col, lw=1.2)
            self._strip_lines.append(line)

        # Roll sharing x-axis (no need for separate plot — share with pitch)
        self._roll_line, = self.ax_pitch.plot([], [], color=AMBER,
                                               lw=1.0, alpha=0.6, ls="--")
        self.ax_pitch.set_title("PITCH (°) cyan  /  ROLL (°) amber dashed",
                                 color=MUTED, fontsize=6,
                                 fontfamily="monospace", pad=2)

    # ── Main animation update ─────────────────────────────────────
    def update(self, _frame):
        t_now = time.time()

        # ── Read sensor ──────────────────────────────────────────
        data = self.imu.read()

        if data is None:
            # Handle NO DATA dynamically 
            self._hud_yaw.set_text("NO IMU DETECTED")
            self._hud_yaw.set_color(RED)
            self._hud_pitch.set_text("")
            self._hud_roll.set_text("")
            self._hud_x.set_text("Check Mbed Pinout:")
            self._hud_y.set_text("SDA -> D14, SCL -> D15")
            self._hud_z.set_text("ADR -> 3.3V (Must be 0x29)")
            self._hud_spd.set_text("")
            self._hud_calib.set_text("")
            self._hud_mode.set_text("OFFLINE")
            self._hud_mode.set_color(RED)
            return []
            
        yaw, pitch, roll, calib, (vx, vy, vz) = data

        self._hud_yaw.set_color(CYAN)
        self._hud_mode.set_text("ONLINE")
        self._hud_mode.set_color(GREEN)

        # ── Dead reckoning ───────────────────────────────────────
        x, y, z = self.dr.update(vx, vy, vz, yaw)

        # ── Speed estimate (magnitude of velocity vector) ─────────
        spd = math.hypot(self.dr.vx, math.hypot(self.dr.vy, self.dr.vz))

        # ── History ──────────────────────────────────────────────
        elapsed = t_now - self._t0
        self.trail_x.append(x);  self.trail_y.append(y);  self.trail_z.append(z)
        self.yaw_hist.append(yaw)
        self.pitch_hist.append(pitch)
        self.roll_hist.append(roll)
        self.spd_hist.append(spd)
        self.t_hist.append(elapsed)
        self._frame += 1

        # ── Update 3D scene ──────────────────────────────────────
        self._update_3d(x, y, z, yaw, pitch, roll, spd, calib)

        # ── Update strip charts ───────────────────────────────────
        self._update_strips()

        # ── FPS counter (every 20 frames) ─────────────────────────
        if self._frame % 20 == 0:
            dt_fps = t_now - self._last_frame_t
            fps = 20.0 / max(dt_fps, 0.001)
            self._hud_fps.set_text(f"FPS {fps:.0f}")
            self._last_frame_t = t_now

        return []   # blitting not used but return list expected

    # ── 3D scene update ──────────────────────────────────────────
    def _update_3d(self, x, y, z, yaw, pitch, roll, spd, calib):
        ax = self.ax3d
        gr = self.GRID_R

        # ── Trail (car-relative: trail scrolls, car stays centred) ──
        tx = np.array(self.trail_x) - x
        ty = np.array(self.trail_y) - y
        tz = np.array(self.trail_z)
        self._trail_line.set_data_3d(tx, ty, tz)

        # Trail colour brightens as it gets longer
        alpha = min(0.9, 0.2 + len(tx) / self.trail_len * 0.7)
        self._trail_line.set_alpha(alpha)

        # ── Car body (rotated with full yaw/pitch/roll) ───────────
        rv = rotate_verts(self._car_verts, yaw, pitch, roll)
        rv[:,0] += x
        rv[:,1] += y
        rv[:,2] += z

        c_polys = []
        c_z_cents = []
        for face_idx in self._car_face_idx:
            poly = rv[face_idx]
            c_polys.append(poly)
            # Simple depth heuristic for top-down-ish camera
            c_z_cents.append(np.mean(poly[:, 2]))

        # Sort faces by Z to draw bottom ones first (Painter's algo)
        sort_idx = np.argsort(c_z_cents)

        sorted_polys = [c_polys[i] for i in sort_idx]
        sorted_colors = [self._car_face_cols[i] for i in sort_idx]
        
        self._car_col.set_verts(sorted_polys)
        self._car_col.set_facecolors(sorted_colors)



        # ── Heading arrow (nose direction, length = speed × 0.6) ──
        nose_local = np.array([[0, CF, CH * 0.5]])
        arr_len    = max(0.06, min(spd * 0.6, 0.45))
        tip_local  = np.array([[0, CF + arr_len, CH * 0.5]])
        nose_w = rotate_verts(nose_local, yaw, pitch, roll)[0]
        tip_w  = rotate_verts(tip_local,  yaw, pitch, roll)[0]
        self._arrow_line.set_data_3d([nose_w[0], tip_w[0]],
                                     [nose_w[1], tip_w[1]],
                                     [nose_w[2], tip_w[2]])

        # ── Shadow ────────────────────────────────────────────────
        self._shadow._offsets3d = ([0], [0], [0.001])

        # ── Axis limits (car-centred window) ──────────────────────
        ax.set_xlim(-gr, gr)
        ax.set_ylim(-gr, gr)

        # ── HUD overlays ──────────────────────────────────────────
        card = ["E","NE","N","NW","W","SW","S","SE"][
            int(((90 - yaw) % 360 + 22.5) / 45) % 8]
        self._hud_yaw.set_text(   f"YAW    {yaw % 360:>7.2f}°  {card}")
        self._hud_pitch.set_text( f"PITCH  {pitch:>+7.2f}°")
        self._hud_roll.set_text(  f"ROLL   {roll:>+7.2f}°")
        self._hud_x.set_text(     f"X      {x:>+8.4f} m")
        self._hud_y.set_text(     f"Y      {y:>+8.4f} m")
        self._hud_z.set_text(     f"Z      {z:>+8.4f} m")
        self._hud_spd.set_text(   f"V      {spd:>8.4f} m/s")

        # Calibration LED row
        _LED_ON  = {"0": "○", "1": "◔", "2": "◑", "3": "●"}
        _LED_COL = ["#5A1010", "#AA5500", "#AAAA00", "#00CC44"]
        sys_c, gyr_c, acc_c, mag_c = [max(0, min(3, int(v))) for v in calib[:4]]
        calib_str = (f"SYS{_LED_ON[str(sys_c)]} "
                     f"GYR{_LED_ON[str(gyr_c)]} "
                     f"ACC{_LED_ON[str(acc_c)]} "
                     f"MAG{_LED_ON[str(mag_c)]}")
        # Colour by lowest calib value
        worst = min(sys_c, gyr_c, acc_c, mag_c)
        self._hud_calib.set_text(calib_str)
        self._hud_calib.set_color(_LED_COL[worst])

    # ── Strip chart update ────────────────────────────────────────
    def _update_strips(self):
        if len(self.t_hist) < 2:
            return
        t  = np.array(self.t_hist)
        # Re-index x-axis relative to oldest visible point
        t_rel = t - t[0]

        def _upd(ax, line, data, ylo, yhi):
            line.set_data(np.linspace(0, 180, len(data)), list(data))
            # Auto-scale Y with padding
            d = np.array(data)
            lo, hi = d.min(), d.max()
            pad = max((hi - lo) * 0.15, 0.5)
            ax.set_ylim(max(ylo, lo - pad), min(yhi, hi + pad))

        _upd(self.ax_yaw,   self._strip_lines[0], self.yaw_hist,  -180, 360)
        _upd(self.ax_pitch, self._strip_lines[1], self.pitch_hist, -45,  45)
        _upd(self.ax_spd,   self._strip_lines[2], self.spd_hist,    0,   1.0)

        # Roll on same axes as pitch
        self._roll_line.set_data(np.linspace(0, 180, len(self.roll_hist)),
                                 list(self.roll_hist))

    # ── Reset button ─────────────────────────────────────────────
    def _on_reset(self, event):
        self.dr.reset()
        self.trail_x.clear();  self.trail_y.clear();  self.trail_z.clear()
        print("[RESET] Dead-reckoning position zeroed.")

    # ── Run ───────────────────────────────────────────────────────
    def run(self):
        from matplotlib.animation import FuncAnimation
        self._anim = FuncAnimation(
            self.fig,
            self.update,
            interval=self.interval,
            blit=False,
            cache_frame_data=False
        )
        print(f"\n[VIS] Running  (close window to exit)\n"
              f"      Mode    : SERIAL STM32 IMU STREAM\n"
              f"      Trail   : {self.trail_len} samples\n"
              f"      Rate    : {int(1000/self.interval)} Hz\n"
              f"      Hotkeys : close window to quit,  RESET button to zero position\n")
        plt.show()


# ==================================================================
# Entry point
# ==================================================================
def main():
    parser = argparse.ArgumentParser(
        description="BNO055 IMU 3D Dead-Reckoning Visualiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python imu_visualizer.py                 # Uses STM32_SerialHandler to grab data
  python imu_visualizer.py --trail 800     # longer yellow trail
  python imu_visualizer.py --rate 15       # 15 Hz (lighter CPU on old Pi)
        """)
    parser.add_argument("--trail", type=int, default=400,
                        help="Number of trail positions to keep (default 400)")
    parser.add_argument("--rate",  type=int, default=30,
                        help="Update rate in Hz (default 30)")
    args = parser.parse_args()

    print("==================================================")
    print("   BNO055 IMU  3D  Dead-Reckoning  Visualiser    ")
    print("==================================================")

    imu = IMUReader()
    dr  = DeadReckoning()
    vis = IMUVisualiser(imu, dr, trail_len=args.trail, update_hz=args.rate)
    vis.run()


if __name__ == "__main__":
    main()