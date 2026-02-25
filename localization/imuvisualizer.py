#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║           BNO055 IMU  —  3D Dead-Reckoning Visualiser            ║
║                                                                  ║
║  Standalone script.  No project files needed.                    ║
║  Works on Raspberry Pi with BNO055 wired to I2C, or on any      ║
║  machine in --sim mode (kinematic simulation).                   ║
║                                                                  ║
║  Usage:                                                          ║
║    python imu_visualizer.py              # real IMU              ║
║    python imu_visualizer.py --sim        # simulation mode       ║
║    python imu_visualizer.py --trail 600  # longer trail          ║
║    python imu_visualizer.py --rate 20    # 20 Hz update rate     ║
║                                                                  ║
║  Dependencies (Pi):                                              ║
║    pip install adafruit-blinka adafruit-circuitpython-bno055     ║
║    pip install matplotlib numpy                                  ║
╚══════════════════════════════════════════════════════════════════╝
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

# ══════════════════════════════════════════════════════════════════
# Colour palette  (dark cockpit theme)
# ══════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════
# IMU Driver  —  robust BNO055 init  +  sim fallback
# ══════════════════════════════════════════════════════════════════
class IMUReader:
    """
    Wraps the BNO055 with:
      • I2C deinit between retries (fixes 'scan empty after Errno 121')
      • No frequency= arg (blinka ignores it on Pi, causes confusion)
      • 1.5 s recovery after Errno 121 (sensor I2C FSM crash)
      • Simulation fallback when --sim or hardware unavailable
    """
    def __init__(self, sim=False):
        self.sim      = sim
        self.imu      = None
        self.calib    = (0, 0, 0, 0)
        self._lock    = threading.Lock()

        # Sim state
        self._sim_yaw   = 0.0
        self._sim_t     = time.time()
        self._sim_pitch = 0.0
        self._sim_roll  = 0.0

        if not sim:
            self.imu = self._init_bno055()
            if self.imu is None:
                print("[IMU] Falling back to simulation mode.")
                self.sim = True

    # ── BNO055 init ──────────────────────────────────────────────
    def _init_bno055(self, retries=4):
        try:
            import board, busio, adafruit_bno055
        except ImportError as e:
            print(f"[IMU] Missing library: {e}")
            print("      pip install adafruit-blinka adafruit-circuitpython-bno055")
            return None

        BNO_ADDRESSES = [0x29, 0x28]

        for attempt in range(1, retries + 1):
            i2c = None
            try:
                i2c = busio.I2C(board.SCL, board.SDA)   # no frequency= — blinka ignores it

                while not i2c.try_lock():
                    pass
                try:
                    found = i2c.scan()
                    print(f"[IMU] I2C scan ({attempt}/{retries}): "
                          f"{[hex(a) for a in found]}")
                finally:
                    i2c.unlock()

                if not found:
                    print(f"[IMU] Bus empty — check wiring "
                          f"SDA=GPIO2(pin3), SCL=GPIO3(pin5), VIN=3.3V(pin1)")
                    continue

                for addr in BNO_ADDRESSES:
                    if addr not in found:
                        continue
                    try:
                        imu_obj = adafruit_bno055.BNO055_I2C(i2c, address=addr)
                        _ = imu_obj.euler           # sanity read
                        print(f"[IMU] BNO055 online at {hex(addr)} ✓")
                        return imu_obj              # success — keep i2c alive
                    except OSError as oe:
                        if getattr(oe, 'errno', -1) == 121:
                            print(f"[IMU] Errno 121 at {hex(addr)} — "
                                  f"sensor I2C crash, waiting 1.5 s…")
                            try: i2c.deinit()
                            except Exception: pass
                            i2c = None
                            time.sleep(1.5)
                        else:
                            print(f"[IMU] {hex(addr)} OSError: {oe}")
                    except Exception as ex:
                        print(f"[IMU] {hex(addr)} error: {ex}")

                print(f"[IMU] No BNO055 responded (attempt {attempt}/{retries})")

            except Exception as e:
                print(f"[IMU] Init error: {type(e).__name__}: {e}")
            finally:
                if i2c is not None:
                    try: i2c.deinit()
                    except Exception: pass

            wait = 1.5 if attempt == 1 else 0.8
            if attempt < retries:
                print(f"[IMU] Retrying in {wait:.1f} s…")
                time.sleep(wait)

        return None

    # ── Read (thread-safe) ───────────────────────────────────────
    def read(self):
        """
        Returns (yaw_deg, pitch_deg, roll_deg, calib_tuple, lin_accel_xyz).
        calib_tuple = (sys, gyro, accel, mag)  each 0-3.
        """
        if self.sim:
            return self._sim_read()

        try:
            with self._lock:
                euler = self.imu.euler        # (yaw, roll, pitch) BNO055 convention
                calib = self.imu.calibration_status
                accel = self.imu.linear_acceleration
            yaw   = euler[0] if euler[0] is not None else 0.0
            pitch = euler[2] if euler[2] is not None else 0.0
            roll  = euler[1] if euler[1] is not None else 0.0
            ax    = accel[0] if accel and accel[0] is not None else 0.0
            ay    = accel[1] if accel and accel[1] is not None else 0.0
            az    = accel[2] if accel and accel[2] is not None else 0.0
            self.calib = calib if calib else (0, 0, 0, 0)
            return yaw, pitch, roll, self.calib, (ax, ay, az)
        except Exception as e:
            print(f"[IMU] Read error: {e}")
            return 0.0, 0.0, 0.0, (0, 0, 0, 0), (0.0, 0.0, 0.0)

    def _sim_read(self):
        """Kinematic simulation: figure-8 path with gentle pitch/roll."""
        t  = time.time()
        dt = t - self._sim_t
        self._sim_t = t
        # Slowly rotating figure-8 heading
        self._sim_yaw   += 25.0 * dt          # 25 deg/s yaw rate
        self._sim_pitch  = 8.0 * math.sin(t * 0.4)
        self._sim_roll   = 5.0 * math.cos(t * 0.3)
        # Fake linear accel from centripetal of circle
        ax = 0.15 * math.cos(math.radians(self._sim_yaw))
        ay = 0.15 * math.sin(math.radians(self._sim_yaw))
        return (self._sim_yaw % 360, self._sim_pitch,
                self._sim_roll, (3, 3, 3, 3), (ax, ay, 0.0))


# ══════════════════════════════════════════════════════════════════
# Dead-Reckoning integrator
# ══════════════════════════════════════════════════════════════════
class DeadReckoning:
    """
    Integrates linear acceleration to produce velocity and position.
    Uses a simple complementary filter to suppress stationary drift.
    """
    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self._prev_t = time.time()
        self._still_count = 0

    def update(self, ax, ay, az, yaw_deg):
        now = time.time()
        dt  = min(now - self._prev_t, 0.1)
        self._prev_t = now
        if dt <= 0:
            return self.x, self.y, self.z

        # Zero-velocity update: if accel magnitude ≈ gravity residual, freeze
        mag = math.hypot(ax, math.hypot(ay, az))
        if mag < 0.08:
            self._still_count += 1
        else:
            self._still_count = 0

        if self._still_count > 8:        # ~0.25 s still → hard zero velocity
            self.vx = self.vy = self.vz = 0.0
        else:
            # Rotate body-frame accel to world frame using yaw only (2-D approx)
            r  = math.radians(yaw_deg)
            c, s = math.cos(r), math.sin(r)
            wx = c * ax - s * ay
            wy = s * ax + c * ay
            self.vx += wx * dt
            self.vy += wy * dt
            self.vz += az * dt
            # Light velocity decay (mimics drag / crude bias correction)
            self.vx *= 0.92
            self.vy *= 0.92
            self.vz *= 0.85

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += max(0.0, self.z + self.vz * dt) - self.z   # clamp to z≥0

        return self.x, self.y, self.z

    def reset(self):
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0


# ══════════════════════════════════════════════════════════════════
# Build car geometry  (8-corner box in local frame)
# ══════════════════════════════════════════════════════════════════
def make_car_verts():
    """
    Returns (8×3 local vertices, face index lists, face colours).
    Local frame: +Y = forward (nose), +X = right, +Z = up.
    """
    L, W, H = 0.23, 0.135, 0.072   # metres — 1:10 scale BFMC car
    CF, CR  = L * 0.55, L * 0.45   # forward/rear half-lengths

    verts = np.array([
        [-W/2, -CR,  0.0],  # 0 rear-left  bottom
        [ W/2, -CR,  0.0],  # 1 rear-right bottom
        [ W/2,  CF,  0.0],  # 2 front-right bottom
        [-W/2,  CF,  0.0],  # 3 front-left  bottom
        [-W/2, -CR,   H ],  # 4 rear-left  top
        [ W/2, -CR,   H ],  # 5 rear-right top
        [ W/2,  CF,   H ],  # 6 front-right top
        [-W/2,  CF,   H ],  # 7 front-left  top
    ])

    face_idx = [
        [3,2,6,7],   # front (nose)     — brightest
        [4,5,6,7],   # roof
        [0,3,7,4],   # left side
        [1,2,6,5],   # right side
        [0,1,5,4],   # rear
        [0,1,2,3],   # bottom           — darkest
    ]
    face_col = [
        "#3A9EE8",   # nose
        "#1E5090",   # roof
        "#1858B0",   # left
        "#1858B0",   # right
        "#0D3060",   # rear
        "#0A1E40",   # bottom
    ]
    return verts, face_idx, face_col


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


# ══════════════════════════════════════════════════════════════════
# Main Visualiser
# ══════════════════════════════════════════════════════════════════
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
        src = "SIMULATION" if self.imu.sim else "HARDWARE"
        self.fig.text(0.50, 0.93, f"SOURCE: {src}",
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

        # Windshield
        self._ws_col = Poly3DCollection(
            [np.zeros((4, 3))],
            facecolors=["#00CCDD"], edgecolors="#00FFFF",
            linewidths=0.5, alpha=0.55, zorder=6)
        ax.add_collection3d(self._ws_col)

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
                                     "SIM" if self.imu.sim else "HW",
                                     color=AMBER if self.imu.sim else GREEN,
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
        yaw, pitch, roll, calib, (ax, ay, az) = self.imu.read()

        # ── Dead reckoning ───────────────────────────────────────
        x, y, z = self.dr.update(ax, ay, az, yaw)

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
        faces = [rv[idx] for idx in self._car_face_idx]
        self._car_col.set_verts(faces)

        # Windshield quad (top-front of car)
        CW = abs(self._car_verts[1, 0]) * 2
        CF = self._car_verts[2, 1]
        CH = self._car_verts[6, 2]
        ws_local = np.array([
            [-CW*0.38, CF,        CH],
            [ CW*0.38, CF,        CH],
            [ CW*0.38, CF * 0.55, CH],
            [-CW*0.38, CF * 0.55, CH],
        ])
        ws_world = rotate_verts(ws_local, yaw, pitch, roll)
        self._ws_col.set_verts([ws_world])

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
              f"      Mode    : {'SIMULATION' if self.imu.sim else 'HARDWARE BNO055'}\n"
              f"      Trail   : {self.trail_len} samples\n"
              f"      Rate    : {int(1000/self.interval)} Hz\n"
              f"      Hotkeys : close window to quit,  RESET button to zero position\n")
        plt.show()


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="BNO055 IMU 3D Dead-Reckoning Visualiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python imu_visualizer.py                 # real BNO055, auto-detect I2C address
  python imu_visualizer.py --sim           # simulation (no hardware needed)
  python imu_visualizer.py --trail 800     # longer yellow trail
  python imu_visualizer.py --rate 15       # 15 Hz (lighter CPU on old Pi)
        """)
    parser.add_argument("--sim",   action="store_true",
                        help="Run in simulation mode (no hardware required)")
    parser.add_argument("--trail", type=int, default=400,
                        help="Number of trail positions to keep (default 400)")
    parser.add_argument("--rate",  type=int, default=30,
                        help="Update rate in Hz (default 30)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║   BNO055 IMU  3D  Dead-Reckoning  Visualiser    ║")
    print("╚══════════════════════════════════════════════════╝")

    imu = IMUReader(sim=args.sim)
    dr  = DeadReckoning()
    vis = IMUVisualiser(imu, dr, trail_len=args.trail, update_hz=args.rate)
    vis.run()


if __name__ == "__main__":
    main()