"""
main.py — BFMC Single-Window Autonomous Pilot
==============================================
Architecture (V3 YOLO Lane Control + SVG Self-Localization):

  ┌─────────────────────────────────────────────────────────┐
  │  Single Tkinter Window                                   │
  │  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐│
  │  │  SVG Map     │  │ YOLO Camera │  │  BEV Lane View  ││
  │  │  (click to  │  │ (raw+YOLO  │  │  (sliding window││
  │  │   set pose) │  │  overlay)  │  │   poly fit)     ││
  │  └──────────────┘  └─────────────┘  └─────────────────┘│
  │  ┌────────────────────────────────────────────────────┐ │
  │  │  Status Bar: Speed  Steer  AnchorMode  Nav  E-STOP │ │
  │  └────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────┘

Self-localization:
  - User LEFT-CLICKS on SVG map → sets (x, y) start pose
  - IMU gyro (gz) integrates yaw continuously at 30 Hz
  - velocity from hardware_io.get_velocity_ms() advances (x, y)
  - Lane tangent provides a soft heading nudge per frame
  - Car dot drifts across SVG as the car moves (no A*)

Lane control (exact V3 YOLO logic):
  - HybridLaneTracker (sliding window → poly search)
  - Pure pursuit steering
  - DividerGuard forcefield
  - JunctionDetector (autonomous decision: LEFT / RIGHT / STRAIGHT)

Traffic (YOLO):
  - ThreadedYOLODetector (async queue, 0.25 conf)
  - TrafficDecisionEngine: red light / stop sign / crosswalk / collision

Usage:
  python main.py [--sim] [--speed SPEED] [--svg PATH_TO_SVG]
"""

import argparse
import logging
import math
import os
import sys
import threading
import time
import queue
from collections import deque

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ── Internal modules ───────────────────────────────────────────────────────
from perception   import VisionPipeline, estimate_heading_from_lanes
from localization import LocalizationEngine
from control      import Controller, ControlOutput
from hardware_io  import HardwareIO
from traffic_module import TrafficDecisionEngine, ThreadedYOLODetector

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("main")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_FPS   = 30
FRAME_PERIOD = 1.0 / TARGET_FPS
PWM_DEADBAND = 14.0   # Motors silent below this PWM (applied after traffic mult)

# Default SVG path (relative to script location)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SVG_PATH_DEFAULT = os.path.join(_SCRIPT_DIR, "..", "Track.svg")

# Map coordinate bounds (metres) — adjust to match your SVG/track size
MAP_W_M = 22.0
MAP_H_M = 15.0

# ═══════════════════════════════════════════════════════════════════════════════
# JunctionDetector (V3 exact)
# ═══════════════════════════════════════════════════════════════════════════════
class AutonomousJunctionPlanner:
    """Decides junction direction from BEV pixel energy."""

    def decide(self, warped_binary, left_fit, right_fit, lane_width_px):
        h, w = warped_binary.shape
        lroi  = warped_binary[0:240, 0:320]
        rroi  = warped_binary[0:240, 320:640]
        sroi  = warped_binary[0:240, 200:440]

        wts   = np.linspace(2.0, 0.5, 240).reshape(-1, 1)
        ls    = np.sum(lroi * wts) / (320 * 240)
        rs    = np.sum(rroi * wts) / (320 * 240)
        ss    = np.sum(sroi * wts) / (240 * 240)

        scores = {"LEFT": ls, "RIGHT": rs, "STRAIGHT": ss}
        best   = max(scores, key=scores.get)
        total  = sum(scores.values())
        conf   = scores[best] / max(total, 1e-6)

        if conf < 0.4:
            return "RIGHT", 0.3
        return best, conf


class JunctionDetector:
    ENTRY_FRAMES     = 5
    EXIT_FRAMES      = 8
    RATIO_EARLY_WARN = 1.7
    MIN_BOT_ENERGY   = 500

    def __init__(self):
        self.state        = "NORMAL"
        self.entry_count  = 0
        self.exit_count   = 0
        self.frames_in_jct = 0
        self.planner      = AutonomousJunctionPlanner()

    def update(self, warped_binary, left_fit, right_fit, lane_width_px, active_labels):
        h, w = warped_binary.shape

        # Detect upcoming wide gap at y=150 (top 30%)
        approaching_wide = False
        if left_fit is not None and right_fit is not None:
            lx = np.polyval(left_fit,  150)
            rx = np.polyval(right_fit, 150)
            if (rx - lx) > lane_width_px * self.RATIO_EARLY_WARN:
                approaching_wide = True
        elif left_fit is not None:
            lx = np.polyval(left_fit, 150)
            if lx < max(0, 320 - lane_width_px * self.RATIO_EARLY_WARN):
                approaching_wide = True
        elif right_fit is not None:
            rx = np.polyval(right_fit, 150)
            if rx > min(640, 320 + lane_width_px * self.RATIO_EARLY_WARN):
                approaching_wide = True

        hist_bot  = float(np.sum(warped_binary[h // 2:, :]))
        hist_top  = float(np.sum(warped_binary[:h // 2, :]))
        cross_e   = False
        if hist_bot > self.MIN_BOT_ENERGY:
            cross_e = (hist_top / hist_bot) > 1.4
        if "crosswalk-sign" in active_labels:
            cross_e = False

        evidence = approaching_wide or cross_e

        if self.state == "NORMAL":
            self.entry_count = self.entry_count + 1 if evidence else 0
            if self.entry_count >= self.ENTRY_FRAMES:
                # Emit JUNCTION_PROMPT; main.py will query the A* plan
                # to override with the correct JUNCTION_LEFT/RIGHT/STRAIGHT.
                self.state         = "JUNCTION_PROMPT"
                self.exit_count    = 0
                self.frames_in_jct = 0

        elif self.state == "JUNCTION_PROMPT":
            # Stays here until main.py replaces it with a direction
            pass

        elif self.state.startswith("JUNCTION_"):
            self.frames_in_jct += 1
            self.exit_count = self.exit_count + 1 if not evidence else 0
            if self.exit_count >= self.EXIT_FRAMES and self.frames_in_jct > 25:
                self.state       = "NORMAL"
                self.entry_count = 0

        return self.state


# ═══════════════════════════════════════════════════════════════════════════════
# SVG Map Panel helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _load_svg_as_cv2(svg_path: str, display_w: int = 600, display_h: int = 500):
    """
    Load SVG and return as a BGR numpy array of shape (display_h, display_w, 3).
    Falls back to a blank grey image if cairosvg / svglib not available.
    """
    # Try cairosvg (fastest, Raspberry Pi installable via pip)
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(
            url=svg_path, output_width=display_w, output_height=display_h
        )
        arr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    # Try svglib + reportlab
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(svg_path)
        img = renderPM.drawToPIL(drawing)
        img = img.convert("RGB").resize((display_w, display_h), Image.LANCZOS)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    log.warning("SVG render unavailable — using blank map. Install cairosvg for SVG support.")
    blank = np.full((display_h, display_w, 3), 50, np.uint8)
    cv2.putText(blank, "SVG render unavailable", (20, display_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(blank, "pip install cairosvg", (20, display_h // 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 180, 130), 1, cv2.LINE_AA)
    return blank


def map_to_pixel(x_m, y_m, img_w, img_h,
                 map_w_m=MAP_W_M, map_h_m=MAP_H_M):
    """Map metres → pixel coords in the displayed SVG image."""
    px = int(x_m / map_w_m * img_w)
    py = int(y_m / map_h_m * img_h)
    return px, py


def pixel_to_map(px, py, img_w, img_h,
                 map_w_m=MAP_W_M, map_h_m=MAP_H_M):
    """Click pixel → map metres."""
    x_m = px / img_w * map_w_m
    y_m = py / img_h * map_h_m
    return x_m, y_m


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════
class Orchestrator:
    """
    Main pilot loop + Tkinter dashboard coordinator.

    Startup sequence:
      1. Window opens, SVG map shown — car dot is NOT yet visible.
      2. User LEFT-CLICKS the SVG to set the start position.
      3. Pilot thread begins immediately.
      4. Pressing E-STOP pauses; RESUME resumes.
    """

    BASE_SPEED = 50     # Default PWM (0-100)
    MAP_W      = 600    # SVG panel width  (pixels)
    MAP_H      = 440    # SVG panel height (pixels)
    CAM_W      = 480    # Camera panel width
    CAM_H      = 360    # Camera panel height

    def __init__(self, sim_mode=False, base_speed=None, svg_path=None):
        self.sim_mode  = sim_mode
        self.base_speed = base_speed or self.BASE_SPEED
        self.svg_path  = svg_path or SVG_PATH_DEFAULT
        self.running   = False
        self._estop    = False
        self._pilot_thread = None

        # ── Hardware ──────────────────────────────────────────────────────────
        self.hw = HardwareIO(sim_mode=sim_mode)

        # ── 2-Click Routing State ─────────────────────────────────────────────
        self._start_node   = None
        self._target_node  = None
        self._planned_path = []

        # ── Perception ────────────────────────────────────────────────────────
        self.vision = VisionPipeline()

        # ── YOLO (threaded) ───────────────────────────────────────────────────
        try:
            # ThreadedYOLODetector loads YOLO internally from model_path.
            # TrafficDecisionEngine wraps it for the full decision pipeline.
            self._threaded_yolo = ThreadedYOLODetector("best.pt")
            self.traffic_engine = TrafficDecisionEngine(self._threaded_yolo)
            log.info("YOLO loaded")
        except Exception as e:
            log.warning(f"YOLO disabled: {e}")
            self._threaded_yolo  = None
            self.traffic_engine  = None


        # ── Navigation state machies ──────────────────────────────────────────
        self.jct_detector  = JunctionDetector()
        self.controller    = Controller()
        self.localizer     = LocalizationEngine()

        # ── Telemetry ─────────────────────────────────────────────────────────
        self._fps          = 0.0
        self._fps_t        = time.time()
        self._steer_hist   = deque(maxlen=120)  # ~4 s at 30 Hz
        self._nav_state    = "NORMAL"
        self._last_ctrl    = ControlOutput(0.0, 0.0, 320.0, "INIT", 200)
        self._last_t_res   = None      # TrafficResult

        # ── SVG map ───────────────────────────────────────────────────────────
        self._svg_base     = _load_svg_as_cv2(self.svg_path, self.MAP_W, self.MAP_H)
        self._start_clicked = False

        # Frame queues for GUI
        self._q_yolo  = queue.Queue(maxsize=1)
        self._q_bev   = queue.Queue(maxsize=1)

        # Camera is opened inside HardwareIO.__init__() automatically.

    # ─────────────────────────────────────────────────────────────────────────
    # GUI
    # ─────────────────────────────────────────────────────────────────────────
    def build_ui(self, root: tk.Tk):
        self._root = root
        root.title("BFMC Pilot — V3 Lane Control")
        root.configure(bg="#0d0d0d")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Top row: map | yolo camera | bev lane ────────────────────────────
        top = tk.Frame(root, bg="#0d0d0d")
        top.pack(padx=8, pady=8)

        # SVG Map panel
        map_frame = tk.LabelFrame(top, text=" MAP — click to set car position ",
                                  bg="#0d0d0d", fg="#00e5ff",
                                  font=("Courier", 9, "bold"))
        map_frame.grid(row=0, column=0, padx=6)
        self._map_label = tk.Label(map_frame, bg="#0d0d0d")
        self._map_label.pack()
        self._map_label.bind("<Button-1>", self._on_map_click)
        self._map_ph = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(self._svg_base, cv2.COLOR_BGR2RGB))
        )
        self._map_label.config(image=self._map_ph)

        # YOLO camera panel
        yolo_frame = tk.LabelFrame(top, text=" CAMERA + YOLO ",
                                   bg="#0d0d0d", fg="#ff9100",
                                   font=("Courier", 9, "bold"))
        yolo_frame.grid(row=0, column=1, padx=6)
        self._yolo_label = tk.Label(yolo_frame, bg="#0d0d0d")
        self._yolo_label.pack()
        blank_cam = np.zeros((self.CAM_H, self.CAM_W, 3), dtype=np.uint8)
        self._yolo_ph = ImageTk.PhotoImage(Image.fromarray(blank_cam))
        self._yolo_label.config(image=self._yolo_ph)

        # BEV lane panel
        bev_frame = tk.LabelFrame(top, text=" LANE VIEW (BEV) ",
                                  bg="#0d0d0d", fg="#69ff47",
                                  font=("Courier", 9, "bold"))
        bev_frame.grid(row=0, column=2, padx=6)
        self._bev_label = tk.Label(bev_frame, bg="#0d0d0d")
        self._bev_label.pack()
        blank_bev = np.zeros((360, 480, 3), dtype=np.uint8)
        self._bev_ph = ImageTk.PhotoImage(Image.fromarray(blank_bev))
        self._bev_label.config(image=self._bev_ph)

        # ── Status bar ────────────────────────────────────────────────────────
        status = tk.Frame(root, bg="#111", pady=6)
        status.pack(fill=tk.X, padx=8, pady=(0, 6))

        self._sv_speed  = tk.StringVar(value="Speed: ---")
        self._sv_steer  = tk.StringVar(value="Steer: ---")
        self._sv_anchor = tk.StringVar(value="Mode: INIT")
        self._sv_nav    = tk.StringVar(value="Nav: ---")
        self._sv_map    = tk.StringVar(value="Map: ---")
        self._sv_pose   = tk.StringVar(value="Pose: not set")
        self._sv_odo    = tk.StringVar(value="VO Yaw: 0.00 rad/s")
        self._sv_fps    = tk.StringVar(value="FPS: ---")

        style_lbl = dict(bg="#111", fg="#eee", font=("Courier", 9))
        for sv in [self._sv_speed, self._sv_steer, self._sv_anchor,
                   self._sv_nav, self._sv_map, self._sv_pose, self._sv_odo, self._sv_fps]:
            tk.Label(status, textvariable=sv, **style_lbl).pack(side=tk.LEFT, padx=10)

        # E-STOP / RESUME buttons
        btn_frame = tk.Frame(status, bg="#111")
        btn_frame.pack(side=tk.RIGHT, padx=10)
        tk.Button(btn_frame, text="⛔ E-STOP",
                  bg="#c0392b", fg="white", font=("Courier", 9, "bold"),
                  command=self._estop_cb).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="▶ RESUME",
                  bg="#27ae60", fg="white", font=("Courier", 9, "bold"),
                  command=self._resume_cb).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="🔄 RESET ROUTE",
                  bg="#2471a3", fg="white", font=("Courier", 9, "bold"),
                  command=self._reset_route).pack(side=tk.LEFT, padx=4)

        # Click instruction overlay
        self._click_hint = tk.Label(root, text="🗺  Click on the map to place the car",
                                    bg="#0d0d0d", fg="#ffcc00",
                                    font=("Courier", 11, "bold"))
        self._click_hint.pack()

        # Start GUI refresh loop
        self._gui_update()

    def _on_map_click(self, event):
        """2-Click Routing: First click sets Start, second sets Destination."""
        if not self.localizer.planner:
            log.warning("No PathPlanner available - cannot map-click.")
            return

        x_m, y_m = pixel_to_map(event.x, event.y, self.MAP_W, self.MAP_H)
        nearest = self.localizer.planner.get_nearest_node(x_m, y_m)
        if not nearest:
            return

        if self._start_node is None:
            self._start_node = nearest
            # Assume 0 yaw for now
            self.localizer.set_pose(x_m, y_m, 0.0)
            if hasattr(self, '_click_hint'):
                self._click_hint.config(text=f"Start: Node {nearest}. Now click Destination.")
            log.info(f"Start set to {nearest} at x={x_m:.1f} y={y_m:.1f}")
            
        elif self._target_node is None:
            self._target_node = nearest
            self._planned_path = self.localizer.planner.plan_route(self._start_node, self._target_node)
            
            if hasattr(self, '_click_hint'):
                self._click_hint.config(text=f"Route Planned! ({len(self._planned_path)} nodes). Driving...")
            
            self._start_clicked = True
            log.info(f"Target set to {nearest}. Path nodes: {len(self._planned_path)}")

            # Launch pilot thread on second click
            if self._pilot_thread is None or not self._pilot_thread.is_alive():
                self.running = True
                self._pilot_thread = threading.Thread(
                    target=self._pilot_loop, daemon=True)
                self._pilot_thread.start()

    def _reset_route(self):
        """Clear the planned route so the user can click a new Start + Destination."""
        self._start_node   = None
        self._target_node  = None
        self._planned_path = []
        self.running       = False   # Stops the pilot loop
        if hasattr(self, '_click_hint'):
            self._click_hint.config(text="🗺  Route reset. Click map to set new Start point.")
        log.info("Route reset by user.")

    def _estop_cb(self):
        self._estop = True
        self.hw.set_speed(0)
        self.hw.set_steering(0)
        log.warning("E-STOP triggered")

    def _resume_cb(self):
        self._estop = False
        log.info("E-STOP cleared — resuming")

    def _on_close(self):
        self.running = False
        self._estop  = True
        try:
            self.hw.set_speed(0)
            self.hw.set_steering(0)
        except Exception:
            pass
        if self._threaded_yolo:
            self._threaded_yolo.stop()
        if hasattr(self.hw, 'shutdown'):
            self.hw.shutdown()
        if hasattr(self, '_root'):
            self._root.destroy()

    # ─────────────────────────────────────────────────────────────────────────
    # GUI refresh  (main thread, 30 Hz)
    # ─────────────────────────────────────────────────────────────────────────
    def _gui_update(self):
        try:
            # ── SVG map with car dot ──────────────────────────────────────────
            map_img = self._svg_base.copy()
            
            if self._planned_path and self.localizer.planner:
                pts = []
                for n in self._planned_path:
                    pos = self.localizer.planner.node_positions.get(n)
                    if pos:
                        px, py = map_to_pixel(pos[0], pos[1], self.MAP_W, self.MAP_H)
                        pts.append([px, py])
                if len(pts) > 1:
                    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(map_img, [pts], False, (200, 50, 255), 3, cv2.LINE_AA)

            if self.localizer.is_initialized():
                x, y, yaw = self.localizer.get_pose()
                px, py = map_to_pixel(x, y, self.MAP_W, self.MAP_H)
                # Car dot
                cv2.circle(map_img, (px, py), 8,  (0, 230, 255), -1, cv2.LINE_AA)
                cv2.circle(map_img, (px, py), 12, (0, 230, 255), 1, cv2.LINE_AA)
                # Heading arrow
                hx = px + int(math.cos(yaw) * 18)
                hy = py + int(math.sin(yaw) * 18)
                cv2.arrowedLine(map_img, (px, py), (hx, hy),
                                (255, 255, 255), 2, tipLength=0.35,
                                line_type=cv2.LINE_AA)
                # Pose text
                self._sv_pose.set(f"x={x:.1f}m y={y:.1f}m yaw={math.degrees(yaw):.0f}°")

            self._map_ph = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)))
            self._map_label.config(image=self._map_ph)

            # ── YOLO frame ────────────────────────────────────────────────────
            try:
                yolo_img = self._q_yolo.get_nowait()
                yolo_img = cv2.resize(yolo_img, (self.CAM_W, self.CAM_H))
                self._yolo_ph = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)))
                self._yolo_label.config(image=self._yolo_ph)
            except queue.Empty:
                pass

            # ── BEV lane frame ────────────────────────────────────────────────
            try:
                bev_img = self._q_bev.get_nowait()
                bev_img = cv2.resize(bev_img, (480, 360))
                self._bev_ph = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)))
                self._bev_label.config(image=self._bev_ph)
            except queue.Empty:
                pass

            # ── Status strings ────────────────────────────────────────────────
            ctrl = self._last_ctrl
            self._sv_speed.set( f"Speed: {ctrl.speed_pwm:5.1f} PWM")
            self._sv_steer.set( f"Steer: {ctrl.steer_angle_deg:+5.1f}°")
            self._sv_anchor.set(f"Anchor: {ctrl.anchor}")
            self._sv_nav.set(   f"Nav: {self._nav_state}")
            
            ahead_str = getattr(self.localizer, "upcoming_curve", "UNKNOWN")
            dir_icon = "⤴" if ahead_str == "LEFT" else "⤵" if ahead_str == "RIGHT" else "⬆"
            self._sv_map.set(f"Map-Ahead: {ahead_str} {dir_icon}")
            
            self._sv_fps.set(   f"FPS: {self._fps:.1f}")

            vo_yaw = getattr(self.localizer, "visual_yaw_rate", 0.0)
            self._sv_odo.set(f"VO Yaw: {vo_yaw:+.3f} r/s")

        except Exception as e:
            log.debug(f"GUI update error: {e}")

        if hasattr(self, '_root') and self._root.winfo_exists():
            self._root.after(33, self._gui_update)  # ~30 Hz

    # ─────────────────────────────────────────────────────────────────────────
    # Pilot loop  (background thread)
    # ─────────────────────────────────────────────────────────────────────────
    def _pilot_loop(self):
        log.info("Pilot loop started")
        t_prev = time.time()
        # Lane-hold safety: consecutive frames with NO lane visible
        _lane_lost_frames = 0
        _LANE_LOST_CRAWL  = 15   # frames: start crawling (speed cap 20 PWM)
        _LANE_LOST_STOP   = 90   # frames (~3 s): full stop + E-STOP if no recovery

        try:
            while self.running:
                t_start = time.time()
                dt      = max(t_start - t_prev, 0.001)
                t_prev  = t_start

                # ── FPS ──────────────────────────────────────────────────────────
                self._fps = 0.7 * self._fps + 0.3 * (1.0 / dt)

                # ── E-STOP guard ─────────────────────────────────────────────────
                if self._estop:
                    self.hw.set_speed(0)
                    self.hw.set_steering(0)
                    time.sleep(0.05)
                    continue

                # ── 1. Capture frame ─────────────────────────────────────────────
                raw_frame = self.hw.read_camera()
                if raw_frame is None:
                    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # ── 2. IMU data (Removed - Visual Odometry Active) ───────────────
                pass

                # ── 3. Hardware velocity ─────────────────────────────────────────
                velocity_ms = self.hw.get_velocity_ms()

                # ── 4. YOLO (traffic) ─────────────────────────────────────────────
                if self.traffic_engine:
                    t_res = self.traffic_engine.process(raw_frame)
                else:
                    from traffic_module import TrafficResult
                    t_res = TrafficResult(
                        state="SYS_GO", reason="NO YOLO",
                        speed_multiplier=1.0, zone_mode="CITY",
                        parking_state="NONE", steer_bias=0.0,
                        pedestrian_blocking=False, light_status="NONE",
                        active_labels=[], yolo_debug_frame=raw_frame.copy()
                    )
                self._last_t_res = t_res

                # ── 5. Lane perception (V3 VisionPipeline) ──────────────────────
                extra_offset = 0.0
                if t_res.state == "SYS_LANE_CHANGE_LEFT":
                    extra_offset = -80.0    # shift target left

                perc = self.vision.process(
                    raw_frame,
                    extra_offset_px=extra_offset,
                    nav_state=self._nav_state
                )

                # ── 6. Junction detection & Path Routing ──────────────────────────
                self._nav_state = self.jct_detector.update(
                    perc.warped_binary,
                    perc.sl, perc.sr,
                    perc.lane_width_px,
                    t_res.active_labels
                )

                # If the detector prompts us for a choice at a junction, query A* path
                if self._nav_state == "JUNCTION_PROMPT":
                    if self._planned_path and self.localizer.planner:
                        x, y, yaw = self.localizer.get_pose()
                    
                        # Find our current nearest node cursor
                        nearest = self.localizer.planner.get_nearest_node(x, y)
                        cursor = 0
                        if nearest in self._planned_path:
                            cursor = self._planned_path.index(nearest)

                        # Determine turn direction
                        action = self.localizer.planner.get_next_action(
                            current_x=x, current_y=y, current_yaw=yaw,
                            path=self._planned_path, cursor=cursor
                        )
                    
                        # E.g. "LEFT" -> "JUNCTION_LEFT"
                        self._nav_state = f"JUNCTION_{action}"
                        log.info(f"📍 Junction Reached at {nearest}. Routing: {action}")
                    else:
                        self._nav_state = "JUNCTION_STRAIGHT"  # default fail-safe

                # ── 7. Camera heading for localizer ─────────────────────────────
                cam_heading = estimate_heading_from_lanes(perc.sl, perc.sr)

                # ── 8. Localizer update ──────────────────────────────────────────
                self.localizer.update(
                    velocity_ms       = velocity_ms,
                    dt                = dt,
                    camera_heading_rad= cam_heading,
                    camera_confidence = perc.confidence
                )

                # ── 9. Controller ────────────────────────────────────────────────
                map_ahead = getattr(self.localizer, "upcoming_curve", "STRAIGHT")
            
                ctrl = self.controller.compute(
                    perc_res      = perc,
                    nav_state     = self._nav_state,
                    traffic_state = t_res.state,
                    base_speed    = float(self.base_speed),
                    traffic_mult  = t_res.speed_multiplier,
                    zone_mode     = t_res.zone_mode,
                    parking_state = t_res.parking_state,
                    steer_bias    = t_res.steer_bias,
                    upcoming_curve= map_ahead,
                    visual_yaw_rate_rps = self.localizer.visual_yaw_rate,
                    velocity_ms   = velocity_ms,
                    dt            = dt,
                )
                self._last_ctrl = ctrl

                # ── LANE-HOLD SAFETY: consecutive blind-frame guard ──────────────
                if perc.sl is None and perc.sr is None:
                    _lane_lost_frames += 1
                else:
                    _lane_lost_frames = 0   # reset as soon as ANY line reappears

                if _lane_lost_frames >= _LANE_LOST_STOP:
                    # 3 consecutive seconds with zero lane — stop the car
                    self.hw.set_speed(0)
                    self.hw.set_steering(0)
                    log.error(f"LANE LOST for {_lane_lost_frames} frames — emergency stop")
                    self._estop = True
                    continue

                # ── 10. PWM deadband + blind-frame speed cap ─────────────────────
                speed = ctrl.speed_pwm
                # Hard cap: if blind > CRAWL threshold, don't exceed 20 PWM
                if _lane_lost_frames >= _LANE_LOST_CRAWL:
                    speed = min(speed, 20.0)
                # PWM deadband
                if 0.0 < speed < PWM_DEADBAND:
                    speed = PWM_DEADBAND

                # ── 11. Send commands ────────────────────────────────────────────
                self.hw.set_speed(speed)
                self.hw.set_steering(ctrl.steer_angle_deg)

                # ── 12. Push frames to GUI queues ────────────────────────────────
                if not self._q_yolo.full():
                    self._q_yolo.put(t_res.yolo_debug_frame)
                if not self._q_bev.full():
                    self._q_bev.put(_annotate_bev(perc, ctrl))

                # ── Frame rate throttle ──────────────────────────────────────────
                elapsed = time.time() - t_start
                sleep_time = max(0.001, FRAME_PERIOD - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            log.error(f"FATAL Pilot loop crash: {e}", exc_info=True)
            self._estop = True

        log.info("Pilot loop exited")
        self.hw.set_speed(0)
        self.hw.set_steering(0)


# ═══════════════════════════════════════════════════════════════════════════════
# BEV annotation helper
# ═══════════════════════════════════════════════════════════════════════════════
def _annotate_bev(perc, ctrl: ControlOutput) -> np.ndarray:
    """Draw polynomial fits, target cross and anchor label on BEV debug frame."""
    dbg = perc.lane_dbg.copy() if perc.lane_dbg is not None else \
          np.zeros((480, 640, 3), np.uint8)

    def draw_poly(fit, color):
        if fit is None:
            return
        ys = np.linspace(0, 479, 240).astype(np.float32)
        xs = np.polyval(fit, ys).astype(np.float32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.int32)
        pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, 639)
        cv2.polylines(dbg, [pts], False, color, 3, cv2.LINE_AA)

    draw_poly(perc.sl, (255, 80, 80))
    draw_poly(perc.sr, (80, 80, 255))

    # Target X cross-hair
    tx = int(ctrl.target_x)
    cv2.line(dbg, (tx, 380), (tx, 420), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(dbg, (tx - 10, 400), (tx + 10, 400), (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(dbg, ctrl.anchor, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(dbg, f"steer={ctrl.steer_angle_deg:+.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
    return dbg


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="BFMC V3 Lane Pilot")
    ap.add_argument("--sim",   action="store_true", help="No hardware — use blank frames")
    ap.add_argument("--speed", type=float, default=50,
                    help="Base PWM speed 0-100 (default 50)")
    ap.add_argument("--svg",   type=str,   default=None,
                    help="Path to Track.svg (default: auto-detect)")
    args = ap.parse_args()

    svg_path = args.svg or SVG_PATH_DEFAULT
    if not os.path.exists(svg_path):
        log.warning(f"SVG not found at {svg_path} — using blank map")
        svg_path = svg_path   # _load_svg_as_cv2 handles missing file gracefully

    orch = Orchestrator(
        sim_mode   = args.sim,
        base_speed = args.speed,
        svg_path   = svg_path,
    )

    root = tk.Tk()
    orch.build_ui(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — shutting down")
    finally:
        orch._on_close()


if __name__ == "__main__":
    main()