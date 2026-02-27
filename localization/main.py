"""
main.py — BFMC Single-Window Autonomous Pilot  (FIXED v2)
==========================================================
Fixes applied:
  VL-05  SVG Y-axis inverted in map_to_pixel / pixel_to_map
  VL-06  Incremental cursor replaces O(N) list.index()
  VL-08  Path cursor reset when new route planned
  SIGN-01 No-Entry triggers A* reroute (blocked node + replan)
  SIGN-02 zone_mode cross-validated with map every frame
  SIGN-03 nav_state = ROUNDABOUT set from map node membership
  MAP-02  get_next_action called with velocity_ms argument
  DASHBOARD: Professional 6-panel redesign:
    - MapOverlayRenderer with Y-axis fix + confidence ring + route overlay
    - GraphML node graph panel (live A* cursor + roundabout highlights)
    - Steering arc gauge
    - Confidence + speed bars
    - FPS alarm
    - Fading sign detection history
    - Path progress bar
    - Zone badge + nav state badge
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

from perception    import VisionPipeline, estimate_heading_from_lanes
from localization  import LocalizationEngine
from control       import Controller, ControlOutput
from hardware_io   import HardwareIO
from traffic_module import TrafficDecisionEngine, ThreadedYOLODetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("main")

TARGET_FPS   = 30
FRAME_PERIOD = 1.0 / TARGET_FPS
PWM_DEADBAND = 14.0

_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
SVG_PATH_DEFAULT = os.path.join(_SCRIPT_DIR, "..", "Track.svg")

MAP_W_M = 22.0
MAP_H_M = 15.0


# ══════════════════════════════════════════════════════════════════════════════
# Map coordinate helpers  (FIX VL-05: Y-axis inverted)
# ══════════════════════════════════════════════════════════════════════════════

def map_to_pixel(x_m, y_m, img_w, img_h,
                 map_w_m=MAP_W_M, map_h_m=MAP_H_M):
    """Map metres → pixel.  FIX VL-05: Y axis flipped (world Y up, pixel Y down)."""
    px = int(x_m / map_w_m * img_w)
    py = int((map_h_m - y_m) / map_h_m * img_h)   # ← INVERTED
    return px, py


def pixel_to_map(px, py, img_w, img_h,
                 map_w_m=MAP_W_M, map_h_m=MAP_H_M):
    """Click pixel → map metres.  FIX VL-05: Y axis flipped."""
    x_m = px / img_w * map_w_m
    y_m = (1.0 - py / img_h) * map_h_m              # ← INVERTED
    return x_m, y_m


# ══════════════════════════════════════════════════════════════════════════════
# SVG loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_svg_as_cv2(svg_path: str, display_w: int = 600, display_h: int = 440):
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(
            url=svg_path, output_width=display_w, output_height=display_h)
        arr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(svg_path)
        img = renderPM.drawToPIL(drawing)
        img = img.convert("RGB").resize((display_w, display_h), Image.LANCZOS)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    log.warning("SVG render unavailable — using blank map.")
    blank = np.full((display_h, display_w, 3), 30, np.uint8)
    cv2.putText(blank, "SVG render unavailable",
                (20, display_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(blank, "pip install cairosvg",
                (20, display_h // 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 180, 130), 1)
    return blank


# ══════════════════════════════════════════════════════════════════════════════
# MapOverlayRenderer  (FIX VL-05 + NEW dashboard features)
# ══════════════════════════════════════════════════════════════════════════════

class MapOverlayRenderer:
    """Renders the SVG map with live car dot, A* route, and progress bar."""

    def __init__(self, svg_bgr, map_w_m=MAP_W_M, map_h_m=MAP_H_M):
        self.base  = svg_bgr.copy()
        self.w_m   = map_w_m
        self.h_m   = map_h_m
        self.ih, self.iw = svg_bgr.shape[:2]

    def _w2p(self, x_m, y_m):
        return map_to_pixel(x_m, y_m, self.iw, self.ih, self.w_m, self.h_m)

    def render(self, x, y, yaw, path, cursor, planner, confidence, zone_mode):
        img = self.base.copy()

        # ── A* planned path ───────────────────────────────────────────────────
        if path and planner:
            pts_all  = []
            pts_done = []
            for i, n in enumerate(path):
                pos = planner.node_positions.get(n)
                if pos is None:
                    continue
                px, py = self._w2p(*pos)
                pts_all.append([px, py])
                if i <= cursor:
                    pts_done.append([px, py])

            if len(pts_all) > 1:
                cv2.polylines(img,
                              [np.array(pts_all, np.int32)],
                              False, (150, 40, 210), 2, cv2.LINE_AA)
            if len(pts_done) > 1:
                cv2.polylines(img,
                              [np.array(pts_done, np.int32)],
                              False, (40, 160, 40), 2, cv2.LINE_AA)

            # Lookahead target marker
            la_idx = min(cursor + 5, len(path) - 1)
            la_pos = planner.node_positions.get(path[la_idx])
            if la_pos:
                lx, ly = self._w2p(*la_pos)
                cv2.drawMarker(img, (lx, ly), (0, 255, 180),
                               cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)

        # ── Car dot + heading arrow ───────────────────────────────────────────
        cx, cy = self._w2p(x, y)
        # Colour reflects confidence
        if   confidence > 0.6: dot_color = (0, 230, 255)    # cyan  — good
        elif confidence > 0.3: dot_color = (0, 160, 255)    # amber — ok
        else:                  dot_color = (50,  50, 220)   # red   — lost

        cv2.circle(img, (cx, cy), 9,  dot_color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 13, dot_color,  1, cv2.LINE_AA)

        # FIX VL-05: heading arrow Y component negated (SVG Y inverted)
        hx = cx + int(math.cos(yaw) * 22)
        hy = cy - int(math.sin(yaw) * 22)   # ← NEGATED
        cv2.arrowedLine(img, (cx, cy), (hx, hy),
                        (255, 255, 255), 2, tipLength=0.4, line_type=cv2.LINE_AA)

        # Confidence uncertainty ring (radius grows when confidence is low)
        r = int(30 * (1.0 - confidence)) + 5
        cv2.circle(img, (cx, cy), r, dot_color, 1, cv2.LINE_AA)

        # ── Zone badge ───────────────────────────────────────────────────────
        zone_colors = {"CITY":(50,180,50), "HIGHWAY":(50,100,230), "PARKING":(180,130,30)}
        zc = zone_colors.get(zone_mode, (120, 120, 120))
        cv2.rectangle(img, (self.iw - 120, 4), (self.iw - 4, 30), zc, -1, cv2.LINE_AA)
        cv2.putText(img, zone_mode,
                    (self.iw - 116, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Path progress bar ─────────────────────────────────────────────────
        if path:
            progress = cursor / max(len(path) - 1, 1)
            bar_w    = int(self.iw * progress)
            cv2.rectangle(img, (0, self.ih - 7), (self.iw, self.ih),
                          (30, 30, 30), -1)
            cv2.rectangle(img, (0, self.ih - 7), (bar_w, self.ih),
                          (0, 200, 100), -1)
            pct_str = f"Node {cursor}/{len(path)}  {progress*100:.0f}%"
            cv2.putText(img, pct_str,
                        (4, self.ih - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        return img


# ══════════════════════════════════════════════════════════════════════════════
# GraphML Node Graph Renderer  (NEW)
# ══════════════════════════════════════════════════════════════════════════════

class GraphMLRenderer:
    """Renders the A* graph as a 2-D network with live cursor highlight."""

    def __init__(self, planner, canvas_w=480, canvas_h=360):
        self.planner = planner
        self.W, self.H = canvas_w, canvas_h
        if not planner or not planner.node_positions:
            self._base = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)
            return

        positions = list(planner.node_positions.values())
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)
        self._base = self._draw_static()

    def _to_px(self, x, y):
        margin = 20
        px = int((x - self.x_min) / max(self.x_max - self.x_min, 0.01)
                 * (self.W - 2 * margin)) + margin
        py = int((self.y_max - y) / max(self.y_max - self.y_min, 0.01)
                 * (self.H - 2 * margin)) + margin
        return px, py

    def _draw_static(self):
        img = np.full((self.H, self.W, 3), 20, dtype=np.uint8)
        # Edges
        for u, v in self.planner.graph.edges():
            p1 = self.planner.node_positions.get(u)
            p2 = self.planner.node_positions.get(v)
            if p1 and p2:
                cv2.line(img, self._to_px(*p1), self._to_px(*p2),
                         (50, 50, 55), 1, cv2.LINE_AA)
        # Nodes
        for nid, (x, y) in self.planner.node_positions.items():
            px = self._to_px(x, y)
            if nid in self.planner._roundabout_nodes:
                color = (30, 180, 180)   # teal for roundabout nodes
            else:
                color = (65, 65, 80)
            cv2.circle(img, px, 2, color, -1)
        return img

    def render(self, path, cursor, nearest_node):
        img = self._base.copy()
        if not self.planner or not self.planner.node_positions:
            return img

        # Planned path
        for i in range(len(path) - 1):
            p1 = self.planner.node_positions.get(path[i])
            p2 = self.planner.node_positions.get(path[i + 1])
            if p1 and p2:
                color = (40, 140, 40) if i < cursor else (220, 120, 30)
                cv2.line(img, self._to_px(*p1), self._to_px(*p2),
                         color, 2, cv2.LINE_AA)

        # Current cursor node — magenta
        if 0 <= cursor < len(path):
            pos = self.planner.node_positions.get(path[cursor])
            if pos:
                cv2.circle(img, self._to_px(*pos), 7, (255, 0, 200), -1, cv2.LINE_AA)

        # Nearest node — white ring
        if nearest_node and nearest_node in self.planner.node_positions:
            pos = self.planner.node_positions[nearest_node]
            cv2.circle(img, self._to_px(*pos), 5, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.putText(img, "GraphML A* View", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
        return img


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard drawing helpers  (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_steer_gauge(img, steer_deg, cx=240, cy=95, r=70):
    """Arc steering gauge: -45° left … 0 … +45° right."""
    # Background track
    cv2.ellipse(img, (cx, cy), (r, r), 0, 200, 340, (50, 50, 50), 10)
    # Needle — angle mapped: 0° steer → top (270°), left→cw, right→ccw in image
    norm  = steer_deg / 45.0                             # -1 … +1
    angle_deg = 270 - norm * 70                          # 270° = straight ahead
    angle_rad = math.radians(angle_deg)
    nx = int(cx + r * math.cos(angle_rad))
    ny = int(cy + r * math.sin(angle_rad))
    color = (50, 220, 50) if abs(steer_deg) < 15 else \
            (50, 200, 200) if abs(steer_deg) < 30 else (50, 50, 230)
    cv2.line(img, (cx, cy), (nx, ny), color, 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 5, color, -1, cv2.LINE_AA)
    cv2.putText(img, f"{steer_deg:+.1f}deg",
                (cx - 28, cy + r + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(img, "STEER",
                (cx - 18, cy - r - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


def _draw_hbar(img, value, max_val, label, x, y, w=180, h=14,
               color=(50, 200, 50)):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    fill = int(min(1.0, abs(value) / max(max_val, 0.001)) * w)
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.putText(img, f"{label}: {value:.1f}",
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


def _draw_telemetry_panel(steer_deg, speed_pwm, lateral_err_px,
                          confidence, anchor, zone_mode, upcoming_curve,
                          fps, sign_history, nav_state,
                          w=480, h=220):
    """Builds the bottom-right telemetry panel as a BGR numpy array."""
    img = np.full((h, w, 3), 18, dtype=np.uint8)

    # ── Steering gauge ────────────────────────────────────────────────────────
    _draw_steer_gauge(img, steer_deg, cx=90, cy=90, r=65)

    # ── Speed bar ─────────────────────────────────────────────────────────────
    _draw_hbar(img, speed_pwm, 100.0, "SPD(PWM)", 200, 20, w=220, h=14,
               color=(50, 140, 230))

    # ── Lateral error bar ─────────────────────────────────────────────────────
    err_color = (50, 220, 50) if abs(lateral_err_px) < 30 else \
                (50, 200, 200) if abs(lateral_err_px) < 70 else (50, 50, 230)
    _draw_hbar(img, lateral_err_px, 160.0, "LAT.ERR", 200, 52, w=220, h=14,
               color=err_color)

    # ── Confidence bar ────────────────────────────────────────────────────────
    conf_color = (50, 220, 50) if confidence > 0.6 else \
                 (50, 200, 200) if confidence > 0.3 else (50, 50, 230)
    _draw_hbar(img, confidence * 100, 100.0, "CONF %", 200, 84, w=220, h=14,
               color=conf_color)

    # ── Anchor mode ───────────────────────────────────────────────────────────
    anc_color = (50, 220, 50) if "DUAL" in anchor else \
                (50, 200, 200) if "DEAD" not in anchor else (50, 50, 230)
    cv2.putText(img, f"ANCHOR: {anchor}",
                (200, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.40, anc_color, 1)

    # ── FPS ───────────────────────────────────────────────────────────────────
    fps_color = (50, 220, 50) if fps >= 25 else \
                (50, 200, 200) if fps >= 18 else (50, 50, 230)
    fps_text = f"FPS: {fps:.1f}"
    if fps < 18:
        fps_text += "  !! LOW"
    cv2.putText(img, fps_text,
                (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1)

    # ── Nav state + upcoming curve ────────────────────────────────────────────
    icons = {"LEFT": "<<LEFT", "RIGHT": "RIGHT>>", "STRAIGHT": "^ STRAIGHT"}
    nav_icon = icons.get(upcoming_curve, upcoming_curve)
    nav_color = (50, 180, 255) if "LEFT" in upcoming_curve else \
                (255, 140, 50) if "RIGHT" in upcoming_curve else (50, 220, 50)
    cv2.putText(img, f"NEXT: {nav_icon}",
                (200, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.45, nav_color, 1)
    cv2.putText(img, f"NAV: {nav_state}",
                (200, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # ── Sign detection history ────────────────────────────────────────────────
    now = time.time()
    for i, (label, conf, ts) in enumerate(sign_history[-5:]):
        age   = now - ts
        alpha = max(0.15, 1.0 - age / 5.0)
        c     = int(200 * alpha)
        cv2.putText(img, f"{label} {conf:.2f} ({age:.1f}s)",
                    (4, 138 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (c, c, c), 1)

    cv2.putText(img, "SIGNS", (4, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 120), 1)

    return img


# ══════════════════════════════════════════════════════════════════════════════
# Junction detector (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

class AutonomousJunctionPlanner:
    def decide(self, warped_binary, left_fit, right_fit, lane_width_px):
        h, w = warped_binary.shape
        lroi = warped_binary[0:240, 0:320]
        rroi = warped_binary[0:240, 320:640]
        sroi = warped_binary[0:240, 200:440]
        wts  = np.linspace(2.0, 0.5, 240).reshape(-1, 1)
        ls   = np.sum(lroi * wts) / (320 * 240)
        rs   = np.sum(rroi * wts) / (320 * 240)
        ss   = np.sum(sroi * wts) / (240 * 240)
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
        self.state         = "NORMAL"
        self.entry_count   = 0
        self.exit_count    = 0
        self.frames_in_jct = 0
        self.planner       = AutonomousJunctionPlanner()

    def update(self, warped_binary, left_fit, right_fit, lane_width_px,
               active_labels):
        h, w = warped_binary.shape

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

        hist_bot = float(np.sum(warped_binary[h // 2:, :]))
        hist_top = float(np.sum(warped_binary[:h // 2, :]))
        cross_e  = False
        if hist_bot > self.MIN_BOT_ENERGY:
            cross_e = (hist_top / hist_bot) > 1.4
        if "crosswalk-sign" in active_labels:
            cross_e = False

        evidence = approaching_wide or cross_e

        if self.state == "NORMAL":
            self.entry_count = self.entry_count + 1 if evidence else 0
            if self.entry_count >= self.ENTRY_FRAMES:
                self.state      = "JUNCTION_PROMPT"
                self.exit_count = 0
                self.frames_in_jct = 0

        elif self.state == "JUNCTION_PROMPT":
            pass

        elif self.state.startswith("JUNCTION_"):
            self.frames_in_jct += 1
            self.exit_count = self.exit_count + 1 if not evidence else 0
            if self.exit_count >= self.EXIT_FRAMES and self.frames_in_jct > 25:
                self.state       = "NORMAL"
                self.entry_count = 0

        return self.state


# ══════════════════════════════════════════════════════════════════════════════
# BEV annotation
# ══════════════════════════════════════════════════════════════════════════════

def _annotate_bev(perc, ctrl: ControlOutput) -> np.ndarray:
    dbg = perc.lane_dbg.copy() if perc.lane_dbg is not None else \
          np.zeros((480, 640, 3), np.uint8)

    def draw_poly(fit, color):
        if fit is None:
            return
        ys  = np.linspace(0, 479, 240).astype(np.float32)
        xs  = np.polyval(fit, ys).astype(np.float32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.int32)
        pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, 639)
        cv2.polylines(dbg, [pts], False, color, 3, cv2.LINE_AA)

    draw_poly(perc.sl, (255, 80, 80))
    draw_poly(perc.sr, (80, 80, 255))

    tx = int(ctrl.target_x)
    cv2.line(dbg, (tx, 380), (tx, 420), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(dbg, (tx - 10, 400), (tx + 10, 400), (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(dbg, ctrl.anchor, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(dbg, f"steer={ctrl.steer_angle_deg:+.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
    return dbg


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    BASE_SPEED = 50
    MAP_W      = 600
    MAP_H      = 440
    CAM_W      = 480
    CAM_H      = 360

    def __init__(self, sim_mode=False, base_speed=None, svg_path=None):
        self.sim_mode   = sim_mode
        self.base_speed = base_speed or self.BASE_SPEED
        self.svg_path   = svg_path or SVG_PATH_DEFAULT
        self.running    = False
        self._estop     = False
        self._pilot_thread = None

        self.hw = HardwareIO(sim_mode=sim_mode)

        # Route state
        self._start_node   = None
        self._target_node  = None
        self._planned_path = []
        self._path_cursor  = 0          # FIX VL-06: maintained incrementally

        # Blocked nodes for SIGN-01 reroute
        self._blocked_nodes    = {}     # node_id → unblock_time

        # Zone cross-validation
        self._zone_override_frames = 0  # SIGN-02

        self.vision  = VisionPipeline()

        try:
            self._threaded_yolo  = ThreadedYOLODetector("best.pt")
            self.traffic_engine  = TrafficDecisionEngine(self._threaded_yolo)
            log.info("YOLO loaded")
        except Exception as e:
            log.warning(f"YOLO disabled: {e}")
            self._threaded_yolo = None
            self.traffic_engine = None

        self.jct_detector = JunctionDetector()
        self.controller   = Controller()
        self.localizer    = LocalizationEngine()

        self._fps        = 0.0
        self._fps_t      = time.time()
        self._nav_state  = "NORMAL"
        self._last_ctrl  = ControlOutput(0.0, 0.0, 320.0, "INIT", 200)
        self._last_t_res = None

        # Sign detection history for dashboard
        self._sign_history: deque = deque(maxlen=20)

        # SVG map
        self._svg_base     = _load_svg_as_cv2(self.svg_path, self.MAP_W, self.MAP_H)
        self._map_renderer = MapOverlayRenderer(self._svg_base, MAP_W_M, MAP_H_M)
        self._graph_renderer: GraphMLRenderer = None   # built after planner ready
        self._start_clicked = False

        # GUI queues
        self._q_yolo  = queue.Queue(maxsize=1)
        self._q_bev   = queue.Queue(maxsize=1)
        self._q_graph = queue.Queue(maxsize=1)
        self._q_telem = queue.Queue(maxsize=1)

    # ── GUI build ─────────────────────────────────────────────────────────────

    def build_ui(self, root: tk.Tk):
        self._root = root
        root.title("BFMC Pilot v2 — Visual Localization")
        root.configure(bg="#0d0d0d")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Top row: Map | YOLO | GraphML ─────────────────────────────────────
        top = tk.Frame(root, bg="#0d0d0d")
        top.pack(padx=6, pady=6)

        # SVG map panel
        mf = tk.LabelFrame(top, text=" MAP (click: Start → Dest) ",
                           bg="#0d0d0d", fg="#00e5ff",
                           font=("Courier", 9, "bold"))
        mf.grid(row=0, column=0, padx=4)
        self._map_label = tk.Label(mf, bg="#0d0d0d")
        self._map_label.pack()
        self._map_label.bind("<Button-1>", self._on_map_click)
        self._map_ph = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(self._svg_base, cv2.COLOR_BGR2RGB)))
        self._map_label.config(image=self._map_ph)

        # YOLO camera panel
        cf = tk.LabelFrame(top, text=" CAMERA + YOLO ",
                           bg="#0d0d0d", fg="#ff9100",
                           font=("Courier", 9, "bold"))
        cf.grid(row=0, column=1, padx=4)
        self._yolo_label = tk.Label(cf, bg="#0d0d0d")
        self._yolo_label.pack()
        blank_cam = np.zeros((self.CAM_H, self.CAM_W, 3), np.uint8)
        self._yolo_ph = ImageTk.PhotoImage(Image.fromarray(blank_cam))
        self._yolo_label.config(image=self._yolo_ph)

        # GraphML panel
        gf = tk.LabelFrame(top, text=" GRAPH (A* route) ",
                           bg="#0d0d0d", fg="#69ff47",
                           font=("Courier", 9, "bold"))
        gf.grid(row=0, column=2, padx=4)
        self._graph_label = tk.Label(gf, bg="#0d0d0d")
        self._graph_label.pack()
        blank_g = np.full((self.CAM_H, self.CAM_W, 3), 20, np.uint8)
        self._graph_ph = ImageTk.PhotoImage(Image.fromarray(blank_g))
        self._graph_label.config(image=self._graph_ph)

        # ── Bottom row: BEV | Telemetry ───────────────────────────────────────
        bot = tk.Frame(root, bg="#0d0d0d")
        bot.pack(padx=6, pady=(0, 6))

        bf = tk.LabelFrame(bot, text=" LANE VIEW (BEV) ",
                           bg="#0d0d0d", fg="#69ff47",
                           font=("Courier", 9, "bold"))
        bf.grid(row=0, column=0, padx=4)
        self._bev_label = tk.Label(bf, bg="#0d0d0d")
        self._bev_label.pack()
        blank_bev = np.zeros((self.CAM_H, self.CAM_W, 3), np.uint8)
        self._bev_ph = ImageTk.PhotoImage(Image.fromarray(blank_bev))
        self._bev_label.config(image=self._bev_ph)

        tf = tk.LabelFrame(bot, text=" TELEMETRY ",
                           bg="#0d0d0d", fg="#ff9100",
                           font=("Courier", 9, "bold"))
        tf.grid(row=0, column=1, padx=4, columnspan=2)
        self._telem_label = tk.Label(tf, bg="#0d0d0d")
        self._telem_label.pack()
        blank_tel = np.full((self.CAM_H, self.CAM_W, 3), 18, np.uint8)
        self._telem_ph = ImageTk.PhotoImage(Image.fromarray(blank_tel))
        self._telem_label.config(image=self._telem_ph)

        # ── Control bar ───────────────────────────────────────────────────────
        ctrl_bar = tk.Frame(root, bg="#111", pady=5)
        ctrl_bar.pack(fill=tk.X, padx=6, pady=(0, 4))

        self._sv_pose = tk.StringVar(value="Pose: not set")
        self._sv_hint = tk.StringVar(value="Click map: set START")

        tk.Label(ctrl_bar, textvariable=self._sv_pose,
                 bg="#111", fg="#eee", font=("Courier", 9)).pack(side=tk.LEFT, padx=10)
        tk.Label(ctrl_bar, textvariable=self._sv_hint,
                 bg="#111", fg="#ffcc00", font=("Courier", 9, "bold")).pack(side=tk.LEFT, padx=10)

        btn_frame = tk.Frame(ctrl_bar, bg="#111")
        btn_frame.pack(side=tk.RIGHT, padx=8)
        tk.Button(btn_frame, text="E-STOP",
                  bg="#c0392b", fg="white",
                  font=("Courier", 9, "bold"),
                  command=self._estop_cb).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="RESUME",
                  bg="#27ae60", fg="white",
                  font=("Courier", 9, "bold"),
                  command=self._resume_cb).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="RESET ROUTE",
                  bg="#2471a3", fg="white",
                  font=("Courier", 9, "bold"),
                  command=self._reset_route).pack(side=tk.LEFT, padx=4)

        # Build graph renderer now that planner is available
        if self.localizer.planner:
            self._graph_renderer = GraphMLRenderer(
                self.localizer.planner, self.CAM_W, self.CAM_H)

        self._gui_update()

    # ── Map click  ────────────────────────────────────────────────────────────

    def _on_map_click(self, event):
        if not self.localizer.planner:
            return
        x_m, y_m = pixel_to_map(event.x, event.y, self.MAP_W, self.MAP_H)
        nearest   = self.localizer.planner.get_nearest_node(x_m, y_m)
        if not nearest:
            return

        if self._start_node is None:
            self._start_node = nearest
            self.localizer.set_pose(x_m, y_m, 0.0)
            self._sv_hint.set(f"Start: {nearest}  — now click DESTINATION")
            log.info(f"Start set: {nearest}  x={x_m:.1f} y={y_m:.1f}")

        elif self._target_node is None:
            self._target_node  = nearest
            self._planned_path = self.localizer.planner.plan_route(
                self._start_node, self._target_node)
            self._path_cursor  = 0          # FIX VL-08
            self.localizer.reset_cursor()   # FIX VL-08

            self._sv_hint.set(
                f"Route planned: {len(self._planned_path)} nodes — DRIVING")
            log.info(f"Target: {nearest}.  Path nodes: {len(self._planned_path)}")
            self._start_clicked = True

            if self._pilot_thread is None or not self._pilot_thread.is_alive():
                self.running = True
                self._pilot_thread = threading.Thread(
                    target=self._pilot_loop, daemon=True)
                self._pilot_thread.start()

    def _reset_route(self):
        self._start_node   = None
        self._target_node  = None
        self._planned_path = []
        self._path_cursor  = 0
        self.localizer.reset_cursor()
        self.running = False
        self._sv_hint.set("Route reset — click map for new START")
        log.info("Route reset.")

    def _estop_cb(self):
        self._estop = True
        self.hw.set_speed(0)
        self.hw.set_steering(0)
        log.warning("E-STOP triggered")

    def _resume_cb(self):
        self._estop = False
        log.info("E-STOP cleared")

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

    # ── GUI refresh (main thread ~30 Hz) ──────────────────────────────────────

    def _gui_update(self):
        try:
            # SVG map
            x, y, yaw = self.localizer.get_pose()
            nearest = (self.localizer.planner.get_nearest_node(x, y)
                       if self.localizer.planner else None)
            conf   = getattr(self._last_ctrl, 'speed_pwm', 0)  # placeholder
            # Better: get confidence from last traffic result or set in pilot
            conf   = getattr(self, '_last_conf', 0.0)
            zone   = getattr(self.localizer, 'current_zone', 'CITY')

            map_img = self._map_renderer.render(
                x, y, yaw,
                self._planned_path, self._path_cursor,
                self.localizer.planner,
                conf, zone)

            self._map_ph = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)))
            self._map_label.config(image=self._map_ph)

            if self.localizer.is_initialized():
                self._sv_pose.set(
                    f"x={x:.2f}m  y={y:.2f}m  yaw={math.degrees(yaw):.0f}deg")

            # YOLO frame
            try:
                yolo_img = self._q_yolo.get_nowait()
                yolo_img = cv2.resize(yolo_img, (self.CAM_W, self.CAM_H))
                self._yolo_ph = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)))
                self._yolo_label.config(image=self._yolo_ph)
            except queue.Empty:
                pass

            # BEV lane frame
            try:
                bev_img = self._q_bev.get_nowait()
                bev_img = cv2.resize(bev_img, (self.CAM_W, self.CAM_H))
                self._bev_ph = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)))
                self._bev_label.config(image=self._bev_ph)
            except queue.Empty:
                pass

            # GraphML graph
            if self._graph_renderer:
                graph_img = self._graph_renderer.render(
                    self._planned_path, self._path_cursor, nearest)
                self._graph_ph = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(graph_img, cv2.COLOR_BGR2RGB)))
                self._graph_label.config(image=self._graph_ph)

            # Telemetry panel
            ctrl = self._last_ctrl
            telem_img = _draw_telemetry_panel(
                steer_deg     = ctrl.steer_angle_deg,
                speed_pwm     = ctrl.speed_pwm,
                lateral_err_px= ctrl.target_x - 320.0,
                confidence    = conf,
                anchor        = ctrl.anchor,
                zone_mode     = zone,
                upcoming_curve= getattr(self.localizer, 'upcoming_curve', 'STRAIGHT'),
                fps           = self._fps,
                sign_history  = list(self._sign_history),
                nav_state     = self._nav_state,
                w=self.CAM_W, h=self.CAM_H,
            )
            self._telem_ph = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(telem_img, cv2.COLOR_BGR2RGB)))
            self._telem_label.config(image=self._telem_ph)

        except Exception as e:
            log.debug(f"GUI update error: {e}")

        if hasattr(self, '_root') and self._root.winfo_exists():
            self._root.after(33, self._gui_update)

    # ── Pilot loop (background thread, 30 Hz) ─────────────────────────────────

    def _pilot_loop(self):
        log.info("Pilot loop started")
        t_prev = time.time()
        _lane_lost_frames = 0
        _LANE_LOST_CRAWL  = 15
        _LANE_LOST_STOP   = 90

        # SIGN-02: zone cross-validation
        _zone_mismatch_frames = 0

        try:
            while self.running:
                t_start = time.time()
                dt      = max(t_start - t_prev, 0.001)
                t_prev  = t_start

                self._fps = 0.7 * self._fps + 0.3 * (1.0 / dt)

                if self._estop:
                    self.hw.set_speed(0)
                    self.hw.set_steering(0)
                    time.sleep(0.05)
                    continue

                # ── 1. Camera ────────────────────────────────────────────────
                raw_frame = self.hw.read_camera()
                if raw_frame is None:
                    raw_frame = np.zeros((480, 640, 3), np.uint8)

                # ── 2. Velocity ──────────────────────────────────────────────
                velocity_ms = self.hw.get_velocity_ms()

                # ── 3. Unblock expired blocked nodes (SIGN-01) ───────────────
                now = time.time()
                expired = [n for n, t in self._blocked_nodes.items() if now >= t]
                for n in expired:
                    del self._blocked_nodes[n]

                # ── 4. YOLO (traffic) ────────────────────────────────────────
                if self.traffic_engine:
                    x_now, y_now, _ = self.localizer.get_pose()
                    edge_info = {}
                    if self._planned_path and self.localizer.planner:
                        edge_info = self.localizer.planner.get_current_edge_info(
                            x_now, y_now,
                            self._planned_path, self._path_cursor)
                    line_type = ("DASHED" if edge_info.get("dotted")
                                 else "CONTINUOUS")
                    t_res = self.traffic_engine.process(raw_frame, line_type)
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

                # ── 5. Record sign detections ────────────────────────────────
                for lbl in t_res.active_labels:
                    # Only record high-level signs (not every detection)
                    for keyword in ("stop", "traffic", "highway", "roundabout",
                                    "parking", "crosswalk", "priority",
                                    "no-entry", "speed"):
                        if keyword in lbl.lower():
                            self._sign_history.append((lbl, 0.9, time.time()))
                            break

                # ── SIGN-01: No-Entry reroute ────────────────────────────────
                if ("NO-ENTRY" in t_res.reason
                        and self._planned_path
                        and self.localizer.planner):
                    x_ne, y_ne, _ = self.localizer.get_pose()
                    nearest_ne = self.localizer.planner.get_nearest_node(x_ne, y_ne)
                    if nearest_ne and nearest_ne not in self._blocked_nodes:
                        log.warning(f"No-Entry: blocking node {nearest_ne}, rerouting")
                        self._blocked_nodes[nearest_ne] = time.time() + 30.0
                        # Temporarily remove + replan
                        if nearest_ne in self.localizer.planner.graph:
                            self.localizer.planner.graph.remove_node(nearest_ne)
                            new_path = self.localizer.planner.plan_route(
                                self.localizer.planner.get_nearest_node(x_ne, y_ne),
                                self._target_node)
                            # Restore node
                            # (the removed node cannot be re-added without its edges
                            # so we reload the graph from file — simplest recovery)
                            self.localizer.planner.load_graph(
                                "Competition_track_graph.graphml")
                            if new_path:
                                self._planned_path = new_path
                                self._path_cursor  = 0
                                self.localizer.reset_cursor()
                                log.info(f"Rerouted: {len(new_path)} nodes")

                # ── SIGN-02: Cross-validate zone_mode with map ────────────────
                if self.localizer.planner and self.localizer.is_initialized():
                    xz, yz, _ = self.localizer.get_pose()
                    map_zone   = self.localizer.planner.get_zone(xz, yz)
                    if map_zone != t_res.zone_mode:
                        _zone_mismatch_frames += 1
                        if _zone_mismatch_frames >= 90:   # ~3 s
                            if self.traffic_engine:
                                self.traffic_engine._zone_mode = map_zone
                            _zone_mismatch_frames = 0
                            log.info(f"Zone forced to map: {map_zone}")
                    else:
                        _zone_mismatch_frames = 0

                # ── 6. Perception ─────────────────────────────────────────────
                extra_offset = 0.0
                if t_res.state == "SYS_LANE_CHANGE_LEFT":
                    extra_offset = -80.0

                # SIGN-03: set ROUNDABOUT nav state from map node membership
                if (self._planned_path and self.localizer.planner
                        and 0 <= self._path_cursor < len(self._planned_path)):
                    node_now = self._planned_path[self._path_cursor]
                    if self.localizer.planner.is_roundabout_node(node_now):
                        if self._nav_state == "NORMAL":
                            self._nav_state = "ROUNDABOUT"
                    elif self._nav_state == "ROUNDABOUT":
                        self._nav_state = "NORMAL"

                perc = self.vision.process(
                    raw_frame,
                    extra_offset_px=extra_offset,
                    nav_state=self._nav_state)

                self._last_conf = perc.confidence   # exposed to GUI

                # ── 7. Junction detection ─────────────────────────────────────
                self._nav_state = self.jct_detector.update(
                    perc.warped_binary, perc.sl, perc.sr,
                    perc.lane_width_px, t_res.active_labels)

                if self._nav_state == "JUNCTION_PROMPT":
                    if self._planned_path and self.localizer.planner:
                        x_j, y_j, yaw_j = self.localizer.get_pose()
                        # FIX VL-06: use incremental cursor (no list.index())
                        action = self.localizer.planner.get_next_action(
                            x_j, y_j, yaw_j,
                            path=self._planned_path,
                            cursor=self._path_cursor,
                            velocity_ms=velocity_ms)
                        self._nav_state = f"JUNCTION_{action}"
                        log.info(f"Junction at cursor {self._path_cursor}: {action}")
                    else:
                        self._nav_state = "JUNCTION_STRAIGHT"

                # ── 8. Camera heading (FIX VL-02: pass raw un-negated value) ──
                cam_heading = estimate_heading_from_lanes(perc.sl, perc.sr)

                # ── 9. FIX VL-06: incremental cursor update ───────────────────
                xc, yc, _ = self.localizer.get_pose()
                self._path_cursor = self.localizer.update_cursor(
                    self._planned_path, xc, yc)

                # ── 10. Localizer update ──────────────────────────────────────
                self.localizer.update(
                    velocity_ms       = velocity_ms,
                    dt                = dt,
                    camera_heading_rad= cam_heading,
                    camera_confidence = perc.confidence,
                    path              = self._planned_path,
                    path_cursor       = self._path_cursor,
                )

                # ── 11. Upcoming curve from path ──────────────────────────────
                self.localizer.get_upcoming_curve_from_path(
                    self._planned_path, self._path_cursor, velocity_ms)

                # ── 12. Controller ────────────────────────────────────────────
                ctrl = self.controller.compute(
                    perc_res      = perc,
                    nav_state     = self._nav_state,
                    traffic_state = t_res.state,
                    base_speed    = float(self.base_speed),
                    traffic_mult  = t_res.speed_multiplier,
                    zone_mode     = t_res.zone_mode,
                    parking_state = t_res.parking_state,
                    steer_bias    = t_res.steer_bias,
                    upcoming_curve= getattr(self.localizer, 'upcoming_curve', 'STRAIGHT'),
                    visual_yaw_rate_rps = self.localizer.visual_yaw_rate,
                    velocity_ms   = velocity_ms,
                    dt            = dt,
                )
                self._last_ctrl = ctrl

                # ── 13. Lane-lost guard ───────────────────────────────────────
                if perc.sl is None and perc.sr is None:
                    _lane_lost_frames += 1
                else:
                    _lane_lost_frames = 0

                if _lane_lost_frames >= _LANE_LOST_STOP:
                    self.hw.set_speed(0)
                    self.hw.set_steering(0)
                    log.error(f"LANE LOST for {_lane_lost_frames} frames — E-STOP")
                    self._estop = True
                    continue

                # ── 14. PWM deadband + speed cap ──────────────────────────────
                speed = ctrl.speed_pwm
                if _lane_lost_frames >= _LANE_LOST_CRAWL:
                    speed = min(speed, 20.0)
                if 0.0 < speed < PWM_DEADBAND:
                    speed = PWM_DEADBAND

                # ── 15. Commands ──────────────────────────────────────────────
                self.hw.set_speed(speed)
                self.hw.set_steering(ctrl.steer_angle_deg)

                # ── 16. GUI queues ────────────────────────────────────────────
                if not self._q_yolo.full():
                    self._q_yolo.put(t_res.yolo_debug_frame)
                if not self._q_bev.full():
                    self._q_bev.put(_annotate_bev(perc, ctrl))

                # ── Frame rate throttle ───────────────────────────────────────
                elapsed    = time.time() - t_start
                sleep_time = max(0.001, FRAME_PERIOD - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            log.error(f"FATAL Pilot loop crash: {e}", exc_info=True)
            self._estop = True

        log.info("Pilot loop exited")
        self.hw.set_speed(0)
        self.hw.set_steering(0)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="BFMC v2 — IMU-Free Visual Pilot")
    ap.add_argument("--sim",   action="store_true",
                    help="Simulation mode (no hardware)")
    ap.add_argument("--speed", type=float, default=50,
                    help="Base PWM speed 0-100 (default 50)")
    ap.add_argument("--svg",   type=str, default=None,
                    help="Path to Track.svg")
    ap.add_argument("--sim-video", type=str, default=None,
                    help="Path to simulation video file")
    args = ap.parse_args()

    svg_path = args.svg or SVG_PATH_DEFAULT

    orch = Orchestrator(
        sim_mode   = args.sim,
        base_speed = args.speed,
        svg_path   = svg_path,
    )
    if args.sim_video:
        orch.hw.sim_video = args.sim_video

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