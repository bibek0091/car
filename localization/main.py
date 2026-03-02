"""
main.py — BFMC Autonomous Pilot  (v5 — SIGN-NAVIGATION DASHBOARD)
===================================================================
v5 DASHBOARD OVERHAUL:

  NEW  GraphMLMapCanvas — replaces SVG-based MapOverlayRenderer.
       Renders nodes/edges directly from PathPlanner, with sign overlays,
       A* path, car pose trail, animated milestone pulse.

  NEW  SignSequencePanel — ordered landmark list displayed on the right.
       Shows which signs lie on the current A* route, highlights the next
       milestone, displays live distance, ticks off completed signs.

  NEW  SignLocalizerBridge — tracks milestone index, fires "REACHED: <sign>"
       announcement when car is within 1.2 m of each sign, stops the car
       when all milestones are done (DESTINATION REACHED).

  NEW  _SIGN_UI — 10 BFMC sign types with colors, glyphs, and behavior strings.

  SIGN EDITOR — 10 color-coded sign-type buttons. PLACE MODE + left-click.
       Right-click removes nearest placed sign. Signs saved to sign_map.json.

  SVG support removed entirely — dashboard works with GraphML only.
  plan_route() now accepts blocked_nodes= for no-entry rerouting.

Fixes from v4 all retained (MAIN-01 to MAIN-08).
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
from PIL import Image, ImageTk

from perception    import VisionPipeline
from localization  import LocalizationEngine
from control       import Controller, ControlOutput
from hardware_io   import HardwareIO
from safety_manager import GlobalSafetyManager

try:
    from behavior_controller import BehaviorController
    _BEHAVIOR_AVAILABLE = True
except ImportError:
    _BEHAVIOR_AVAILABLE = False
    BehaviorController = None

try:
    from sign_map import SignMap, SIGN_TYPES
    _SIGNMAP_AVAILABLE = True
except ImportError:
    _SIGNMAP_AVAILABLE = False
    SignMap   = None
    SIGN_TYPES = ["stop","parking","crosswalk","priority",
                  "highway-entry","highway-exit","no-entry",
                  "roundabout","speed-limit","traffic-light"]

try:
    from traffic_module import TrafficDecisionEngine, ThreadedYOLODetector, TrafficResult
    _TRAFFIC_AVAILABLE = True
except ImportError:
    _TRAFFIC_AVAILABLE = False
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class TrafficResult:
        state: str = "SYS_GO"
        reason: str = "OK"
        speed_multiplier: float = 1.0
        active_labels: list = field(default_factory=list)
        yolo_debug_frame: object = None
        sign_approach_m: float = 99.0
        zone_mode: str = "CITY"

    class ThreadedYOLODetector:
        def __init__(self, *a, **kw): pass
        def stop(self): pass

    class TrafficDecisionEngine:
        def __init__(self, *a, **kw): pass
        def process(self, frame, *a, **kw):
            return TrafficResult(yolo_debug_frame=frame.copy())

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("main")

TARGET_FPS   = 30
FRAME_PERIOD = 1.0 / TARGET_FPS
PWM_DEADBAND = 14.0
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
# SVG removed — GraphML canvas renders the map directly
MAP_W_M = 22.0
MAP_H_M = 15.0

# BGR colours — refined neon-on-dark palette
C_CYAN  = (255, 240,  10)   # electric yellow-cyan
C_GREEN = ( 20, 230,  80)   # vivid mint green
C_AMBER = ( 30, 165, 255)   # warm amber
C_RED   = ( 40,  40, 240)   # alert red
C_WHITE = (240, 240, 245)   # off-white
C_DGRAY = ( 50,  50,  58)   # dark charcoal
C_BG    = ( 12,  12,  20)   # deep navy-black


def map_to_pixel(x_m, y_m, img_w, img_h, map_w_m=MAP_W_M, map_h_m=MAP_H_M):
    """World-map coords (x right, y up) → image pixels (origin top-left)."""
    return int(x_m / map_w_m * img_w), int((map_h_m - y_m) / map_h_m * img_h)

def pixel_to_map(px, py, img_w, img_h, map_w_m=MAP_W_M, map_h_m=MAP_H_M):
    """Image pixels → world-map coords. Matches the y-flip in map_to_pixel."""
    return px / img_w * map_w_m, (1.0 - py / img_h) * map_h_m


def push_latest(q, item):
    """Forcefully pushes the newest frame, dropping the oldest if full."""
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    q.put(item)




# ── Drawing primitives ────────────────────────────────────────────────────────

def _lbl(img, txt, x, y, scale=0.38, color=C_WHITE, t=1):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, t, cv2.LINE_AA)

def _badge(img, txt, x, y, color, w=None):
    tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
    bw = w or tw + 10
    cv2.rectangle(img, (x, y-15), (x+bw, y+4), color, -1, cv2.LINE_AA)
    cv2.putText(img, txt, (x+4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1, cv2.LINE_AA)

def _hbar(img, val, maxv, x, y, w, h, color):
    cv2.rectangle(img, (x, y), (x+w, y+h), C_DGRAY, -1)
    fill = int(np.clip(abs(val)/max(abs(maxv),1e-6), 0, 1)*w)
    cv2.rectangle(img, (x, y), (x+fill, y+h), color, -1)

def _spark(img, vals, x, y, w, h, color, scale=None):
    if len(vals) < 2: return
    arr = np.array(list(vals), dtype=float)
    if scale is None:
        mn, mx = arr.min(), arr.max(); rng = max(mx-mn, 1e-6)
    else:
        mn, mx, rng = -scale, scale, 2*scale
    cv2.rectangle(img, (x, y), (x+w, y+h), (28,28,28), -1)
    zy = y+h - int((-mn)/rng*h)
    cv2.line(img, (x, zy), (x+w, zy), (55,55,55), 1)
    pts = []
    for i, v in enumerate(arr):
        px_ = x + int(i/max(len(arr)-1,1)*w)
        py_ = y+h - int(np.clip((v-mn)/rng, 0, 1)*h)
        pts.append([px_, py_])
    cv2.polylines(img, [np.array(pts,dtype=np.int32)], False, color, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# Sign type catalogue (visual properties for dashboard)
# ══════════════════════════════════════════════════════════════════════════════

_SIGN_UI = {
    "traffic-light":  {"glyph": "TL", "color_bgr": (50,  50, 230), "hex": "#e63232", "action": "STOP_AT_RED"},
    "stop":           {"glyph": "S",  "color_bgr": (30,  30, 210), "hex": "#c0392b", "action": "STOP_3S"},
    "parking":        {"glyph": "P",  "color_bgr": (20, 130, 220), "hex": "#e67e22", "action": "PARK"},
    "crosswalk":      {"glyph": "X",  "color_bgr": (200, 190,  30), "hex": "#00ced1", "action": "SLOW"},
    "priority":       {"glyph": "!",  "color_bgr": ( 30, 180,  30), "hex": "#27ae60", "action": "PROCEED"},
    "highway-entry":  {"glyph": "H+", "color_bgr": ( 20, 140,  20), "hex": "#1e8449", "action": "HIGHWAY_ON"},
    "highway-exit":   {"glyph": "H-", "color_bgr": (180,  80,  30), "hex": "#2980b9", "action": "HIGHWAY_OFF"},
    "one-way":        {"glyph": "O",  "color_bgr": (200,  40, 160), "hex": "#8e44ad", "action": "ONEWAY"},
    "roundabout":     {"glyph": "R",  "color_bgr": (180,  60, 200), "hex": "#e91e8c", "action": "CCW"},
    "no-entry":       {"glyph": "N",  "color_bgr": ( 30,  30, 220), "hex": "#e74c3c", "action": "BLOCK"},
}

# ══════════════════════════════════════════════════════════════════════════════
# GraphMLMapCanvas  (replaces MapOverlayRenderer which needed SVG)
# Draws directly from PathPlanner.node_positions on a dark canvas.
# ══════════════════════════════════════════════════════════════════════════════

class GraphMLMapCanvas:
    """
    Renders the BFMC GraphML map as a canvas: nodes, edges, A* path, car pose,
    trail, sign overlays, and uncertainty ellipse — all without any SVG file.
    """
    TRAIL_LEN = 300

    def __init__(self, planner, w=640, h=480, map_w_m=MAP_W_M, map_h_m=MAP_H_M):
        self.planner  = planner
        self.W, self.H = w, h
        self.map_w_m  = map_w_m
        self.map_h_m  = map_h_m
        self._trail   = deque(maxlen=self.TRAIL_LEN)

        # Pre-compute node-pixel positions and build static base image
        self._px_cache: dict = {}   # node_id -> (px, py)
        self._base    = self._build_base()
        self._frame   = 0   # animation counter

    def _m2p(self, x_m, y_m):
        """Map world coords → canvas pixels (origin top-left, y flipped)."""
        return (int(x_m / self.map_w_m * self.W),
                int((self.map_h_m - y_m) / self.map_h_m * self.H))

    def _build_base(self):
        img = np.full((self.H, self.W, 3), 8, np.uint8)
        # Subtle vignette gradient
        for y in range(self.H):
            alpha = int(4 * abs(y - self.H // 2) / self.H)
            img[y, :] = np.clip(img[y, :].astype(int) + alpha, 0, 255).astype(np.uint8)

        if not self.planner or not self.planner.node_positions:
            cv2.putText(img, "No GraphML map loaded",
                        (20, self.H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 80, 120), 1, cv2.LINE_AA)
            return img

        # Cache pixel positions
        for nid, (x, y) in self.planner.node_positions.items():
            self._px_cache[nid] = self._m2p(x, y)

        # Edges — refined palette: highway teal, roundabout violet, normal indigo
        for u, v, d in self.planner.graph.edges(data=True):
            pu = self._px_cache.get(u)
            pv = self._px_cache.get(v)
            if pu and pv:
                hw = (u in self.planner._highway_nodes or v in self.planner._highway_nodes)
                rb = (u in self.planner._roundabout_nodes or v in self.planner._roundabout_nodes)
                col = ((20, 100, 60) if hw else
                       (80, 40, 90) if rb else (35, 35, 65))
                thick = 2 if hw else 1
                cv2.line(img, pu, pv, col, thick, cv2.LINE_AA)

        # Nodes — bright accent dots
        for nid, (x, y) in self.planner.node_positions.items():
            px_, py_ = self._px_cache[nid]
            col = ((0, 210, 190) if nid in self.planner._roundabout_nodes else
                   (0, 170, 255) if nid in self.planner._highway_nodes else
                   (60, 60, 110))
            r = 3 if nid in self.planner._highway_nodes else 2
            cv2.circle(img, (px_, py_), r, col, -1, cv2.LINE_AA)

        # Fine grid with dotted lines
        for xg in np.arange(0, self.map_w_m, 2.0):
            gx = int(xg / self.map_w_m * self.W)
            for gy in range(0, self.H, 6):
                cv2.line(img, (gx, gy), (gx, gy + 2), (22, 22, 38), 1)
        for yg in np.arange(0, self.map_h_m, 2.0):
            gy = int((self.map_h_m - yg) / self.map_h_m * self.H)
            for gx in range(0, self.W, 6):
                cv2.line(img, (gx, gy), (gx + 2, gy), (22, 22, 38), 1)

        # Title with accent underline
        cv2.rectangle(img, (0, 0), (self.W, 22), (15, 15, 30), -1)
        cv2.line(img, (0, 22), (self.W, 22), (0, 180, 120), 1)
        cv2.putText(img, "BFMC ARENA MAP  \u2014  GraphML",
                    (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (0, 200, 140), 1, cv2.LINE_AA)
        return img

    def add_trail_point(self, x, y):
        self._trail.append((x, y))

    def draw_map(self):
        """Re-draws the static base map (e.g., after clearing signs)."""
        self._base = self._build_base()

    def render(self, car_x, car_y, car_yaw, path, cursor, conf,
               zone, snap_miss, heading_conf, sign_map=None,
               path_signs=None, sign_milestone_idx=0):
        """Render one frame of the map canvas."""
        img = self._base.copy()
        self._frame += 1

        # ── A* path ──────────────────────────────────────────────────────────
        if path and self.planner:
            done_pts, ahead_pts = [], []
            for i, nid in enumerate(path):
                pp = self._px_cache.get(nid)
                if pp:
                    (done_pts if i <= cursor else ahead_pts).append(list(pp))
            if len(done_pts) > 1:
                cv2.polylines(img, [np.array(done_pts, np.int32)],
                              False, (20, 110, 20), 2, cv2.LINE_AA)
            if len(ahead_pts) > 1:
                n = max(len(ahead_pts), 1)
                for j in range(len(ahead_pts) - 1):
                    frac = j / n
                    # Cyan → electric blue gradient
                    col  = (int(200*(1-frac) + 255*frac),
                            int(220*(1-frac) +  80*frac),
                            int(10*(1-frac)  +   0*frac))
                    cv2.line(img,
                             tuple(ahead_pts[j]), tuple(ahead_pts[j+1]),
                             col, 2, cv2.LINE_AA)
            # Lookahead marker — bright crosshair
            la_idx = min(cursor + 6, len(path) - 1)
            la_pp  = self._px_cache.get(path[la_idx])
            if la_pp:
                cv2.drawMarker(img, la_pp, (0, 255, 160),
                               cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
                cv2.circle(img, la_pp, 8, (0, 255, 160), 1, cv2.LINE_AA)
                cv2.putText(img, f"+{la_idx - cursor}",
                            (la_pp[0]+7, la_pp[1]-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                            (0, 255, 160), 1, cv2.LINE_AA)

        # ── Trail ─────────────────────────────────────────────────────────────
        trail = list(self._trail)
        for i, (tx, ty) in enumerate(trail):
            frac  = i / max(len(trail) - 1, 1)
            alpha = int(30 + frac * 200)
            tp = self._m2p(tx, ty)
            cv2.circle(img, tp, max(1, int(1 + frac * 2)),
                       (int(alpha * .3), int(alpha * .6), alpha),
                       -1, cv2.LINE_AA)

        # ── Sign overlays ─────────────────────────────────────────────────────
        if sign_map:
            for s in sign_map.signs:
                sp = self._m2p(s['x_m'], s['y_m'])
                ui = _SIGN_UI.get(s['type'], {"glyph": "?", "color_bgr": (160,160,160)})
                col = ui['color_bgr']
                # Pulsing ring for next milestone
                is_next = (path_signs and sign_milestone_idx < len(path_signs) and
                           path_signs[sign_milestone_idx]['sign']['id'] == s['id'])
                if is_next:
                    pulse_r = 14 + (self._frame % 8)
                    cv2.circle(img, sp, pulse_r, col, 2, cv2.LINE_AA)
                cv2.circle(img, sp, 11, (15, 15, 20), -1)
                cv2.circle(img, sp, 10, col, 2, cv2.LINE_AA)
                g = ui['glyph']
                tx_ = sp[0] - (7 if len(g) > 1 else 4)
                cv2.putText(img, g, (tx_, sp[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
                # Short label below
                cv2.putText(img, s['type'][:7], (sp[0] - 18, sp[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.26, col, 1, cv2.LINE_AA)

        # ── Car pose ──────────────────────────────────────────────────────────
        cx, cy  = self._m2p(car_x, car_y)
        cx = max(8, min(self.W - 8, cx))
        cy = max(8, min(self.H - 8, cy))
        dot_col = ((0, 230, 90) if conf > 0.6 else
                   ((0, 165, 255) if conf > 0.3 else (40, 40, 240)))

        # Uncertainty ellipse — softened glow
        miss = min(snap_miss, 60)
        r_u  = 6 + int(miss * 0.8)
        uc   = ((0, 200, 80) if miss < 5 else
                (0, 130, 200) if miss < 20 else (40, 40, 230))
        # Outer glow ring
        cv2.ellipse(img, (cx, cy), (r_u + 3, max(5, r_u // 2 + 2)),
                    int(math.degrees(car_yaw)), 0, 360, (uc[0]//3, uc[1]//3, uc[2]//3), 2, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (r_u, max(4, r_u // 2)),
                    int(math.degrees(car_yaw)), 0, 360, uc, 1, cv2.LINE_AA)

        # Heading confidence arc
        if heading_conf > 0.05:
            arc_span = int(heading_conf * 180)
            cv2.ellipse(img, (cx, cy), (16, 16),
                        int(math.degrees(car_yaw) - 90), 0, arc_span,
                        (180, 180, 60), 2, cv2.LINE_AA)

        # Car dot + heading arrow — sleek neon look
        cv2.circle(img, (cx, cy), 11, (dot_col[0]//4, dot_col[1]//4, dot_col[2]//4), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 8, dot_col, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 13, dot_col, 1, cv2.LINE_AA)
        # MAIN-FIX-07: image y increases DOWN but map y increases UP.
        # cos(yaw) is correct for x; sin(yaw) must be NEGATED for image y.
        hx = cx + int(math.cos(car_yaw) * 26)
        hy = cy - int(math.sin(car_yaw) * 26)   # negate sin for image frame
        cv2.arrowedLine(img, (cx, cy), (hx, hy), (240, 240, 255), 2,
                        tipLength=0.35, line_type=cv2.LINE_AA)

        # ── Zone badge ────────────────────────────────────────────────────────
        zc = {"CITY": (20, 130, 20), "HIGHWAY": (160, 80, 10),
              "PARKING": (10, 90, 160), "ROUNDABOUT": (110, 30, 130),
              "SPEED_OVAL": (10, 120, 160)}.get(zone, (55, 55, 70))
        _badge(img, zone, self.W - 96, 22, zc, w=92)

        # ── Progress bar — premium gradient look ──────────────────────────────
        if path:
            prog = cursor / max(len(path) - 1, 1)
            bw   = int(self.W * prog)
            cv2.rectangle(img, (0, self.H - 10), (self.W, self.H), (14, 14, 22), -1)
            # Gradient fill: teal → electric green
            for px_i in range(bw):
                frac = px_i / max(bw, 1)
                col_bar = (int(0*(1-frac)), int(190*(1-frac)+230*frac), int(140*(1-frac)+50*frac))
                cv2.line(img, (px_i, self.H - 10), (px_i, self.H), col_bar, 1)
            cv2.putText(img, f"{cursor}/{len(path)}  {prog*100:.0f}%",
                        (4, self.H - 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.30, (140, 140, 160), 1, cv2.LINE_AA)

        return img


# ══════════════════════════════════════════════════════════════════════════════
# SignSequencePanel  — ordered landmark list from A* path
# ══════════════════════════════════════════════════════════════════════════════

class SignSequencePanel:
    """
    Renders the ordered sequence of signs found on the current A* path.
    Shows which sign is next, which are done, and the destination.
    """
    def __init__(self, w=280, h=480):
        self.W, self.H = w, h
        self._frame = 0

    def render(self, path_signs: list, milestone_idx: int,
               car_x: float, car_y: float, destination_reached: bool):
        self._frame += 1
        img = np.full((self.H, self.W, 3), 10, np.uint8)

        # Header — refined dark bar with accent line
        cv2.rectangle(img, (0, 0), (self.W, 36), (14, 14, 30), -1)
        cv2.line(img, (0, 36), (self.W, 36), (0, 180, 120), 1)
        cv2.putText(img, "SIGN SEQUENCE",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 200, 140), 1, cv2.LINE_AA)
        cv2.putText(img, f"  {len(path_signs)} landmarks on path",
                    (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                    (80, 80, 120), 1, cv2.LINE_AA)

        if not path_signs:
            cv2.putText(img, "No signs on this path.",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (80, 80, 120), 1, cv2.LINE_AA)
            cv2.putText(img, "Place signs on the map then",
                        (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (60, 60, 90), 1, cv2.LINE_AA)
            cv2.putText(img, "plan a route.",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (60, 60, 90), 1, cv2.LINE_AA)
            return img

        # Sign entries
        y0 = 46
        row_h = min(44, (self.H - 85) // max(len(path_signs), 1))

        for i, entry in enumerate(path_signs):
            s = entry['sign']
            ui = _SIGN_UI.get(s['type'], {"glyph": "?", "color_bgr": (160,160,160)})
            col = ui['color_bgr']

            done   = i < milestone_idx
            is_next = i == milestone_idx and not destination_reached
            y      = y0 + i * row_h

            # Row background — layered depth
            if done:
                bg = (14, 28, 14)
            elif is_next:
                bg = (20, 22, 40)
            else:
                bg = (16, 16, 28)
            cv2.rectangle(img, (4, y), (self.W - 4, y + row_h - 2), bg, -1)
            # Bottom separator
            cv2.line(img, (4, y + row_h - 2), (self.W - 4, y + row_h - 2), (30, 30, 50), 1)

            if is_next:
                # Animated neon border
                pulse = int(140 + 80 * math.sin(self._frame * 0.2))
                border_col = (0, pulse, int(pulse * 0.6))
                cv2.rectangle(img, (3, y), (self.W - 3, y + row_h - 2), border_col, 1)
                # Left accent bar
                cv2.rectangle(img, (3, y), (5, y + row_h - 2), (0, 220, 130), -1)

            # Glyph circle — premium filled/ring style
            gx, gy = 20, y + row_h // 2
            dim_col = tuple(int(c * 0.35) for c in col) if done else col
            cv2.circle(img, (gx, gy), 12, (col[0]//5, col[1]//5, col[2]//5), -1, cv2.LINE_AA)
            cv2.circle(img, (gx, gy), 11, dim_col, 1 if done else -1 if is_next else 2, cv2.LINE_AA)
            g = ui['glyph']
            cv2.putText(img, g, (gx - (7 if len(g) > 1 else 4), gy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (80, 80, 80) if done else (255, 255, 255),
                        1, cv2.LINE_AA)

            # Sign type text
            if done:
                label_col = (50, 80, 50)
            elif is_next:
                label_col = (20, 230, 130)
            else:
                label_col = (140, 140, 190)
            cv2.putText(img, s['type'].upper(), (36, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, label_col, 1, cv2.LINE_AA)

            # Distance info
            d_str = f"\u25b8  {entry['path_dist_m']:.1f} m along route"
            cv2.putText(img, d_str, (36, y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, (65, 65, 95), 1, cv2.LINE_AA)

            # Done checkmark / live distance
            if done:
                cv2.putText(img, "\u2713 DONE", (self.W - 56, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (30, 160, 60), 1, cv2.LINE_AA)
            elif is_next:
                d_live = math.hypot(s['x_m'] - car_x, s['y_m'] - car_y)
                live_col = ((0, 240, 120) if d_live < 1.5 else
                            (0, 165, 255) if d_live < 3.0 else (140, 140, 200))
                cv2.putText(img, f"{d_live:.1f}m", (self.W - 46, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, live_col, 1, cv2.LINE_AA)

        # Destination banner
        by = y0 + len(path_signs) * row_h + 8
        if destination_reached:
            cc = int(80 + 60 * abs(math.sin(self._frame * 0.15)))
            cv2.rectangle(img, (4, by), (self.W - 4, by + 40), (10, cc//5, 20), -1)
            cv2.line(img, (4, by), (self.W - 4, by), (0, 220, 80), 1)
            cv2.putText(img, "DESTINATION REACHED", (10, by + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 90), 2, cv2.LINE_AA)
            cv2.putText(img, "vehicle stopped",
                        (10, by + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (0, 160, 60), 1, cv2.LINE_AA)
        elif by + 30 < self.H:
            cv2.rectangle(img, (4, by), (self.W - 4, by + 28), (16, 16, 28), -1)
            cv2.line(img, (4, by), (self.W - 4, by), (30, 30, 55), 1)
            cv2.putText(img, "\u25a1  DESTINATION", (10, by + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (70, 70, 110), 1, cv2.LINE_AA)
            cv2.putText(img, "end of planned route",
                        (10, by + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                        (45, 45, 70), 1, cv2.LINE_AA)

        return img


# ══════════════════════════════════════════════════════════════════════════════
# SignLocalizerBridge  — milestone tracking + localization snapping
# ══════════════════════════════════════════════════════════════════════════════

class SignLocalizerBridge:
    """
    Tracks the ordered sign milestones on the current path.
    - Advances milestone pointer when car is within REACH_M of next sign
    - Announces "REACHED: <type>" on each advance
    - Signals destination_reached when all milestones passed
    """
    REACH_M = 1.2   # metres from sign centre to trigger "REACHED"

    def __init__(self):
        self.path_signs: list   = []   # from sign_map.get_signs_on_path()
        self.milestone_idx: int = 0
        self.destination_reached: bool = False
        self._last_announced:     str  = ""

    def reset(self, path_signs: list):
        self.path_signs         = path_signs
        self.milestone_idx      = 0
        self.destination_reached = False
        self._last_announced     = ""
        log.info("SignLocalizerBridge: %d signs on path", len(path_signs))

    def update(self, car_x: float, car_y: float) -> str | None:
        """
        Call every pilot-loop tick. Returns an announcement string when a sign
        is reached, otherwise None.
        """
        if not self.path_signs or self.destination_reached:
            return None
        if self.milestone_idx >= len(self.path_signs):
            self.destination_reached = True
            return "DESTINATION REACHED"

        nxt = self.path_signs[self.milestone_idx]
        s   = nxt['sign']
        d   = math.hypot(s['x_m'] - car_x, s['y_m'] - car_y)
        if d <= self.REACH_M:
            msg = f"REACHED: {s['type'].upper()}"
            if msg != self._last_announced:
                self._last_announced = msg
                log.info("SignLocalizerBridge: %s at (%.2f, %.2f)", msg, car_x, car_y)
            self.milestone_idx += 1
            if self.milestone_idx >= len(self.path_signs):
                self.destination_reached = True
                return "DESTINATION REACHED"
            return msg
        return None

    def get_action_for_next_sign(self) -> str:
        """Returns the behavior action string for the upcoming sign."""
        if not self.path_signs or self.milestone_idx >= len(self.path_signs):
            return "STRAIGHT"
        s  = self.path_signs[self.milestone_idx]['sign']
        ui = _SIGN_UI.get(s['type'], {})
        return ui.get('action', 'STRAIGHT')



# ══════════════════════════════════════════════════════════════════════════════
# VIZ-05 — GraphMLRenderer
# ══════════════════════════════════════════════════════════════════════════════

class GraphMLRenderer:
    def __init__(self, planner, canvas_w=480, canvas_h=360):
        self.planner = planner
        self.W, self.H = canvas_w, canvas_h
        self._jct_nodes = set()
        if not planner or not planner.node_positions:
            self._base = np.full((canvas_h, canvas_w, 3), 18, np.uint8)
            return
        positions = list(planner.node_positions.values())
        xs = [p[0] for p in positions]; ys = [p[1] for p in positions]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)
        for nid in planner.graph.nodes():
            if planner.is_at_junction(nid): self._jct_nodes.add(nid)
        self._base = self._draw_static()

    def _p(self, x, y):
        m = 22
        return (int((x-self.x_min)/max(self.x_max-self.x_min,0.01)*(self.W-2*m))+m,
                int((self.y_max-y)/max(self.y_max-self.y_min,0.01)*(self.H-2*m))+m)

    def _draw_static(self):
        img = np.full((self.H, self.W, 3), 18, np.uint8)
        for u, v, d in self.planner.graph.edges(data=True):
            p1 = self.planner.node_positions.get(u)
            p2 = self.planner.node_positions.get(v)
            if p1 and p2:
                cv2.line(img, self._p(*p1), self._p(*p2),
                         (38,38,50) if not d.get('dotted') else (28,28,42), 1, cv2.LINE_AA)
        for nid, (x,y) in self.planner.node_positions.items():
            col = ((160,160,30) if nid in self.planner._roundabout_nodes else
                   (30,180,180) if nid in self._jct_nodes else (55,55,70))
            cv2.circle(img, self._p(x,y), 2, col, -1)
        return img

    def render(self, path, cursor, nearest_node, velocity_ms=0.3):
        img = self._base.copy()
        if not self.planner: return img
        for i in range(len(path)-1):
            p1 = self.planner.node_positions.get(path[i])
            p2 = self.planner.node_positions.get(path[i+1])
            if not p1 or not p2: continue
            if i < cursor:
                col, thick = (30,110,30), 1
            elif i == cursor:
                col, thick = (0,210,255), 3
            else:
                frac = min((i-cursor)/max(len(path)-cursor,1), 1.0)
                col  = (30, int(190*(1-frac)), int(200*frac)); thick = 1
            cv2.line(img, self._p(*p1), self._p(*p2), col, thick, cv2.LINE_AA)

        # Lookahead circle
        la_m = max(2.5, velocity_ms*6.0)
        if 0 <= cursor < len(path):
            pos = self.planner.node_positions.get(path[cursor])
            if pos:
                ppm = (self.W-44) / max(self.x_max-self.x_min, 0.01)
                cv2.circle(img, self._p(*pos), int(la_m*ppm), (45,45,75), 1, cv2.LINE_AA)
                cv2.circle(img, self._p(*pos), 7, (200,50,200), -1, cv2.LINE_AA)

        if nearest_node and nearest_node in self.planner.node_positions:
            cv2.circle(img, self._p(*self.planner.node_positions[nearest_node]),
                       5, C_WHITE, 1, cv2.LINE_AA)

        _lbl(img, f"A* GRAPH  cursor={cursor}", 4, 14, scale=0.36, color=(110,110,140))
        _lbl(img, "[teal=rbt  cyan=jct  orange=active]",
             4, self.H-6, scale=0.28, color=(70,70,90))
        return img


# ══════════════════════════════════════════════════════════════════════════════
# VIZ-03 — Localization Panel
# ══════════════════════════════════════════════════════════════════════════════

class LocalizationPanel:
    HIST = 80

    def __init__(self, w=520, h=400):
        self.W, self.H = w, h
        self._yr_hist  = deque(maxlen=self.HIST)
        self._le_hist  = deque(maxlen=self.HIST)
        self._x_hist   = deque(maxlen=self.HIST)
        self._y_hist   = deque(maxlen=self.HIST)
        self._dr_dist  = 0.0

    def push(self, yaw_rate, lat_err, x, y, snap_active):
        self._yr_hist.append(yaw_rate)
        self._le_hist.append(lat_err)
        if len(self._x_hist) >= 1 and not snap_active:
            dx = x - (self._x_hist[-1] if self._x_hist else x)
            dy = y - (self._y_hist[-1] if self._y_hist else y)
            self._dr_dist += math.hypot(dx, dy)
        else:
            if snap_active: self._dr_dist = 0.0
        self._x_hist.append(x); self._y_hist.append(y)

    def render(self, x, y, yaw_deg, yaw_rate, heading_conf, snap_miss,
               confidence, upcoming_curve, curve_dist_m, lat_err_px,
               velocity_ms, zone, nav_state, l1, l2, l4):
        img = np.full((self.H, self.W, 3), 10, np.uint8)

        # Header — deep navy with teal accent line
        cv2.rectangle(img, (0, 0), (self.W, 32), (12, 12, 26), -1)
        cv2.line(img, (0, 32), (self.W, 32), (0, 180, 120), 1)
        cv2.putText(img, "LOCALIZATION  \u2014  LIVE TELEMETRY",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 140), 1, cv2.LINE_AA)

        # Pose readout — large, color-coded by confidence
        pc = ((0, 220, 90) if confidence > 0.6 else
              ((0, 165, 255) if confidence > 0.3 else (40, 40, 240)))
        cv2.putText(img, f"X: {x:+8.3f} m", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.68, pc, 2, cv2.LINE_AA)
        cv2.putText(img, f"Y: {y:+8.3f} m", (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.68, pc, 2, cv2.LINE_AA)
        cv2.putText(img, f"YAW: {yaw_deg:+7.1f}\u00b0", (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.68, pc, 2, cv2.LINE_AA)
        _lbl(img, f"v={velocity_ms:.3f} m/s   zone={zone}   nav={nav_state}",
             10, 142, scale=0.37, color=(130, 130, 155))

        # Yaw-rate sparkline
        _lbl(img, "YAW-RATE (rad/s)", 8, 164, scale=0.35, color=(100, 130, 200))
        _spark(img, self._yr_hist, 8, 167, self.W - 16, 50, (100, 210, 255), scale=1.2)
        yrc = ((0, 220, 90) if abs(yaw_rate) < 0.3 else
               ((0, 165, 255) if abs(yaw_rate) < 0.8 else (40, 40, 240)))
        _lbl(img, f"{yaw_rate:+.3f} r/s", self.W - 92, 208, scale=0.40, color=yrc)

        # Lateral error sparkline
        _lbl(img, "LATERAL ERROR (px)", 8, 232, scale=0.35, color=(100, 200, 130))
        _spark(img, self._le_hist, 8, 235, self.W - 16, 42, (60, 230, 100), scale=160.0)
        lec = ((0, 220, 90) if abs(lat_err_px) < 30 else
               ((0, 165, 255) if abs(lat_err_px) < 70 else (40, 40, 240)))
        _lbl(img, f"{lat_err_px:+.1f} px", self.W - 82, 269, scale=0.40, color=lec)

        # Layer badges — pill style
        _lbl(img, "LAYERS:", 8, 295, scale=0.38, color=(110, 110, 145))
        bx = 72
        for name, active in [("L1:YAW", l1), ("L2:PATH", l2), ("L3:DR", True), ("L4:SNAP", l4)]:
            ac = (0, 160, 70) if active else (30, 30, 60)
            fc = (0, 240, 110) if active else (60, 60, 90)
            tw = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0][0]
            cv2.rectangle(img, (bx - 2, 282), (bx + tw + 8, 298), ac, -1, cv2.LINE_AA)
            cv2.rectangle(img, (bx - 2, 282), (bx + tw + 8, 298), (fc[0]//2, fc[1]//2, fc[2]//2), 1, cv2.LINE_AA)
            _lbl(img, name, bx + 3, 295, scale=0.34, color=fc)
            bx += tw + 14

        # Heading confidence bar
        hcc = ((0, 220, 90) if heading_conf > 0.5 else
               ((0, 165, 255) if heading_conf > 0.25 else (40, 40, 240)))
        _lbl(img, f"HEADING CONF: {heading_conf:.2f}", 8, 318, scale=0.37, color=(170, 165, 100))
        _hbar(img, heading_conf, 1.0, 8, 321, self.W - 16, 9, hcc)

        # Map snap status
        snc = ((0, 220, 90) if snap_miss < 5 else
               ((0, 165, 255) if snap_miss < 20 else (40, 40, 240)))
        snap_txt = f"MAP SNAP: {'ACTIVE' if snap_miss < 5 else 'LOST'}  ({snap_miss} misses)"
        _lbl(img, snap_txt, 8, 348, scale=0.38, color=snc)

        # DR distance
        drc = ((0, 220, 90) if self._dr_dist < 0.3 else
               ((0, 165, 255) if self._dr_dist < 1.0 else (40, 40, 240)))
        _lbl(img, f"DR DIST: {self._dr_dist:.3f} m since snap", 8, 365, scale=0.37, color=drc)

        # Upcoming curve
        cc = ((0, 220, 90) if upcoming_curve == "STRAIGHT" else
              ((0, 165, 255) if "LEFT" in upcoming_curve else (40, 40, 240)))
        ds = f"{curve_dist_m:.1f} m" if curve_dist_m < 99 else "---"
        _lbl(img, f"CURVE: {upcoming_curve}  @ {ds}", self.W // 2, 348, scale=0.42, color=cc)

        # Confidence mini bar
        _lbl(img, f"CONF: {confidence:.2f}", self.W // 2, 365, scale=0.37, color=pc)
        _hbar(img, confidence, 1.0, self.W // 2, 368, self.W // 2 - 12, 9, pc)

        return img


# ══════════════════════════════════════════════════════════════════════════════
# VIZ-04 — Telemetry panel
# ══════════════════════════════════════════════════════════════════════════════

def _steer_gauge(img, steer, cx, cy, r):
    # Background arc fill
    cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 360, (22, 22, 36), 14)
    cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 360, (40, 40, 65), 2)
    # Tick marks
    for td in [-45, -30, -15, 0, 15, 30, 45]:
        ang = math.radians(270 - (td / 45.0) * 90)
        oi  = (cx + int(r * math.cos(ang)), cy + int(r * math.sin(ang)))
        ii  = (cx + int((r - 10) * math.cos(ang)), cy + int((r - 10) * math.sin(ang)))
        tc  = (100, 100, 130) if td != 0 else (160, 160, 190)
        cv2.line(img, ii, oi, tc, 1 if td != 0 else 2, cv2.LINE_AA)
        if td in (-45, 0, 45):
            lx = cx + int((r - 22) * math.cos(ang)) - 8
            ly = cy + int((r - 22) * math.sin(ang)) + 4
            cv2.putText(img, str(td), (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.26, (80, 80, 110), 1, cv2.LINE_AA)
    # Needle
    ang = math.radians(270 - (steer / 45.0) * 90)
    nx  = int(cx + r * math.cos(ang))
    ny  = int(cy + r * math.sin(ang))
    col = ((0, 220, 90) if abs(steer) < 15 else ((0, 165, 255) if abs(steer) < 30 else (40, 40, 240)))
    # Needle shadow
    cv2.line(img, (cx + 1, cy + 1), (nx + 1, ny + 1), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(img, (cx, cy), (nx, ny), col, 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 6, col, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 8, (30, 30, 50), 1, cv2.LINE_AA)
    _lbl(img, f"{steer:+.1f}\u00b0", cx - 26, min(cy + r + 20, img.shape[0] - 4), scale=0.46, color=col)
    _lbl(img, "STEER", cx - 20, max(cy - r - 8, 12), scale=0.37, color=(100, 100, 130))


def _draw_telemetry_panel(steer, speed_pwm, lat_err, conf, anchor,
                          zone, upcoming_curve, fps, sign_history,
                          nav_state, l_conf, r_conf, curvature,
                          velocity_ms, w=480, h=360):
    img = np.full((h, w, 3), 10, np.uint8)
    # Header
    cv2.rectangle(img, (0, 0), (w, 32), (12, 12, 26), -1)
    cv2.line(img, (0, 32), (w, 32), (0, 150, 255), 1)
    _lbl(img, "TELEMETRY", 8, 22, scale=0.52, color=(0, 170, 255), t=1)

    _steer_gauge(img, steer, cx=100, cy=110, r=80)

    # Speed / velocity
    _lbl(img, f"PWM:{speed_pwm:.0f}  v={velocity_ms:.3f} m/s", 200, 52, scale=0.38, color=(100, 165, 230))
    _hbar(img, speed_pwm, 100, 200, 55, w - 212, 11, (30, 140, 230))

    # Lateral error
    lec = ((0, 220, 90) if abs(lat_err) < 30 else ((0, 165, 255) if abs(lat_err) < 70 else (40, 40, 240)))
    _lbl(img, f"LAT ERR: {lat_err:+.1f} px", 200, 83, scale=0.38, color=lec)
    _hbar(img, lat_err, 160, 200, 86, w - 212, 11, lec)

    # Lane confidence
    cc = ((0, 220, 90) if conf > 0.6 else ((0, 165, 255) if conf > 0.3 else (40, 40, 240)))
    _lbl(img, f"LANE CONF: {conf:.2f}", 200, 113, scale=0.38, color=cc)
    _hbar(img, conf, 1.0, 200, 116, w - 212, 11, cc)

    # L/R lane confidence
    hw = (w - 212) // 2 - 4
    _lbl(img, f"L:{l_conf:.2f}", 200, 142, scale=0.34, color=(170, 80, 80))
    _hbar(img, l_conf, 1.0, 200, 145, hw, 9, (180, 70, 70))
    _lbl(img, f"R:{r_conf:.2f}", 200 + hw + 6, 142, scale=0.34, color=(80, 90, 200))
    _hbar(img, r_conf, 1.0, 200 + hw + 6, 145, hw, 9, (70, 85, 195))

    # Anchor
    ac = ((0, 220, 90) if "DUAL" in anchor else ((0, 165, 255) if "DEAD" not in anchor else (40, 40, 240)))
    _lbl(img, f"ANCHOR: {anchor}", 200, 167, scale=0.38, color=ac)

    # Curvature
    curvc = ((0, 220, 90) if curvature < 0.001 else ((0, 165, 255) if curvature < 0.003 else (40, 40, 240)))
    _lbl(img, f"CURV: {curvature:.5f}", 200, 185, scale=0.37, color=curvc)

    # FPS
    fpsc = ((0, 220, 90) if fps >= 25 else ((0, 165, 255) if fps >= 18 else (40, 40, 240)))
    _lbl(img, f"FPS: {fps:.1f}" + ("  \u26a0 LOW" if fps < 18 else ""), 200, 207, scale=0.44, color=fpsc)

    # Nav state
    nc = ((0, 220, 90) if nav_state == "NORMAL" else ((0, 165, 255) if "JUNCTION" in nav_state else (10, 240, 240)))
    _lbl(img, f"NAV: {nav_state}", 200, 227, scale=0.37, color=nc)

    # Upcoming curve
    ucc = ((0, 220, 90) if upcoming_curve == "STRAIGHT" else ((0, 165, 255) if "LEFT" in upcoming_curve else (40, 40, 240)))
    _lbl(img, f"NEXT: {upcoming_curve}", 200, 247, scale=0.42, color=ucc)

    # Detections list
    _lbl(img, "DETECTIONS:", 8, 235, scale=0.36, color=(80, 90, 125))
    # Subtle separator
    cv2.line(img, (8, 240), (185, 240), (25, 25, 45), 1)
    now_ = time.time()
    for i, (lbl, cf, ts) in enumerate(sign_history[-5:]):
        age = now_ - ts
        alpha = max(0.12, 1.0 - age / 5.0)
        c = int(200 * alpha)
        ca = int(130 * alpha)
        _lbl(img, f"{lbl}  ({age:.1f}s)", 8, 252 + i * 18, scale=0.33, color=(ca, c, ca))

    return img


# ══════════════════════════════════════════════════════════════════════════════
# VIZ-06 — BEV annotation
# ══════════════════════════════════════════════════════════════════════════════

def _annotate_bev(perc, ctrl):
    dbg = perc.lane_dbg.copy() if perc.lane_dbg is not None else np.zeros((480,640,3),np.uint8)

    def draw_poly(fit, color):
        if fit is None: return
        ys  = np.linspace(40,479,240).astype(np.float32)
        xs  = np.clip(np.polyval(fit,ys),0,639).astype(np.float32)
        pts = np.stack([xs,ys],axis=1).reshape(-1,1,2).astype(np.int32)
        cv2.polylines(dbg,[pts],False,color,3,cv2.LINE_AA)

    draw_poly(perc.sl,(255,80,80))
    draw_poly(perc.sr,(80,80,255))

    # Lane width annotation
    if perc.sl is not None and perc.sr is not None:
        lx = int(np.clip(np.polyval(perc.sl,400),0,639))
        rx = int(np.clip(np.polyval(perc.sr,400),0,639))
        cv2.line(dbg,(lx,400),(rx,400),(70,170,70),1,cv2.LINE_AA)
        _lbl(dbg,f"w={perc.lane_width_px:.0f}px",(lx+rx)//2-20,396,scale=0.34,color=(70,170,70))

    # y_eval dashed row
    yrow = int(perc.y_eval)
    yc   = C_GREEN if "DUAL" in perc.anchor else (C_AMBER if "DEAD" not in perc.anchor else C_RED)
    for xi in range(0,640,18): cv2.line(dbg,(xi,yrow),(xi+9,yrow),yc,1,cv2.LINE_AA)

    # Target dashed crosshair
    tx = max(4, min(636, int(ctrl.target_x)))
    for yi in range(360,440,12): cv2.line(dbg,(tx,yi),(tx,yi+6),(0,255,255),2,cv2.LINE_AA)
    cv2.line(dbg,(tx-12,yrow),(tx+12,yrow),(0,255,255),2,cv2.LINE_AA)

    # Curvature arc
    curv = perc.curvature
    if curv > 1e-5:
        R = min(int(1.0/curv),1400)
        if R < 700:
            sign = 1 if (perc.sl is not None and perc.sl[0]>0) else -1
            cv2.ellipse(dbg,(tx+sign*R,400),(R,R),0,84,96,(190,70,170),2,cv2.LINE_AA)

    _lbl(dbg,ctrl.anchor,10,25,scale=0.50,color=C_WHITE)
    _lbl(dbg,f"steer={ctrl.steer_angle_deg:+.1f}  la={ctrl.lookahead_px:.0f}px",10,50,scale=0.44,color=(70,225,70))
    _lbl(dbg,f"conf={perc.confidence:.2f}  curv={perc.curvature:.5f}",10,72,scale=0.37,color=(150,150,150))
    return dbg


# ══════════════════════════════════════════════════════════════════════════════
# VIZ-07 — Status bar
# ══════════════════════════════════════════════════════════════════════════════

def _status_bar(w, estop, fps, zone, nav_state, upcoming_curve, curve_dist_m,
                conf, snap_miss):
    img = np.full((34, w, 3), 12, np.uint8)
    cv2.line(img, (0, 33), (w, 33), (0, 120, 80), 1)
    _badge(img, " E-STOP " if estop else " RUNNING ",
           4, 24, (0, 0, 150) if estop else (10, 130, 40), w=82)
    _badge(img, f" {fps:.0f} fps ",
           96, 24, (10, 140, 30) if fps >= 25 else ((15, 110, 170) if fps >= 18 else (0, 0, 160)), w=66)
    zc = {"CITY": (15, 110, 20), "HIGHWAY": (120, 60, 10), "PARKING": (10, 80, 150)}.get(zone, (50, 50, 70))
    _badge(img, f" {zone} ", 172, 24, zc, w=84)
    nc = (10, 120, 30) if nav_state == "NORMAL" else ((10, 90, 165) if "JUNCTION" in nav_state else (80, 50, 15))
    _badge(img, f" {nav_state} ", 266, 24, nc, w=140)
    cd = f"{curve_dist_m:.1f}m" if curve_dist_m < 99 else "--"
    ucc = (10, 120, 30) if upcoming_curve == "STRAIGHT" else ((10, 90, 165) if "LEFT" in upcoming_curve else (10, 50, 140))
    _badge(img, f" {upcoming_curve}@{cd} ", 416, 24, ucc, w=126)
    sc = (10, 120, 30) if snap_miss < 5 else ((10, 90, 165) if snap_miss < 20 else (0, 0, 160))
    _badge(img, f" SNAP:{snap_miss} ", 552, 24, sc, w=84)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Junction detection removed — map_planner.get_next_action() handles turns
# directly from the A* path cursor. No visual fallback needed.
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    BASE_SPEED = 22    # city base speed — matches CITY_SPEED_PWM in BehaviorController
    MAP_W=600; MAP_H=440
    CAM_W=480; CAM_H=360
    LOC_W=520; LOC_H=400
    TEL_W=480; TEL_H=360

    def __init__(self, sim_mode=False, base_speed=None):
        self.sim_mode   = sim_mode
        self.base_speed = base_speed or self.BASE_SPEED
        self.running    = False
        self._estop     = False
        self._pilot_thread = None

        self.hw           = HardwareIO(sim_mode=sim_mode)
        self._start_node  = None
        self._target_node = None
        self._planned_path= []
        self._path_cursor = 0
        self._blocked_nodes = {}

        self.vision         = VisionPipeline()
        try:
            if _TRAFFIC_AVAILABLE:
                self._threaded_yolo = ThreadedYOLODetector("best.pt")
                self.traffic_engine = TrafficDecisionEngine(self._threaded_yolo)
            else: raise RuntimeError
        except Exception:
            self._threaded_yolo = None
            self.traffic_engine = TrafficDecisionEngine(None)

        # jct_detector removed — direct map cursor used instead
        self.controller     = Controller()
        self.behavior       = BehaviorController() if _BEHAVIOR_AVAILABLE else None
        self.localizer      = LocalizationEngine()

        self._fps        = 0.0
        self._nav_state  = "NORMAL"
        self._last_ctrl  = ControlOutput(0.0, 0.0, 320.0, "INIT", 200)
        self._last_perc  = None
        self._last_conf  = 0.0
        self._last_t_res = None
        self._sign_history = deque(maxlen=20)

        # Safety Manager for Fused Confidence
        self.safety_manager = GlobalSafetyManager()

        # Sign map — placed signs persist across restarts
        _sm_path = os.path.join(_SCRIPT_DIR, "sign_map.json")
        self.sign_map = SignMap(_sm_path) if _SIGNMAP_AVAILABLE else None
        self._sign_place_mode    = False    # toggled by [PLACE MODE] button
        self._selected_sign_type = None     # tk.StringVar set in build_ui
        self._sign_place_btn     = None     # reference to toggle button

        # Proximity YOLO gating
        self._YOLO_GATE_M  = 6.0   # wider gate — more margin for localization drift
        self._SLOW_SIGN_M  = 3.5   # start slowing at this distance from a sign
        self._last_snap_id = None  # ID of last sign that triggered a loc snap

        # GraphML map canvas (replaces SVG + MapOverlayRenderer)
        self._map_canvas     = None   # built in build_ui after planner is ready
        self._graph_renderer = None
        self._loc_panel      = LocalizationPanel(self.LOC_W, self.LOC_H)

        # Sign milestone tracker + localization bridge
        self._sign_bridge    = SignLocalizerBridge()
        self._path_signs     : list = []  # [{'sign':..,'node_idx':..,'path_dist_m':..}]
        self._dest_reached   = False
        self._announce_msg   = ""
        self._announce_ts    = 0.0

        # Panel sizes
        self.MAP_W = 640; self.MAP_H = 480
        self.SEQ_W = 280; self.SEQ_H = 480

        self._q_yolo  = queue.Queue(maxsize=1)
        self._q_bev   = queue.Queue(maxsize=1)
        self._q_loc   = queue.Queue(maxsize=1)

    def build_ui(self, root):
        """BFMC v5 dashboard — scrollable, works on any screen size."""
        self._root = root
        root.title("BFMC v5  \u2014  Autonomous Navigation Pilot")
        root.configure(bg="#080810")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Screen geometry ───────────────────────────────────────────────────
        SW  = root.winfo_screenwidth()
        SH  = root.winfo_screenheight()
        self._status_bar_w = max(SW, 1200)

        # Content always uses full screen width; minimum 1200 for usability.
        COL_GAP   = 2
        COL_W     = (SW - COL_GAP * 2) // 3

        # Panel height: each column stacks 2 panels.
        # Chrome (status+editor+control) ~= 140px → each panel gets half the rest.
        # Cap at 500, floor at 280 so it's always usable.
        CHROME_H  = 140
        PANEL_H   = max(280, min(500, (SH - CHROME_H) // 2))

        self.MAP_W = COL_W;  self.MAP_H = PANEL_H
        self.CAM_W = COL_W;  self.CAM_H = PANEL_H
        self.LOC_W = COL_W;  self.LOC_H = PANEL_H
        self.TEL_W = COL_W;  self.TEL_H = PANEL_H
        self.SEQ_W = COL_W;  self.SEQ_H = PANEL_H

        self._loc_panel = LocalizationPanel(self.LOC_W, self.LOC_H)

        # ── Colours ───────────────────────────────────────────────────────────
        BG      = "#080810"
        PANEL   = "#08080f"
        SIGN_BG = "#06060e"
        _ACCENT    = "#00e5a0"
        _FG_DIM    = "#444460"
        _FG_MID    = "#7878a0"
        _FG_BRIGHT = "#c8c8e8"

        # ── StringVars ────────────────────────────────────────────────────────
        self._sv_pose        = tk.StringVar(value="Pose: not set")
        self._sv_hint        = tk.StringVar(value="Click START then DESTINATION on map")
        self._sv_active_sign = tk.StringVar(value="TRAFFIC LIGHT")
        self._sv_sign_count  = tk.StringVar(
            value=f"{len(self.sign_map) if self.sign_map else 0} signs")
        self._sv_announce    = tk.StringVar(value="")

        # ═══════════════════════════════════════════════════════════════════
        # FIXED TOP CHROME — status bar + sign editor (never scrolls)
        # ═══════════════════════════════════════════════════════════════════
        top_chrome = tk.Frame(root, bg=BG)
        top_chrome.pack(fill=tk.X, side=tk.TOP)

        # Status bar
        self._sf = tk.Frame(top_chrome, bg=BG, height=26)
        self._sf.pack(fill=tk.X)
        self._sf.pack_propagate(False)
        self._sl = tk.Label(self._sf, bg=BG, anchor=tk.W)
        self._sl.pack(fill=tk.BOTH, expand=True)
        self._refresh_status_label(np.full((26, max(SW, 1200), 3), 12, np.uint8))

        # Sign editor
        se_outer = tk.Frame(top_chrome, bg=SIGN_BG)
        se_outer.pack(fill=tk.X)
        tk.Frame(se_outer, bg=_ACCENT, height=1).pack(fill=tk.X)

        ctrl_row = tk.Frame(se_outer, bg=SIGN_BG)
        ctrl_row.pack(fill=tk.X, padx=4, pady=(2, 1))

        self._sign_place_mode = False
        def _toggle_place_mode():
            self._sign_place_mode = not self._sign_place_mode
            st = self._selected_sign_type.get()
            if self._sign_place_mode:
                self._sign_place_btn.config(
                    bg="#6a0000", fg="#ffffff",
                    text="\u25cf PLACE ON  \u2190 click map",
                    relief=tk.SUNKEN, bd=1)
                self._sv_hint.set(
                    f"PLACE MODE \u2014 click map to drop [{st}]   right-click removes")
                self._map_label.config(cursor="crosshair")
            else:
                self._sign_place_btn.config(
                    bg="#0b220b", fg=_ACCENT,
                    text="\u25a1 PLACE OFF",
                    relief=tk.FLAT, bd=0)
                self._sv_hint.set("Click START \u2192 DESTINATION on map")
                self._map_label.config(cursor="arrow")

        self._sign_place_btn = tk.Button(ctrl_row,
            text="\u25a1 PLACE OFF",
            bg="#0b220b", fg=_ACCENT,
            font=("Consolas", 9, "bold"),
            relief=tk.FLAT, bd=0, padx=8, pady=2,
            activebackground="#1a4a1a", activeforeground="#00ffaa",
            command=_toggle_place_mode)
        self._sign_place_btn.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(ctrl_row, text="Sign:", bg=SIGN_BG, fg=_FG_MID,
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        tk.Label(ctrl_row, textvariable=self._sv_active_sign,
                 bg="#061510", fg="#00ffcc",
                 font=("Consolas", 8, "bold"),
                 width=14, relief=tk.FLAT, bd=0,
                 padx=4, pady=1).pack(side=tk.LEFT, padx=(2, 8))

        def _undo_sign():
            if self.sign_map and self.sign_map.remove_last():
                self._refresh_path_signs()
                self._sv_sign_count.set(f"{len(self.sign_map)} signs")
                self._sv_hint.set(f"Removed last sign ({len(self.sign_map)} remain)")
        def _clear_signs():
            if self.sign_map:
                self.sign_map.clear(); self._refresh_path_signs()
                self._sv_sign_count.set("0 signs"); self._sv_hint.set("All signs cleared")
        def _save_signs():
            if self.sign_map:
                self.sign_map.save()
                self._sv_hint.set(f"Saved {len(self.sign_map)} signs \u2192 sign_map.json")

        for lbl, bg, abg, fn in [
            ("\u21a9 Undo",  "#182030", "#253545", _undo_sign),
            ("\u2715 Clear", "#280e0e", "#401616", _clear_signs),
            ("\u25a4 Save",  "#0c1e0c", "#143014", _save_signs),
        ]:
            tk.Button(ctrl_row, text=lbl, bg=bg, fg=_FG_BRIGHT,
                      font=("Consolas", 9, "bold"),
                      relief=tk.FLAT, bd=0, padx=7, pady=2,
                      activebackground=abg, activeforeground="#fff",
                      command=fn).pack(side=tk.LEFT, padx=2)

        tk.Label(ctrl_row, textvariable=self._sv_sign_count,
                 bg=SIGN_BG, fg=_ACCENT,
                 font=("Consolas", 8)).pack(side=tk.LEFT, padx=8)
        tk.Label(ctrl_row,
                 text="\u2460 type  \u2461 place  \u2462 L-click  \u2463 R-removes",
                 bg=SIGN_BG, fg=_FG_DIM,
                 font=("Consolas", 7)).pack(side=tk.RIGHT, padx=4)

        SIGN_COLORS = {
            "traffic-light": ("#e63232","#fff"), "stop":    ("#c0392b","#fff"),
            "parking":       ("#e67e22","#fff"), "crosswalk":("#00ced1","#000"),
            "priority":      ("#27ae60","#fff"), "highway-entry":("#1e8449","#fff"),
            "highway-exit":  ("#2980b9","#fff"), "one-way": ("#8e44ad","#fff"),
            "roundabout":    ("#e91e8c","#fff"), "no-entry":("#e74c3c","#fff"),
        }
        SIGN_LABELS = {
            "traffic-light": "\U0001f6a6 TRF LIGHT", "stop":    "\U0001f6d1 STOP",
            "parking":       "\U0001f17f PARKING",   "crosswalk":"\u2b1c XWALK",
            "priority":      "\u25b2 PRIORITY",      "highway-entry":"H\u207a HWY IN",
            "highway-exit":  "H\u207b HWY OUT",      "one-way": "\u2192 ONE-WAY",
            "roundabout":    "\u21ba ROUNDABOUT",    "no-entry":"\u2296 NO ENTRY",
        }
        SIGN_DESCR = {
            "traffic-light":"Stop at RED, go on GREEN","stop":"Halt 3 s at intersection",
            "parking":"Slow, find spot & park","crosswalk":"Slow down at crossing",
            "priority":"Enter junction without stopping","highway-entry":"Switch to highway rules",
            "highway-exit":"Switch back to city rules","one-way":"Follow one-way direction",
            "roundabout":"Follow CCW roundabout rules","no-entry":"Block node, reroute",
        }
        self._selected_sign_type = tk.StringVar(value=SIGN_TYPES[0])
        self._sign_btns = {}

        def _make_select(stype):
            def _fn():
                self._selected_sign_type.set(stype)
                self._sv_active_sign.set(stype.upper().replace("-", " "))
                for t2, b2 in self._sign_btns.items():
                    nc2, fc2 = SIGN_COLORS.get(t2, ("#333","#fff"))
                    is_sel = (t2 == stype)
                    b2.config(bg="#e8e8e8" if is_sel else nc2,
                              fg="#000000" if is_sel else fc2,
                              relief=tk.SUNKEN if is_sel else tk.FLAT,
                              bd=1 if is_sel else 0)
                hint = SIGN_DESCR.get(stype, "")
                if self._sign_place_mode:
                    self._sv_hint.set(f"PLACE [{stype}] \u2014 {hint}")
                else:
                    self._sv_hint.set(f"{stype} | {hint} | enable PLACE then click map")
            return _fn

        btn_area = tk.Frame(se_outer, bg=SIGN_BG)
        btn_area.pack(fill=tk.X, padx=4, pady=(0, 2))
        sr0 = tk.Frame(btn_area, bg=SIGN_BG); sr0.pack(fill=tk.X)
        sr1 = tk.Frame(btn_area, bg=SIGN_BG); sr1.pack(fill=tk.X)
        for i, st in enumerate(SIGN_TYPES):
            nc, fc = SIGN_COLORS.get(st, ("#333","#fff"))
            lb     = SIGN_LABELS.get(st, st.upper())
            parent = sr0 if i < 5 else sr1
            b = tk.Button(parent, text=lb, bg=nc, fg=fc,
                          font=("Consolas", 8, "bold"),
                          relief=tk.FLAT, bd=0, padx=7, pady=2,
                          activebackground="#e8e8e8", activeforeground="#000",
                          command=_make_select(st))
            b.pack(side=tk.LEFT, padx=1, pady=0)
            self._sign_btns[st] = b
        _make_select(SIGN_TYPES[0])()
        tk.Frame(se_outer, bg="#1a1a2a", height=1).pack(fill=tk.X)

        # Announcement banner (hidden until triggered, sits in top chrome)
        self._announce_lbl = tk.Label(top_chrome, textvariable=self._sv_announce,
            bg="#030f06", fg="#00ff88", font=("Consolas", 10, "bold"),
            anchor=tk.CENTER, pady=2, relief=tk.FLAT)

        # ═══════════════════════════════════════════════════════════════════
        # CONTROL BAR — inside top_chrome so it is ALWAYS visible
        # ═══════════════════════════════════════════════════════════════════
        tk.Frame(top_chrome, bg="#1a2a1a", height=1).pack(fill=tk.X)
        cb = tk.Frame(top_chrome, bg="#060610", height=28)
        cb.pack(fill=tk.X)
        cb.pack_propagate(False)

        tk.Label(cb, textvariable=self._sv_pose, bg="#060610", fg="#606080",
                 font=("Consolas", 7)).pack(side=tk.LEFT, padx=6)
        tk.Label(cb, textvariable=self._sv_hint, bg="#060610", fg="#d0b040",
                 font=("Consolas", 8, "bold")).pack(side=tk.LEFT, padx=4)

        for _lbl, _bg, _abg, _fn in [
            ("\u26d4 E-STOP",        "#500000", "#800000", self._estop_cb),
            ("\u25b6 RESUME",        "#0b220b", "#183018", self._resume_cb),
            ("\u21ba RESET",         "#091830", "#102848", self._reset_route),
            ("\u25b6\u25b6 START",   "#1e0038", "#300060", self._start_pilot),
        ]:
            tk.Button(cb, text=_lbl, bg=_bg, fg="#d8d8f0",
                      font=("Consolas", 9, "bold"), relief=tk.RIDGE, bd=1,
                      padx=10, pady=2,
                      activebackground=_abg, activeforeground="#fff",
                      command=_fn).pack(side=tk.RIGHT, padx=3)

        tk.Frame(top_chrome, bg="#001a0a", height=1).pack(fill=tk.X)



        # ═══════════════════════════════════════════════════════════════════
        # SCROLLABLE CONTENT AREA
        # Canvas + scrollbars.  Mouse-wheel scrolls vertically.
        # Shift+wheel or horizontal drag scrolls horizontally.
        # ═══════════════════════════════════════════════════════════════════
        scroll_outer = tk.Frame(root, bg=BG)
        scroll_outer.pack(fill=tk.BOTH, expand=True)

        v_scroll = tk.Scrollbar(scroll_outer, orient=tk.VERTICAL,
                                bg="#141420", troughcolor="#0a0a14",
                                activebackground=_ACCENT, width=10)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        h_scroll = tk.Scrollbar(scroll_outer, orient=tk.HORIZONTAL,
                                bg="#141420", troughcolor="#0a0a14",
                                activebackground=_ACCENT, width=10)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self._scroll_canvas = tk.Canvas(
            scroll_outer, bg=BG, highlightthickness=0,
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.config(command=self._scroll_canvas.yview)
        h_scroll.config(command=self._scroll_canvas.xview)

        # ── Inner frame — all panels live here ──────────────────────────────
        inner = tk.Frame(self._scroll_canvas, bg=BG)
        _inner_id = self._scroll_canvas.create_window(
            (0, 0), window=inner, anchor=tk.NW)

        # ── Scroll helpers ────────────────────────────────────────────────────
        def _vscroll(event):
            if event.num == 4:
                self._scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self._scroll_canvas.yview_scroll(1, "units")
            else:
                self._scroll_canvas.yview_scroll(int(-event.delta / 120), "units")

        def _hscroll(event):
            if event.num == 4:
                self._scroll_canvas.xview_scroll(-1, "units")
            elif event.num == 5:
                self._scroll_canvas.xview_scroll(1, "units")
            else:
                self._scroll_canvas.xview_scroll(int(-event.delta / 120), "units")

        def _bind_scroll(widget):
            """Recursively bind mousewheel on widget and all descendants."""
            widget.bind("<MouseWheel>",       _vscroll, add="+")
            widget.bind("<Button-4>",         _vscroll, add="+")
            widget.bind("<Button-5>",         _vscroll, add="+")
            widget.bind("<Shift-MouseWheel>", _hscroll, add="+")
            widget.bind("<Shift-Button-4>",   _hscroll, add="+")
            widget.bind("<Shift-Button-5>",   _hscroll, add="+")
            for child in widget.winfo_children():
                _bind_scroll(child)

        # Bind now on root & canvas; re-bind on inner after all children exist
        for w in (root, self._scroll_canvas):
            _bind_scroll(w)

        def _update_scrollregion(*_):
            self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all"))

        def _on_canvas_resize(event):
            # Keep inner frame at least as wide as the canvas viewport
            req = inner.winfo_reqwidth()
            new_w = max(req, event.width)
            self._scroll_canvas.itemconfig(_inner_id, width=new_w)
            _update_scrollregion()

        inner.bind("<Configure>", _update_scrollregion)
        self._scroll_canvas.bind("<Configure>", _on_canvas_resize)

        # ── Panel grid inside the scrollable inner frame ─────────────────────
        def _panel_label(parent, title, color):
            """16-px accent header bar + image label — zero border waste."""
            hdr = tk.Frame(parent, bg="#0d0d1c", height=16)
            hdr.pack(fill=tk.X)
            hdr.pack_propagate(False)
            tk.Label(hdr, text=f"  {title}", bg="#0d0d1c", fg=color,
                     font=("Consolas", 7, "bold"), anchor=tk.W).pack(
                         side=tk.LEFT, fill=tk.Y)
            lbl = tk.Label(parent, bg=PANEL, cursor="arrow")
            lbl.pack()
            return lbl

        main_grid = tk.Frame(inner, bg=BG)
        main_grid.pack(anchor=tk.NW)

        # Column 0 — MAP + BEV
        col0 = tk.Frame(main_grid, bg=BG)
        col0.grid(row=0, column=0, padx=(0, COL_GAP), sticky="nw")
        self._map_label = _panel_label(
            col0, "MAP \u2014 GraphML Arena  [L-click place \u2022 R-click remove]", "#00e5ff")
        self._map_label.bind("<Button-1>", self._on_map_click)
        self._map_label.bind("<Button-3>", self._on_map_right_click)
        blank_map = np.full((self.MAP_H, self.MAP_W, 3), 8, np.uint8)
        self._map_ph = ImageTk.PhotoImage(Image.fromarray(blank_map))
        self._map_label.config(image=self._map_ph)

        self._bev_label = _panel_label(col0, "LANE VIEW \u2014 Bird's Eye", "#69ff47")
        blank_bev = np.zeros((self.CAM_H, self.CAM_W, 3), np.uint8)
        self._bev_ph = ImageTk.PhotoImage(Image.fromarray(blank_bev))
        self._bev_label.config(image=self._bev_ph)

        # Column 1 — CAMERA + LOCALIZATION
        col1 = tk.Frame(main_grid, bg=BG)
        col1.grid(row=0, column=1, padx=(0, COL_GAP), sticky="nw")
        self._yolo_label = _panel_label(col1, "CAMERA \u2014 YOLO Detection", "#ff9100")
        blank_cam = np.zeros((self.CAM_H, self.CAM_W, 3), np.uint8)
        self._yolo_ph = ImageTk.PhotoImage(Image.fromarray(blank_cam))
        self._yolo_label.config(image=self._yolo_ph)

        self._loc_label = _panel_label(col1, "LOCALIZATION ENGINE", "#cc44ff")
        blank_loc = np.full((self.LOC_H, self.LOC_W, 3), 10, np.uint8)
        self._loc_ph = ImageTk.PhotoImage(Image.fromarray(blank_loc))
        self._loc_label.config(image=self._loc_ph)

        # Column 2 — SIGN SEQUENCE + TELEMETRY
        col2 = tk.Frame(main_grid, bg=BG)
        col2.grid(row=0, column=2, sticky="nw")
        self._seq_panel = SignSequencePanel(self.SEQ_W, self.SEQ_H)
        self._seq_label = _panel_label(col2, "SIGN ROUTE SEQUENCE", "#00e5a0")
        blank_seq = np.full((self.SEQ_H, self.SEQ_W, 3), 10, np.uint8)
        self._seq_ph = ImageTk.PhotoImage(Image.fromarray(blank_seq))
        self._seq_label.config(image=self._seq_ph)

        self._telem_label = _panel_label(col2, "TELEMETRY", "#ff9100")
        blank_tel = np.full((self.TEL_H, self.TEL_W, 3), 10, np.uint8)
        self._telem_ph = ImageTk.PhotoImage(Image.fromarray(blank_tel))
        self._telem_label.config(image=self._telem_ph)

        # Bind scroll on all inner widgets now that they exist
        _bind_scroll(inner)

        # Force scrollregion after layout settles (next event-loop tick)
        root.after(100, _update_scrollregion)

        # Window: fill screen, let OS handle taskbar offset
        root.geometry(f"{SW}x{SH - 40}+0+0")

        # ═══════════════════════════════════════════════════════════════════
        # Init map canvas renderer
        # ═══════════════════════════════════════════════════════════════════
        if self.localizer.planner:
            self._map_canvas = GraphMLMapCanvas(
                self.localizer.planner, self.MAP_W, self.MAP_H)

        self._gui_update()


    def _refresh_status_label(self, img):
        ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
        self._sl.config(image=ph); self._sl.photo = ph  # keep reference

    def _on_map_click(self, event):
        # ─ Sign placement mode ───────────────────────────────────────────────
        if self._sign_place_mode and self.sign_map is not None:
            x_m, y_m = pixel_to_map(event.x, event.y, self.MAP_W, self.MAP_H)
            stype = (self._selected_sign_type.get()
                     if self._selected_sign_type else "stop")
            self.sign_map.add_sign(stype, x_m, y_m)
            count = len(self.sign_map)
            self._sv_sign_count.set(f"{count} sign{'s' if count != 1 else ''}")
            self._sv_hint.set(f"Placed {stype} @ ({x_m:.2f},{y_m:.2f})  [{count} total]")
            self._refresh_path_signs()
            return

        # ─ Normal route-planning click ───────────────────────────────────────
        if not self.localizer.planner: return
        if len(self.localizer.planner.graph.nodes) == 0:
            self._sv_hint.set("Graph empty"); return
        x_m, y_m = pixel_to_map(event.x, event.y, self.MAP_W, self.MAP_H)
        nearest   = self.localizer.planner.get_nearest_node(x_m, y_m)
        if not nearest: return

        if self._start_node is None:
            self._start_node = nearest
            self.localizer.set_pose(x_m, y_m, 0.0)
            self._sv_hint.set(f"Start: node {nearest}  → click DESTINATION")
        elif self._target_node is None:
            self._target_node = nearest
            planned = self.localizer.planner.plan_route(
                self._start_node, nearest, blocked_nodes=self._blocked_nodes)
            if not planned:
                self._sv_hint.set(f"No path to {nearest}. Try another dest.")
                self._target_node = None; return
            self._planned_path = planned; self._path_cursor = 0
            self.localizer.reset_cursor()
            self._refresh_path_signs()
            self._sv_hint.set(
                f"Route: {len(planned)} nodes, {len(self._path_signs)} signs. Press START.")
        else:
            self._target_node = nearest
            cx, cy, _ = self.localizer.get_pose()
            ns = self.localizer.planner.get_nearest_node(cx, cy)
            planned = self.localizer.planner.plan_route(
                ns, nearest, blocked_nodes=self._blocked_nodes)
            if planned:
                self._planned_path = planned; self._path_cursor = 0
                self.localizer.reset_cursor()
                self._refresh_path_signs()
                self._sv_hint.set(
                    f"Re-routed: {len(planned)} nodes, {len(self._path_signs)} signs.")

    def _on_map_right_click(self, event):
        """Right-click on map removes nearest sign within 1.5 m."""
        if self.sign_map is None: return
        x_m, y_m = pixel_to_map(event.x, event.y, self.MAP_W, self.MAP_H)
        if self.sign_map.remove_nearest(x_m, y_m, max_dist_m=1.5):
            self._sv_hint.set(f"Removed sign near ({x_m:.2f},{y_m:.2f})  [{len(self.sign_map)} remain]")
            self._refresh_path_signs()

    def _refresh_path_signs(self):
        """Recompute which signs lie on the current A* path."""
        if (self.sign_map and self._planned_path and
                self.localizer.planner and self.localizer.planner.node_positions):
            self._path_signs = self.sign_map.get_signs_on_path(
                self._planned_path,
                self.localizer.planner.node_positions,
                threshold_m=2.0)
            self._sign_bridge.reset(self._path_signs)
            self._dest_reached = False
        else:
            self._path_signs = []
        if self._sv_sign_count:
            self._sv_sign_count.set(f"{len(self.sign_map) if self.sign_map else 0} signs")


    def _gui_update(self):
        try:
            x, y, yaw  = self.localizer.get_pose()
            conf        = self._last_conf
            loc_data    = self.localizer.get_pose_for_dashboard()
            zone        = loc_data.get("zone", "CITY")
            snap_miss   = getattr(self.localizer, '_snap_miss_frames', 0)
            hconf_raw   = getattr(self.localizer, '_cam_yaw_smoothed', 0.0)

            if self.localizer.is_initialized() and self._map_canvas:
                self._map_canvas.add_trail_point(x, y)

            # ── Map canvas ──────────────────────────────────────────────────
            if self._map_canvas:
                map_img = self._map_canvas.render(
                    x, y, yaw,
                    path=self._planned_path, cursor=self._path_cursor,
                    conf=conf, zone=zone, snap_miss=snap_miss,
                    heading_conf=abs(hconf_raw) * 2.0,
                    sign_map=self.sign_map,
                    path_signs=self._path_signs,
                    sign_milestone_idx=self._sign_bridge.milestone_idx,
                )
            else:
                map_img = np.full((self.MAP_H, self.MAP_W, 3), 10, np.uint8)
                cv2.putText(map_img, "Waiting for map...",
                            (20, self.MAP_H//2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (80,80,120), 1)

            self._map_ph = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)))
            self._map_label.config(image=self._map_ph)

            # Pose readout
            if self.localizer.is_initialized():
                self._sv_pose.set(
                    f"x={x:.3f}m  y={y:.3f}m  yaw={math.degrees(yaw):.1f}°  snap={snap_miss}")

            # ── Sign sequence panel ─────────────────────────────────────────
            seq_img = self._seq_panel.render(
                self._path_signs,
                self._sign_bridge.milestone_idx,
                x, y,
                self._dest_reached)
            self._seq_ph = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(seq_img, cv2.COLOR_BGR2RGB)))
            self._seq_label.config(image=self._seq_ph)

            # ── Announcement banner ─────────────────────────────────────────
            if self._announce_msg:
                age = time.time() - self._announce_ts
                if age < 4.0:
                    self._sv_announce.set(self._announce_msg)
                    if not self._announce_lbl.winfo_ismapped():
                        self._announce_lbl.pack(fill=tk.X, padx=6, pady=0)
                else:
                    self._sv_announce.set("")
                    self._announce_lbl.pack_forget()

            # ── Camera / YOLO frame ─────────────────────────────────────────
            yi = None
            try:
                yi = self._q_yolo.get_nowait()
            except queue.Empty:
                if not self.running:
                    yi = self.hw.read_camera()

            if yi is not None and getattr(self, "CAM_W", None):
                yi = cv2.resize(yi, (self.CAM_W, self.CAM_H))
                self._yolo_ph = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(yi, cv2.COLOR_BGR2RGB)))
                self._yolo_label.config(image=self._yolo_ph)

            # ── BEV lane frame ─────────────────────────────────────────────
            bi = None
            try:
                bi = self._q_bev.get_nowait()
            except queue.Empty:
                if not self.running:
                    # BEV is usually warped, but before start we just show raw
                    bi = self.hw.read_camera()

            if bi is not None and getattr(self, "CAM_W", None):
                bi = cv2.resize(bi, (self.CAM_W, self.CAM_H))
                self._bev_ph = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(bi, cv2.COLOR_BGR2RGB)))
                self._bev_label.config(image=self._bev_ph)

            # ── Localization panel ─────────────────────────────────────────
            try:
                li = self._q_loc.get_nowait()
                li = cv2.resize(li, (self.LOC_W, self.LOC_H))
                self._loc_ph = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(li, cv2.COLOR_BGR2RGB)))
                self._loc_label.config(image=self._loc_ph)
            except queue.Empty:
                pass

            # ── Telemetry panel ────────────────────────────────────────────
            ctrl  = self._last_ctrl
            perc  = self._last_perc
            vm    = loc_data.get("speed_ms", 0.0)
            # MAIN-FIX-02: left/right lane confidence from tracker pixel counts
            l_conf_norm = min(1.0, perc.confidence) if perc else 0.0
            r_conf_norm = min(1.0, perc.confidence) if perc else 0.0
            if perc and hasattr(perc, 'sl') and hasattr(perc, 'sr'):
                # Distinguish left vs right lane visibility from anchor
                anchor_str = ctrl.anchor if ctrl else ""
                if "DUAL" in anchor_str:
                    l_conf_norm = r_conf_norm = min(1.0, perc.confidence)
                elif perc.sl is not None and perc.sr is None:
                    l_conf_norm = min(1.0, perc.confidence)
                    r_conf_norm = 0.0
                elif perc.sr is not None and perc.sl is None:
                    l_conf_norm = 0.0
                    r_conf_norm = min(1.0, perc.confidence)
            ti = _draw_telemetry_panel(
                steer         = ctrl.steer_angle_deg,
                speed_pwm     = ctrl.speed_pwm,
                lat_err       = 320.0 - ctrl.target_x,   # MAIN-FIX-08: match perception sign
                conf          = conf,
                anchor        = ctrl.anchor,
                zone          = zone,
                upcoming_curve= getattr(self.localizer, 'upcoming_curve', 'STRAIGHT'),
                fps           = self._fps,
                sign_history  = list(self._sign_history),
                nav_state     = self._nav_state,
                l_conf        = l_conf_norm,
                r_conf        = r_conf_norm,
                curvature     = perc.curvature  if perc else 0.0,
                velocity_ms   = vm,
                w=self.TEL_W, h=self.TEL_H)
            self._telem_ph = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(ti, cv2.COLOR_BGR2RGB)))
            self._telem_label.config(image=self._telem_ph)

            # ── Status bar ────────────────────────────────────────────────
            cd  = getattr(self.localizer, 'curve_dist_m', 99.0)
            uc  = getattr(self.localizer, 'upcoming_curve', 'STRAIGHT')
            _status_w = getattr(self, '_status_bar_w', 1280)
            si = _status_bar(_status_w, self._estop, self._fps, zone,
                              self._nav_state, uc, cd, conf, snap_miss)
            self._refresh_status_label(si)

            # Sign count refresh
            if self._sv_sign_count and self.sign_map is not None:
                self._sv_sign_count.set(f"{len(self.sign_map)} signs")

        except Exception as e:
            log.debug("GUI update error: %s", e)

        if hasattr(self, '_root') and self._root.winfo_exists():
            self._root.after(33, self._gui_update)


    def _pilot_loop(self):
        log.info("Pilot loop started")
        startup_time = time.time()   # reference for calibration phases
        t_prev = time.time()
        _ll = 0; _LLC = 15; _LLS = 90; _zmf = 0
        # low resolution flag to shed load if FPS drops
        self._low_res_mode = False

        try:
            while self.running:
                ts = time.time()
                dt = max(ts-t_prev, 0.001)
                t_prev = ts
                elapsed_run = ts - startup_time

                # Auto-Scaling FPS Shield: Downsample if we fell behind last frame
                # (e.g. if the previous frame took too long to process)
                if dt > 0.060: # 60ms is 16.6 FPS, 0.060 is 16.6 FPS
                    if not self._low_res_mode:
                        log.warning("FPS drop detected (dt=%.3f), entering low-res mode", dt)
                    self._low_res_mode = True
                elif dt < 0.040: # 40ms is 25 FPS
                    if self._low_res_mode:
                        log.info("FPS recovered (dt=%.3f), exiting low-res mode", dt)
                    self._low_res_mode = False

                # Always read camera & velocity so dashboard stays live
                raw_frame = self.hw.read_camera()
                velocity_ms = self.hw.get_velocity_ms()

                # --- EXTRACT PREDICTIVE MAP DATA (available even during E-STOP) ---
                upcoming_curve = getattr(self.localizer, 'upcoming_curve', 'STRAIGHT')
                curve_dist_m   = getattr(self.localizer, 'curve_dist_m',   99.0)
                map_curvature  = 0.0
                if self._planned_path and self.localizer.planner:
                    try:
                        map_curvature = self.localizer.planner.get_path_curvature(
                            self.localizer.x, self.localizer.y,
                            self._planned_path,
                            cursor=self._path_cursor,
                            window_m=1.0)
                    except Exception:
                        pass

                if not self._estop:
                    self._fps = 0.7*self._fps + 0.3*(1.0/dt)

                # --- PROXIMITY-GATED YOLO (saves CPU, focuses detection) ---
                # When signs have been placed on the map, only run full YOLO
                # inference when the car is within YOLO_GATE_M of a sign.
                # If no signs placed, run YOLO unconditionally (no map yet).
                if self.traffic_engine:
                    x0, y0, _ = self.localizer.get_pose()
                    ei = {}
                    if self._planned_path and self.localizer.planner:
                        ei = self.localizer.planner.get_current_edge_info(
                            x0, y0, self._planned_path, self._path_cursor)

                    run_yolo = True
                    nearby_signs = []
                    if self.sign_map and self.sign_map.signs:
                        # Only gate YOLO if we trust our position estimate.
                        # If localization is uninitialized or low-confidence,
                        # the position could be wrong → always run YOLO to
                        # avoid missing signs.
                        # Run YOLO unconditionally to ensure signs are never missed,
                        # regardless of whether they have been placed on the map.
                        run_yolo = True

                    if run_yolo:
                        t_res = self.traffic_engine.process(
                            raw_frame, "DASHED" if ei.get("dotted") else "CONTINUOUS")
                    else:
                        # Reuse a minimal neutral result to keep dashboard alive
                        t_res = TrafficResult(
                            state="SYS_GO", reason="SIGN_GATE_INACTIVE",
                            speed_multiplier=1.0,
                            yolo_debug_frame=raw_frame.copy())
                else:
                    x0, y0 = 0.0, 0.0
                    nearby_signs = []
                    t_res = TrafficResult(yolo_debug_frame=raw_frame.copy())

                # --- SIGN-TRIGGERED LOCALIZATION SNAP ---
                if self.sign_map and t_res.active_labels:
                    x_snap, y_snap, _ = self.localizer.get_pose()
                    snap_found = False
                    for lbl in t_res.active_labels:
                        matched = self.sign_map.match_detection(
                            lbl, x_snap, y_snap, radius_m=3.0)
                        if matched:
                            if matched["id"] != self._last_snap_id:
                                _, _, yaw_now = self.localizer.get_pose()
                                d_obs = min(t_res.sign_approach_m, 4.0)
                                d_est = math.hypot(matched["x_m"] - x_snap, matched["y_m"] - y_snap)
                                shift_m = d_est - d_obs
                                true_x = x_snap + shift_m * math.cos(yaw_now)
                                true_y = y_snap + shift_m * math.sin(yaw_now)
                                self.localizer.set_pose(true_x, true_y, yaw_now)
                                self._last_snap_id = matched["id"]
                                log.info("SIGN SNAP: %s → shifted %+.2fm", lbl, shift_m)
                            snap_found = True
                            break  # Found a match — stop searching
                    if not snap_found:
                        # MAIN-FIX-05: only clear lock when NO labels matched anything
                        self._last_snap_id = None

                self._last_t_res = t_res

                # ── Sign milestone check ───────────────────────────────────
                _cx0, _cy0, _ = self.localizer.get_pose()
                _msg = self._sign_bridge.update(_cx0, _cy0)
                if _msg:
                    self._announce_msg = _msg
                    self._announce_ts  = time.time()
                    self._sign_history.append((_msg, 1.0, time.time()))
                    if self._sign_bridge.destination_reached:
                        self._dest_reached = True
                        self.hw.set_speed(0)
                        self.hw.set_steering(0)
                        self._estop = True
                        log.info("DESTINATION REACHED — stopping")


                if self._estop:
                    # Halted — keep motors off, reuse last perception for dashboard
                    self.hw.set_speed(0); self.hw.set_steering(0)
                    perc = self._last_perc if self._last_perc else self.vision.process(raw_frame)
                    ctrl = self._last_ctrl

                    # F-15: Auto-recovery — attempt re-detection every loop tick.
                    # If lanes are visible again, clear E-STOP and resume driving.
                    try:
                        recovery_perc = self.vision.process(raw_frame, dt=dt)
                        if recovery_perc.confidence > 0.4 and not self._dest_reached:
                            self._estop = False
                            _ll = 0
                            log.info("F-15: E-STOP cleared — lanes re-detected (conf=%.2f)", recovery_perc.confidence)
                    except Exception:
                        pass

                else:
                    # --- NORMAL DRIVING ---
                    now = time.time()
                    for n,t in list(self._blocked_nodes.items()):
                        if now>=t: del self._blocked_nodes[n]

                    for lbl in t_res.active_labels:
                        for kw in ("stop","traffic","highway","roundabout","parking",
                                   "crosswalk","priority","no-entry","speed"):
                            if kw in lbl.lower():
                                self._sign_history.append((lbl,0.9,time.time())); break

                    if ("NO-ENTRY" in t_res.reason and self._planned_path and self.localizer.planner):
                        xne,yne,_ = self.localizer.get_pose()
                        nn = self.localizer.planner.get_nearest_node(xne,yne)
                        if nn and nn not in self._blocked_nodes:
                            # Fix-8: block node via cost-gating only — no graph reload.
                            # plan_route() will skip blocked nodes (infinite cost).
                            self._blocked_nodes[nn] = time.time() + 30.0
                            log.warning("NO-ENTRY: blocking node %s for 30 s", nn)
                            ns2 = self.localizer.planner.get_nearest_node(xne, yne)
                            np2 = self.localizer.planner.plan_route(
                                ns2, self._target_node,
                                blocked_nodes=self._blocked_nodes)
                            if np2:
                                self._planned_path = np2
                                self._path_cursor  = 0
                                self.localizer.reset_cursor()

                    if self.localizer.planner and self.localizer.is_initialized():
                        xz,yz,_ = self.localizer.get_pose()
                        mz = self.localizer.planner.get_zone(xz,yz)
                        _zmf = _zmf+1 if mz!=t_res.zone_mode else 0
                        if _zmf>=90 and self.traffic_engine:
                            self.traffic_engine._zone_mode=mz; _zmf=0

                    extra_offset = -80.0 if t_res.state=="SYS_LANE_CHANGE_LEFT" else 0.0

                    if (self._planned_path and self.localizer.planner
                            and 0<=self._path_cursor<len(self._planned_path)):
                        node_now = self._planned_path[self._path_cursor]
                        if self.localizer.planner.is_roundabout_node(node_now):
                            if self._nav_state=="NORMAL": self._nav_state="ROUNDABOUT"
                        elif self._nav_state=="ROUNDABOUT": self._nav_state="NORMAL"

                    # Fix-5: pitch proxy from velocity derivative (no IMU needed)
                    _accel_proxy = (velocity_ms - getattr(self, '_v_prev', velocity_ms)) / max(dt, 0.001)
                    self._v_prev = velocity_ms
                    _pitch_rad = float(max(-0.15, min(0.15, -_accel_proxy * 0.08)))

                    # Fix-6: Tunnel dark-mode shortcut — skip lane detection entirely
                    # when the warped frame is almost completely dark (mean_L < 40).
                    _raw_lab = None
                    try:
                        import cv2 as _cv2
                        _raw_bgr = raw_frame if raw_frame.shape[:2] == (480, 640) else _cv2.resize(raw_frame, (640, 480))
                        _raw_lab = _cv2.cvtColor(_raw_bgr, _cv2.COLOR_BGR2LAB)
                        _mean_l  = float(_raw_lab[:, :, 0].mean())
                    except Exception:
                        _mean_l = 128.0

                    if _mean_l < 40.0:
                        # TUNNEL_ENTRY: use last known perception + extend dead-reckoning
                        perc = self._last_perc if self._last_perc else self.vision.process(
                            raw_frame, dt=dt,
                            nav_state=self._nav_state, velocity_ms=velocity_ms,
                            last_steering=getattr(self._last_ctrl,'steer_angle_deg',0.0),
                            upcoming_curve=getattr(self.localizer,'upcoming_curve','STRAIGHT'),
                            pitch_rad=_pitch_rad)
                        if hasattr(perc, 'confidence'):
                            object.__setattr__(perc, 'confidence', 0.0)  # mark as lost
                        if self._nav_state not in ('TUNNEL_ENTRY',):
                            self._nav_state = 'TUNNEL_ENTRY'
                            log.info("Fix-6: TUNNEL_ENTRY — darkness detected (mean_L=%.1f)", _mean_l)
                    else:
                        if self._nav_state == 'TUNNEL_ENTRY':
                            self._nav_state = 'NORMAL'
                            log.info("Fix-6: TUNNEL_EXIT — lanes resuming")
                        
                        perc = self.vision.process(
                            raw_frame,
                            dt=dt,
                            extra_offset_px=extra_offset,
                            nav_state=self._nav_state,
                            velocity_ms=velocity_ms,
                            last_steering=getattr(self._last_ctrl,'steer_angle_deg',0.0),
                            upcoming_curve=getattr(self.localizer,'upcoming_curve','STRAIGHT'),
                            pitch_rad=_pitch_rad)
                    
                    self._last_conf = perc.confidence
                    self._last_perc = perc

                    # --- PATH-ACTION & NEARBY SIGNS ---
                    nearby_signs = []
                    if self.sign_map:
                        sx, sy, _ = self.localizer.get_pose()
                        nearby_signs = self.sign_map.get_nearby_signs(sx, sy, radius_m=2.0)

                    # Fix-2: direct map-cursor action — JunctionDetector removed.
                    map_action = "STRAIGHT"
                    if self._planned_path and self.localizer.planner:
                        _cur_x, _cur_y, _cur_yaw = self.localizer.get_pose()
                        map_action = self.localizer.planner.get_next_action(
                            _cur_x, _cur_y, _cur_yaw,
                            path=self._planned_path,
                            cursor=self._path_cursor, velocity_ms=velocity_ms)
                        if map_action not in ("STRAIGHT", ""):
                            self._nav_state = f"JUNCTION_{map_action}"
                        elif self._nav_state.startswith("JUNCTION_"):
                            self._nav_state = "NORMAL"

                    self.localizer.update(
                        velocity_ms=velocity_ms, dt=dt,
                        camera_heading_rad=perc.heading_rad,
                        camera_confidence=perc.confidence,
                        heading_conf=perc.heading_conf,
                        path=self._planned_path,
                        optical_yaw_rate=perc.optical_yaw_rate,
                        optical_vel=perc.optical_vel)
                    self._path_cursor = self.localizer.path_cursor

                    self.localizer.get_upcoming_curve_from_path(
                        self._planned_path,self._path_cursor,velocity_ms)

                    ctrl = self.controller.compute(
                        perc_res=perc, nav_state=self._nav_state,
                        base_speed=float(self.base_speed),
                        traffic_mult=t_res.speed_multiplier,
                        velocity_ms=velocity_ms, dt=dt,
                        map_curvature=map_curvature,
                        upcoming_curve=upcoming_curve,
                        curve_dist_m=curve_dist_m)

                    # --- AUTO-SLOW NEAR PLACED SIGNS ---
                    # Smoothly reduce speed when car is approaching any mapped sign.
                    # Linear ramp: full speed at SLOW_SIGN_M, 50% at 0 m (sign centre).
                    # This is independent of YOLO — works from map distance alone.
                    if nearby_signs and ctrl.speed_pwm > 0:
                        closest_dist = nearby_signs[0]["dist"]   # already sorted
                        if closest_dist < self._SLOW_SIGN_M:
                            slow_mult = max(0.50, closest_dist / self._SLOW_SIGN_M)
                            ctrl.speed_pwm *= slow_mult
                            log.debug("SIGN SLOW: %.2fm → %.0f%% speed",
                                      closest_dist, slow_mult * 100)


                    # BehaviorController evaluates the full priority hierarchy
                    # (Emergency > Mandatory > Legal > Mission > Normal).
                    # If any layer fires above NORMAL priority it overrides
                    # the Stanley controller's speed and steer output.
                    _loc_x_beh, _loc_y_beh = 0.0, 0.0
                    try:
                        _loc_x_beh, _loc_y_beh, _ = self.localizer.get_pose()
                    except Exception:
                        pass

                    if self.behavior:
                        beh = self.behavior.compute(
                            perc_res=perc,
                            t_res=t_res,
                            dt=dt,
                            base_steer=ctrl.steer_angle_deg,
                            planner=self.localizer.planner if self.localizer else None,
                            map_action=map_action,
                            cursor=self._path_cursor,
                            path=self._planned_path,
                            loc_x=_loc_x_beh,
                            loc_y=_loc_y_beh,
                        )

                        if beh.priority < self.behavior.PRI_NORMAL:
                            # Higher-priority command wins
                            ctrl.speed_pwm       = beh.speed_pwm
                            ctrl.steer_angle_deg = beh.steer_deg
                            log.debug("BEH[%d] %s: %s",
                                      beh.priority, beh.state, beh.reason)

                    # --- GLOBAL SAFETY ENFORCEMENT ---
                    snap_miss_ratio = getattr(self.localizer, '_snap_miss_frames', 0)
                    sign_in_range   = (len(nearby_signs) > 0)
                    yolo_active     = (len(t_res.active_labels) > 0)

                    # --- DYNAMIC REPLANNING TRIGGER ---
                    # If localizer is utterly lost for 1 full second (30 frames), force a graphical re-route
                    if snap_miss_ratio > 30 and self._planned_path and self.localizer.planner:
                        log.warning("REPLAN: Lost map snap for >30 frames. Forcing A* recalculation.")
                        _rx, _ry, _ = self.localizer.get_pose()
                        _nearest = self.localizer.planner.get_nearest_node(_rx, _ry)
                        if _nearest and _nearest != self._target_node:
                            _new_plan = self.localizer.planner.plan_route(
                                _nearest, self._target_node, blocked_nodes=self._blocked_nodes)
                            if _new_plan:
                                self._planned_path = _new_plan
                                self._path_cursor = 0
                                self.localizer.reset_cursor()
                                self.localizer._snap_miss_frames = 0
                                log.info("REPLAN SUCCESS: New route from %s to %s.", _nearest, self._target_node)
                            else:
                                log.error("REPLAN FAILED: Cannot find path from %s.", _nearest)

                    global_conf = self.safety_manager.update(
                        lane_conf=perc.confidence,
                        loc_conf=self.localizer.confidence,
                        yolo_active=yolo_active,
                        snap_success=(self._last_snap_id is not None),
                        sign_in_range=sign_in_range
                    )

                    # Clamp speed based on fused global confidence
                    safe_speed = self.safety_manager.apply_speed_limits(ctrl.speed_pwm, global_conf)
                    
                    # Apply Localization Hardening limits
                    loc_pos_var = getattr(self.localizer, 'pos_var', 0.0)
                    loc_slip    = getattr(self.localizer, 'wheel_slip', False)
                    
                    if loc_slip:
                        log.warning("SAFETY [HARDENING]: Wheel slip detected. Forcing halt.")
                        safe_speed = 0.0
                    elif loc_pos_var > 1.0:
                        log.warning("SAFETY [HARDENING]: Position covariance too high (%.2f). Halving speed.", loc_pos_var)
                        safe_speed = min(safe_speed, ctrl.speed_pwm * 0.5)

                    if safe_speed < ctrl.speed_pwm and not loc_slip and loc_pos_var <= 1.0:
                        if safe_speed == 0.0:
                            log.warning("SAFETY: Low confidence (%.2f). Force-stopping car.", global_conf)
                        else:
                            log.info("SAFETY: Marginal confidence (%.2f). Halving speed (%.1f -> %.1f).", 
                                     global_conf, ctrl.speed_pwm, safe_speed)
                                     
                    ctrl.speed_pwm = safe_speed

                    # --- STARTUP CALIBRATION OVERRIDE ---
                    # Stage 1 (0-3 s): hold stationary — let AE/AWB settle.
                    # Stage 2 (3-6 s): crawl at ≤15 PWM — warm up EMA lane tracker.
                    if elapsed_run < 3.0:
                        ctrl.speed_pwm       = 0.0
                        ctrl.steer_angle_deg = 0.0
                        cv2.putText(perc.lane_dbg,
                            f"CAM CALIB: {3.0 - elapsed_run:.1f}s",
                            (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    elif elapsed_run < 6.0:
                        ctrl.speed_pwm = min(ctrl.speed_pwm, 17.0)
                        cv2.putText(perc.lane_dbg,
                            f"LANE CALIB: {6.0 - elapsed_run:.1f}s",
                            (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

                    self._last_ctrl = ctrl

                    _ll = _ll+1 if (perc.sl is None and perc.sr is None) else 0

                    if _ll >= _LLS and elapsed_run > 6.0:
                        self.hw.set_speed(0); self.hw.set_steering(0)
                        self._estop=True
                    else:
                        speed = ctrl.speed_pwm
                        if _ll>=_LLC: speed = min(speed,20.0)
                        if 0.0<speed<PWM_DEADBAND: speed=PWM_DEADBAND
                        _is_hwy = (self._nav_state == "HIGHWAY")
                        self.hw.set_speed(speed, highway_mode=_is_hwy)
                        self.hw.set_steering(ctrl.steer_angle_deg)

                # --- DASHBOARD TELEMETRY (runs always, even in E-STOP) ---
                yolo_frame_to_push = t_res.yolo_debug_frame if getattr(t_res, 'yolo_debug_frame', None) is not None else raw_frame
                push_latest(self._q_yolo, yolo_frame_to_push)
                
                # perc and ctrl might be None if estop triggered on very first frame, use fallback
                _perc_to_draw = perc if perc else getattr(self, '_last_perc', None)
                _ctrl_to_draw = ctrl if ctrl else getattr(self, '_last_ctrl', None)
                if _perc_to_draw and _ctrl_to_draw:
                    push_latest(self._q_bev, _annotate_bev(_perc_to_draw, _ctrl_to_draw))
                else:
                    push_latest(self._q_bev, raw_frame)

                # VIZ-03: localization panel
                sm     = getattr(self.localizer,'_snap_miss_frames',0)
                lx,ly,lyaw = self.localizer.get_pose()
                yr     = self.localizer.visual_yaw_rate
                
                _p_conf  = perc.confidence if perc else 0.0
                _p_hconf = perc.heading_conf if perc else 0.0
                _p_lat   = perc.lateral_error_px if perc else 0.0

                self._loc_panel.push(yr, _p_lat, lx, ly, sm==0)
                loc_img = self._loc_panel.render(
                    x=lx, y=ly, yaw_deg=math.degrees(lyaw),
                    yaw_rate=yr, heading_conf=_p_hconf,
                    snap_miss=sm, confidence=_p_conf,
                    upcoming_curve=getattr(self.localizer,'upcoming_curve','STRAIGHT'),
                    curve_dist_m=getattr(self.localizer,'curve_dist_m',99.0),
                    lat_err_px=_p_lat, velocity_ms=velocity_ms,
                    zone=self.localizer.current_zone, nav_state=self._nav_state,
                    l1=(_p_conf>0.3 and _p_hconf>=0.35),
                    l2=bool(self._planned_path and _p_conf>0.5),
                    l4=sm<5)
                push_latest(self._q_loc, loc_img)

                elapsed = time.time()-ts
                
                # FPS Monitor & Auto-Scaling
                # If loop takes > 60ms (<16 FPS), shed resolution next tick
                if elapsed > 0.060 and not self._low_res_mode:
                    log.warning("MAIN LOOP WARNING: Latency spiked to %.0f ms. Engaging low-res mode.", elapsed * 1000)
                    self._low_res_mode = True
                elif elapsed < 0.025 and self._low_res_mode:
                    log.info("MAIN LOOP RECOVERY: Latency returned to %.0f ms. Restoring high-res mode.", elapsed * 1000)
                    self._low_res_mode = False

                time.sleep(max(0.001, FRAME_PERIOD-elapsed))

        except Exception as e:
            log.critical(f"FATAL EXCEPTION SHIELD: Pilot crashed due to {e}. Halting.", exc_info=True)
            self._estop = True
            
        finally:
            log.info("Pilot loop exited")
            self.hw.set_speed(0)
            self.hw.set_steering(0)


    def _start_pilot(self):
        if self.running: return
        if not self.localizer.is_initialized():
            self._sv_hint.set("Set start position on map first!"); return
        self.running=True; self._estop=False
        self._pilot_thread = threading.Thread(target=self._pilot_loop,daemon=True,name="pilot")
        self._pilot_thread.start(); self._sv_hint.set("Pilot running…")

    def _estop_cb(self):
        self._estop=True; self.hw.set_speed(0); self.hw.set_steering(0)
        self._sv_hint.set("E-STOP engaged")

    def _resume_cb(self):
        if not self.running: self._start_pilot()
        else: self._estop=False; self._sv_hint.set("Resumed")

    def _reset_route(self, event=None):
        self._start_node  = None
        self._target_node = None
        self._planned_path= []
        self._path_cursor = 0
        if self._map_canvas:
            self._map_canvas._trail.clear()
            self._map_canvas.draw_map()
        self.localizer.reset_cursor()
        self._sv_hint.set("Route cleared. Click two nodes to plan.")
        log.info("Route reset by user.")

    def _on_close(self):
        self.running=False; self._estop=True
        self.hw.set_speed(0); self.hw.set_steering(0)
        time.sleep(0.15); self.hw.shutdown()
        if self._threaded_yolo: self._threaded_yolo.stop()
        if hasattr(self,'_root'): self._root.destroy()


def main():
    ap = argparse.ArgumentParser(description="BFMC v5 — Sign-Navigation Pilot")
    ap.add_argument("--sim",       action="store_true")
    ap.add_argument("--speed",     type=float, default=50)
    ap.add_argument("--svg",       type=str,   default=None,
                    help="(ignored in v5 — SVG replaced by GraphML canvas)")
    ap.add_argument("--sim-video", type=str,   default=None)
    args = ap.parse_args()

    orch = Orchestrator(sim_mode=args.sim, base_speed=args.speed)
    if args.sim_video:
        orch.hw.sim_video = args.sim_video
        orch.hw.video_cap = cv2.VideoCapture(args.sim_video)

    root = tk.Tk()
    orch.build_ui(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        orch._on_close()

if __name__ == "__main__":
    main()