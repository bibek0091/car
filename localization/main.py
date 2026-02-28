"""
main.py — BFMC Single-Window Autonomous Pilot  (v4 — VISUALIZATION OVERHAUL)
=============================================================================
VISUALIZATION UPGRADES in v4:

  VIZ-01  LIVE POSE TRAIL on SVG map: rolling deque of last 200 (x,y) drawn
          as a colour-gradient trail (bright cyan=recent, faded blue=old).

  VIZ-02  UNCERTAINTY ELLIPSE around car dot: radius proportional to
          snap_miss_frames. Green=snapped, amber=drifting, red=lost.

  VIZ-03  DEDICATED LOCALIZATION PANEL (520×400):
          - Large X/Y/YAW readout with quality colour coding
          - Yaw-rate sparkline (80 frames)
          - Lateral error sparkline
          - Layer status badges (L1=yaw-rate, L2=path-nudge, L3=dead-reck, L4=snap)
          - Heading confidence bar
          - Snap status + miss count
          - Dead-reckoning distance since last snap
          - Upcoming curve + distance indicator

  VIZ-04  ENHANCED TELEMETRY PANEL:
          - 180° semicircle steering gauge with tick marks at 0,±15,±30,±45°
          - Speed shows PWM + m/s side by side
          - Separate L/R lane confidence bars
          - Curvature display + nav state

  VIZ-05  GRAPHML PANEL UPGRADED:
          - Roundabout nodes=teal, junction nodes=yellow
          - Active segment highlighted bright orange (width=3)
          - Lookahead circle radius drawn around cursor node
          - Gradient: green=done, orange→red=ahead

  VIZ-06  BEV PANEL UPGRADED:
          - Dashed target crosshair
          - y_eval row pulsed with anchor colour
          - Curvature arc drawn in pink
          - Lane-width dimension annotation

  VIZ-07  STATUS BAR across full window top:
          - E-STOP/RUNNING, FPS, Zone, NavState, UpcomingCurve, SnapMiss

  VIZ-08  MAP heading-confidence arc + numbered lookahead waypoint label

Fixes from v3 all retained (MAIN-01 to MAIN-05).
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
        state:               str   = "SYS_GO"
        reason:              str   = "NO TRAFFIC MODULE"
        speed_multiplier:    float = 1.0
        zone_mode:           str   = "CITY"
        parking_state:       str   = "NONE"
        steer_bias:          float = 0.0
        pedestrian_blocking: bool  = False
        light_status:        str   = "NONE"
        active_labels:       List  = field(default_factory=list)
        yolo_debug_frame:    object = None

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
SVG_PATH_DEFAULT = os.path.join(_SCRIPT_DIR, "..", "Track.svg")
MAP_W_M = 22.0
MAP_H_M = 15.0

# BGR colours
C_CYAN  = (255, 230,   0)
C_GREEN = ( 50, 220,  50)
C_AMBER = ( 50, 190, 255)
C_RED   = ( 50,  50, 230)
C_WHITE = (230, 230, 230)
C_DGRAY = ( 55,  55,  55)
C_BG    = ( 18,  18,  18)


def map_to_pixel(x_m, y_m, img_w, img_h, map_w_m=MAP_W_M, map_h_m=MAP_H_M):
    return int(x_m / map_w_m * img_w), int((map_h_m - y_m) / map_h_m * img_h)

def pixel_to_map(px, py, img_w, img_h, map_w_m=MAP_W_M, map_h_m=MAP_H_M):
    return px / img_w * map_w_m, (1.0 - py / img_h) * map_h_m


def push_latest(q, item):
    """Forcefully pushes the newest frame, dropping the oldest if full."""
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    q.put(item)


def _load_svg_as_cv2(svg_path, display_w=600, display_h=440):
    try:
        import cairosvg
        arr = np.frombuffer(
            cairosvg.svg2png(url=svg_path, output_width=display_w, output_height=display_h),
            np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None: return img
    except Exception: pass
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        img = renderPM.drawToPIL(svg2rlg(svg_path))
        img = img.convert("RGB").resize((display_w, display_h), Image.LANCZOS)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception: pass
    blank = np.full((display_h, display_w, 3), 30, np.uint8)
    cv2.putText(blank, "SVG unavailable — pip install cairosvg",
                (12, display_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
    return blank


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
# VIZ-01/02/08 — MapOverlayRenderer
# ══════════════════════════════════════════════════════════════════════════════

class MapOverlayRenderer:
    TRAIL_LEN = 200

    def __init__(self, svg_bgr, map_w_m=MAP_W_M, map_h_m=MAP_H_M):
        self.base = svg_bgr.copy()
        self.w_m, self.h_m = map_w_m, map_h_m
        self.ih, self.iw = svg_bgr.shape[:2]
        self._trail = deque(maxlen=self.TRAIL_LEN)

    def _w2p(self, x, y):
        return map_to_pixel(x, y, self.iw, self.ih, self.w_m, self.h_m)

    def add_trail_point(self, x, y):
        self._trail.append((x, y))

    def render(self, x, y, yaw, path, cursor, planner, conf, zone,
               snap_miss=0, heading_conf=0.0):
        img = self.base.copy()

        # Planned path
        if path and planner:
            pts_all, pts_done = [], []
            for i, n in enumerate(path):
                pos = planner.node_positions.get(n)
                if pos:
                    pts_all.append(list(self._w2p(*pos)))
                    if i <= cursor: pts_done.append(list(self._w2p(*pos)))
            if len(pts_all) > 1:
                cv2.polylines(img,[np.array(pts_all,np.int32)],False,(120,40,170),2,cv2.LINE_AA)
            if len(pts_done) > 1:
                cv2.polylines(img,[np.array(pts_done,np.int32)],False,(40,160,40),2,cv2.LINE_AA)
            # VIZ-08: numbered lookahead waypoint
            la_idx = min(cursor+5, len(path)-1)
            la_pos = planner.node_positions.get(path[la_idx])
            if la_pos:
                lx, ly = self._w2p(*la_pos)
                cv2.drawMarker(img,(lx,ly),(0,255,180),cv2.MARKER_CROSS,12,2,cv2.LINE_AA)
                _lbl(img, f"+{la_idx-cursor}", lx+4, ly-4, scale=0.30, color=(0,255,180))

        # VIZ-01: faded trail
        trail = list(self._trail)
        for i, (tx, ty) in enumerate(trail):
            frac  = i / max(len(trail)-1, 1)
            alpha = int(40 + frac*200)
            px_, py_ = self._w2p(tx, ty)
            cv2.circle(img, (px_,py_), max(1,int(1+frac*2)),
                       (int(alpha*.5), int(alpha*.7), alpha), -1, cv2.LINE_AA)

        # Car dot
        cx, cy = self._w2p(x, y)
        cx = max(6, min(self.iw-6, cx)); cy = max(6, min(self.ih-6, cy))
        dot_col = C_CYAN if conf>0.6 else (C_AMBER if conf>0.3 else C_RED)

        # VIZ-02: uncertainty ellipse
        miss = min(snap_miss, 60)
        r_u  = 6 + int(miss*0.75)
        uc   = (50,160,50) if miss<5 else ((50,130,180) if miss<20 else (50,50,200))
        cv2.ellipse(img,(cx,cy),(r_u,max(4,r_u//2)),
                    int(math.degrees(yaw)),0,360,uc,1,cv2.LINE_AA)

        # VIZ-08: heading-confidence arc
        if heading_conf > 0.05:
            arc_span = int(heading_conf * 180)
            cv2.ellipse(img,(cx,cy),(14,14),
                        int(math.degrees(yaw)-90),0,arc_span,(180,180,80),2,cv2.LINE_AA)

        cv2.circle(img,(cx,cy), 9, dot_col, -1, cv2.LINE_AA)
        cv2.circle(img,(cx,cy),13, dot_col,  1, cv2.LINE_AA)
        hx = cx + int(math.cos(yaw)*22); hy = cy - int(math.sin(yaw)*22)
        cv2.arrowedLine(img,(cx,cy),(hx,hy),C_WHITE,2,tipLength=0.4,line_type=cv2.LINE_AA)

        # Zone badge
        zc = {"CITY":(40,110,40),"HIGHWAY":(160,70,20),"PARKING":(30,90,150)}.get(zone,(80,80,80))
        _badge(img, zone, self.iw-86, 22, zc, w=82)

        # Progress bar
        if path:
            prog = cursor / max(len(path)-1, 1)
            bw   = int(self.iw * prog)
            cv2.rectangle(img,(0,self.ih-8),(self.iw,self.ih),(25,25,25),-1)
            cv2.rectangle(img,(0,self.ih-8),(bw,self.ih),(0,170,80),-1)
            _lbl(img, f"{cursor}/{len(path)}  {prog*100:.0f}%",
                 4, self.ih-11, scale=0.32)
        return img


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
        img = np.full((self.H, self.W, 3), 18, np.uint8)

        # Title
        cv2.rectangle(img,(0,0),(self.W,28),(28,28,44),-1)
        _lbl(img, "LOCALIZATION ENGINE — LIVE TELEMETRY",
             8, 18, scale=0.48, color=(150,150,220), t=1)

        # Pose readout
        pc = C_GREEN if confidence>0.6 else (C_AMBER if confidence>0.3 else C_RED)
        cv2.putText(img,f"X: {x:+8.3f} m",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.68,pc,2,cv2.LINE_AA)
        cv2.putText(img,f"Y: {y:+8.3f} m",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.68,pc,2,cv2.LINE_AA)
        cv2.putText(img,f"YAW: {yaw_deg:+7.1f}\u00b0",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.68,pc,2,cv2.LINE_AA)
        _lbl(img, f"v={velocity_ms:.3f}m/s   zone={zone}   nav={nav_state}",
             10, 140, scale=0.37, color=(150,150,150))

        # Yaw-rate sparkline
        _lbl(img,"YAW-RATE (rad/s)",8,162,scale=0.35,color=(130,130,180))
        _spark(img,self._yr_hist,8,165,self.W-16,50,(60,190,255),scale=1.2)
        yrc = C_GREEN if abs(yaw_rate)<0.3 else (C_AMBER if abs(yaw_rate)<0.8 else C_RED)
        _lbl(img,f"{yaw_rate:+.3f}r/s",self.W-90,206,scale=0.40,color=yrc)

        # Lateral error sparkline
        _lbl(img,"LATERAL ERROR (px)",8,230,scale=0.35,color=(130,180,130))
        _spark(img,self._le_hist,8,233,self.W-16,42,(60,220,60),scale=160.0)
        lec = C_GREEN if abs(lat_err_px)<30 else (C_AMBER if abs(lat_err_px)<70 else C_RED)
        _lbl(img,f"{lat_err_px:+.1f}px",self.W-80,267,scale=0.40,color=lec)

        # Layer badges
        _lbl(img,"LAYERS:",8,292,scale=0.38,color=(130,130,155))
        bx = 68
        for name, active in [("L1:YAW",l1),("L2:PATH",l2),("L3:DR",True),("L4:SNAP",l4)]:
            col = (35,145,35) if active else (35,35,95)
            tw  = cv2.getTextSize(name,cv2.FONT_HERSHEY_SIMPLEX,0.34,1)[0][0]
            cv2.rectangle(img,(bx,279),(bx+tw+8,296),col,-1,cv2.LINE_AA)
            _lbl(img,name,bx+4,292,scale=0.34,color=(0,0,0))
            bx += tw+14

        # Heading confidence bar
        hcc = C_GREEN if heading_conf>0.5 else (C_AMBER if heading_conf>0.25 else C_RED)
        _lbl(img,f"HEADING CONF: {heading_conf:.2f}",8,316,scale=0.37,color=(170,170,110))
        _hbar(img,heading_conf,1.0,8,319,self.W-16,10,hcc)

        # Map snap status
        snc = C_GREEN if snap_miss<5 else (C_AMBER if snap_miss<20 else C_RED)
        _lbl(img,
             f"MAP SNAP: {'ACTIVE' if snap_miss<5 else 'LOST'} ({snap_miss} misses)",
             8,346,scale=0.38,color=snc)

        # DR distance
        drc = C_GREEN if self._dr_dist<0.3 else (C_AMBER if self._dr_dist<1.0 else C_RED)
        _lbl(img,f"DR DIST: {self._dr_dist:.3f} m since snap",8,362,scale=0.37,color=drc)

        # Upcoming curve
        cc = C_GREEN if upcoming_curve=="STRAIGHT" else (C_AMBER if "LEFT" in upcoming_curve else C_RED)
        ds = f"{curve_dist_m:.1f}m" if curve_dist_m<99 else "---"
        _lbl(img,f"CURVE: {upcoming_curve}  @{ds}",self.W//2,346,scale=0.42,color=cc)

        # Confidence mini bar
        _lbl(img,f"CONF: {confidence:.2f}",self.W//2,362,scale=0.37,color=pc)
        _hbar(img,confidence,1.0,self.W//2,365,self.W//2-10,10,pc)

        return img


# ══════════════════════════════════════════════════════════════════════════════
# VIZ-04 — Telemetry panel
# ══════════════════════════════════════════════════════════════════════════════

def _steer_gauge(img, steer, cx, cy, r):
    cv2.ellipse(img,(cx,cy),(r,r),0,180,360,(45,45,45),10)
    for td in [-45,-30,-15,0,15,30,45]:
        ang = math.radians(270-(td/45.0)*90)
        oi  = (cx+int(r*math.cos(ang)),   cy+int(r*math.sin(ang)))
        ii  = (cx+int((r-8)*math.cos(ang)),cy+int((r-8)*math.sin(ang)))
        cv2.line(img,ii,oi,(75,75,75) if td!=0 else (110,110,110),
                 1 if td!=0 else 2, cv2.LINE_AA)
    ang = math.radians(270-(steer/45.0)*90)
    nx  = int(cx+r*math.cos(ang)); ny = int(cy+r*math.sin(ang))
    col = C_GREEN if abs(steer)<15 else (C_AMBER if abs(steer)<30 else C_RED)
    cv2.line(img,(cx,cy),(nx,ny),col,3,cv2.LINE_AA)
    cv2.circle(img,(cx,cy),5,col,-1,cv2.LINE_AA)
    _lbl(img,f"{steer:+.1f}\u00b0",cx-24,min(cy+r+18,img.shape[0]-4),scale=0.44,color=col)
    _lbl(img,"STEER",cx-18,max(cy-r-6,12),scale=0.37,color=(120,120,120))


def _draw_telemetry_panel(steer, speed_pwm, lat_err, conf, anchor,
                          zone, upcoming_curve, fps, sign_history,
                          nav_state, l_conf, r_conf, curvature,
                          velocity_ms, w=480, h=360):
    img = np.full((h,w,3),18,np.uint8)
    cv2.rectangle(img,(0,0),(w,28),(28,28,40),-1)
    _lbl(img,"TELEMETRY",8,18,scale=0.48,color=(160,160,220),t=1)

    _steer_gauge(img, steer, cx=100, cy=108, r=78)

    _lbl(img,f"PWM:{speed_pwm:.0f}  v={velocity_ms:.3f}m/s",200,52,scale=0.38,color=(130,175,220))
    _hbar(img,speed_pwm,100,200,55,w-210,12,(50,135,220))

    lec = C_GREEN if abs(lat_err)<30 else (C_AMBER if abs(lat_err)<70 else C_RED)
    _lbl(img,f"LAT ERR: {lat_err:+.1f}px",200,83,scale=0.38,color=lec)
    _hbar(img,lat_err,160,200,86,w-210,12,lec)

    cc = C_GREEN if conf>0.6 else (C_AMBER if conf>0.3 else C_RED)
    _lbl(img,f"LANE CONF: {conf:.2f}",200,113,scale=0.38,color=cc)
    _hbar(img,conf,1.0,200,116,w-210,12,cc)

    hw = (w-210)//2 - 4
    _lbl(img,f"L:{l_conf:.2f}",200,142,scale=0.34,color=(180,90,90))
    _hbar(img,l_conf,1.0,200,145,hw,9,(175,75,75))
    _lbl(img,f"R:{r_conf:.2f}",200+hw+6,142,scale=0.34,color=(80,95,190))
    _hbar(img,r_conf,1.0,200+hw+6,145,hw,9,(75,90,185))

    ac = C_GREEN if "DUAL" in anchor else (C_AMBER if "DEAD" not in anchor else C_RED)
    _lbl(img,f"ANCHOR: {anchor}",200,167,scale=0.38,color=ac)

    curvc = C_GREEN if curvature<0.001 else (C_AMBER if curvature<0.003 else C_RED)
    _lbl(img,f"CURV: {curvature:.5f}",200,185,scale=0.37,color=curvc)

    fpsc = C_GREEN if fps>=25 else (C_AMBER if fps>=18 else C_RED)
    _lbl(img,f"FPS: {fps:.1f}" + ("  !! LOW" if fps<18 else ""),200,207,scale=0.44,color=fpsc)

    nc = C_GREEN if nav_state=="NORMAL" else (C_AMBER if "JUNCTION" in nav_state else C_CYAN)
    _lbl(img,f"NAV: {nav_state}",200,227,scale=0.37,color=nc)

    ucc = C_GREEN if upcoming_curve=="STRAIGHT" else (C_AMBER if "LEFT" in upcoming_curve else C_RED)
    _lbl(img,f"NEXT: {upcoming_curve}",200,247,scale=0.42,color=ucc)

    _lbl(img,"DETECTIONS:",8,232,scale=0.35,color=(90,90,115))
    now_ = time.time()
    for i,(lbl,cf,ts) in enumerate(sign_history[-5:]):
        age = now_-ts; alpha=max(0.15,1.0-age/5.0); c=int(200*alpha)
        _lbl(img,f"{lbl} ({age:.1f}s)",8,248+i*18,scale=0.33,color=(c,c,c))

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
    yc   = C_GREEN if "DUAL" in ctrl.anchor else (C_AMBER if "DEAD" not in ctrl.anchor else C_RED)
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
    img = np.full((32, w, 3), 28, np.uint8)
    _badge(img," E-STOP " if estop else " RUNNING ",4,22,
           (0,0,170) if estop else (30,120,30), w=76)
    _badge(img,f" {fps:.0f}fps ",90,22,
           (30,120,30) if fps>=25 else ((30,100,160) if fps>=18 else (0,0,160)), w=62)
    zc = {"CITY":(40,100,40),"HIGHWAY":(130,55,20),"PARKING":(25,80,140)}.get(zone,(60,60,60))
    _badge(img,f" {zone} ",162,22,zc,w=80)
    nc = (30,110,30) if nav_state=="NORMAL" else ((25,90,160) if "JUNCTION" in nav_state else (90,55,20))
    _badge(img,f" {nav_state} ",252,22,nc,w=130)
    cd = f"{curve_dist_m:.1f}m" if curve_dist_m<99 else "--"
    ucc=(30,110,30) if upcoming_curve=="STRAIGHT" else ((25,90,160) if "LEFT" in upcoming_curve else (20,55,140))
    _badge(img,f" {upcoming_curve}@{cd} ",392,22,ucc,w=120)
    sc = (30,110,30) if snap_miss<5 else ((25,90,160) if snap_miss<20 else (0,0,160))
    _badge(img,f" SNAP:{snap_miss} ",522,22,sc,w=80)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Junction detector
# ══════════════════════════════════════════════════════════════════════════════

class AutonomousJunctionPlanner:
    def decide(self, warped_binary, left_fit, right_fit, lane_width_px):
        lroi=warped_binary[0:240,0:320]; rroi=warped_binary[0:240,320:640]; sroi=warped_binary[0:240,200:440]
        wts=np.linspace(2.0,0.5,240).reshape(-1,1)
        ls=np.sum(lroi*wts)/(320*240); rs=np.sum(rroi*wts)/(320*240); ss=np.sum(sroi*wts)/(240*240)
        scores={"LEFT":ls,"RIGHT":rs,"STRAIGHT":ss}
        best=max(scores,key=scores.get); total=sum(scores.values())
        conf=scores[best]/max(total,1e-6)
        return ("RIGHT",0.3) if conf<0.4 else (best,conf)

class JunctionDetector:
    ENTRY_FRAMES=5; EXIT_FRAMES=8; RATIO_EARLY_WARN=1.7; MIN_BOT_ENERGY=500
    def __init__(self):
        self.state="NORMAL"; self.entry_count=0; self.exit_count=0
        self.frames_in_jct=0; self.planner=AutonomousJunctionPlanner()
    def update(self, warped_binary, left_fit, right_fit, lane_width_px, active_labels):
        h,w=warped_binary.shape
        approaching_wide=False
        if left_fit is not None and right_fit is not None:
            if (np.polyval(right_fit,150)-np.polyval(left_fit,150))>lane_width_px*self.RATIO_EARLY_WARN:
                approaching_wide=True
        elif left_fit is not None:
            if np.polyval(left_fit,150)<max(0,320-lane_width_px*self.RATIO_EARLY_WARN): approaching_wide=True
        elif right_fit is not None:
            if np.polyval(right_fit,150)>min(640,320+lane_width_px*self.RATIO_EARLY_WARN): approaching_wide=True
        hist_bot=float(np.sum(warped_binary[h//2:,:])); hist_top=float(np.sum(warped_binary[:h//2,:]))
        cross_e=False
        if hist_bot>self.MIN_BOT_ENERGY and (hist_top/hist_bot)>1.4: cross_e=True
        if "crosswalk-sign" in active_labels: cross_e=False
        evidence=approaching_wide or cross_e
        if self.state=="NORMAL":
            self.entry_count=self.entry_count+1 if evidence else 0
            if self.entry_count>=self.ENTRY_FRAMES:
                self.state="JUNCTION_PROMPT"; self.exit_count=0; self.frames_in_jct=0
        elif self.state.startswith("JUNCTION_"):
            self.frames_in_jct+=1
            self.exit_count=self.exit_count+1 if not evidence else 0
            if self.exit_count>=self.EXIT_FRAMES and self.frames_in_jct>25:
                self.state="NORMAL"; self.entry_count=0
        return self.state


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    BASE_SPEED = 50
    MAP_W=600; MAP_H=440
    CAM_W=480; CAM_H=360
    LOC_W=520; LOC_H=400
    TEL_W=480; TEL_H=360

    def __init__(self, sim_mode=False, base_speed=None, svg_path=None):
        self.sim_mode   = sim_mode
        self.base_speed = base_speed or self.BASE_SPEED
        self.svg_path   = svg_path or SVG_PATH_DEFAULT
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

        self.jct_detector   = JunctionDetector()
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

        # Sign map — placed signs persist across restarts
        _sm_path = os.path.join(_SCRIPT_DIR, "sign_map.json")
        self.sign_map = SignMap(_sm_path) if _SIGNMAP_AVAILABLE else None
        self._sign_place_mode    = False    # toggled by [PLACE MODE] button
        self._selected_sign_type = None     # tk.StringVar set in build_ui
        self._sign_place_btn     = None     # reference to toggle button

        # Proximity YOLO gating
        self._YOLO_GATE_M  = 4.0   # run YOLO only within this radius of a sign
        self._SLOW_SIGN_M  = 3.5   # start slowing at this distance from a sign
        self._last_snap_id = None  # ID of last sign that triggered a loc snap

        self._svg_base     = _load_svg_as_cv2(self.svg_path, self.MAP_W, self.MAP_H)
        self._map_renderer = MapOverlayRenderer(self._svg_base, MAP_W_M, MAP_H_M)
        self._graph_renderer = None
        self._loc_panel    = LocalizationPanel(self.LOC_W, self.LOC_H)

        self._q_yolo  = queue.Queue(maxsize=1)
        self._q_bev   = queue.Queue(maxsize=1)
        self._q_loc   = queue.Queue(maxsize=1)

    def build_ui(self, root):
        self._root = root
        root.title("BFMC Pilot v4 — Live Localization Dashboard")
        root.configure(bg="#0d0d0d")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Status bar (VIZ-07)
        self._sf = tk.Frame(root, bg="#1a1a1a"); self._sf.pack(fill=tk.X)
        self._sl = tk.Label(self._sf, bg="#1a1a1a"); self._sl.pack(fill=tk.X)
        self._refresh_status_label(np.full((32,1290,3),28,np.uint8))

        # Row 1: Map | YOLO | GraphML
        r1 = tk.Frame(root, bg="#0d0d0d"); r1.pack(padx=6, pady=4)
        for col, title, fg, attr, blank in [
            (0," MAP — click: Start→Dest ","#00e5ff","_map_label",
             cv2.cvtColor(self._svg_base,cv2.COLOR_BGR2RGB)),
            (1," CAMERA + YOLO ","#ff9100","_yolo_label",
             np.zeros((self.CAM_H,self.CAM_W,3),np.uint8)),
            (2," GRAPH (A* route) ","#69ff47","_graph_label",
             np.full((self.CAM_H,self.CAM_W,3),18,np.uint8)),
        ]:
            fr = tk.LabelFrame(r1,text=title,bg="#0d0d0d",fg=fg,font=("Courier",9,"bold"))
            fr.grid(row=0,column=col,padx=4)
            lbl = tk.Label(fr,bg="#0d0d0d"); lbl.pack()
            ph  = ImageTk.PhotoImage(Image.fromarray(blank))
            lbl.config(image=ph)
            setattr(self,attr,lbl)
            setattr(self,attr.replace("label","ph"),ph)
        self._map_label.bind("<Button-1>", self._on_map_click)

        # Row 2: BEV | Localization | Telemetry
        r2 = tk.Frame(root, bg="#0d0d0d"); r2.pack(padx=6, pady=(0,4))
        for col, title, fg, attr, blank in [
            (0," LANE VIEW (BEV) ","#69ff47","_bev_label",
             np.zeros((self.CAM_H,self.CAM_W,3),np.uint8)),
            (1," LOCALIZATION ENGINE ","#ff4fd8","_loc_label",
             np.full((self.LOC_H,self.LOC_W,3),18,np.uint8)),
            (2," TELEMETRY ","#ff9100","_telem_label",
             np.full((self.TEL_H,self.TEL_W,3),18,np.uint8)),
        ]:
            fr = tk.LabelFrame(r2,text=title,bg="#0d0d0d",fg=fg,font=("Courier",9,"bold"))
            fr.grid(row=0,column=col,padx=4)
            lbl = tk.Label(fr,bg="#0d0d0d"); lbl.pack()
            ph  = ImageTk.PhotoImage(Image.fromarray(blank))
            lbl.config(image=ph)
            setattr(self,attr,lbl)
            setattr(self,attr.replace("label","ph"),ph)

        # Control bar
        cb = tk.Frame(root,bg="#111",pady=5); cb.pack(fill=tk.X,padx=6,pady=(0,4))
        self._sv_pose = tk.StringVar(value="Pose: not set")
        self._sv_hint = tk.StringVar(value="Click map: set START")
        tk.Label(cb,textvariable=self._sv_pose,bg="#111",fg="#eee",font=("Courier",9)).pack(side=tk.LEFT,padx=10)
        tk.Label(cb,textvariable=self._sv_hint,bg="#111",fg="#ffcc00",font=("Courier",9,"bold")).pack(side=tk.LEFT,padx=10)
        bf = tk.Frame(cb,bg="#111"); bf.pack(side=tk.RIGHT,padx=8)
        for txt,bg,cmd in [("E-STOP","#c0392b",self._estop_cb),
                            ("RESUME","#27ae60",self._resume_cb),
                            ("RESET ROUTE","#2471a3",self._reset_route),
                            ("START","#8e44ad",self._start_pilot)]:
            tk.Button(bf,text=txt,bg=bg,fg="white",font=("Courier",9,"bold"),
                      command=cmd).pack(side=tk.LEFT,padx=4)

        if self.localizer.planner:
            self._graph_renderer = GraphMLRenderer(
                self.localizer.planner, self.CAM_W, self.CAM_H)

        # ── Sign Editor panel ────────────────────────────────────────────
        se = tk.Frame(root, bg="#0d1a0d", pady=4)
        se.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(se, text=" SIGN EDITOR ", bg="#0d1a0d", fg="#69ff47",
                 font=("Courier", 9, "bold")).pack(side=tk.LEFT, padx=6)

        # Sign type dropdown
        self._selected_sign_type = tk.StringVar(value=SIGN_TYPES[0])
        om = tk.OptionMenu(se, self._selected_sign_type, *SIGN_TYPES)
        om.config(bg="#1a2b1a", fg="#cce8cc", font=("Courier", 8),
                  highlightthickness=0, bd=0, relief=tk.FLAT,
                  activebackground="#2a3b2a")
        om["menu"].config(bg="#1a2b1a", fg="#cce8cc", font=("Courier", 8))
        om.pack(side=tk.LEFT, padx=4)

        # PLACE MODE toggle
        def _toggle_place_mode():
            self._sign_place_mode = not self._sign_place_mode
            if self._sign_place_mode:
                self._sign_place_btn.config(bg="#c0392b", text="PLACE MODE ●")
                self._sv_hint.set(f"Click map to place: {self._selected_sign_type.get()}")
            else:
                self._sign_place_btn.config(bg="#1a3a1a", text="PLACE MODE")
                self._sv_hint.set("Click map: set START")

        self._sign_place_btn = tk.Button(
            se, text="PLACE MODE", bg="#1a3a1a", fg="white",
            font=("Courier", 8, "bold"), command=_toggle_place_mode)
        self._sign_place_btn.pack(side=tk.LEFT, padx=4)

        def _undo_sign():
            if self.sign_map and self.sign_map.remove_last():
                self._sv_hint.set(f"Removed last sign ({len(self.sign_map)} remain)")

        def _clear_signs():
            if self.sign_map:
                self.sign_map.clear()
                self._sv_hint.set("All signs cleared")

        tk.Button(se, text="UNDO LAST", bg="#2c3e50", fg="white",
                  font=("Courier", 8, "bold"),
                  command=_undo_sign).pack(side=tk.LEFT, padx=4)
        tk.Button(se, text="CLEAR ALL", bg="#6d2c2c", fg="white",
                  font=("Courier", 8, "bold"),
                  command=_clear_signs).pack(side=tk.LEFT, padx=4)

        # Live count
        self._sv_sign_count = tk.StringVar(
            value=f"{len(self.sign_map) if self.sign_map else 0} signs placed")
        tk.Label(se, textvariable=self._sv_sign_count, bg="#0d1a0d",
                 fg="#99ccaa", font=("Courier", 8)).pack(side=tk.LEFT, padx=10)

        # Update type hint when dropdown changes
        def _on_type_change(*_):
            if self._sign_place_mode:
                self._sv_hint.set(
                    f"Click map to place: {self._selected_sign_type.get()}")
        self._selected_sign_type.trace_add("write", _on_type_change)

        self._gui_update()

    def _refresh_status_label(self, img):
        ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
        self._sl.config(image=ph); self._sl.photo = ph  # keep reference

    def _on_map_click(self, event):
        # ─ Sign placement mode: left-click places a sign, not start/dest ─────
        if self._sign_place_mode and self.sign_map is not None:
            x_m, y_m = pixel_to_map(event.x, event.y, self.MAP_W, self.MAP_H)
            stype = (self._selected_sign_type.get()
                     if self._selected_sign_type else "stop")
            self.sign_map.add_sign(stype, x_m, y_m)
            count = len(self.sign_map)
            self._sv_sign_count.set(f"{count} sign{'s' if count != 1 else ''} placed")
            self._sv_hint.set(
                f"Placed {stype} @ ({x_m:.2f}, {y_m:.2f})  [{count} total]")
            return  # don't fall through to start/dest logic

        # ─ Normal route-planning click ─────────────────────────────────────
        if not self.localizer.planner: return
        if len(self.localizer.planner.graph.nodes)==0:
            self._sv_hint.set("Graph empty"); return
        x_m, y_m = pixel_to_map(event.x, event.y, self.MAP_W, self.MAP_H)
        nearest   = self.localizer.planner.get_nearest_node(x_m, y_m)
        if not nearest: return

        if self._start_node is None:
            self._start_node = nearest
            self.localizer.set_pose(x_m, y_m, 0.0)
            self._sv_hint.set(f"Start: {nearest}  → click DESTINATION")
        elif self._target_node is None:
            self._target_node = nearest
            planned = self.localizer.planner.plan_route(self._start_node, nearest)
            if not planned:
                self._sv_hint.set(f"No path to {nearest}. Try another dest.")
                self._target_node = None; return
            self._planned_path = planned; self._path_cursor = 0
            self.localizer.reset_cursor()
            self._sv_hint.set(f"Route: {len(planned)} nodes. Press START.")
        else:
            self._target_node = nearest
            cx,cy,_ = self.localizer.get_pose()
            ns = self.localizer.planner.get_nearest_node(cx,cy)
            planned = self.localizer.planner.plan_route(ns, nearest)
            if planned:
                self._planned_path = planned; self._path_cursor = 0
                self.localizer.reset_cursor()
                self._sv_hint.set(f"Re-routed: {len(planned)} nodes.")

    def _gui_update(self):
        try:
            x, y, yaw = self.localizer.get_pose()
            conf      = self._last_conf
            nearest   = (self.localizer.planner.get_nearest_node(x,y)
                         if self.localizer.planner else None)
            loc_data  = self.localizer.get_pose_for_dashboard()
            zone      = loc_data.get("zone","CITY")
            snap_miss = getattr(self.localizer,'_snap_miss_frames',0)
            hconf_raw = getattr(self.localizer,'_cam_yaw_smoothed',0.0)

            if self.localizer.is_initialized():
                self._map_renderer.add_trail_point(x, y)

            map_img = self._map_renderer.render(
                x,y,yaw,self._planned_path,self._path_cursor,
                self.localizer.planner,conf,zone,
                snap_miss=snap_miss,
                heading_conf=abs(hconf_raw)*2.0)
            # Overlay placed signs on the map image
            if self.sign_map:
                self.sign_map.draw_on_image(map_img, MAP_W_M, MAP_H_M)
                self._sv_sign_count.set(
                    f"{len(self.sign_map)} sign{'s' if len(self.sign_map)!=1 else ''} placed")  # scale to 0-1 approx
            self._map_ph = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(map_img,cv2.COLOR_BGR2RGB)))
            self._map_label.config(image=self._map_ph)

            if self.localizer.is_initialized():
                self._sv_pose.set(
                    f"x={x:.3f}m  y={y:.3f}m  yaw={math.degrees(yaw):.1f}°  snap={snap_miss}")

            try:
                yi = self._q_yolo.get_nowait()
                yi = cv2.resize(yi,(self.CAM_W,self.CAM_H))
                self._yolo_ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(yi,cv2.COLOR_BGR2RGB)))
                self._yolo_label.config(image=self._yolo_ph)
            except queue.Empty: pass

            try:
                bi = self._q_bev.get_nowait()
                bi = cv2.resize(bi,(self.CAM_W,self.CAM_H))
                self._bev_ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bi,cv2.COLOR_BGR2RGB)))
                self._bev_label.config(image=self._bev_ph)
            except queue.Empty: pass

            try:
                li = self._q_loc.get_nowait()
                li = cv2.resize(li,(self.LOC_W,self.LOC_H))
                self._loc_ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(li,cv2.COLOR_BGR2RGB)))
                self._loc_label.config(image=self._loc_ph)
            except queue.Empty: pass

            if self._graph_renderer:
                vm = self.hw.get_velocity_ms() if self.running else 0.0
                gi = self._graph_renderer.render(
                    self._planned_path,self._path_cursor,nearest,velocity_ms=vm)
                self._graph_ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(gi,cv2.COLOR_BGR2RGB)))
                self._graph_label.config(image=self._graph_ph)

            ctrl  = self._last_ctrl
            perc  = self._last_perc
            vm    = loc_data.get("speed_ms",0.0)
            ti = _draw_telemetry_panel(
                steer         = ctrl.steer_angle_deg,
                speed_pwm     = ctrl.speed_pwm,
                lat_err       = ctrl.target_x-320.0,
                conf          = conf,
                anchor        = ctrl.anchor,
                zone          = zone,
                upcoming_curve= getattr(self.localizer,'upcoming_curve','STRAIGHT'),
                fps           = self._fps,
                sign_history  = list(self._sign_history),
                nav_state     = self._nav_state,
                l_conf        = perc.confidence if perc else 0.0,
                r_conf        = perc.confidence if perc else 0.0,
                curvature     = perc.curvature if perc else 0.0,
                velocity_ms   = vm,
                w=self.TEL_W, h=self.TEL_H)
            self._telem_ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(ti,cv2.COLOR_BGR2RGB)))
            self._telem_label.config(image=self._telem_ph)

            cd  = getattr(self.localizer,'curve_dist_m',99.0)
            uc  = getattr(self.localizer,'upcoming_curve','STRAIGHT')
            si  = _status_bar(1290,self._estop,self._fps,zone,self._nav_state,uc,cd,conf,snap_miss)
            self._refresh_status_label(si)

        except Exception as e:
            log.debug(f"GUI update error: {e}")

        if hasattr(self,'_root') and self._root.winfo_exists():
            self._root.after(33, self._gui_update)

    def _pilot_loop(self):
        log.info("Pilot loop started")
        startup_time = time.time()   # reference for calibration phases
        t_prev = time.time()
        _ll = 0; _LLC = 15; _LLS = 90; _zmf = 0
        try:
            while self.running:
                ts = time.time(); dt = max(ts-t_prev, 0.001); t_prev = ts
                elapsed_run = ts - startup_time

                # Always read camera & velocity so dashboard stays live
                raw_frame   = self.hw.read_camera()
                if raw_frame is None:
                    raw_frame = np.zeros((480, 640, 3), np.uint8)
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
                        nearby_signs = self.sign_map.get_nearby(
                            x0, y0, radius_m=self._YOLO_GATE_M)
                        run_yolo = bool(nearby_signs)
                        if not run_yolo:
                            log.debug("YOLO SKIPPED — no signs within %.1fm", self._YOLO_GATE_M)

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
                # When YOLO detects a label that matches a placed map sign within
                # 3 m of the car's current estimated position, snap the localizer
                # to that sign's known map coordinates to correct odometry drift.
                if self.sign_map and t_res.active_labels:
                    x_snap, y_snap, _ = self.localizer.get_pose()
                    for lbl in t_res.active_labels:
                        matched = self.sign_map.match_detection(
                            lbl, x_snap, y_snap, radius_m=3.0)
                        if matched and matched["id"] != self._last_snap_id:
                            _, _, yaw_now = self.localizer.get_pose()
                            self.localizer.set_pose(
                                matched["x_m"], matched["y_m"], yaw_now)
                            self._last_snap_id = matched["id"]
                            log.info(
                                "SIGN SNAP: %s → (%.2f, %.2f) dist=%.2fm",
                                lbl, matched["x_m"], matched["y_m"],
                                matched["dist"])
                            break  # one snap per frame
                    else:
                        # Reset snap lock when no sign detected nearby
                        self._last_snap_id = None

                self._last_t_res = t_res


                if self._estop:
                    # Halted — keep motors off, reuse last perception for dashboard
                    self.hw.set_speed(0); self.hw.set_steering(0)
                    perc = self._last_perc if self._last_perc else self.vision.process(raw_frame)
                    ctrl = self._last_ctrl

                    # F-15: Auto-recovery — attempt re-detection every loop tick.
                    # If lanes are visible again, clear E-STOP and resume driving.
                    try:
                        recovery_perc = self.vision.process(raw_frame, dt=dt)
                        if recovery_perc.confidence > 0.4:
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
                            self._blocked_nodes[nn] = time.time()+30.0
                            if nn in self.localizer.planner.graph:
                                self.localizer.planner.graph.remove_node(nn)
                                rem = [n for n in self.localizer.planner._node_ids
                                       if n in self.localizer.planner.graph]
                                import scipy.spatial
                                self.localizer.planner._node_ids = rem
                                self.localizer.planner._kdtree = scipy.spatial.KDTree(
                                    [self.localizer.planner.node_positions[n] for n in rem])
                                ns2 = self.localizer.planner.get_nearest_node(xne,yne)
                                np2 = self.localizer.planner.plan_route(ns2,self._target_node)
                                self.localizer.planner.load_graph("Competition_track_graph.graphml")
                                if np2:
                                    self._planned_path=np2; self._path_cursor=0
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

                    perc = self.vision.process(
                        raw_frame,
                        dt=dt,
                        extra_offset_px=extra_offset,
                        nav_state=self._nav_state,
                        velocity_ms=velocity_ms,
                        last_steering=getattr(self._last_ctrl,'steer_angle_deg',0.0),
                        upcoming_curve=getattr(self.localizer,'upcoming_curve','STRAIGHT'))
                    self._last_conf = perc.confidence; self._last_perc = perc

                    self._nav_state = self.jct_detector.update(
                        perc.warped_binary,perc.sl,perc.sr,perc.lane_width_px,t_res.active_labels)

                    if self._nav_state=="JUNCTION_PROMPT":
                        if self._planned_path and self.localizer.planner:
                            xj,yj,yj2 = self.localizer.get_pose()
                            action = self.localizer.planner.get_next_action(
                                xj,yj,yj2,path=self._planned_path,
                                cursor=self._path_cursor,velocity_ms=velocity_ms)
                            self._nav_state = f"JUNCTION_{action}"
                        else: self._nav_state="JUNCTION_STRAIGHT"

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
                    if self.behavior:
                        beh = self.behavior.compute(
                            perc_res=perc,
                            t_res=t_res,
                            dt=dt,
                            base_steer=ctrl.steer_angle_deg,
                        )
                        if beh.priority < self.behavior.PRI_NORMAL:
                            # Higher-priority command wins
                            ctrl.speed_pwm       = beh.speed_pwm
                            ctrl.steer_angle_deg = beh.steer_deg
                            log.debug("BEH[%d] %s: %s",
                                      beh.priority, beh.state, beh.reason)

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
                        ctrl.speed_pwm = min(ctrl.speed_pwm, 15.0)
                        cv2.putText(perc.lane_dbg,
                            f"LANE CALIB: {6.0 - elapsed_run:.1f}s",
                            (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

                    self._last_ctrl = ctrl

                    _ll = _ll+1 if (perc.sl is None and perc.sr is None) else 0
                    if _ll>=_LLS:
                        self.hw.set_speed(0); self.hw.set_steering(0)
                        self._estop=True
                    else:
                        speed = ctrl.speed_pwm
                        if _ll>=_LLC: speed = min(speed,20.0)
                        if 0.0<speed<PWM_DEADBAND: speed=PWM_DEADBAND
                        self.hw.set_speed(speed); self.hw.set_steering(ctrl.steer_angle_deg)

                # --- DASHBOARD TELEMETRY (runs always, even in E-STOP) ---
                push_latest(self._q_yolo,
                            t_res.yolo_debug_frame if t_res.yolo_debug_frame is not None
                            else raw_frame)
                push_latest(self._q_bev, _annotate_bev(perc, ctrl))

                # VIZ-03: localization panel
                sm     = getattr(self.localizer,'_snap_miss_frames',0)
                lx,ly,lyaw = self.localizer.get_pose()
                yr     = self.localizer.visual_yaw_rate
                self._loc_panel.push(yr, perc.lateral_error_px, lx, ly, sm==0)
                loc_img = self._loc_panel.render(
                    x=lx, y=ly, yaw_deg=math.degrees(lyaw),
                    yaw_rate=yr, heading_conf=perc.heading_conf,
                    snap_miss=sm, confidence=perc.confidence,
                    upcoming_curve=getattr(self.localizer,'upcoming_curve','STRAIGHT'),
                    curve_dist_m=getattr(self.localizer,'curve_dist_m',99.0),
                    lat_err_px=perc.lateral_error_px, velocity_ms=velocity_ms,
                    zone=self.localizer.current_zone, nav_state=self._nav_state,
                    l1=(perc.confidence>0.3 and perc.heading_conf>=0.35),
                    l2=bool(self._planned_path and perc.confidence>0.5),
                    l4=sm<5)
                push_latest(self._q_loc, loc_img)

                elapsed = time.time()-ts
                time.sleep(max(0.001, FRAME_PERIOD-elapsed))

        except Exception as e:
            log.error(f"FATAL Pilot crash: {e}", exc_info=True); self._estop=True

        log.info("Pilot loop exited")
        self.hw.set_speed(0); self.hw.set_steering(0)


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

    def _reset_route(self):
        self._start_node=None; self._target_node=None
        self._planned_path=[]; self._path_cursor=0
        self.localizer.reset_cursor(); self._map_renderer._trail.clear()
        self._sv_hint.set("Route cleared. Click map: set START")

    def _on_close(self):
        self.running=False; self._estop=True
        self.hw.set_speed(0); self.hw.set_steering(0)
        time.sleep(0.15); self.hw.shutdown()
        if self._threaded_yolo: self._threaded_yolo.stop()
        if hasattr(self,'_root'): self._root.destroy()


def main():
    ap = argparse.ArgumentParser(description="BFMC v4 — Visual Localization Pilot")
    ap.add_argument("--sim",       action="store_true")
    ap.add_argument("--speed",     type=float, default=50)
    ap.add_argument("--svg",       type=str,   default=None)
    ap.add_argument("--sim-video", type=str,   default=None)
    args = ap.parse_args()

    orch = Orchestrator(sim_mode=args.sim, base_speed=args.speed,
                        svg_path=args.svg or SVG_PATH_DEFAULT)
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