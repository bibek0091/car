"""
BFMC Autonomous Pilot — main.py  (Improved v2)
==============================================
Key fixes over original:
  1. STARTUP LOCALIZATION WIZARD: User places car on map before driving.
  2. CORRECT MAP RENDERING: Equal-axis, no distortion, exact track proportions.
  3. CALIBRATION PHASE: Clearly shows 6-second countdown with IMU zero.
  4. CLEAN THREAD SAFETY: All GUI updates via queue — no cross-thread Tk calls.
  5. MAP CLICK LOCALISATION: Click any node on the map to teleport car position.
  6. REROUTE DIALOG: Re-plan A* to a new target without restarting.
"""

import logging
import threading
import queue
import time
import csv
import argparse
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from datetime import datetime
import numpy as np
import cv2
import math

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Custom Modules ─────────────────────────────────────────────────────────────
from map_planner import PathPlanner
from localization import LocalizationEngine
from perception import VisionPipeline
from control import Controller
from hardware_io import HardwareIO
from traffic_module import ThreadedYOLODetector, TrafficDecisionEngine

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Global e-stop event — any thread can halt the car
estop_event = threading.Event()

# ── Global Constants & Utilities ─────────────────────────────────────────────
BFMC_SIGNS = {
    # format: "label": [(x, y, "char", "color")]
    "STOP":        [(4.17, 6.89, "STP", "#FF2020"),   
                    (2.35, 3.84, "STP", "#FF2020")],   
    "TRAFFIC_LT":  [(5.71, 6.89, "TL", "#FF8000"),   
                    (4.94, 3.83, "TL", "#FF8000")],   
    "CROSSWALK":   [(3.50, 6.89, "CW", "#FFFFFF"),
                    (4.94, 5.50, "CW", "#FFFFFF")],
    "SPEED_30":    [(7.00, 3.83, "30", "#00AAFF")],
    "ONE_WAY":     [(10.0, 3.83, "->", "#AAAAFF")],
    "PRIORITY":    [(15.48, 3.83, "YIE", "#FFFF00")],
}

import matplotlib.collections as mcoll

def draw_rich_map(ax, planner, signs_dict=None):
    if signs_dict is None:
        signs_dict = BFMC_SIGNS

    G = planner.graph
    pos = planner.node_positions

    ax.clear()
    ax.set_facecolor("#121215")

    # 2.5 Parking Zone
    parking_nodes = {k: p for k, p in pos.items() if p[1] > 9.0}
    if parking_nodes:
        xs = [p[0] for p in parking_nodes.values()]
        ys = [p[1] for p in parking_nodes.values()]
        min_x, max_x = min(xs) - 0.4, max(xs) + 0.4
        min_y, max_y = min(ys) - 0.4, max(ys) + 0.4
        rect = mpatches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                  fc="#0A1A0A", ec="none", alpha=0.3, zorder=1)
        ax.add_patch(rect)
        ax.text((min_x+max_x)/2, min_y + 0.3, "PARKING ZONE", color="#305030", 
                fontsize=16, ha="center", fontfamily="monospace", zorder=2)

    # 2.8 Zone Labels
    ax.text(16.0, 4.5, "HIGHWAY", color="white", alpha=0.12, fontsize=32, ha="center", va="center", fontfamily="sans-serif", rotation=0, zorder=1)
    ax.text(4.5, 5.5, "URBAN", color="white", alpha=0.12, fontsize=32, ha="center", va="center", fontfamily="sans-serif", rotation=0, zorder=1)

    # 2.1 & 2.2 Edges
    solid_edges = []
    dotted_edges = []
    divider_edges = []
    
    for u, v, d in G.edges(data=True):
        pu, pv = pos.get(u), pos.get(v)
        if not pu or not pv: continue
        is_dotted = d.get("dotted", False)
        if isinstance(is_dotted, str): is_dotted = is_dotted.lower() == "true"
        
        if is_dotted:
            dotted_edges.append([(pu[0], pu[1]), (pv[0], pv[1])])
            if pu[1] > 9.0 and pv[1] > 9.0 and abs(pu[0]-pv[0]) < 0.1:
                divider_edges.append([(pu[0], pu[1]), (pv[0], pv[1])])
        else:
            solid_edges.append([(pu[0], pu[1]), (pv[0], pv[1])])

    ax.add_collection(mcoll.LineCollection(divider_edges, colors="#808080", linewidths=1.0, zorder=3))
    ax.add_collection(mcoll.LineCollection(solid_edges, colors="#3A3A42", linewidths=16, capstyle="round", joinstyle="round", zorder=2))
    ax.add_collection(mcoll.LineCollection(solid_edges, colors="#1E1E24", linewidths=10, capstyle="round", joinstyle="round", zorder=3))
    ax.add_collection(mcoll.LineCollection(dotted_edges, colors="#FFFF99", linewidths=1.5, linestyles=(0, (6, 4)), capstyle="round", zorder=4))

    # 2.3 Road Direction Arrows
    for i, edge in enumerate(solid_edges):
        if i % 5 == 0:
            (x1, y1), (x2, y2) = edge
            mx, my = (x1+x2)/2.0, (y1+y2)/2.0
            dx, dy = x2-x1, y2-y1
            l = math.hypot(dx, dy)
            if l > 0:
                ax.annotate("", xy=(mx + dx/l*0.1, my + dy/l*0.1), xytext=(mx - dx/l*0.1, my - dy/l*0.1),
                            arrowprops=dict(arrowstyle="->", color="#404050", lw=1.2), zorder=4)

    # 2.4 Roundabouts
    rbt_clusters = [
        ("RBT-A", 4.94, 6.71), ("RBT-B", 2.70, 6.70), ("RBT-C", 2.70, 3.84),
        ("RBT-D", 4.94, 3.83), ("RBT-E", 15.48, 3.83)
    ]
    for name, cx, cy in rbt_clusters:
        ax.add_patch(mpatches.Circle((cx, cy), 0.65, color="#FF6600", fill=False, lw=2.0, zorder=5))
        ax.add_patch(mpatches.Circle((cx, cy), 0.30, color="#2A1A00", fill=True, alpha=0.8, zorder=5))
        ax.text(cx, cy - 0.75, name, color="#FF8030", fontsize=7, ha="center", fontfamily="monospace", zorder=6)
        for angle in [0, 90, 180, 270]:
            rad = math.radians(angle)
            tx, ty = cx + 0.65 * math.cos(rad), cy + 0.65 * math.sin(rad)
            ax.add_patch(mpatches.RegularPolygon((tx, ty), 3, radius=0.08, orientation=rad+math.pi, color="#FF6600", zorder=6))

    # All nodes
    for px, py in pos.values():
        ax.plot(px, py, "o", color="#2A2A32", ms=2.5, zorder=4)

    # 2.6 Traffic Signs
    sign_artists = {}
    for stype, sign_list in signs_dict.items():
        for i, data in enumerate(sign_list):
            if len(data) == 4:
                sx, sy, char, color = data
            else:
                sx, sy, char, color = data[0], data[1], data[2], data[3]
                
            min_d, nx, ny = float('inf'), sx, sy
            best_n = None
            for nid, (px, py) in pos.items():
                d = math.hypot(px-sx, py-sy)
                if d < min_d:
                    min_d, nx, ny, best_n = d, px, py, nid
                    
            if stype == "CROSSWALK" and best_n is not None:
                connected = [v for u, v in G.edges(best_n)] + [u for u, v in G.edges() if v == best_n]
                angle = 0
                if connected:
                    cx, cy = pos[connected[0]]
                    angle = math.atan2(cy - ny, cx - nx)
                
                for j in range(-2, 2):
                    offset = j * 0.08
                    ox = nx + offset * math.cos(angle)
                    oy = ny + offset * math.sin(angle)
                    c = "#FFFFFF" if j % 2 == 0 else "#808080"
                    
                    rect = mpatches.Rectangle((ox - 0.175*math.sin(angle), oy + 0.175*math.cos(angle)), 
                                              0.35, 0.05, angle=math.degrees(angle)-90,
                                              color=c, zorder=3, alpha=0.8)
                    ax.add_patch(rect)
                
            line, = ax.plot([sx, nx], [sy, ny], color="#606060", lw=1.0, zorder=7)
            # Add picker=5 for easy selection
            circ = mpatches.Circle((sx, sy), 0.18, color=color, zorder=8, picker=True)
            ax.add_patch(circ)
            txt = ax.text(sx, sy, char, color="black" if color in ["#FFFFFF", "#FFFF00"] else "white", 
                          ha="center", va="center", fontsize=9, fontweight="bold", zorder=9)
                          
            sign_artists[f"{stype}_{i}"] = {
                "circ": circ, "txt": txt, "line": line, "type": stype, "idx": i, 
                "color": color, "char": char, "nx": nx, "ny": ny
            }
            
    # CRITICAL — equal aspect, invert Y
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 1.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    
    ax.set_xlabel("X (m)", color="#787880", fontsize=8)
    ax.set_ylabel("Y (m)", color="#787880", fontsize=8)
    ax.tick_params(colors="#505058", labelsize=7)
    for s in ax.spines.values(): s.set_color("#303038")

    return sign_artists


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP LOCALIZATION WIZARD
# ══════════════════════════════════════════════════════════════════════════════
class StartupOverlay(tk.Frame):
    """
    Full-screen overlay shown BEFORE the pilot loop starts.
    User must:
      1. Click a node to set START position (where the car physically is).
      2. Click a node to set TARGET (where A* should route to).
    Both selections are required before dismissing.
    """

    def __init__(self, parent, planner: "PathPlanner", on_confirm_callback):
        super().__init__(parent, bg="#0C0C0C")
        self.win = self  # Alias for straightforward UI initialization updates
        self.planner = planner
        self.on_confirm_callback = on_confirm_callback
        self.start_node = None
        self.target_node = None
        self.phase = "START"   # "START" → "TARGET" → "DONE"

        self._build_ui()
        self._draw_map()

    # ── Build wizard UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top instruction bar
        self.frm_top = tk.Frame(self.win, bg="#161618", height=60)
        self.frm_top.pack(fill=tk.X, padx=0, pady=0)

        self.lbl_phase = tk.Label(
            self.frm_top,
            text="STEP 1 / 2 — Click the node where your car is physically placed",
            font=("Courier", 14, "bold"), fg="#00C8FF", bg="#161618"
        )
        self.lbl_phase.pack(side=tk.LEFT, padx=20, pady=15)

        self.lbl_sel = tk.Label(
            self.frm_top, text="START: —    TARGET: —",
            font=("Courier", 12), fg="#FFE000", bg="#161618"
        )
        self.lbl_sel.pack(side=tk.RIGHT, padx=20, pady=15)

        # Manual entry row
        frm_entry = tk.Frame(self.win, bg="#0C0C0C")
        frm_entry.pack(fill=tk.X, padx=10, pady=4)

        tk.Label(frm_entry, text="Or type node ID — Start:", font=("Courier", 11),
                 fg="#787880", bg="#0C0C0C").pack(side=tk.LEFT, padx=6)
        self.ent_start = tk.Entry(frm_entry, width=8, font=("Courier", 11),
                                  bg="#222226", fg="white", insertbackground="white")
        self.ent_start.pack(side=tk.LEFT, padx=4)

        tk.Label(frm_entry, text="Target:", font=("Courier", 11),
                 fg="#787880", bg="#0C0C0C").pack(side=tk.LEFT, padx=6)
        self.ent_target = tk.Entry(frm_entry, width=8, font=("Courier", 11),
                                   bg="#222226", fg="white", insertbackground="white")
        self.ent_target.pack(side=tk.LEFT, padx=4)

        tk.Button(frm_entry, text="Apply", font=("Courier", 11, "bold"),
                  bg="#00C8FF", fg="black",
                  command=self._apply_manual_entry).pack(side=tk.LEFT, padx=10)

        # Map canvas
        self.fig, self.ax = plt.subplots(figsize=(11.5, 6.8), facecolor="#0C0C0C")
        self.ax.set_facecolor("#0C0C0C")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        
        self._dragging_sign = None
        self._hover_ann = None
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        # Bottom confirm button (disabled until both are set)
        self.btn_confirm = tk.Button(
            self.win, text="▶  CONFIRM & LAUNCH CALIBRATION",
            font=("Courier", 13, "bold"), bg="#1A3A1A", fg="#64DC64",
            state=tk.DISABLED, command=self._confirm
        )
        self.btn_confirm.pack(fill=tk.X, padx=10, pady=8)

        # Artist handles updated during selection
        self._start_dot = None
        self._target_dot = None
        self._ann_start = None
        self._ann_target = None
        self._hover_dot = None

    # ── Draw the exact track map ───────────────────────────────────────────────
    def _draw_map(self):
        G = self.planner.graph
        pos = self.planner.node_positions

        self.ax.clear()
        self.ax.set_facecolor("#0C0C0C")

        # ── Edges ─────────────────────────────────────────────────────────────
        for u, v, d in G.edges(data=True):
            pu, pv = pos.get(u), pos.get(v)
            if pu is None or pv is None:
                continue
            is_dotted = d.get("dotted", False)
            if isinstance(is_dotted, str):
                is_dotted = is_dotted.lower() == "true"
            style = {"color": "#4A4A56", "lw": 1.0, "ls": "--", "dashes": (4, 3)} \
                    if is_dotted else \
                    {"color": "#5A5A68", "lw": 1.4, "ls": "-"}
            self.ax.plot([pu[0], pv[0]], [pu[1], pv[1]], **style,
                         solid_capstyle="round", zorder=2)

        # ── All nodes (small, subtle) ─────────────────────────────────────────
        for node, (x, y) in pos.items():
            self.ax.plot(x, y, "o", color="#303038", ms=3, zorder=3)

        # ── Roundabout markers ─────────────────────────────────────────────────
        rbt_clusters = [
            ("RBT-A", 4.94, 6.71), ("RBT-B", 2.70, 6.70),
            ("RBT-C", 2.70, 3.84), ("RBT-D", 4.94, 3.83),
            ("RBT-E", 15.48, 3.83),
        ]
        for name, cx, cy in rbt_clusters:
            circle = plt.Circle((cx, cy), 0.30, color="#FF6600",
                                 fill=False, lw=1.5, linestyle="-", zorder=4, alpha=0.7)
            self.ax.add_patch(circle)
            self.ax.text(cx, cy - 0.45, name, color="#FF8030",
                         fontsize=7, ha="center", zorder=5,
                         fontfamily="monospace")

        # ── Axes: CRITICAL — equal aspect, invert Y so map-north is up ────────
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        pad = 0.8
        self.ax.set_xlim(min(xs) - pad, max(xs) + pad)
        self.ax.set_ylim(min(ys) - pad, max(ys) + pad)
        self.ax.set_aspect("equal", adjustable="box")   # ← prevents distortion
        self.ax.invert_yaxis()                           # ← Y=0 is physical top

        self.ax.set_xlabel("X (metres)", color="#606068", fontsize=9)
        self.ax.set_ylabel("Y (metres)", color="#606068", fontsize=9)
        self.ax.tick_params(colors="#505058", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#303038")

        # Legend
        legend_elements = [
            Line2D([0], [0], color="#5A5A68", lw=1.4, label="Track lane"),
            Line2D([0], [0], color="#4A4A56", lw=1.0, ls="--", dashes=(4, 3), label="Dashed (junction)"),
            Line2D([0], [0], marker="o", color="w", ms=8, markerfacecolor="#00FF80", label="START node"),
            Line2D([0], [0], marker="o", color="w", ms=8, markerfacecolor="#FF4040", label="TARGET node"),
        ]
        self.ax.legend(handles=legend_elements, loc="upper right",
                       facecolor="#161618", edgecolor="#303038",
                       labelcolor="white", fontsize=8)

        self.ax.set_title("Click to place car (START), then click TARGET",
                          color="#787880", fontsize=10, pad=8)
        self.canvas.draw()

    # ── Interaction handlers ──────────────────────────────────────────────────
    def _on_press(self, event):
        if event.xdata is None or event.ydata is None:
            return
            
        # Check if clicking a sign
        if hasattr(self, '_sign_artists'):
            for key, artist in self._sign_artists.items():
                contains, _ = artist["circ"].contains(event)
                if contains:
                    self._dragging_sign = key
                    return
                
        # Existing node click logic
        nearest = self.planner.get_nearest_node(event.xdata, event.ydata)
        if nearest is None: return

        if self.phase == "START":
            self._set_start(nearest)
        elif self.phase == "TARGET":
            self._set_target(nearest)

    def _on_motion(self, event):
        if event.xdata is None or event.ydata is None:
            if self._hover_ann:
                self._hover_ann.set_visible(False)
                self.canvas.draw_idle()
            return
            
        # Handle dragging
        if self._dragging_sign:
            key = self._dragging_sign
            artist = self._sign_artists[key]
            
            # Update circle and text
            artist["circ"].center = (event.xdata, event.ydata)
            artist["txt"].set_position((event.xdata, event.ydata))
            
            # Find nearest node to update snap line
            min_d, nx, ny = float('inf'), event.xdata, event.ydata
            for nid, (px, py) in self.planner.node_positions.items():
                d = math.hypot(px-event.xdata, py-event.ydata)
                if d < min_d: min_d, nx, ny = d, px, py
                
            artist["line"].set_data([event.xdata, nx], [event.ydata, ny])
            artist["nx"], artist["ny"] = nx, ny
            self.canvas.draw_idle()
            return
            
        # Handle tooltip hover
        hovered = False
        if hasattr(self, '_sign_artists'):
            for key, artist in self._sign_artists.items():
                contains, _ = artist["circ"].contains(event)
                if contains:
                    if not self._hover_ann:
                        self._hover_ann = self.ax.annotate("", xy=(0,0), xytext=(10,10),
                                                           textcoords="offset points",
                                                           bbox=dict(boxstyle="round", fc="#2A2A38", ec="#505058"),
                                                           color="white", fontsize=9, zorder=20)
                    self._hover_ann.set_text(artist["type"])
                    self._hover_ann.xy = artist["circ"].center
                    self._hover_ann.set_visible(True)
                    self.canvas.draw_idle()
                    hovered = True
                    break
                
        if not hovered and self._hover_ann and self._hover_ann.get_visible():
            self._hover_ann.set_visible(False)
            self.canvas.draw_idle()

    def _on_release(self, event):
        if self._dragging_sign:
            key = self._dragging_sign
            artist = self._sign_artists[key]
            stype, idx = artist["type"], artist["idx"]
            
            cx, cy = artist["circ"].center
            # Update global BFMC_SIGNS format: (x, y, char, color)
            old_curr = BFMC_SIGNS[stype][idx]
            if len(old_curr) == 4:
                char, color = old_curr[2], old_curr[3]
            else:
                char, color = old_curr[0][2], old_curr[0][3]
            BFMC_SIGNS[stype][idx] = (cx, cy, char, color)
            
            self._dragging_sign = None
            self.canvas.draw_idle()

    def _set_start(self, node_id):
        pos = self.planner.node_positions[node_id]
        if self._start_dot:
            self._start_dot.remove()
        if self._ann_start:
            self._ann_start.remove()
        self._start_dot, = self.ax.plot(pos[0], pos[1], "o",
                                         color="#00FF80", ms=14, zorder=10,
                                         markeredgecolor="white", markeredgewidth=1.5)
        self._ann_start = self.ax.annotate(
            f"START\nNode {node_id}\n({pos[0]:.2f},{pos[1]:.2f})",
            pos, xytext=(12, 12), textcoords="offset points",
            color="#00FF80", fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#0A200A", ec="#00FF80", lw=1),
            zorder=11
        )
        self.start_node = node_id
        self.phase = "TARGET"
        self.lbl_phase.config(
            text="STEP 2 / 2 — Now click your TARGET (destination) node",
            fg="#FFE000"
        )
        self._update_sel_label()
        self.canvas.draw_idle()

    def _set_target(self, node_id):
        if node_id == self.start_node:
            messagebox.showwarning("Same node", "Target must differ from Start.", parent=self.win)
            return
            
        # Validate reachability before accepting
        if self.start_node:
            test_path = self.planner.plan_route(self.start_node, node_id)
            if not test_path:
                messagebox.showerror("Unreachable",
                    f"Node {node_id} cannot be reached from node {self.start_node}.\n"
                    "Choose a different target.", parent=self.win)
                return
                
        pos = self.planner.node_positions[node_id]
        if self._target_dot:
            self._target_dot.remove()
        if self._ann_target:
            self._ann_target.remove()
        self._target_dot, = self.ax.plot(pos[0], pos[1], "o",
                                          color="#FF4040", ms=14, zorder=10,
                                          markeredgecolor="white", markeredgewidth=1.5)
        self._ann_target = self.ax.annotate(
            f"TARGET\nNode {node_id}\n({pos[0]:.2f},{pos[1]:.2f})",
            pos, xytext=(12, -28), textcoords="offset points",
            color="#FF4040", fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#200A0A", ec="#FF4040", lw=1),
            zorder=11
        )
        self.target_node = node_id
        self.phase = "DONE"
        self.lbl_phase.config(
            text="✓  Both nodes set — press CONFIRM to begin",
            fg="#64DC64"
        )
        self.btn_confirm.config(state=tk.NORMAL, bg="#163A16")
        self._update_sel_label()
        self.canvas.draw_idle()

    def _apply_manual_entry(self):
        s = self.ent_start.get().strip()
        t = self.ent_target.get().strip()
        if s and s in self.planner.node_positions:
            self._set_start(s)
        if t and t in self.planner.node_positions:
            if self.phase in ("TARGET", "DONE"):
                self._set_target(t)

    def _update_sel_label(self):
        sn = f"Node {self.start_node}" if self.start_node else "—"
        tn = f"Node {self.target_node}" if self.target_node else "—"
        self.lbl_sel.config(text=f"START: {sn}    TARGET: {tn}")

    def _confirm(self):
        if self.start_node and self.target_node:
            self.on_confirm_callback(self.start_node, self.target_node)
            self.place_forget()
            self.destroy()

    def _on_force_close(self):
        if not (self.start_node and self.target_node):
            messagebox.showwarning(
                "Selection Required",
                "You must select both START and TARGET before continuing.",
                parent=self.win
            )
        else:
            self._confirm()

    def wait_for_result(self):
        """Block (in Tk mainloop) until user confirms. Call from main thread."""
        self.win.wait_window()
        return self.start_node, self.target_node


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
from matplotlib.patches import FancyArrow

class DashboardApp:
    """
    4-panel live dashboard.  Runs entirely on the Tk main thread.
    Pilot thread pushes telemetry via telem_q (maxsize=2, non-blocking).
    """

    # Colour palette
    BG        = "#0C0C0C"
    PANEL_BG  = "#161618"
    CYAN      = "#00C8FF"
    AMBER     = "#FF8C00"
    RED_C     = "#F03030"
    GREEN_C   = "#64DC64"
    MUTED     = "#787880"
    WHITE     = "#F5F5FA"
    YELLOW    = "#FFE000"

    def __init__(self, root, orchestrator: "Orchestrator"):
        self.root = root
        self.orch = orchestrator
        self.telem_q: queue.Queue = queue.Queue(maxsize=2)

        self.root.title("BFMC Autonomous Pilot V4 — Live Dashboard")
        self.root.geometry("1920x760")
        self.root.configure(bg=self.BG)

        # Map artist handles (updated without full redraw)
        self._car_dot = None
        self._la_dot = None
        self._route_line = None
        self._calib_text = None

        # Steering history
        self._steer_hist: list[float] = []
        self._steer_line = None

        # Photo image references (prevent GC)
        self._yolo_img = None
        self._bev_img = None

        self._build_layout()

    # ── Build 3-panel layout ───────────────────────────────────────────────────
    def _build_layout(self):
        root = self.root

        # ── LEFT panel: Status ────────────────────────────────────────────────
        pnl_stat = tk.Frame(root, bg=self.PANEL_BG, width=320)
        pnl_stat.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=8)
        pnl_stat.pack_propagate(False)

        tk.Label(pnl_stat, text="SYSTEM STATUS", font=("Courier", 10),
                 fg=self.MUTED, bg=self.PANEL_BG).pack(pady=(16, 2))

        self.lbl_traffic = tk.Label(pnl_stat, text="WAITING",
                                    font=("Courier", 26, "bold"),
                                    fg=self.YELLOW, bg=self.PANEL_BG)
        self.lbl_traffic.pack(pady=4)

        self.lbl_reason = tk.Label(pnl_stat, text="—",
                                   font=("Courier", 10),
                                   fg=self.MUTED, bg=self.PANEL_BG)
        self.lbl_reason.pack(pady=2)

        tk.Frame(pnl_stat, bg="#303038", height=1).pack(fill=tk.X, padx=16, pady=8)

        self.lbl_speed = tk.Label(pnl_stat, text="Speed: 0.0 %",
                                  font=("Courier", 14), fg=self.CYAN, bg=self.PANEL_BG)
        self.lbl_speed.pack(pady=6)

        self.lbl_steer = tk.Label(pnl_stat, text="Steer: 0.0°",
                                  font=("Courier", 14), fg=self.YELLOW, bg=self.PANEL_BG)
        self.lbl_steer.pack(pady=6)

        self.lbl_pose = tk.Label(pnl_stat, text="Pose: x=0.00  y=0.00  ψ=0°",
                                 font=("Courier", 10), fg=self.MUTED, bg=self.PANEL_BG)
        self.lbl_pose.pack(pady=4)

        self.lbl_anchor = tk.Label(pnl_stat, text="Anchor: —",
                                   font=("Courier", 10), fg=self.MUTED, bg=self.PANEL_BG)
        self.lbl_anchor.pack(pady=2)

        self.lbl_node = tk.Label(pnl_stat, text="Nearest Node: —",
                                 font=("Courier", 10), fg=self.MUTED, bg=self.PANEL_BG)
        self.lbl_node.pack(pady=2)

        tk.Frame(pnl_stat, bg="#303038", height=1).pack(fill=tk.X, padx=16, pady=8)

        # Calibration countdown
        self.lbl_calib = tk.Label(pnl_stat, text="CALIBRATING\n6.0 s",
                                  font=("Courier", 18, "bold"),
                                  fg=self.RED_C, bg=self.PANEL_BG)
        self.lbl_calib.pack(pady=6)

        # FPS
        self.lbl_fps = tk.Label(pnl_stat, text="FPS: —",
                                font=("Courier", 10), fg=self.MUTED, bg=self.PANEL_BG)
        self.lbl_fps.pack(pady=2)

        tk.Frame(pnl_stat, bg="#303038", height=1).pack(fill=tk.X, padx=16, pady=8)

        # Steering graph
        tk.Label(pnl_stat, text="STEERING HISTORY", font=("Courier", 9),
                 fg=self.MUTED, bg=self.PANEL_BG).pack()

        self.fig_steer, self.ax_steer = plt.subplots(figsize=(2.8, 1.4),
                                                      facecolor=self.PANEL_BG)
        self.ax_steer.set_facecolor(self.PANEL_BG)
        self.ax_steer.axhline(0, color="#404048", lw=0.8)
        self.ax_steer.set_ylim(-45, 45)
        self.ax_steer.set_xlim(0, 120)
        self.ax_steer.tick_params(colors=self.MUTED, labelsize=7)
        self.ax_steer.spines[:].set_color("#303038")
        self._steer_line, = self.ax_steer.plot([], [], color=self.CYAN, lw=1.5)
        self.fig_steer.tight_layout(pad=0.3)

        self.canvas_steer = FigureCanvasTkAgg(self.fig_steer, master=pnl_stat)
        self.canvas_steer.get_tk_widget().pack(pady=4)

        tk.Frame(pnl_stat, bg="#303038", height=1).pack(fill=tk.X, padx=16, pady=6)

        # Control buttons
        self.btn_estop = tk.Button(pnl_stat, text="⛔  E-STOP",
                              font=("Courier", 13, "bold"),
                              bg=self.RED_C, fg="white", activebackground="#A01010",
                              command=self._trigger_estop)
        self.btn_estop.pack(fill=tk.X, padx=16, pady=4)

        self.btn_pause = tk.Button(pnl_stat, text="⏸  PAUSE",
                                   font=("Courier", 11, "bold"),
                                   bg="#2A2A2E", fg=self.WHITE,
                                   command=self._toggle_pause)
        self.btn_pause.pack(fill=tk.X, padx=16, pady=4)

        btn_reroute = tk.Button(pnl_stat, text="🗺  RE-ROUTE",
                                font=("Courier", 11, "bold"),
                                bg="#2A2A2E", fg=self.CYAN,
                                command=self._reroute_dialog)
        btn_reroute.pack(fill=tk.X, padx=16, pady=4)

        btn_locate = tk.Button(pnl_stat, text="📍  RE-LOCALIZE",
                               font=("Courier", 11, "bold"),
                               bg="#2A2A2E", fg=self.AMBER,
                               command=self._relocalize_dialog)
        btn_locate.pack(fill=tk.X, padx=16, pady=4)

        # ── CENTRE panel: Map ────────────────────────────────────────────────
        pnl_map = tk.Frame(root, bg=self.PANEL_BG)
        pnl_map.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=8)

        tk.Label(pnl_map, text="COMPETITION TRACK MAP",
                 font=("Courier", 10), fg=self.MUTED,
                 bg=self.PANEL_BG).pack(pady=(8, 2))

        # Map figure — sized to fill available space
        self.fig_map, self.ax_map = plt.subplots(figsize=(9, 7),
                                                  facecolor=self.PANEL_BG)
        self.ax_map.set_facecolor("#0C0C0C")
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=pnl_map)
        self.canvas_map.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.canvas_map.mpl_connect("button_press_event", self._on_map_click)

        self._draw_static_map()

        # Waypoint readout
        self.lbl_wp = tk.Label(pnl_map, text="Next waypoints: —",
                               font=("Courier", 9), fg=self.MUTED, bg=self.PANEL_BG)
        self.lbl_wp.pack(pady=2)

        # ── RIGHT panel: Camera + BEV ─────────────────────────────────────────
        pnl_cam = tk.Frame(root, bg=self.PANEL_BG, width=480)
        pnl_cam.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=8)
        pnl_cam.pack_propagate(False)

        tk.Label(pnl_cam, text="YOLO CAMERA FEED",
                 font=("Courier", 9), fg=self.MUTED, bg=self.PANEL_BG).pack(pady=(10, 2))
        self.lbl_yolo = tk.Label(pnl_cam, bg="black")
        self.lbl_yolo.pack(pady=2)

        tk.Label(pnl_cam, text="BEV LANE TRACKER",
                 font=("Courier", 9), fg=self.MUTED, bg=self.PANEL_BG).pack(pady=(8, 2))
        self.lbl_bev = tk.Label(pnl_cam, bg="black")
        self.lbl_bev.pack(pady=2)

        tk.Label(pnl_cam, text="TRAFFIC / NAV STATE",
                 font=("Courier", 9), fg=self.MUTED, bg=self.PANEL_BG).pack(pady=(8, 2))
        self.lbl_nav = tk.Label(pnl_cam, text="NAV: —",
                                font=("Courier", 11, "bold"),
                                fg=self.CYAN, bg=self.PANEL_BG)
        self.lbl_nav.pack(pady=2)

        self.lbl_light = tk.Label(pnl_cam, text="LIGHT: NONE",
                                  font=("Courier", 11),
                                  fg=self.MUTED, bg=self.PANEL_BG)
        self.lbl_light.pack(pady=2)

        self.lbl_labels = tk.Label(pnl_cam, text="Detections: —",
                                   font=("Courier", 9),
                                   fg=self.MUTED, bg=self.PANEL_BG,
                                   wraplength=440, justify=tk.LEFT)
        self.lbl_labels.pack(pady=2, padx=8)

        # ── Schedule GUI update loop ───────────────────────────────────────────
        self.root.after(33, self._update_gui)

    # ── Draw static map elements (once at startup) ─────────────────────────────
    def _draw_static_map(self):
        self._sign_artists = draw_rich_map(self.ax_map, self.orch.planner, BFMC_SIGNS)
        
        # Cache limits for potential runtime viewport resets
        pos = self.orch.planner.node_positions
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        pad = 1.0
        self._map_xlim = (min(xs) - pad, max(xs) + pad)
        self._map_ylim = (min(ys) - pad, max(ys) + pad)

        # Initialise dynamic artists (car dot, route overlay)
        # 2.11 Car polygon
        self._car_poly = mpatches.Polygon(np.zeros((5, 2)), closed=True, color=self.YELLOW, zorder=12)
        self.ax_map.add_patch(self._car_poly)
        
        self._ghost_x = []
        self._ghost_y = []
        self._ghost_scatter = self.ax_map.scatter([], [], c=[], s=8, zorder=9, edgecolors='none')
        
        # Heading arrow patch
        self._heading_patch = FancyArrow(0, 0, 0, 0, width=0.1, color=self.YELLOW, zorder=13)
        self.ax_map.add_patch(self._heading_patch)

        # 2.10 Glowing route line
        self._route_outer, = self.ax_map.plot([], [], color="#00FFFF", lw=6, zorder=6, alpha=0.2)
        self._route_mid, = self.ax_map.plot([], [], color="#00FFFF", lw=3, zorder=7, alpha=0.5)
        self._route_inner, = self.ax_map.plot([], [], color="#FFFFFF", lw=1.5, zorder=8, alpha=0.9)
        self._route_done, = self.ax_map.plot([], [], color="#404048", lw=3, zorder=9, alpha=0.8)

        self._la_dot, = self.ax_map.plot([], [], "o", color="#FF00FF",
                                ms=6, zorder=10, alpha=0.8)

        # 2.9 Start / target markers
        self._start_flag = self.ax_map.text(0, 0, "", color="#00FF80", fontsize=18, ha="center", va="bottom", zorder=14)
        self._start_pulse = mpatches.Circle((0, 0), 0.2, color="#00FF80", fill=False, lw=1.5, zorder=11, alpha=0.8)
        self._start_pulse.set_visible(False)
        self.ax_map.add_patch(self._start_pulse)
        self._pulse_phase = 0.0

        self._target_flag = self.ax_map.text(0, 0, "", color="#FF4040", fontsize=18, ha="center", va="bottom", zorder=14)
        self._target_ring = mpatches.Circle((0, 0), 0.3, color="#FF4040", fill=False, lw=1.5, ls="--", zorder=11)
        self._target_ring.set_visible(False)
        self.ax_map.add_patch(self._target_ring)

        self._hint_line, = self.ax_map.plot([], [], color="#00FFFF", lw=1.0, ls="--", alpha=0.5, zorder=5)

        self.fig_map.tight_layout(pad=0.4)
        self.canvas_map.draw()

    def draw_route_on_map(self, path, start_node, target_node):
        """Called once after A* is computed. Draws the planned route."""
        pos = self.orch.planner.node_positions
        if path:
            rx = [pos[n][0] for n in path if n in pos]
            ry = [pos[n][1] for n in path if n in pos]
            self._route_outer.set_data(rx, ry)
            self._route_mid.set_data(rx, ry)
            self._route_inner.set_data(rx, ry)
            self._route_done.set_data(rx[:1], ry[:1])

        sp, tp = None, None
        if start_node and start_node in pos:
            sp = pos[start_node]
            self._start_flag.set_position((sp[0], sp[1]))
            self._start_flag.set_text("⚑")
            self._start_pulse.center = (sp[0], sp[1])
            self._start_pulse.set_visible(True)

        if target_node and target_node in pos:
            tp = pos[target_node]
            self._target_flag.set_position((tp[0], tp[1]))
            self._target_flag.set_text("⚑")
            self._target_ring.center = (tp[0], tp[1])
            self._target_ring.set_visible(True)

        if sp and tp:
            self._hint_line.set_data([sp[0], tp[0]], [sp[1], tp[1]])

        self.canvas_map.draw_idle()

    # ── GUI update loop (33 ms = ~30 Hz) ──────────────────────────────────────
    def _update_gui(self):
        try:
            telem = self.telem_q.get_nowait()
            self._apply_telemetry(telem)
        except queue.Empty:
            pass
        self.root.after(33, self._update_gui)

    def _apply_telemetry(self, t: dict):
        speed   = t.get("speed", 0.0)
        steer   = t.get("steer", 0.0)
        nav_st  = t.get("nav_state", "—")
        trf_st  = t.get("traffic", "SYS_GO")
        reason  = t.get("reason", "—")
        x, y, yaw = t.get("x", 0), t.get("y", 0), t.get("yaw", 0)
        fps     = t.get("fps", 0.0)
        anchor  = t.get("anchor", "—")
        nearest = t.get("nearest_node", "—")
        calib_remain = t.get("calib_remain", -1.0)
        light_st = t.get("light_status", "NONE")
        act_lbl  = t.get("active_labels", [])
        waypoints = t.get("waypoints", [])
        yolo_frame = t.get("yolo_frame")
        bev_frame  = t.get("bev_frame")

        # ── Status labels ──────────────────────────────────────────────────────
        short = trf_st.replace("SYS_", "")
        colour = self.GREEN_C if "GO" in short else \
                 self.RED_C   if "STOP" in short else self.AMBER
        self.lbl_traffic.config(text=short, fg=colour)
        self.lbl_reason.config(text=reason[:40])
        self.lbl_speed.config(text=f"Speed: {speed:.1f} %")
        self.lbl_steer.config(text=f"Steer: {steer:+.1f}°")
        yaw_deg = math.degrees(yaw) if abs(yaw) < 10 else yaw
        self.lbl_pose.config(text=f"Pose: x={x:.2f}  y={y:.2f}  ψ={yaw_deg:.1f}°")
        self.lbl_anchor.config(text=f"Anchor: {anchor}")
        self.lbl_node.config(text=f"Nearest Node: {nearest}")
        self.lbl_fps.config(text=f"FPS: {fps:.1f}")
        self.lbl_nav.config(text=f"NAV: {nav_st}")
        lc = self.GREEN_C if "GREEN" in light_st else \
             self.RED_C   if "RED" in light_st else self.MUTED
        self.lbl_light.config(text=f"LIGHT: {light_st}", fg=lc)
        lbl_str = "  ".join(act_lbl[:6]) if act_lbl else "—"
        self.lbl_labels.config(text=f"Detections: {lbl_str}")

        # Calibration overlay
        if calib_remain > 0:
            self.lbl_calib.config(
                text=f"CALIBRATING\n{calib_remain:.1f} s", fg=self.RED_C
            )
        else:
            self.lbl_calib.config(text="RUNNING ✓", fg=self.GREEN_C)

        # Waypoints readout
        if waypoints:
            wp_str = "  ".join([f"({w[0]:.1f},{w[1]:.1f})" for w in waypoints[:4]])
            self.lbl_wp.config(text=f"→ {wp_str}")

        # ── Steering history graph ──────────────────────────────────────────────
        self._steer_hist.append(steer)
        if len(self._steer_hist) > 120:
            self._steer_hist.pop(0)
        xs_h = list(range(len(self._steer_hist)))
        self._steer_line.set_data(xs_h, self._steer_hist)
        self.ax_steer.set_xlim(0, max(120, len(self._steer_hist)))
        self.canvas_steer.draw_idle()

        # ── Map: car poly + heading arrow + ghost trail ────────────────────────
        short_state = trf_st.replace("SYS_", "")
        car_color = self.GREEN_C if "GO" in short_state else self.RED_C if "STOP" in short_state else self.AMBER
        self._car_poly.set_facecolor(car_color)
        
        yaw_r = yaw if abs(yaw) < 10 else math.radians(yaw)
        L, W = 0.28, 0.14
        pts_local = np.array([[L/2, 0], [L/4, W/2], [-L/2, W/2], [-L/2, -W/2], [L/4, -W/2]])
        c_rot, s_rot = math.cos(yaw_r), math.sin(yaw_r)
        R_mat = np.array([[c_rot, -s_rot], [s_rot, c_rot]])
        pts_global = np.dot(pts_local, R_mat.T) + np.array([x, y])
        self._car_poly.set_xy(pts_global)

        # Update heading arrow
        arr_dx = 0.35 * math.cos(yaw_r)
        arr_dy = 0.35 * math.sin(yaw_r)
        self._heading_patch.set_data(x=x, y=y, dx=arr_dx, dy=-arr_dy)

        # Ghost trail
        self._ghost_x.append(x)
        self._ghost_y.append(y)
        if len(self._ghost_x) > 30:
            self._ghost_x.pop(0)
            self._ghost_y.pop(0)
        colors = np.zeros((len(self._ghost_x), 4))
        for i in range(len(self._ghost_x)):
            alpha = 0.6 * (i / 30.0)
            colors[i] = (1.0, 1.0, 0.0, alpha)
        self._ghost_scatter.set_offsets(np.c_[self._ghost_x, self._ghost_y])
        self._ghost_scatter.set_facecolors(colors)

        # 2.9 Pulse effect
        self._pulse_phase += 0.2
        self._start_pulse.set_radius(0.2 + 0.1 * math.sin(self._pulse_phase))

        # 2.10 Route done (grey out path)
        rx = self._route_outer.get_xdata()
        ry = self._route_outer.get_ydata()
        if len(rx) > 0 and nearest in self.orch.planner.node_positions:
            px, py = self.orch.planner.node_positions[nearest]
            for idx, (px_r, py_r) in enumerate(zip(rx, ry)):
                if abs(px - px_r) < 0.05 and abs(py - py_r) < 0.05:
                    self._route_done.set_data(rx[:idx+1], ry[:idx+1])
                    break

        # Lookahead dot (first waypoint)
        if waypoints:
            wx, wy = waypoints[0]
            self._la_dot.set_data([wx], [wy])
        else:
            self._la_dot.set_data([], [])

        self.canvas_map.draw_idle()

        # ── Camera feeds ────────────────────────────────────────────────────────
        if yolo_frame is not None:
            self._yolo_img = self._cv2tk(yolo_frame, 460, 280)
            self.lbl_yolo.config(image=self._yolo_img)

        if bev_frame is not None:
            self._bev_img = self._cv2tk(bev_frame, 460, 240)
            self.lbl_bev.config(image=self._bev_img)

    def _cv2tk(self, cv_img: np.ndarray, w: int, h: int):
        from PIL import Image, ImageTk
        cv_img = cv2.resize(cv_img, (w, h))
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(cv_img))

    # ── Map click → re-localize ────────────────────────────────────────────────
    def _on_map_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        nearest = self.orch.planner.get_nearest_node(event.xdata, event.ydata)
        if nearest:
            pos = self.orch.planner.node_positions[nearest]
            self.orch.localizer.set_pose(pos[0], pos[1], self.orch.localizer.yaw)
            log.info(f"MAP CLICK → Re-localized to node {nearest} @ {pos}")

    # ── Button callbacks ───────────────────────────────────────────────────────
    def _trigger_estop(self):
        estop_event.set()
        log.critical("E-STOP triggered from dashboard")

    def _toggle_pause(self):
        if hasattr(self.orch, "_paused") and self.orch._paused:
            self.orch._paused = False
            self.btn_pause.config(text="⏸  PAUSE")
        else:
            self.orch._paused = True
            self.btn_pause.config(text="▶  RESUME")

    def _reroute_dialog(self):
        new_target = simpledialog.askstring(
            "Re-Route", "Enter new TARGET node ID:",
            parent=self.root
        )
        if new_target and new_target.strip() in self.orch.planner.node_positions:
            new_target = new_target.strip()
            nearest_start = self.orch.planner.get_nearest_node(
                self.orch.localizer.x, self.orch.localizer.y
            )
            new_path = self.orch.planner.plan_route(nearest_start, new_target)
            if new_path:
                with self.orch.path_lock:
                    self.orch.planned_path = new_path
                    self.orch.target_node = new_target
                self.draw_route_on_map(new_path, nearest_start, new_target)
                log.info(f"Re-routed → target node {new_target}, path length {len(new_path)}")
            else:
                messagebox.showerror("No Path", f"Cannot reach node {new_target}.")
        elif new_target:
            messagebox.showerror("Invalid", f"Node '{new_target}' not found in map.")

    def _relocalize_dialog(self):
        node_id = simpledialog.askstring(
            "Re-Localize", "Enter current node ID (or leave blank to click map):",
            parent=self.root
        )
        if node_id and node_id.strip() in self.orch.planner.node_positions:
            node_id = node_id.strip()
            pos = self.orch.planner.node_positions[node_id]
            self.orch.localizer.set_pose(pos[0], pos[1], self.orch.localizer.yaw)
            log.info(f"Re-localized to node {node_id} @ {pos}")
        elif node_id:
            messagebox.showerror("Invalid", f"Node '{node_id}' not found.")


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR (Pilot Thread)
# ══════════════════════════════════════════════════════════════════════════════
class Orchestrator:
    CSV_SCHEMA_VER = 3
    CSV_HEADER = [
        "schema_ver", "timestamp", "fps", "yolo_ms", "vis_ms",
        "speed_pwm", "velocity_ms", "steer_angle",
        "nav_state", "traffic_state", "x_est", "y_est", "yaw_est",
        "imu_yaw_raw", "nearest_node", "curvature", "lane_l_conf", "lane_r_conf", "estop"
    ]

    def __init__(self, args):
        self.args = args
        self.planner   = PathPlanner("Competition_track_graph.graphml")
        self.hw        = HardwareIO(sim_mode=args.sim,
                                     sim_video=getattr(args, "sim_video", None))
        self.yolo_worker = ThreadedYOLODetector()
        self.traffic   = TrafficDecisionEngine(self.yolo_worker)
        self.vision    = VisionPipeline()
        self.localizer = LocalizationEngine()
        self.controller = Controller()

        self.path_lock = threading.Lock()
        self.planned_path: list = []
        self._path_cursor = 0
        self.start_node:  str  = None
        self.target_node: str  = None
        self._paused = False

        self.t0   = time.time()
        self._dt  = 0.033
        self._fps = 0.0
        self._fps_alpha = 0.2

        fname = f'bfmc_telemetry_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.csv_file   = open(fname, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self.CSV_HEADER)

    # ── Main pilot loop (runs in daemon thread) ────────────────────────────────
    def run_pilot_loop(self, dashboard: DashboardApp):
        log.info("Pilot loop started")
        frame_idx   = 0
        self._last_nearest_node = None
        map_fuse_ctr = 0
        speed  = 0.0
        steer  = 0.0
        anchor = "INIT"
        waypoints = []

        while True:
            t_start  = time.time()
            elapsed  = t_start - self.t0
            calib_remain = max(0.0, 6.0 - elapsed)
            
            # Default tracking values for CSV / GUI
            pose = None
            nearest_node = "—"
            yolo_ms = 0.0
            vis_ms = 0.0
            imu_yaw_deg = 0.0
            velocity_ms = self.hw.get_velocity_ms()
            v_res_curv = 0.0
            l_conf = 0.0
            r_conf = 0.0
            anchor = "HOLD"
            waypoints = []

            # ── E-STOP check ────────────────────────────────────────────────────
            if estop_event.is_set():
                self.hw.set_speed(0.0)
                self.hw.set_steering(0.0)
                nav_state  = "E-STOP"
                trf_result = None
                speed, steer = 0.0, 0.0
                yolo_frame = bev_frame = None
                traffic_str = "SYS_STOP"
                reason = "E-STOP"
                light_st = "NONE"
                act_lbl  = []

            elif self._paused:
                self.hw.set_speed(0.0)
                self.hw.set_steering(0.0)
                nav_state = "PAUSED"
                speed, steer = 0.0, 0.0
                yolo_frame = bev_frame = None
                traffic_str = "SYS_STOP"
                reason = "PAUSED"
                light_st = "NONE"
                act_lbl  = []

            elif calib_remain > 0:
                # ── Calibration phase: Pre-flight checklist ──────────────────
                self.hw.set_speed(0.0)
                self.hw.set_steering(0.0)
                nav_state = f"CALIBRATING"
                traffic_str = "CALIBRATING"
                if not hasattr(self, '_calib_started'):
                    self._calib_started = True
                    log.info("--- PRE-FLIGHT CALIBRATION STARTED ---")
                
                # Try to pull one frame to warm up camera and YOLO
                try:
                    frame = self.hw.capture_frame()
                    if frame is not None:
                        # Feed the pipeline once to compile/warmup allocators
                        _ = self.traffic.process(frame)
                        _ = self.vision.process(frame)
                        yolo_frame = frame  # show raw frame in GUI during calib
                        bev_frame = None
                except Exception:
                    pass
                
                reason = f"{calib_remain:.1f} s remaining"
                speed, steer = 0.0, 0.0
                light_st = "NONE"
                act_lbl  = []

                # Zero IMU yaw at t=5.5s (camera settled, ready to zero)
                if 5.4 < elapsed < 5.6:
                    raw_yaw, calib_status = self.hw.read_imu()
                    if not getattr(self, '_imu_zeroed', False):
                        self.hw.zero_imu_yaw(raw_yaw)
                        self._imu_zeroed = True
                        log.info(f"IMU yaw zeroed at {raw_yaw:.1f}°. Status: Sys={calib_status[0]} Gyro={calib_status[1]} Accel={calib_status[2]} Mag={calib_status[3]}")
                        if calib_status[1] < 2:
                            log.warning("IMU Gyro calibration logic is LOW. Do not move the car.")

                # Auto-tune SPEED_CALIB
                if not self.hw.sim_mode:
                    if 2.0 <= elapsed < 3.0:
                        self.hw.set_speed(50.0)
                        self.hw.set_steering(0.0)
                        if not hasattr(self, "_calib_speed_start_dist"):
                            self._calib_speed_start_dist = 0.0
                        self._calib_speed_start_dist += self.hw.get_velocity_ms() * self._dt
                    elif 3.0 <= elapsed < 3.1:
                        if hasattr(self, "_calib_speed_start_dist") and self._calib_speed_start_dist > 0:
                            # SPEED_CALIB = measured_distance / (50 - 12)
                            self.hw.SPEED_CALIB = self._calib_speed_start_dist / (50.0 - 12.0)
                            log.info(f"Auto-tuned SPEED_CALIB to {self.hw.SPEED_CALIB:.4f}")
                            self._calib_speed_start_dist = 0.0 # prevent recalculation

            else:
                # ── Full autonomous driving ────────────────────────────────────
                
                # Degradation tracking flags
                vision_ok = True
                yolo_ok = True

                # 1. Capture raw frame
                try:
                    frame = self.hw.capture_frame()
                    if frame is None: raise ValueError("Empty frame")
                except Exception as e:
                    log.error(f"Camera failure: {e}")
                    self.hw.set_speed(0.0)
                    self.hw.set_steering(0.0)
                    nav_state = "CAM_ERROR"
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # 2. Traffic YOLO (async)
                yolo_t0 = time.time()
                try:
                    t_res = self.traffic.process(frame)
                except Exception as e:
                    log.warning(f"YOLO failure: {e}")
                    yolo_ok = False
                    # Fallback empty traffic result
                    from traffic_module import TrafficResult
                    t_res = TrafficResult("SYS_GO", "YOLO_DEAD", 0.5)
                yolo_ms = (time.time() - yolo_t0) * 1000.0

                # 3. Vision / lane perception
                vis_t0 = time.time()
                try:
                    v_res = self.vision.process(frame)
                except Exception as e:
                    log.warning(f"Vision failure: {e}")
                    from perception import PerceptionResult
                    v_res = PerceptionResult(
                        warped_binary=None, lane_dbg=None,
                        sl=None, sr=None, lateral_error_px=0.0,
                        anchor="DEAD", confidence=0.0, lane_width_px=300.0,
                        curvature=0.0, l_conf=0.0, r_conf=0.0
                    )
                    vision_ok = False
                vis_ms = (time.time() - vis_t0) * 1000.0
                
                # Periodically log profiling metrics
                if frame_idx % 30 == 0:
                    log.info(f"Pipeline Profiling: YOLO={yolo_ms:.1f}ms, Vision={vis_ms:.1f}ms")
                    
                # 4. IMU heading
                imu_yaw_deg, imu_calib = self.hw.get_fused_imu_yaw()
                
                # Assign visual confidences for the CSV later
                v_res_curv = v_res.curvature
                l_conf = v_res.l_conf
                r_conf = v_res.r_conf

                # Snapshot the planned path to avoid mid-frame GUI mutations
                with self.path_lock:
                    current_path = list(self.planned_path)

                # 5. Map-matching correction (every 6 frames ≈ 5 Hz)
                map_fuse_ctr += 1
                if map_fuse_ctr >= 6:
                    map_fuse_ctr = 0
                    self.localizer.fuse_map_correction(
                        current_path,
                        self.planner.node_positions,
                        max_snap_m=0.60,
                        lane_conf=v_res.confidence
                    )

                # 6. Localizer update (fuses IMU + dead-reckoning + vision)
                accel = self.hw.get_imu_accel()

                if self.localizer.detect_slip(accel, velocity_ms, self._dt):
                    log.warning("Slip detected! Encoder velocity discarded.")
                    velocity_ms = 0.0

                pose = self.localizer.update(
                    velocity_ms     = velocity_ms,
                    steer_angle_deg = steer,
                    imu_yaw_deg     = imu_yaw_deg,
                    lane_error_px   = v_res.lateral_error_px,
                    lane_width_px   = v_res.lane_width_px,
                    conf            = v_res.confidence,
                    dt              = self._dt
                )

                # 7. Lookahead waypoints from A* path
                waypoints, self._path_cursor = self.planner.get_lookahead_waypoints(
                    pose[0], pose[1], current_path, cursor=self._path_cursor, lookahead_m=0.8
                )

                # 8. Nearest node (for telemetry)
                nearest_node = self.planner.get_nearest_node(pose[0], pose[1])
                
                # 8.1 Node Reset
                if nearest_node != self._last_nearest_node and nearest_node in self.planner.node_positions:
                    node_pos = self.planner.node_positions[nearest_node]
                    self.localizer.node_reset(node_pos[0], node_pos[1])
                    self._last_nearest_node = nearest_node
                
                # 8a. Lap completion check
                if nearest_node == self.target_node:
                    self._at_target_frames = getattr(self, '_at_target_frames', 0) + 1
                    if self._at_target_frames >= 15:  # ~0.5 seconds at target
                        log.info("TARGET REACHED - halting.")
                        self.hw.set_speed(0.0)
                        self.hw.set_steering(0.0)
                        estop_event.set()
                else:
                    self._at_target_frames = 0

                # 8b. Junction map override logic
                if self.planner.is_at_junction(nearest_node):
                    next_node = self.planner.get_junction_branch(nearest_node, current_path, cursor=self._path_cursor)
                    if next_node:
                        # Project next_node into BEV and bias target_x toward it
                        next_pos = self.planner.node_positions[next_node]
                        waypoints = [next_pos] + waypoints  # prepend as priority target

                # 8c. Lookahead map curvature
                map_curv = self.planner.get_path_curvature(pose[0], pose[1], current_path, cursor=self._path_cursor, window_m=1.0)

                # 9. Control
                nav_state = "PILOTING"
                ctrl = self.controller.compute(
                    v_res, pose, waypoints, nav_state,
                    t_res.state, base_speed=50.0, map_curvature=map_curv,
                    velocity_ms=velocity_ms
                )
                
                # Apply level degradation speed overrides
                speed  = ctrl.speed_pwm * t_res.speed_multiplier
                if not yolo_ok:
                    speed *= 0.40 # Level 1 degrade
                    reason = "DEGRADED: YOLO DEAD"
                if not vision_ok:
                    # Level 2 degrade: crawl briefly to see if we regain it, else dead-reckon
                    speed *= 0.25
                    reason = "DEGRADED: VISION DEAD"
                    
                steer  = ctrl.steer_angle_deg
                anchor = ctrl.anchor

                # 10. Send to hardware
                self.hw.set_speed(speed)
                self.hw.set_steering(steer)
                self._last_speed_pwm = speed

                yolo_frame  = t_res.yolo_debug_frame
                bev_frame   = v_res.lane_dbg
                traffic_str = t_res.state
                reason      = t_res.reason
                light_st    = t_res.light_status
                act_lbl     = t_res.active_labels

            # ── CSV log & Telemetry Export (Run every frame) ─────────────
            px, py, pyaw = (self.localizer.x, self.localizer.y, self.localizer.yaw) if pose is None else pose
            self.csv_writer.writerow([
                self.CSV_SCHEMA_VER,
                round(t_start, 4), round(self._fps, 1),
                round(yolo_ms, 1), round(vis_ms, 1),
                round(speed, 2), round(velocity_ms, 3), round(steer, 2),
                nav_state, traffic_str,
                round(px, 4), round(py, 4), round(pyaw, 4),
                round(imu_yaw_deg, 2), nearest_node,
                round(v_res_curv, 5), round(l_conf, 3), round(r_conf, 3), int(estop_event.is_set())
            ])

            # ── FPS calculation ──────────────────────────────────────────
            elapsed_this_frame = time.time() - t_start
            self._dt  = max(0.001, elapsed_this_frame)
            self._fps = self._fps_alpha * (1.0 / self._dt) + \
                        (1.0 - self._fps_alpha) * self._fps

            # ── Push telemetry to GUI (non-blocking) ──────────────────────
            telem = {
                "fps": self._fps,
                "speed": speed, "steer": steer,
                "nav_state": nav_state,
                "traffic": traffic_str, "reason": reason,
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "calib_remain": calib_remain,
                "anchor": anchor,
                "nearest_node": nearest_node,
                "light_status": light_st,
                "active_labels": act_lbl,
                "waypoints": waypoints,
                "yolo_frame": yolo_frame,
                "bev_frame":  bev_frame,
            }

            if pose is not None:
                telem["x"], telem["y"], telem["yaw"] = pose
            else:
                telem["x"] = self.localizer.x
                telem["y"] = self.localizer.y
                telem["yaw"] = self.localizer.yaw

            if not dashboard.telem_q.full():
                dashboard.telem_q.put(telem)

            frame_idx += 1
            sleep_sec = max(0.001, 0.033 - (time.time() - t_start))
            time.sleep(sleep_sec)

    def shutdown(self):
        self.hw.set_speed(0.0)
        self.hw.set_steering(0.0)
        self.hw.shutdown()
        if self.yolo_worker:
            self.yolo_worker.stop()
        try:
            self.csv_file.close()
        except Exception:
            pass
        log.info("Orchestrator shut down cleanly.")


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="BFMC Autonomous Pilot V4")
    p.add_argument("--sim",       action="store_true", help="Simulation mode (no hardware)")
    p.add_argument("--sim-video", type=str,            help="Path to .mp4 for camera sim")
    p.add_argument("--start",     type=str,            help="Start node ID (skips wizard)")
    p.add_argument("--target",    type=str,            help="Target node ID (skips wizard)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()

    # ── 1. Build orchestrator (loads map, inits hardware) ──────────────────────
    orch = Orchestrator(args)

    # ── 2. Start Tk root ───────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("BFMC Autonomous Dashboard")
    try:
        root.state("zoomed")
    except Exception:
        root.geometry("1400x800")
    root.configure(bg="#0C0C0C")

    # ── 3. Build and show main dashboard ──────────────────────────────────────
    dash = DashboardApp(root, orch)
    
    # Disable controls until startup is confirmed
    dash.btn_estop.config(state=tk.DISABLED)
    dash.btn_pause.config(state=tk.DISABLED)

    # ── 4. Define confirmation callback ───────────────────────────────────────
    def on_startup_confirmed(start_node, target_node):
        orch.start_node = start_node
        orch.target_node = target_node
        orch.planned_path = orch.planner.plan_route(start_node, target_node)
        
        if not orch.planned_path:
            log.warning(f"A* found no path {start_node}→{target_node}. Driving on vision only.")
        else:
            log.info(f"A* path: {len(orch.planned_path)} nodes  ({start_node} → {target_node})")

        sp = orch.planner.node_positions.get(start_node, (0.0, 0.0))
        orch.localizer.set_pose(sp[0], sp[1], 0.0)
        log.info(f"Initial pose: node {start_node} @ ({sp[0]:.2f}, {sp[1]:.2f})")
        
        dash.draw_route_on_map(orch.planned_path, start_node, target_node)
        
        dash.btn_estop.config(state=tk.NORMAL)
        dash.btn_pause.config(state=tk.NORMAL)

        # Start pilot in daemon thread
        pilot_t = threading.Thread(
            target=orch.run_pilot_loop, args=(dash,), daemon=True
        )
        pilot_t.start()

    # ── 5. Run localization wizard overlay (unless CLI args provided) ─────────
    if args.start and args.target:
        on_startup_confirmed(args.start, args.target)
    else:
        overlay = StartupOverlay(root, orch.planner, on_startup_confirmed)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

    # ── 6. Tkinter main loop ───────────────────────────────────────────────────
    def on_close():
        estop_event.set()
        time.sleep(0.1)
        orch.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()