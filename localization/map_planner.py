"""
map_planner.py — GraphML Path Planner  (TRACK-AWARE v4)
=========================================================
Upgrades from v3 → v4 based on BFMC track geography analysis:

  TRACK-01  Zone geometry rewritten to match the actual track layout:
            - HIGHWAY:  x > 10.0  (A1 Transilvania Highway, top-centre/right)
            - PARKING:  dedicated bounding box near START area (bottom-centre)
            - SPEED_OVAL: Constantia/Ploiesti outer loop (top-right)
            - ROUNDABOUT: Mihai Viteazu Square (top-right) + Uniri Square (bottom)
            - CITY: urban grid (left/centre)

  TRACK-02  get_zone() now returns one of six values:
            "HIGHWAY" | "SPEED_OVAL" | "ROUNDABOUT" | "PARKING" | "CITY" | "START"
            Callers (behavior, localizer) receive richer context for speed limits.

  TRACK-03  get_lane_offset_px() added — returns a lateral pixel offset to keep
            the car on the RIGHT side of a two-way road, varying by zone:
            - City streets (two-way, dashed): +20 px right bias
            - Highway (multi-lane): +35 px right bias (outermost lane)
            - Speed oval: +25 px
            - Roundabout: 0 px (CCW flow dictates position)

  TRACK-04  BUS_LANE bounding box defined from track description.  Edges inside
            the bus-lane rectangle near "21 Decembrie 1989 Boulevard" are tagged
            automatically if 'bus_lane' attribute is missing from GraphML.

  TRACK-05  CROSSWALK_NODES set: known node IDs near crosswalks (151, 165, 252
            from map description). get_crosswalk_approach() returns True when the
            cursor is within N nodes of a crosswalk node.

  TRACK-06  get_speed_limit_ms() returns the zone-appropriate speed in m/s so
            the behavior controller can enforce track-specific limits without
            hardcoding them everywhere.

  TRACK-07  get_next_action() now also returns "CCW" for roundabout navigation
            (previously only LEFT/RIGHT/STRAIGHT). Caller checks this to trigger
            the roundabout behavior mode.

  TRACK-08  is_highway_node() / is_speed_oval_node() helpers added for
            behavior_controller zone mode transitions.

Fixes carried forward from v3 (MAP-01, MAP-02, MAP-03 all retained).
"""

import networkx as nx
import numpy as np
import logging
import math
import time
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

# ── Track Zone Geometry ───────────────────────────────────────────────────────
# Coordinate system: x increases RIGHT, y increases UP (GraphML world frame).
# These bounds were derived from the BFMC track map description.

# Highway "A1 Transilvania" — top strip of the map
_HW_X_MIN       = 10.0
_HW_Y_MIN       =  7.0   # above this line = highway

# Speed Oval — top-right continuous loop (Constantia / Ploiesti Streets)
_OVAL_X_MIN     = 13.0
_OVAL_Y_MIN     =  4.5
_OVAL_X_MAX     = 22.0
_OVAL_Y_MAX     = 10.0

# Parking area — bottom-centre near START / Eroilor Boulevard
_PARK_X_MIN     =  3.0
_PARK_X_MAX     =  9.0
_PARK_Y_MAX     =  1.5   # below this y = parking zone

# START area
_START_X_MIN    =  4.0
_START_X_MAX    =  7.5
_START_Y_MIN    = -0.5
_START_Y_MAX    =  1.0

# Bus lane — "21 Decembrie 1989 Boulevard" red restricted zone, bottom-centre
_BUS_X_MIN      =  5.5
_BUS_X_MAX      =  8.5
_BUS_Y_MIN      =  1.2
_BUS_Y_MAX      =  2.2

# ── Roundabout Geometry ───────────────────────────────────────────────────────
# Two main roundabouts:
#   Mihai Viteazu Square — top-right, connects highway to city
#   Uniri Square         — bottom-centre plaza
# Plus the four smaller city-grid roundabouts (same as v3).
_ROUNDABOUT_CENTRES = [
    # City grid mini-roundabouts (v3 values retained)
    (4.94,  6.71),
    (2.70,  6.70),
    (2.70,  3.84),
    (4.94,  3.83),
    # Mihai Viteazu Square (highway ↔ city interchange)
    (15.48, 3.83),
    # Uniri Square (bottom-centre)
    (6.50,  2.20),
]
_ROUNDABOUT_RADIUS = 0.70   # tightened in v3 to avoid bypass-node false positives

# ── Crosswalk Nodes (from map description) ───────────────────────────────────
# These are node IDs adjacent to zebra crossings.  Stored as strings because
# NetworkX reads GraphML node IDs as strings.
_CROSSWALK_NODE_IDS: set = {"151", "165", "252"}
_CROSSWALK_APPROACH_N = 4   # warn when cursor is within this many nodes

# ── Zone Speed Limits ─────────────────────────────────────────────────────────
# m/s targets — converted to PWM by behavior_controller using SPEED_CALIB.
ZONE_SPEED_MS = {
    "CITY"       : 0.22,   # ~22 cm/s — urban grid with pedestrians
    "HIGHWAY"    : 0.40,   # ~40 cm/s — A1 straight sections
    "SPEED_OVAL" : 0.35,   # ~35 cm/s — continuous loop, no pedestrians
    "ROUNDABOUT" : 0.16,   # ~16 cm/s — tight CCW navigation
    "PARKING"    : 0.12,   # ~12 cm/s — low-speed maneuver
    "START"      : 0.18,   # ~18 cm/s — START/PIT area
}

# ── Right-Lane Bias by Zone ───────────────────────────────────────────────────
# Extra pixel offset applied to target_x to keep car on the right side of
# two-way roads.  Positive = shift right (toward outer road edge).
ZONE_LANE_OFFSET_PX = {
    "CITY"       : 0,    # user requested exact center of road
    "HIGHWAY"    : 0,
    "SPEED_OVAL" : 0,
    "ROUNDABOUT" : 0,
    "PARKING"    : 0,
    "START"      : 0,
}


class PathPlanner:

    def __init__(self, graphml_path="Competition_track_graph.graphml"):
        self.graph          = None
        self.node_positions = {}
        self._roundabout_nodes: set  = set()
        self._highway_nodes:   set   = set()
        self._oval_nodes:      set   = set()
        self._bus_lane_nodes:  set   = set()
        self.load_graph(graphml_path)

    # ── Graph loading ─────────────────────────────────────────────────────────

    def load_graph(self, path):
        try:
            self.graph = nx.read_graphml(path)
            for node, data in self.graph.nodes(data=True):
                self.node_positions[node] = (
                    float(data.get('x', 0)),
                    float(data.get('y', 0))
                )

            for u, v, data in self.graph.edges(data=True):
                up = self.node_positions[u]
                vp = self.node_positions[v]
                dist = math.hypot(up[0] - vp[0], up[1] - vp[1])
                self.graph[u][v]['weight'] = dist

                # Normalise 'dotted' to bool
                if 'dotted' not in data:
                    self.graph[u][v]['dotted'] = False
                elif isinstance(data['dotted'], str):
                    self.graph[u][v]['dotted'] = (
                        data['dotted'].lower() == 'true')

                # TRACK-04: auto-tag bus_lane edges if not in GraphML
                if 'bus_lane' not in data:
                    mid_x = (up[0] + vp[0]) / 2.0
                    mid_y = (up[1] + vp[1]) / 2.0
                    self.graph[u][v]['bus_lane'] = _in_bus_zone(mid_x, mid_y)

            self._node_ids = list(self.node_positions.keys())
            coords         = [self.node_positions[n] for n in self._node_ids]
            self._kdtree   = KDTree(coords)

            # Classify every node into feature sets
            self._classify_nodes()

            log.info(
                "Loaded map: %d nodes, %d edges | "
                "roundabout=%d  highway=%d  oval=%d  bus_lane=%d",
                len(self.graph.nodes), len(self.graph.edges),
                len(self._roundabout_nodes), len(self._highway_nodes),
                len(self._oval_nodes),       len(self._bus_lane_nodes),
            )
        except Exception as e:
            log.error("Failed to load map %s: %s", path, e)
            self.graph = nx.DiGraph()

    def _classify_nodes(self):
        """TRACK-01/08: Classify every node into named feature sets."""
        self._roundabout_nodes.clear()
        self._highway_nodes.clear()
        self._oval_nodes.clear()
        self._bus_lane_nodes.clear()

        for nid, (nx_, ny_) in self.node_positions.items():
            # Roundabout membership
            for cx, cy in _ROUNDABOUT_CENTRES:
                if math.hypot(nx_ - cx, ny_ - cy) < _ROUNDABOUT_RADIUS:
                    self._roundabout_nodes.add(nid)
                    break

            # ── Speed oval ──
            in_oval = (_OVAL_X_MIN <= nx_ <= _OVAL_X_MAX and
                       _OVAL_Y_MIN <= ny_ <= _OVAL_Y_MAX)
            if in_oval:
                self._oval_nodes.add(nid)

            # ── Highway (excluding oval) ──
            if not in_oval and nx_ > _HW_X_MIN and ny_ > _HW_Y_MIN:
                self._highway_nodes.add(nid)

            # Bus lane
            if _in_bus_zone(nx_, ny_):
                self._bus_lane_nodes.add(nid)

    # ── Zone API ──────────────────────────────────────────────────────────────

    def get_zone(self, x: float, y: float) -> str:
        """
        TRACK-02: Returns one of six zone labels based on world coordinates.

        Priority order (most-specific first):
          ROUNDABOUT > SPEED_OVAL > HIGHWAY > PARKING > START > CITY
        """
        # Check roundabout first (can overlap other zones)
        for cx, cy in _ROUNDABOUT_CENTRES:
            if math.hypot(x - cx, y - cy) < _ROUNDABOUT_RADIUS + 0.3:
                return "ROUNDABOUT"

        if (_OVAL_X_MIN <= x <= _OVAL_X_MAX and
                _OVAL_Y_MIN <= y <= _OVAL_Y_MAX):
            return "SPEED_OVAL"

        if x > _HW_X_MIN and y > _HW_Y_MIN:
            return "HIGHWAY"

        if (_PARK_X_MIN <= x <= _PARK_X_MAX and y < _PARK_Y_MAX):
            return "PARKING"

        if (_START_X_MIN <= x <= _START_X_MAX and
                _START_Y_MIN <= y <= _START_Y_MAX):
            return "START"

        return "CITY"

    def get_zone_speed_ms(self, zone: str) -> float:
        """TRACK-06: Return target speed in m/s for a given zone label."""
        return ZONE_SPEED_MS.get(zone, ZONE_SPEED_MS["CITY"])

    def get_lane_offset_px(self, zone: str) -> int:
        """TRACK-03: Return rightward pixel bias for target_x in a given zone."""
        return ZONE_LANE_OFFSET_PX.get(zone, 0)

    def is_in_bus_lane(self, x: float, y: float) -> bool:
        """TRACK-04: True if coordinate is inside the bus-lane restricted zone."""
        return _in_bus_zone(x, y)

    # ── Node-type helpers ──────────────────────────────────────────────────────

    def is_roundabout_node(self, node_id: str) -> bool:
        return node_id in self._roundabout_nodes

    def is_highway_node(self, node_id: str) -> bool:
        """TRACK-08: True if this node lies on the A1 highway."""
        return node_id in self._highway_nodes

    def is_speed_oval_node(self, node_id: str) -> bool:
        """TRACK-08: True if this node is on the Constantia/Ploiesti oval."""
        return node_id in self._oval_nodes

    def is_bus_lane_node(self, node_id: str) -> bool:
        return node_id in self._bus_lane_nodes

    def is_at_junction(self, node_id: str) -> bool:
        if not self.graph or node_id not in self.graph:
            return False
        # MAP-02 fix: count both in- and out-edges (non-dotted)
        out_edges = [(u, v, d) for u, v, d in self.graph.edges(node_id, data=True)
                     if not d.get('dotted', False)]
        in_edges  = [(u, v, d) for u, v, d in self.graph.in_edges(node_id, data=True)
                     if not d.get('dotted', False)]
        return len(out_edges) > 1 or len(in_edges) > 1

    # ── Crosswalk detection ───────────────────────────────────────────────────

    def get_crosswalk_approach(self, path: list, cursor: int) -> bool:
        """
        TRACK-05: Returns True when the cursor is within _CROSSWALK_APPROACH_N
        nodes of a known crosswalk node.  Triggers crosswalk slow-down.
        """
        if not path:
            return False
        start = max(0, cursor)
        end   = min(len(path), cursor + _CROSSWALK_APPROACH_N + 1)
        for nid in path[start:end]:
            if nid in _CROSSWALK_NODE_IDS:
                return True
        return False

    def get_crosswalk_nodes(self) -> set:
        """Returns the set of known crosswalk node IDs."""
        return _CROSSWALK_NODE_IDS.copy()

    # ── Nearest-node lookup ───────────────────────────────────────────────────

    def get_nearest_node(self, x: float, y: float) -> str | None:
        if not hasattr(self, '_kdtree'):
            return None
        _, idx = self._kdtree.query([x, y])
        return self._node_ids[idx]

    # ── Route planning ────────────────────────────────────────────────────────

    def heuristic(self, u: str, v: str) -> float:
        u_pos = self.node_positions[u]
        v_pos = self.node_positions[v]
        return math.hypot(u_pos[0] - v_pos[0], u_pos[1] - v_pos[1])

    def plan_route(self, start_id: str, target_id: str,
                   blocked_nodes: dict = None) -> list:
        """
        A* path from start_id to target_id.
        blocked_nodes: optional dict of {node_id: expiry_time}. Nodes whose
        expiry_time is in the future are given an infinite edge weight so A*
        naturally routes around them without graph mutation.
        """
        if (not self.graph
                or start_id not in self.graph
                or target_id not in self.graph):
            log.error("Invalid start or target node for A*.")
            return []

        now = time.time() if blocked_nodes else 0.0
        active_blocked: set = set()
        if blocked_nodes:
            active_blocked = {nid for nid, exp in blocked_nodes.items()
                              if exp > time.time()}

        if active_blocked:
            # Build a view with high-cost edges touching blocked nodes
            def weight_fn(u, v, d):
                if u in active_blocked or v in active_blocked:
                    return 1e9
                return d.get('weight', 1.0)
            try:
                return nx.astar_path(
                    self.graph, start_id, target_id,
                    heuristic=self.heuristic, weight=weight_fn)
            except nx.NetworkXNoPath:
                log.error("No path (with blocks): %s → %s", start_id, target_id)
                return []
            except Exception as e:
                log.error("A* error: %s", e)
                return []
        else:
            try:
                return nx.astar_path(
                    self.graph, start_id, target_id,
                    heuristic=self.heuristic, weight='weight')
            except nx.NetworkXNoPath:
                log.error("No path: %s → %s", start_id, target_id)
                return []
            except Exception as e:
                log.error("A* error: %s", e)
                return []


    # ── Waypoint helpers ──────────────────────────────────────────────────────

    def get_lookahead_waypoints(self, current_x: float, current_y: float,
                                path: list, cursor: int = 0,
                                lookahead_m: float = 0.8) -> tuple:
        if not path or not self.node_positions:
            return [], cursor

        search_start = max(0, cursor - 2)
        closest_idx  = search_start
        min_d        = float('inf')

        for i in range(search_start, len(path)):
            node = path[i]
            pos  = self.node_positions.get(node)
            if pos is None:
                continue
            d = math.hypot(pos[0] - current_x, pos[1] - current_y)
            if d < min_d and i >= cursor - 2:
                min_d       = d
                closest_idx = i

        waypoints        = []
        accumulated_dist = 0.0
        curr_x, curr_y   = current_x, current_y

        for i in range(closest_idx, len(path)):
            node = path[i]
            pos  = self.node_positions.get(node)
            if pos is None:
                continue
            waypoints.append(pos)
            d = math.hypot(pos[0] - curr_x, pos[1] - curr_y)
            accumulated_dist += d
            curr_x, curr_y = pos
            if accumulated_dist >= lookahead_m and len(waypoints) >= 2:
                break

        new_cursor = max(cursor, closest_idx)
        return waypoints, new_cursor

    def get_path_curvature(self, current_x: float, current_y: float,
                           path: list, cursor: int = 0,
                           window_m: float = 1.2) -> float:
        """
        Signed Menger curvature κ = ±1/R averaged over the lookahead window.
        Positive = turning left (CCW), negative = turning right (CW).
        MAP-FIX-01: was always returning unsigned |κ|, causing the feed-forward
        to always steer in the same direction regardless of curve side.
        """
        waypoints, _ = self.get_lookahead_waypoints(
            current_x, current_y, path, cursor=cursor, lookahead_m=window_m
        )
        if len(waypoints) < 3:
            return 0.0

        total_curvature = 0.0
        count           = 0
        for i in range(1, len(waypoints) - 1):
            p1 = np.array(waypoints[i - 1])
            p2 = np.array(waypoints[i])
            p3 = np.array(waypoints[i + 1])
            l1 = np.linalg.norm(p2 - p1)
            l2 = np.linalg.norm(p3 - p2)
            l3 = np.linalg.norm(p3 - p1)
            if l1 < 1e-4 or l2 < 1e-4 or l3 < 1e-4:
                continue
            # Signed cross product (z-component) determines turn direction
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            area  = abs(cross) / 2.0
            curvature_i = (4.0 * area) / max(l1 * l2 * l3, 1e-8)
            # Positive cross = CCW = left turn → positive curvature
            signed_curv = curvature_i if cross > 0 else -curvature_i
            total_curvature += signed_curv
            count += 1

        return total_curvature / max(count, 1)

    # ── Edge info ─────────────────────────────────────────────────────────────

    def get_current_edge_info(self, x: float, y: float,
                              path: list, cursor: int = 0) -> dict:
        """
        Returns a dict describing the edge nearest to (x, y) on the path.
        Roundabout and zone checks use edge midpoint (MAP-03 fix retained).
        """
        info = {
            "dotted"       : False,
            "zone"         : self.get_zone(x, y),
            "in_roundabout": False,
            "bus_lane"     : False,
            "crosswalk"    : self.get_crosswalk_approach(path, cursor),
        }
        if not path or len(path) < 2:
            return info

        search_start = max(0, cursor - 2)
        search_end   = min(len(path) - 1, cursor + 6)

        min_dist  = float('inf')
        best_edge = None

        for i in range(search_start, search_end):
            n1, n2 = path[i], path[i + 1]
            if n1 not in self.node_positions or n2 not in self.node_positions:
                continue
            p1 = self.node_positions[n1]
            p2 = self.node_positions[n2]
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            d = math.hypot(mid_x - x, mid_y - y)
            if d < min_dist:
                min_dist  = d
                best_edge = (n1, n2)

        if best_edge and self.graph.has_edge(*best_edge):
            edata  = self.graph[best_edge[0]][best_edge[1]]
            dotted = edata.get('dotted', False)
            if isinstance(dotted, str):
                dotted = dotted.lower() == 'true'
            info["dotted"]   = bool(dotted)
            info["bus_lane"] = bool(edata.get('bus_lane', False))

        if best_edge:
            p1 = self.node_positions[best_edge[0]]
            p2 = self.node_positions[best_edge[1]]
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            info["zone"] = self.get_zone(mid_x, mid_y)

            for cx, cy in _ROUNDABOUT_CENTRES:
                if math.hypot(mid_x - cx, mid_y - cy) < _ROUNDABOUT_RADIUS:
                    info["in_roundabout"] = True
                    break

        return info

    # ── Junction helpers ──────────────────────────────────────────────────────

    def get_junction_branch(self, node_id: str, target_path: list,
                            cursor: int = 0) -> str:
        if not target_path or cursor >= len(target_path) - 1:
            return node_id
        return target_path[cursor + 1]

    # ── Next action prediction ────────────────────────────────────────────────

    def get_next_action(self, current_x: float, current_y: float,
                        current_yaw: float, path: list,
                        cursor: int = 0, velocity_ms: float = 0.3) -> str:
        """
        TRACK-07: Returns one of STRAIGHT | LEFT | RIGHT | CCW | HIGHWAY_MERGE.

        CCW         — cursor is at/approaching a roundabout node.
        HIGHWAY_MERGE — cursor is approaching the highway on-ramp.

        MAP-02 fix retained: velocity-adaptive lookahead, min 2.5 m.
        """
        if not path or cursor >= len(path) - 1:
            return "STRAIGHT"

        # Check roundabout approach first
        current_node = path[cursor]
        if self.is_roundabout_node(current_node):
            return "CCW"
        # Also check a few nodes ahead for early roundabout warning
        for look in range(1, min(5, len(path) - cursor)):
            nid = path[cursor + look]
            if self.is_roundabout_node(nid):
                return "CCW"

        # Highway merge detection
        if (not self.is_highway_node(current_node) and
                cursor + 3 < len(path) and
                self.is_highway_node(path[cursor + 3])):
            return "HIGHWAY_MERGE"

        # Standard directional action using adaptive lookahead
        la_m = max(2.5, velocity_ms * 6.0)
        waypoints, _ = self.get_lookahead_waypoints(
            current_x, current_y, path,
            cursor=cursor, lookahead_m=la_m
        )
        if len(waypoints) < 2:
            return "STRAIGHT"

        target_wp  = waypoints[-1]
        dx = target_wp[0] - current_x
        dy = target_wp[1] - current_y
        target_yaw = math.atan2(dy, dx)
        angle_diff = (target_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
        diff_deg   = math.degrees(angle_diff)

        if diff_deg > 20.0:
            return "LEFT"
        elif diff_deg < -20.0:
            return "RIGHT"
        return "STRAIGHT"


# ── Module-level helpers ──────────────────────────────────────────────────────

def _in_bus_zone(x: float, y: float) -> bool:
    """Returns True if (x, y) is inside the 21 Decembrie bus-lane rectangle."""
    return (_BUS_X_MIN <= x <= _BUS_X_MAX and
            _BUS_Y_MIN <= y <= _BUS_Y_MAX)