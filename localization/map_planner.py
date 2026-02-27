"""
map_planner.py — GraphML Path Planner  (FIXED v3)
==================================================
Fixes applied (v2 → v3):
  MAP-01  Dead variable `start_pos` removed from get_next_action
  MAP-02  is_at_junction now checks both in_edges and out_edges on DiGraph
          so merge-only nodes (many incoming, one outgoing) are detected
  MAP-03  get_path_curvature normalisation comment clarified

Fixes carried forward from v2:
  MAP-01(v2)  get_nearest_node uses KDTree — O(log N)
  MAP-02(v2)  get_next_action uses velocity-adaptive 2.5 m lookahead
  MAP-03(v2)  get_current_edge_info roundabout check uses edge midpoint
"""

import networkx as nx
import numpy as np
import logging
import math
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

_HW_X_THRESHOLD   = 10.0
_PARK_Y_THRESHOLD =  9.0

_ROUNDABOUT_CENTRES = [
    (4.94,  6.71),
    (2.70,  6.70),
    (2.70,  3.84),
    (4.94,  3.83),
    (15.48, 3.83),
]
_ROUNDABOUT_RADIUS = 0.70   # A-04: tightened from 1.0 m — avoids misclassifying bypass nodes


class PathPlanner:

    def __init__(self, graphml_path="Competition_track_graph.graphml"):
        self.graph          = None
        self.node_positions = {}
        self._roundabout_nodes: set = set()
        self.load_graph(graphml_path)

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

                if 'dotted' not in data:
                    self.graph[u][v]['dotted'] = False
                elif isinstance(data['dotted'], str):
                    self.graph[u][v]['dotted'] = (
                        data['dotted'].lower() == 'true')

            self._node_ids = list(self.node_positions.keys())
            coords         = [self.node_positions[n] for n in self._node_ids]
            self._kdtree   = KDTree(coords)

            self._roundabout_nodes = set()
            for nid, (nx_, ny_) in self.node_positions.items():
                for cx, cy in _ROUNDABOUT_CENTRES:
                    if math.hypot(nx_ - cx, ny_ - cy) < _ROUNDABOUT_RADIUS:
                        self._roundabout_nodes.add(nid)
                        break

            log.info(
                f"Loaded map: {len(self.graph.nodes)} nodes, "
                f"{len(self.graph.edges)} edges, "
                f"{len(self._roundabout_nodes)} roundabout nodes."
            )
        except Exception as e:
            log.error(f"Failed to load map {path}: {e}")
            self.graph = nx.DiGraph()

    # ── Zone helpers ──────────────────────────────────────────────────────────

    def get_zone(self, x, y):
        if y > _PARK_Y_THRESHOLD:
            return "PARKING"
        if x > _HW_X_THRESHOLD:
            return "HIGHWAY"
        return "CITY"

    def get_current_edge_info(self, x, y, path, cursor=0):
        """
        FIX MAP-03: roundabout check now uses edge midpoint for consistency
        with the zone check (both use the same spatial reference point).
        """
        info = {
            "dotted"       : False,
            "zone"         : self.get_zone(x, y),
            "in_roundabout": False,
            "bus_lane"     : False,
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

        # FIX MAP-03: use edge midpoint for roundabout check (was nearest node)
        if best_edge:
            p1 = self.node_positions[best_edge[0]]
            p2 = self.node_positions[best_edge[1]]
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0

            # Zone from midpoint
            info["zone"] = self.get_zone(mid_x, mid_y)

            # Roundabout from midpoint — consistent with zone check
            for cx, cy in _ROUNDABOUT_CENTRES:
                if math.hypot(mid_x - cx, mid_y - cy) < _ROUNDABOUT_RADIUS:
                    info["in_roundabout"] = True
                    break

        return info

    # ── Planner API ───────────────────────────────────────────────────────────

    def get_nearest_node(self, x, y):
        if not hasattr(self, '_kdtree'):
            return None
        _, idx = self._kdtree.query([x, y])
        return self._node_ids[idx]

    def heuristic(self, u, v):
        u_pos = self.node_positions[u]
        v_pos = self.node_positions[v]
        return math.hypot(u_pos[0] - v_pos[0], u_pos[1] - v_pos[1])

    def plan_route(self, start_id, target_id):
        if (not self.graph
                or start_id not in self.graph
                or target_id not in self.graph):
            log.error("Invalid start or target node.")
            return []
        try:
            path = nx.astar_path(
                self.graph, start_id, target_id,
                heuristic=self.heuristic, weight='weight'
            )
            return path
        except nx.NetworkXNoPath:
            log.error(f"No path: {start_id} -> {target_id}")
            return []
        except Exception as e:
            log.error(f"A* Error: {e}")
            return []

    def get_lookahead_waypoints(self, current_x, current_y, path,
                                cursor=0, lookahead_m=0.8):
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

    def get_path_curvature(self, current_x, current_y, path,
                           cursor=0, window_m=1.2):
        waypoints, _ = self.get_lookahead_waypoints(
            current_x, current_y, path, cursor=cursor, lookahead_m=window_m
        )
        if len(waypoints) < 3:
            return 0.0

        # F-11: Menger circumradius curvature κ = 1/R = 4·Area / (|p1p2|·|p2p3|·|p1p3|)
        # Eliminates the angle/avg_length approximation that over-amplified short segments.
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
            # Signed area of triangle (cross product magnitude)
            area = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])) / 2.0
            # Circumradius R = (l1*l2*l3) / (4*Area); curvature κ = 1/R
            curvature_i = (4.0 * area) / max(l1 * l2 * l3, 1e-8)
            total_curvature += curvature_i
            count += 1

        return total_curvature / max(count, 1)

    def is_at_junction(self, node_id):
        if not self.graph or node_id not in self.graph:
            return False
        # FIX MAP-02: DiGraph.edges() returns only outgoing edges.
        # A merge node (many incoming, one outgoing) was missed.
        # Count non-dotted outgoing AND incoming edges for full detection.
        out_edges = [(u, v, d) for u, v, d in self.graph.edges(node_id, data=True)
                     if not d.get('dotted', False)]
        in_edges  = [(u, v, d) for u, v, d in self.graph.in_edges(node_id, data=True)
                     if not d.get('dotted', False)]
        return len(out_edges) > 1 or len(in_edges) > 1

    def is_roundabout_node(self, node_id):
        return node_id in self._roundabout_nodes

    def get_junction_branch(self, node_id, target_path, cursor=0):
        if not target_path or cursor >= len(target_path) - 1:
            return node_id
        return target_path[cursor + 1]

    def get_next_action(self, current_x, current_y, current_yaw,
                        path, cursor=0, velocity_ms=0.3):
        """
        FIX MAP-02: velocity-adaptive lookahead (min 2.5 m).
        Old value was 1.0 m — too short, often returned STRAIGHT at junctions.
        """
        if not path or cursor >= len(path) - 1:
            return "STRAIGHT"

        # FIX MAP-02: adaptive lookahead
        la_m = max(2.5, velocity_ms * 6.0)

        waypoints, _ = self.get_lookahead_waypoints(
            current_x, current_y, path,
            cursor=cursor, lookahead_m=la_m
        )

        if len(waypoints) < 2:
            return "STRAIGHT"

        # FIX MAP-01: start_pos was computed but never used — removed dead variable
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