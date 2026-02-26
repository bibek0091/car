import networkx as nx
import numpy as np
import logging
import math
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

# ── Zone geometry constants (derived from Competition_track_graph.graphml) ───
# x > _HW_X_THRESHOLD  → HIGHWAY zone  (205 nodes confirmed)
# y > _PARK_Y_THRESHOLD → PARKING zone  (222 nodes confirmed)
_HW_X_THRESHOLD   = 10.0
_PARK_Y_THRESHOLD =  9.0

# Roundabout centre positions (metres, map frame) — from BFMC_SIGNS in main.py
_ROUNDABOUT_CENTRES = [
    (4.94, 6.71),   # RBT-A
    (2.70, 6.70),   # RBT-B
    (2.70, 3.84),   # RBT-C
    (4.94, 3.83),   # RBT-D
    (15.48, 3.83),  # RBT-E
]
_ROUNDABOUT_RADIUS = 1.0   # metres — nodes within this radius are in-roundabout


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

            # Edge weights + dotted normalisation
            for u, v, data in self.graph.edges(data=True):
                up = self.node_positions[u]
                vp = self.node_positions[v]
                dist = math.hypot(up[0] - vp[0], up[1] - vp[1])
                self.graph[u][v]['weight'] = dist

                if 'dotted' not in data:
                    self.graph[u][v]['dotted'] = False
                elif isinstance(data['dotted'], str):
                    self.graph[u][v]['dotted'] = (
                        data['dotted'].lower() == 'true'
                    )

            # KDTree for nearest-node queries
            self._node_ids = list(self.node_positions.keys())
            coords         = [self.node_positions[n] for n in self._node_ids]
            self._kdtree   = KDTree(coords)

            # Pre-compute roundabout node set
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
        """
        Returns the driving zone at a given map-frame position:
          "HIGHWAY"  — x > 10.0 m
          "PARKING"  — y > 9.0 m
          "CITY"     — everything else
        Highway entry/exit signs override this via TrafficDecisionEngine,
        but this provides a map-based fallback that remains accurate even
        if the sign is missed.
        """
        if y > _PARK_Y_THRESHOLD:
            return "PARKING"
        if x > _HW_X_THRESHOLD:
            return "HIGHWAY"
        return "CITY"

    def get_current_edge_info(self, x, y, path, cursor=0):
        """
        Returns a dict describing the edge the car is currently on:
          {
            "dotted"   : bool   — True → dashed centre line → overtake OK
            "zone"     : str    — "CITY" | "HIGHWAY" | "PARKING"
            "in_roundabout": bool
            "bus_lane" : bool   — True → do not enter (map attr if present)
          }
        Uses the same cursor-windowed search as fuse_map_correction so it
        is O(1) and safe to call every frame.
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
            # Mid-point distance as a fast proxy
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            d = math.hypot(mid_x - x, mid_y - y)
            if d < min_dist:
                min_dist  = d
                best_edge = (n1, n2)

        if best_edge and self.graph.has_edge(*best_edge):
            edata = self.graph[best_edge[0]][best_edge[1]]
            dotted = edata.get('dotted', False)
            if isinstance(dotted, str):
                dotted = dotted.lower() == 'true'
            info["dotted"]    = bool(dotted)
            # bus_lane attribute — only present if competition map is annotated
            info["bus_lane"]  = bool(edata.get('bus_lane', False))

        # Roundabout check using pre-computed node set
        nearest = self.get_nearest_node(x, y)
        if nearest and nearest in self._roundabout_nodes:
            info["in_roundabout"] = True

        # Zone override from edge midpoint (most accurate)
        if best_edge:
            p1 = self.node_positions[best_edge[0]]
            p2 = self.node_positions[best_edge[1]]
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            info["zone"] = self.get_zone(mid_x, mid_y)

        return info

    # ── Standard planner API ──────────────────────────────────────────────────

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
        if not self.graph or start_id not in self.graph or target_id not in self.graph:
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
            pos  = self.node_positions[node]
            d    = math.hypot(pos[0] - current_x, pos[1] - current_y)
            if d < min_d and i >= cursor - 2:
                min_d       = d
                closest_idx = i

        waypoints        = []
        accumulated_dist = 0.0
        curr_x, curr_y  = current_x, current_y

        for i in range(closest_idx, len(path)):
            node = path[i]
            pos  = self.node_positions[node]
            waypoints.append(pos)
            d  = math.hypot(pos[0] - curr_x, pos[1] - curr_y)
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

        total_curvature = 0.0
        for i in range(1, len(waypoints) - 1):
            p1 = np.array(waypoints[i - 1])
            p2 = np.array(waypoints[i])
            p3 = np.array(waypoints[i + 1])
            v1 = p2 - p1
            v2 = p3 - p2
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)
            if l1 < 1e-4 or l2 < 1e-4:
                continue
            dot   = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
            angle = math.acos(dot)
            total_curvature += angle / ((l1 + l2) / 2.0)

        return total_curvature / max(1, len(waypoints) - 2)

    def is_at_junction(self, node_id):
        if not self.graph or node_id not in self.graph:
            return False
        edges = [
            (u, v, d) for u, v, d in self.graph.edges(node_id, data=True)
            if not d.get('dotted', False)
        ]
        return len(edges) > 1

    def is_roundabout_node(self, node_id):
        """Returns True if node_id is inside a roundabout."""
        return node_id in self._roundabout_nodes

    def get_junction_branch(self, node_id, target_path, cursor=0):
        if not target_path or cursor >= len(target_path) - 1:
            return node_id
        return target_path[cursor + 1]