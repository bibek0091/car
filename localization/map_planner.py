import networkx as nx
import numpy as np
import logging
import math
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

class PathPlanner:
    def __init__(self, graphml_path="Competition_track_graph.graphml"):
        self.graph = None
        self.node_positions = {}
        self.load_graph(graphml_path)

    def load_graph(self, path):
        try:
            self.graph = nx.read_graphml(path)
            for node, data in self.graph.nodes(data=True):
                self.node_positions[node] = (float(data.get('x', 0)), float(data.get('y', 0)))
            
            # Compute Edge weights
            for u, v, data in self.graph.edges(data=True):
                u_pos = self.node_positions[u]
                v_pos = self.node_positions[v]
                dist = math.hypot(u_pos[0] - v_pos[0], u_pos[1] - v_pos[1])
                self.graph[u][v]['weight'] = dist
                
                # ensure dotted is explicitly tracked
                if 'dotted' not in data:
                    self.graph[u][v]['dotted'] = False
                elif isinstance(data['dotted'], str):
                    self.graph[u][v]['dotted'] = (data['dotted'].lower() == 'true')
                    
            # After loading positions:
            self._node_ids = list(self.node_positions.keys())
            coords = [self.node_positions[n] for n in self._node_ids]
            self._kdtree = KDTree(coords)
            
            log.info(f"Loaded map: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges.")
        except Exception as e:
            log.error(f"Failed to load map {path}: {e}")
            self.graph = nx.DiGraph()

    def get_nearest_node(self, x, y):
        if not hasattr(self, '_kdtree'): return None
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
            path = nx.astar_path(self.graph, start_id, target_id, heuristic=self.heuristic, weight='weight')
            return path
        except nx.NetworkXNoPath:
            log.error(f"No path found between {start_id} and {target_id}")
            return []
        except Exception as e:
            log.error(f"A* Error: {e}")
            return []

    def get_lookahead_waypoints(self, current_x, current_y, path, cursor=0, lookahead_m=0.8):
        """Returns sequence of (x,y) reaching at least lookahead_m ahead and the tracked cursor update."""
        if not path or not self.node_positions: return [], cursor
        
        # Start search from cursor
        search_start = max(0, cursor - 2)  # small lookback for robustness
        closest_idx = search_start
        min_d = float('inf')
        
        for i in range(search_start, len(path)):
            node = path[i]
            pos = self.node_positions[node]
            d = math.hypot(pos[0] - current_x, pos[1] - current_y)
            if d < min_d and i >= cursor - 2: # never go far backwards
                min_d = d
                closest_idx = i
                
        # Walk forward until distance accumulated >= lookahead_m
        waypoints = []
        accumulated_dist = 0.0
        
        curr_x, curr_y = current_x, current_y
        
        for i in range(closest_idx, len(path)):
            node = path[i]
            pos = self.node_positions[node]
            waypoints.append(pos)
            
            d = math.hypot(pos[0] - curr_x, pos[1] - curr_y)
            accumulated_dist += d
            curr_x, curr_y = pos[0], pos[1]
            
            if accumulated_dist >= lookahead_m and len(waypoints) >= 2:
                break
                
        return waypoints, closest_idx

    def get_path_curvature(self, current_x, current_y, path, cursor=0, window_m=1.2):
        """Compute average angular change per meter ahead in path."""
        waypoints, _ = self.get_lookahead_waypoints(current_x, current_y, path, cursor=cursor, lookahead_m=window_m)
        if len(waypoints) < 3: return 0.0
        
        total_curvature = 0.0
        for i in range(1, len(waypoints)-1):
            p1 = np.array(waypoints[i-1])
            p2 = np.array(waypoints[i])
            p3 = np.array(waypoints[i+1])
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)
            if l1 < 1e-4 or l2 < 1e-4: continue
                
            dot = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
            angle = math.acos(dot)
            total_curvature += angle / ((l1 + l2)/2.0)
            
        return total_curvature / max(1, len(waypoints)-2)

    def is_at_junction(self, node_id):
        if not self.graph or node_id not in self.graph: return False
        # Count non-dotted out edges
        edges = [(u, v, d) for u, v, d in self.graph.edges(node_id, data=True) if not d.get('dotted', False)]
        return len(edges) > 1

    def get_junction_branch(self, node_id, target_path, cursor=0):
        """If current node is a junction, which next node belongs to the path?"""
        if not target_path or cursor >= len(target_path) - 1: return node_id
        return target_path[cursor + 1]
