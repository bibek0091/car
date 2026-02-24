import os
import re
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import KDTree

class GlobalMap:
    def __init__(self, svg_path="Track.svg", cache_path="map_cache.npz"):
        self.svg_path = svg_path
        self.cache_path = cache_path
        
        self.points = None
        self.headings = None
        self.curvatures = None
        self.kdtree = None
        
        self.load_map()

    def load_map(self):
        if os.path.exists(self.cache_path):
            print(f"[GlobalMap] Loading map from cache: {self.cache_path}")
            data = np.load(self.cache_path)
            self.points = data['points']
            self.headings = data['headings']
            self.curvatures = data['curvatures']
        else:
            print(f"[GlobalMap] Fast Parsing raw SVG map: {self.svg_path}")
            self._fast_parse()
            
        print(f"[GlobalMap] Loaded {len(self.points)} point cloud nodes.")
        self.kdtree = KDTree(self.points)

    def _fast_parse(self):
        tree = ET.parse(self.svg_path)
        root = tree.getroot()
        
        # Handle namespaces
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        paths = root.findall('.//svg:path', ns)
        if not paths:
            paths = root.findall('.//path')
            
        points = []
        
        for path in paths:
            d = path.get('d', '')
            if not d: continue
            
            # Regex to tokenize the svg path
            tokens = re.findall(r'([a-zA-Z])|([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', d)
            
            parsed_tokens = []
            for t in tokens:
                if t[0]: parsed_tokens.append(t[0])
                if t[1]: parsed_tokens.append(float(t[1]))
                
            curr_x = 0.0
            curr_y = 0.0
            cmd = None
            i = 0
            
            while i < len(parsed_tokens):
                if isinstance(parsed_tokens[i], str):
                    cmd = parsed_tokens[i]
                    i += 1
                else:
                    if cmd == 'M': # Absolute Move
                        curr_x = parsed_tokens[i]
                        curr_y = parsed_tokens[i+1]
                        points.append([curr_x, curr_y])
                        i += 2
                    elif cmd == 'm': # Relative Move
                        curr_x += parsed_tokens[i]
                        curr_y += parsed_tokens[i+1]
                        points.append([curr_x, curr_y])
                        i += 2
                    elif cmd == 'L': # Absolute Line
                        curr_x = parsed_tokens[i]
                        curr_y = parsed_tokens[i+1]
                        points.append([curr_x, curr_y])
                        i += 2
                    elif cmd == 'l': # Relative Line
                        curr_x += parsed_tokens[i]
                        curr_y += parsed_tokens[i+1]
                        points.append([curr_x, curr_y])
                        i += 2
                    elif cmd == 'H': # Absolute horizontal
                        curr_x = parsed_tokens[i]
                        points.append([curr_x, curr_y])
                        i += 1
                    elif cmd == 'h': # Relative horizontal
                        curr_x += parsed_tokens[i]
                        points.append([curr_x, curr_y])
                        i += 1
                    elif cmd == 'V': # Absolute vertical
                        curr_y = parsed_tokens[i]
                        points.append([curr_x, curr_y])
                        i += 1
                    elif cmd == 'v': # Relative vertical
                        curr_y += parsed_tokens[i]
                        points.append([curr_x, curr_y])
                        i += 1
                    elif cmd == 'A': # Absolute Arc
                        curr_x = parsed_tokens[i+5]
                        curr_y = parsed_tokens[i+6]
                        points.append([curr_x, curr_y])
                        i += 7
                    elif cmd == 'a': # Relative Arc
                        curr_x += parsed_tokens[i+5]
                        curr_y += parsed_tokens[i+6]
                        points.append([curr_x, curr_y])
                        i += 7
                    elif cmd in ['C', 'S', 'Q', 'T']:
                        args = {'C': 6, 'S': 4, 'Q': 4, 'T': 2}
                        n = args[cmd]
                        curr_x = parsed_tokens[i+n-2]
                        curr_y = parsed_tokens[i+n-1]
                        points.append([curr_x, curr_y])
                        i += n
                    elif cmd in ['c', 's', 'q', 't']:
                        args = {'c': 6, 's': 4, 'q': 4, 't': 2}
                        n = args[cmd]
                        curr_x += parsed_tokens[i+n-2]
                        curr_y += parsed_tokens[i+n-1]
                        points.append([curr_x, curr_y])
                        i += n
                    else:
                        i += 1

        pts = np.array(points)
        headings = np.zeros(len(pts))
        curvatures = np.zeros(len(pts))
        
        # Vectorized calculation for headings and curvatures
        if len(pts) > 1:
            diffs = np.diff(pts, axis=0) # dx, dy
            headings[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
            headings[-1] = headings[-2]
            
        self.points = pts
        self.headings = headings
        self.curvatures = curvatures
        
        np.savez(self.cache_path, points=self.points, headings=self.headings, curvatures=self.curvatures)
        print(f"[GlobalMap] Parsed and saved cache to {self.cache_path}")

    def get_nearest_spline_index(self, x, y):
        dist, index = self.kdtree.query([x, y])
        return index, dist

    def get_trajectory_window(self, start_index, num_points=20):
        end_index = min(start_index + num_points, len(self.points))
        pts = self.points[start_index:end_index]
        head = self.headings[start_index:end_index]
        curv = self.curvatures[start_index:end_index]
        return pts, head, curv

if __name__ == "__main__":
    import time
    start = time.time()
    gmap = GlobalMap()
    end = time.time()
    print(f"Map extraction completed in {end - start:.4f} seconds.")
    if len(gmap.points) > 0:
        print(f"First parsed point: {gmap.points[0]}")
        print(f"Last parsed point: {gmap.points[-1]}")
