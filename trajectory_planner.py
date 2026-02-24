import numpy as np
import math

class MapTrajectoryPlanner:
    def __init__(self, global_map, lookahead_meters=0.8):
        self.gmap = global_map
        self.lookahead = lookahead_meters
        
    def get_local_target(self, car_x, car_y, car_yaw):
        """
        Given the car's global pose, finds the target waypoint on the map
        and transforms it into the vehicle's local frame.
        Local frame:
          x is right
          y is forward
          Origin is the car center.
        """
        # 1. Find the nearest point on the track to the car
        idx, _ = self.gmap.get_nearest_spline_index(car_x, car_y)
        
        # 2. Extract a window of points ahead
        # Assuming points are roughly 1-2cm apart, let's grab 100 points
        pts, heads, curv = self.gmap.get_trajectory_window(idx, num_points=100)
        
        if len(pts) < 10:
            return None
        
        # 3. Find a point that is 'lookahead' meters away from the car along the spline
        target_pt = pts[-1] # Default to the furthest if lookahead isn't reached
        target_head = heads[-1]
        
        car_pos = np.array([car_x, car_y])
        for i in range(len(pts)):
            dist = np.linalg.norm(pts[i] - car_pos)
            # Assuming map scale is 1 unit = 1 pixel approx, need to map to physical meters
            # The SVG track is scaled such that lane width ~ 430 units.
            # Real lane width is 0.35m. So 1 unit = 0.35 / 430 = 0.0008 meters = 0.8 mm.
            # Let's say map_scale = 0.0008
            dist_m = dist * 0.0008
            if dist_m >= self.lookahead:
                target_pt = pts[i]
                target_head = heads[i]
                break
                
        # 4. Transform target_pt into vehicle coordinate frame
        # Car's heading: car_yaw (0 = right, pi/2 = down in SVG space).
        # We need to compute dx, dy in map space
        dx = target_pt[0] - car_x
        dy = target_pt[1] - car_y
        
        # Rotate by -car_yaw to align with vehicle forward axis
        # In a standard right-hand frame:
        # local_x = dx * cos(-yaw) - dy * sin(-yaw)
        # local_y = dx * sin(-yaw) + dy * cos(-yaw)
        c = math.cos(-car_yaw)
        s = math.sin(-car_yaw)
        
        # Using SVG coordinates where Y is down, it acts a bit like a left-handed system
        # If car yaw is 0, car faces +X. Target at +Y is to its right.
        local_x = dx * c - dy * s
        local_y = dx * s + dy * c
        
        # Convert local_x from meters to Pixels in the BEV frame
        # In our BEV frame, the car is at (320, 480).
        # Forward is -Y, Right is +X.
        # So local_y (forward) needs to be mapped to pixels, local_x (right) to pixels.
        # Lane width is 280px or whatever trackbar says. Let's use 280 roughly.
        px_per_m = 280.0 / 0.35
        
        # Target in BEV coordinates
        target_x_bev = 320.0 + (local_x * px_per_m)
        target_y_bev = 480.0 - (local_y * px_per_m)
        
        return target_x_bev, target_y_bev, target_pt, target_head
