import math
import numpy as np
import logging
import threading

log = logging.getLogger(__name__)

class LocalizationEngine:
    """3-Layer Sensor Fusion for tracking car (x, y, yaw) avoiding drift."""
    
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0  # radians
        self.wheelbase = 0.23
        self.pose_lock = threading.Lock()
        
    def set_pose(self, x, y, yaw):
        with self.pose_lock:
            self.x = x
            self.y = y
            self.yaw = yaw

    def node_reset(self, node_x, node_y):
        # Hard set x,y to known node position — eliminates accumulated along-path drift
        with self.pose_lock:
            self.x = node_x
            self.y = node_y

    def get_pose(self):
        with self.pose_lock:
            return self.x, self.y, self.yaw
        
    def update(self, velocity_ms, steer_angle_deg, imu_yaw_deg, lane_error_px, lane_width_px, conf, dt):
        """
        Layer 1: IMU yaw
        Layer 2: Dead Reckoning translation
        Layer 3a: Camera lateral snap
        """
        # 1. Yaw priority from IMU (smoothing the discontinuity wrap)
        new_yaw = math.radians(imu_yaw_deg)
        
        with self.pose_lock:
            delta = new_yaw - self.yaw
            delta = (delta + math.pi) % (2 * math.pi) - math.pi  # clamp to (-pi, pi)
            self.yaw = self.yaw + delta  # smooth update, no discontinuity
            
            # 2. Dead-reckoning for translation
            self.x += velocity_ms * dt * math.cos(self.yaw)
            self.y += velocity_ms * dt * math.sin(self.yaw)
            
            # 3a. Vision Correction
            if conf > 0.5 and lane_width_px > 0:
                lane_error_m = lane_error_px * (0.35 / lane_width_px)
                # Perpendicular-right unit vector = (sin(yaw), -cos(yaw))
                # Positive lane_error_m means car is left of center -> push pose rightward
                perp_x = math.sin(self.yaw)
                perp_y = -math.cos(self.yaw)
                self.x += 0.08 * lane_error_m * perp_x
                self.y += 0.08 * lane_error_m * perp_y
                
            return self.x, self.y, self.yaw

    def _nearest_point_on_segment(self, p, a, b):
        """Closest point on segment a->b to point p."""
        ab = (b[0]-a[0], b[1]-a[1])
        t = max(0, min(1, ((p[0]-a[0])*ab[0] + (p[1]-a[1])*ab[1]) / 
                          max(ab[0]**2 + ab[1]**2, 1e-9)))
        return (a[0] + t*ab[0], a[1] + t*ab[1])

    def fuse_map_correction(self, planned_path, node_positions, max_snap_m=0.60, lane_conf=1.0):
        """Snaps estimated position laterally toward the planned A* path segment."""
        if not planned_path or len(planned_path) < 2: return
        if lane_conf < 0.3: return # Skip snap if visual localization is poor
        
        # Find which segment of the path we are closest to
        min_dist = float('inf')
        best_pt = None
        
        for i in range(len(planned_path) - 1):
            n1, n2 = planned_path[i], planned_path[i+1]
            if n1 not in node_positions or n2 not in node_positions: continue
            
            p1 = node_positions[n1]
            p2 = node_positions[n2]
            
            pt = self._nearest_point_on_segment((self.x, self.y), p1, p2)
            d = math.hypot(pt[0] - self.x, pt[1] - self.y)
            
            if d < min_dist:
                min_dist = d
                best_pt = pt
                
        if best_pt and min_dist < max_snap_m:  # Only snap if reasonably close to track
            with self.pose_lock:
                self.x = self.x * 0.60 + best_pt[0] * 0.40
                self.y = self.y * 0.60 + best_pt[1] * 0.40

    def detect_slip(self, imu_accel_ms2, velocity_ms, dt):
        if abs(imu_accel_ms2) > 3.0 and velocity_ms < 0.05:
            return True
        return False
