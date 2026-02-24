import math
import numpy as np

class LocalizationEngine:
    def __init__(self, start_x=0.0, start_y=0.0, start_yaw=0.0):
        self.x = start_x
        self.y = start_y
        self.yaw = start_yaw
        
        # Vehicle characteristics (competition dimensions)
        self.wheelbase_cm = 26.0 
        
    def reset_pose(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

    def update_dead_reckoning(self, speed_cm_s, steering_deg, dt):
        """
        Bicycle Model kinematics update.
        steering_deg is the wheel angle.
        """
        steering_rad = math.radians(steering_deg)
        dist = speed_cm_s * dt
        
        # Avoid division by zero
        if abs(steering_rad) > 1e-4:
            turning_radius = self.wheelbase_cm / math.tan(steering_rad)
            yaw_rate = speed_cm_s / turning_radius
            
            # Update position (arc)
            self.x += turning_radius * (math.sin(self.yaw + yaw_rate * dt) - math.sin(self.yaw))
            self.y -= turning_radius * (math.cos(self.yaw + yaw_rate * dt) - math.cos(self.yaw))
            self.yaw += yaw_rate * dt
        else:
            # Update position (straight line)
            self.x += dist * math.cos(self.yaw)
            self.y += dist * math.sin(self.yaw)
            
        # Normalize yaw
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
        return self.x, self.y, self.yaw
        
    def fuse_lane_correction(self, lane_error_cm, heading_error_rad):
        """
        Fuse local vision data to gently pull the global pose towards the track center.
        """
        # Cross-track error pushes the car sideways
        self.x += lane_error_cm * math.cos(self.yaw + math.pi/2)
        self.y += lane_error_cm * math.sin(self.yaw + math.pi/2)
        
        # Heading error lightly overrides the yaw
        self.yaw = self.yaw * 0.90 + heading_error_rad * 0.10
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
