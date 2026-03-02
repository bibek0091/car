"""
safety_manager.py — Global Confidence & Safety Enforcer
=========================================================
Fuses independent confidence metrics from Lane Tracking, Localization,
Map Snapping, and YOLO Object Detection into a single Global Confidence score.

This score governs the vehicle's maximum allowed speed, enforcing strict
safety bounds (like `force_speed = 0.0` or `zone_speed * 0.5`) when confidence
drops, preventing dangerous blind actions.
"""

from collections import deque
import numpy as np

class GlobalSafetyManager:
    def __init__(self, yolo_buffer=10, snap_buffer=5):
        self.yolo_history = deque(maxlen=yolo_buffer)
        self.snap_history = deque(maxlen=snap_buffer)
        
    def update(self, lane_conf: float, loc_conf: float, 
               yolo_active: bool, snap_success: bool,
               sign_in_range: bool) -> float:
        """
        Computes the fused global confidence score.
        
        Args:
           lane_conf     : 0.0 to 1.0 (from perception.py)
           loc_conf      : 0.0 to 1.0 (from EKF / localization.py)
           yolo_active   : True if YOLO detects *any* valid object this frame
           snap_success  : True if a map-snap successfully matched a YOLO detection
           sign_in_range : True if YOLO sees a sign AND it is near enough to expect a snap
           
        Returns:
           Global confidence score [0.0 to 1.0]
        """
        
        # 1. YOLO Stability Score
        self.yolo_history.append(1.0 if yolo_active else 0.0)
        yolo_stability = np.mean(self.yolo_history) if len(self.yolo_history) > 0 else 1.0
        
        # 2. Map Snap Miss Ratio (measured only when signs are in range)
        if sign_in_range:
            self.snap_history.append(1.0 if snap_success else 0.0)
            
        snap_hit_rate = np.mean(self.snap_history) if len(self.snap_history) > 0 else 1.0
        snap_miss_ratio = 1.0 - snap_hit_rate
        
        # 3. Fuse the confidences
        # If a snap was expected but missed, that drops the map_snap component.
        map_snap_score = 1.0 if (not sign_in_range or snap_success) else 0.5
        
        global_confidence = min(
            lane_conf,
            loc_conf,
            map_snap_score,
            yolo_stability,
            1.0 - snap_miss_ratio
        )
        
        return float(max(0.0, global_confidence))

    def apply_speed_limits(self, base_pwm: float, global_conf: float) -> float:
        """
        Enforces speed caps based on the global confidence score.
        """
        if global_conf < 0.3:
            return 0.0
        elif global_conf < 0.5:
            return base_pwm * 0.5
        return base_pwm
