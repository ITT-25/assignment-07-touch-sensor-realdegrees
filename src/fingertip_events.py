import time
from typing import Tuple, List, Dict
import math

from src.fingertip_data import FingertipData

class FingertipEventDetector:
    """Detect events based on fingertip histories"""
    
    def __init__(self, dt: float):
        self.last_click_time = 0
        self.dt = dt
        self.last_positions: Dict[int, Tuple[int, int]] = {}  # Track last positions by fingertip ID
    
    def detect_events(self, histories: List[List[FingertipData]]) -> Dict:
        """Detect events based on fingertip histories"""
        if not histories:
            return {}

        events = {
            "tap": 0
        }
        
        # Process only the latest history entry (one finger)
        if histories and histories[0]:
            history = histories[0]
            latest_tip = history[-1]
            
            # Detect movement event if we have enough history
            if len(history) >= 3:
                events["movement"] = {
                    "x": latest_tip.center[0],
                    "y": latest_tip.center[1],
                }
            
            # Detect click events
            current_time = time.time()
            # convert to seconds
            if len(history) >= 3 and self._detect_click(history):
                # Check if we're past the cooldown period
                if current_time - self.last_click_time > 1.0:  # 1 second cooldown
                    events["tap"] = 1
                    self.last_click_time = current_time
        
        return events
    
    def _detect_click(self, history: List[FingertipData]) -> bool:
        """Detect click event based on high radius and low circularity"""
        if len(history) < 1:  # Need at least one data point
            return False

        # Calculate average radius and circularity over the last 0.1 seconds
        time_window = 0.1  # seconds
        num_samples = int(time_window / self.dt)
        num_samples = max(1, min(num_samples, len(history)))  # Ensure valid sample count

        recent_history = history[-num_samples:]
        avg_radius = sum(tip.radius for tip in recent_history) / num_samples
        avg_circularity = sum(tip.circularity for tip in recent_history) / num_samples

        # Define thresholds
        radius_threshold = 30.0
        circularity_threshold = 0.9

        # Check if the latest data meets the criteria for a tap
        if int(avg_radius) > radius_threshold and avg_circularity < circularity_threshold:
            return True

        return False
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
