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
        """Detect click event based on radius increase from lowest point, with circularity check"""
        if len(history) < 3:  # Need minimum history to detect pattern
            return False
        
        # Get radii from recent history
        radii = [tip.radius for tip in history]
        circularities = [tip.circularity for tip in history]

        # Find the lowest radius and its index
        min_radius = min(radii)
        min_radius_index = radii.index(min_radius)
        
        # Define thresholds
        circularity_threshold = 0.85
        min_radius_growth_factor = 1.5

        # Look for a radius peak after the minimum, with time buffer
        max_radius_after = -1
        max_radius_after_index = -1
        
        # Start looking after the buffer period from the minimum
        buffer_frames = max(1, int(0.3 / self.dt))
        start_search_index = min(min_radius_index + buffer_frames, len(radii) - 1)
        
        for i in range(start_search_index, len(radii)):
            if radii[i] > max_radius_after:
                max_radius_after = radii[i]
                max_radius_after_index = i
        
        # Check if we found a valid peak after the minimum
        if max_radius_after_index == -1:
            return False

        # Check if the max radius is at least 1.5x larger than the minimum radius
        if max_radius_after < min_radius * min_radius_growth_factor:
            return False
        
        # Check if radius returns to near original low after the peak
        radius_return_variance = 1.2  # Allow 20% variance
        max_allowed_return_radius = min_radius * radius_return_variance
        
        # Look for radius returning to low after the peak
        radius_returned = False
        circularity_returned = False
        
        for i in range(max_radius_after_index + 1, len(radii)):
            if radii[i] <= max_allowed_return_radius:
                radius_returned = True
                # Check if circularity also returned above threshold at this point or later
                for j in range(i, len(circularities)):
                    if circularities[j] > circularity_threshold:
                        circularity_returned = True
                        break
                break
        
        if not radius_returned or not circularity_returned:
            return False
        
        # Check circularity conditions at key points
        circularity_at_min = circularities[min_radius_index]
        circularity_at_max = circularities[max_radius_after_index]
        
        # Circularity at lowest radius must be above 0.8
        if circularity_at_min <= circularity_threshold:
            return False
        
        # Circularity at highest radius must be below 0.8
        if circularity_at_max >= circularity_threshold:
            return False

        return True
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
