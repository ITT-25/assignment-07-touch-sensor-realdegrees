import time
from typing import Tuple, List
import math

from src.fingertip_data import FingertipData

class FingertipEventDetector:
    """Detect events based on fingertip histories"""
    
    def __init__(self, dt: float):
        self.last_click_time = 0
        self.dt = dt
    
    def detect_events(self, histories: List[List[FingertipData]]) -> List[str]:
        """Detect events based on fingertip histories"""
        if not histories:
            return []
        
        events = []
        
        # Detect click events
        current_time = time.time()
        for history in histories:
            if len(history) >= 3 and self._detect_click(history):
                # Check if we're past the cooldown period
                if current_time - self.last_click_time > 1.0:  # 1 second cooldown
                    events.append("click")
                    self.last_click_time = current_time
        
        return events
    
    def _detect_click(self, history: List[FingertipData]) -> bool:
        """Detect click event based on radius increase that stays large"""
        if len(history) < 3:  # Need minimum history to detect pattern
            return False
                
        # Get radii from recent history
        recent_history = history[-int(1.0 / self.dt):] if self.dt > 0 else history
        radii = [tip.radius for tip in recent_history]
        
        # Starting radius
        start_radius = radii[0]
        
        # Current radius
        current_radius = radii[-1]
        
        # Check if there was a significant increase that maintained until the end
        significant_increase = current_radius >= start_radius * 1.5
        
        # Check if the radius has been consistently large for the last few frames
        consistent_large = all(r >= start_radius * 1.4 for r in radii[-min(3, len(radii)):])
        
        return significant_increase and consistent_large
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
