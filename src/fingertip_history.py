import time
from typing import Deque, Tuple, List
from collections import deque
import math

from src.fingertip_data import FingertipData

class FingertipHistory:
    """Track fingertip history and maintain stable fingertips"""
    
    def __init__(self, history_duration: float, dt: float):
        self.histories: List[Deque[FingertipData]] = []
        self.history_duration = history_duration
        self.dt = dt

    def update(self, fingertips: List[FingertipData]) -> None:
        """Update histories with current fingertips"""        
        # Match current fingertips with existing histories
        matched_indices = self._match_fingertips(fingertips)

        # Add unmatched fingertips as new histories
        self._add_new_fingertips(fingertips, matched_indices)
        
        # Remove old data
        self._clean_old_data()
    
    def _match_fingertips(self, current_data: List[FingertipData]) -> set:
        """Match current fingertips with existing histories"""
        matched_indices = set()
        used_fingertips = set()
        
        for j, history in enumerate(self.histories):
            if not history:
                continue
                
            best_match = None
            best_distance = float('inf')
            best_fingertip_idx = -1
            
            for i, fingertip in enumerate(current_data):
                if i in used_fingertips:
                    continue

                distance = self._calculate_distance(fingertip.center, history[-1].center)

                # Dynamic threshold based on recent movement
                if len(history) >= 2:
                    recent_movement = self._calculate_distance(history[-1].center, history[-2].center)
                    threshold = max(150, recent_movement * 2)
                else:
                    threshold = 150

                if distance < threshold and distance < best_distance:
                    best_match = fingertip
                    best_distance = distance
                    best_fingertip_idx = i
            
            if best_match is not None:
                history.append(best_match)
                matched_indices.add(j)
                used_fingertips.add(best_fingertip_idx)
        
        return matched_indices
    
    def _add_new_fingertips(self, current_data: List[FingertipData], matched_indices: set) -> None:
        """Add unmatched fingertips as new histories"""
        used_fingertips = set()
        
        # Find which fingertips were already matched
        for j in matched_indices:
            if j < len(self.histories) and self.histories[j]:
                last_tip: FingertipData = self.histories[j][-1]
                for i, fingertip in enumerate(current_data):
                    if (fingertip.center == last_tip.center and
                        fingertip.radius == last_tip.radius and
                        abs(fingertip.timestamp - last_tip.timestamp) < 0.1):
                        used_fingertips.add(i)
                        break
        
        # Add unmatched fingertips as new histories
        for i, fingertip in enumerate(current_data):
            if i not in used_fingertips:
                new_history = deque(maxlen=int(self.history_duration / self.dt))
                new_history.append(fingertip)
                self.histories.append(new_history)
    
    def _clean_old_data(self) -> None:
        """Remove histories that haven't been updated recently"""
        for i in range(len(self.histories) - 1, -1, -1):
            if self.histories[i] and (time.time() - self.histories[i][-1].timestamp) > 0.5:
                del self.histories[i]

    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_stable_fingertips(self) -> List[List[FingertipData]]:
        """Get all stable fingertip histories"""
        return [list(hist) for hist in self.histories if len(hist) >= 3]

