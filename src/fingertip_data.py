from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class FingertipData:
    """Data structure to store fingertip information"""
    center: Tuple[int, int]
    radius: int
    contour: np.ndarray
    timestamp: float
    