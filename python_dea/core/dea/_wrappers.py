__all__ = ["DEAResult"]

from dataclasses import dataclass

import numpy as np


@dataclass
class DEAResult:
    efficiency: np.ndarray
    lambdas: np.ndarray
    slack: np.ndarray
