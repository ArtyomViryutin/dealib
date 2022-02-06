__all__ = ["LPPResult"]

from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class LPPResult:
    f: float
    x: NDArray[float]
    slack: NDArray[float]
