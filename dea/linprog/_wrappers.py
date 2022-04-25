__all__ = ["LPPResult", "LPP"]

from dataclasses import dataclass
from typing import Optional

from numpy.typing import NDArray


@dataclass
class LPPResult:
    f: float
    x: NDArray[float]
    slack: NDArray[float]
    dual: NDArray[float]


@dataclass
class LPP:
    c: Optional[NDArray[float]] = None
    A_ub: Optional[NDArray[float]] = None
    b_ub: Optional[NDArray[float]] = None
    A_eq: Optional[NDArray[float]] = None
    b_eq: Optional[NDArray[float]] = None
