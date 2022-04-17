__all__ = ["Efficiency", "Malmquist"]

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._options import RTS, Orientation


@dataclass
class Efficiency:
    rts: RTS = None
    orientation: Orientation = None
    transpose: bool = None
    direct: NDArray[float] = None
    eff: NDArray[float] = None
    objval: NDArray[float] = None
    lambdas: NDArray[float] = None
    sx: NDArray[float] = None
    sy: NDArray[float] = None
    slack: NDArray[float] = None
    ux: NDArray[float] = None
    vy: NDArray[float] = None


@dataclass
class Malmquist:
    m: NDArray[float]
    tc: NDArray[float]
    ec: NDArray[float]
    mq: NDArray[float]
    e00: NDArray[float]
    e10: NDArray[float]
    e11: NDArray[float]
    e01: NDArray[float]
