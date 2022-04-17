__all__ = ["Efficiency", "Malmquist"]

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._options import RTS, Orientation


class Efficiency:
    def __init__(
        self,
        rts: RTS,
        orientation: Orientation,
        k: int,
        kr: int,
        m: int,
        n: int,
        dual: bool = False,
    ):
        self.rts: RTS = rts
        self.orientation: Orientation = orientation
        self.eff: NDArray[float] = np.zeros(k, dtype=float)
        self.objval: NDArray[float] = np.zeros(k, dtype=float)
        self.k: int = k
        self.m: int = m
        self.n: int = n
        self.dual: bool = dual
        if not dual:
            self.lambdas: NDArray[float] = np.zeros((k, kr), dtype=float)
            self.slack: NDArray[float] = np.zeros((k, m + n), dtype=float)
        else:
            self.lambdas: NDArray[float] = np.zeros((kr, m + n), dtype=float)
            self.slack: NDArray[float] = np.zeros((kr, kr), dtype=float)

    @property
    def sx(self):
        if self.dual:
            return None
        return self.slack[: self.m]

    @property
    def sy(self):
        if self.dual:
            return None
        return self.slack[self.m : self.m + self.n]


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
