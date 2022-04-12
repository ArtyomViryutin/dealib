__all__ = ["Efficiency"]

import numpy as np
from numpy.typing import NDArray

from ._options import RTS, Orientation


class Efficiency:
    def __init__(
        self,
        rts: RTS,
        orientation: Orientation,
        k: int,
        m: int,
        n: int,
        dual: bool = False,
    ):
        self.rts: RTS = rts
        self.orientation: Orientation = orientation
        self.k = k
        self.m = m
        self.n = n
        self.eff: NDArray[float] = np.zeros(k)
        self.objval: NDArray[float] = np.zeros(k)
        if not dual:
            self.lambdas: NDArray[float] = np.zeros((k, k))
            self.slack: NDArray[float] = np.zeros((k, m + n))
        else:
            self.lambdas: NDArray[float] = np.zeros((k, m + n))
            self.slack: NDArray[float] = np.zeros((k, k))

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
