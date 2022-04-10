__all__ = ["Efficiency"]

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ._options import RTS, Model, Orientation


class Efficiency:
    def __init__(
        self,
        model: Model,
        orientation: Orientation,
        rts: RTS,
        k: int,
        m: int,
        n: int,
    ):
        self.model = model
        self.orientation: Orientation = orientation
        self.rts: RTS = rts
        self.k = k
        self.m = m
        self.n = n
        self.direct: Optional[NDArray[float]] = None
        self.eff: NDArray[float] = np.zeros(k)
        self.objval: NDArray[float] = np.zeros(k)
        if model == Model.envelopment:
            self.lambdas: NDArray[float] = np.zeros((k, k))
            self.slack: NDArray[float] = np.zeros((k, m + n))
        else:
            self.lambdas: NDArray[float] = np.zeros((k, m + n))
            self.slack: NDArray[float] = np.zeros((k, k))

    @property
    def sx(self):
        if self.model != Model.envelopment:
            raise ValueError("Multiplier efficiency has not sx")
        return self.slack[: self.m]

    @property
    def sy(self):
        if self.model != Model.envelopment:
            raise ValueError("Multiplier efficiency has not sy")
        return self.slack[self.m : self.m + self.n]
