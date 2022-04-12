__all__ = ["add"]

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from python_dea.dea.core._slack import slack

from .._options import RTS, Orientation
from .._wrappers import Efficiency


def add(
    x: Union[ArrayLike, NDArray[float]],
    y: Union[ArrayLike, NDArray[float]],
    rts: Union[str, RTS] = RTS.vrs,
) -> Efficiency:
    rts = RTS.get(rts)

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    k, m = x.shape
    n = y.shape[1]

    eff = Efficiency(rts, Orientation.input, k, m, n)

    eff.eff = np.ones(k)

    eff = slack(x, y, eff)

    return eff
