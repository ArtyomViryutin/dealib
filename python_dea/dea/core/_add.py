__all__ = ["add"]

import numpy as np
from numpy.typing import ArrayLike

from python_dea.dea.core._slack import slack

from .._options import RTS, Model, Orientation
from .._wrappers import Efficiency


def add(
    inputs: ArrayLike,
    outputs: ArrayLike,
    rts: RTS = RTS.vrs,
) -> Efficiency:
    rts = RTS.get(rts)

    x, y = np.asarray(inputs), np.asarray(outputs)

    k, m = x.shape
    n = y.shape[1]

    eff = Efficiency(Model.envelopment, Orientation.input, rts, k, m, n)

    eff.eff = np.ones(k)

    eff = slack(x, y, eff)

    return eff
