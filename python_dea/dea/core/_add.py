__all__ = ["add"]

from typing import Optional

import numpy as np

from python_dea.dea.core._slack import slack

from .._options import RTS, Orientation
from .._types import MATRIX, RTS_T
from .._wrappers import Efficiency


def add(
    x: MATRIX,
    y: MATRIX,
    *,
    rts: RTS_T = RTS.vrs,
    xref: Optional[MATRIX] = None,
    yref: Optional[MATRIX] = None,
    transpose: Optional[bool] = False,
) -> Efficiency:
    k = x.shape[0]
    m = x.shape[1]
    n = y.shape[1]
    kr = xref.shape[0]
    eff = Efficiency(rts, Orientation.input, k, kr, m, n)
    eff.eff = np.ones(k)

    eff = slack(x, y, eff, rts=rts, xref=xref, yref=yref, transpose=transpose)

    return eff
