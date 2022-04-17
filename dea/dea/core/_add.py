__all__ = ["add"]

from typing import Optional

import numpy as np

from dea.dea.core._slack import slack

from .._options import RTS
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
    if transpose is True:
        m = x.shape[0]
        n = y.shape[0]
        k = x.shape[1]
    else:
        m = x.shape[1]
        n = y.shape[1]
        k = x.shape[0]

    if xref is not None:
        xref = np.asarray(xref, dtype=float)
        if transpose is True:
            kr = xref.shape[1]
        else:
            kr = xref.shape[0]
    else:
        kr = k

    e = Efficiency(
        rts=rts,
        transpose=transpose,
        eff=np.ones(k),
        objval=np.zeros(k),
        lambdas=np.zeros((k, kr)),
        sx=np.zeros((k, m)),
        sy=np.zeros((k, n)),
    )

    return slack(x, y, e, rts=rts, xref=xref, yref=yref, transpose=transpose)
