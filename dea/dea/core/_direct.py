__all__ = ["direct"]

from typing import Optional

import numpy as np

from dea.dea._types import DIRECTION, MATRIX, ORIENTATION_T, RTS_T

from .._options import RTS, Orientation
from ._dea import dea


def direct(
    x: MATRIX,
    y: MATRIX,
    direct_: DIRECTION,
    *,
    rts: RTS_T = RTS.vrs,
    orientation: ORIENTATION_T = Orientation.input,
    xref: Optional[MATRIX] = None,
    yref: Optional[MATRIX] = None,
    two_phase: bool = False,
    transpose: bool = False,
):
    e = dea(
        x=x,
        y=y,
        rts=rts,
        orientation=orientation,
        direct=direct_,
        xref=xref,
        yref=yref,
        two_phase=two_phase,
        transpose=transpose,
    )

    mm = np.outer(e.objval, e.direct)
    if orientation == Orientation.input:
        div = np.asarray(x)
    else:
        div = np.asarray(y)

    not_nulls = div != 0
    mm[not_nulls] /= div[not_nulls]
    mm[np.logical_not(not_nulls)] = np.inf

    if orientation == Orientation.input:
        e.eff = 1 - mm
    else:
        e.eff = 1 + mm

    if e.eff.shape[1] == 1:
        e.eff = e.eff.flatten()
    return e
