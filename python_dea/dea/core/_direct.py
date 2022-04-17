__all__ = ["direct"]

from typing import Optional

import numpy as np

from python_dea.dea._types import DIRECTION, MATRIX, ORIENTATION_T, RTS_T

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
    x = np.asarray(x)
    y = np.asarray(y)
    direct_ = np.asarray(direct_)

    eff = dea(
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

    mm = np.outer(eff.objval, direct_)
    if orientation == Orientation.input:
        div = x
    else:
        div = y

    not_nulls = div != 0
    mm[not_nulls] /= div[not_nulls]
    mm[np.logical_not(not_nulls)] = np.inf

    if orientation == Orientation.input:
        eff.eff = 1 - mm
    else:
        eff.eff = 1 + mm

    if eff.eff.shape[1] == 1:
        eff.eff = eff.eff.flatten()
    return eff
