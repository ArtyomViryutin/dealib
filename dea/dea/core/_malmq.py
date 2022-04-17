__all__ = ["malmq"]

from typing import Optional

import numpy as np

from dea.dea._types import DIRECTION, MATRIX, ORIENTATION_T, RTS_T

from .._options import RTS, Orientation
from .._wrappers import Malmquist
from ._dea import dea


def malmq(
    x0: MATRIX,
    y0: MATRIX,
    x1: MATRIX,
    y1: MATRIX,
    *,
    rts: RTS_T = RTS.vrs,
    orientation: ORIENTATION_T = Orientation.input,
    direct: Optional[DIRECTION] = None,
    transpose: bool = False,
) -> Malmquist:
    e00 = dea(
        x=x0,
        y=y0,
        rts=rts,
        orientation=orientation,
        direct=direct,
        transpose=transpose,
    ).eff

    e10 = dea(
        x=x1,
        y=y1,
        rts=rts,
        orientation=orientation,
        xref=x0,
        yref=y0,
        direct=direct,
        transpose=transpose,
    ).eff

    e11 = dea(
        x=x1,
        y=y1,
        rts=rts,
        orientation=orientation,
        direct=direct,
        transpose=transpose,
    ).eff

    e01 = dea(
        x=x0,
        y=y0,
        rts=rts,
        orientation=orientation,
        xref=x1,
        yref=y1,
        direct=direct,
        transpose=transpose,
    ).eff

    k = e00.shape[0]

    tc = np.zeros(k)
    valid = np.logical_and(e11 != 0, e01 != 0)
    tc[valid] = np.sqrt(e10[valid] / e11[valid] * e00[valid] / e01[valid])

    ec = np.zeros(k)
    valid = e00 != 0
    ec[valid] = e11[valid] / e00[valid]

    m = tc * ec

    mq = np.zeros(k)
    valid = np.logical_and(e00 != 0, e01 != 0)
    mq[valid] = np.sqrt(e10[valid] / e00[valid] * e11[valid] / e01[valid])

    return Malmquist(
        m=m, tc=tc, ec=ec, mq=mq, e00=e00, e10=e10, e11=e11, e01=e01
    )
