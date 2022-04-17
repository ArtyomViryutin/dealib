__all__ = ["malmq"]

import numpy as np

from python_dea.dea._types import MATRIX, ORIENTATION_T, RTS_T

from .._options import RTS, Orientation
from .._wrappers import Malmquist
from ._dea import dea


def malmq(
    x0: MATRIX,
    y0: MATRIX,
    x1: MATRIX,
    y1: MATRIX,
    *,
    orientation: ORIENTATION_T = Orientation.input,
    rts: RTS_T = RTS.vrs,
    transpose: bool = False,
) -> Malmquist:
    e00 = dea(
        x0, y0, rts=rts, orientation=orientation, transpose=transpose
    ).eff
    e10 = dea(
        x1,
        y1,
        rts=rts,
        orientation=orientation,
        xref=x0,
        yref=y0,
        transpose=transpose,
    ).eff
    e11 = dea(
        x1, y1, rts=rts, orientation=orientation, transpose=transpose
    ).eff
    e01 = dea(
        x0,
        y0,
        rts=rts,
        orientation=orientation,
        xref=x1,
        yref=y1,
        transpose=transpose,
    ).eff

    tc = np.sqrt(e10 / e11 * e00 / e01)
    ec = e11 / e00
    m = tc * ec
    mq = np.sqrt(e10 / e00 * e11 / e01)

    return Malmquist(m, tc, ec, mq, e00, e10, e11, e01)
