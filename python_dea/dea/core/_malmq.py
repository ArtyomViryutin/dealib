__all__ = ["malmq"]

from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._options import RTS, Orientation
from .._wrappers import Malmquist
from ._dea import dea


def malmq(
    x0: Union[List[List[float]], ArrayLike, NDArray[float]],
    y0: Union[List[List[float]], ArrayLike, NDArray[float]],
    x1: Union[List[List[float]], ArrayLike, NDArray[float]],
    y1: Union[List[List[float]], ArrayLike, NDArray[float]],
    *,
    orientation: Orientation = Orientation.input,
    rts: RTS = RTS.vrs,
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
