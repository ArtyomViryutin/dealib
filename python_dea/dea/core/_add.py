__all__ = ["add"]

from typing import List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from python_dea.dea.core._slack import slack

from .._options import RTS, Orientation
from .._wrappers import Efficiency


def add(
    x: Union[List[List[float]], ArrayLike, NDArray[float]],
    y: Union[List[List[float]], ArrayLike, NDArray[float]],
    *,
    rts: Union[str, RTS] = RTS.vrs,
    xref: Optional[Union[List[List[float]], ArrayLike, NDArray[float]]] = None,
    yref: Optional[Union[List[List[float]], ArrayLike, NDArray[float]]] = None,
    transpose: Optional[bool] = False,
) -> Efficiency:
    k = x.shape[0]
    m = x.shape[1]
    n = y.shape[1]
    eff = Efficiency(rts, Orientation.input, k, m, n)
    eff.eff = np.ones(k)

    eff = slack(x, y, eff, rts=rts, xref=xref, yref=yref, transpose=transpose)

    return eff
