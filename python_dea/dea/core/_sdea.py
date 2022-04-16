__all__ = ["sdea"]

from typing import List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._options import RTS, Orientation
from .._utils import validate_data
from .._wrappers import Efficiency
from ._dea import dea


def _solve_sdea(
    *,
    x: Union[List[List[float]], ArrayLike, NDArray[float]],
    y: Union[List[List[float]], ArrayLike, NDArray[float]],
    rts: Union[str, RTS] = RTS.vrs,
    orientation: Union[str, Orientation] = Orientation.input,
    direct: Optional[
        Union[List[List[float]], ArrayLike, NDArray[float]]
    ] = None,
    transpose: bool = False,
) -> Efficiency:
    k = x.shape[0]
    m = x.shape[1]
    n = y.shape[1]

    eff = Efficiency(rts, orientation, k, m, n)
    mask = np.ones(k, dtype=bool)

    for i in range(k):
        mask[i] = False
        e = dea(
            x[i, :][np.newaxis, :],
            y[i, :][np.newaxis, :],
            rts=rts,
            orientation=orientation,
            xref=x[mask, :],
            yref=y[mask, :],
            direct=direct,
            transpose=transpose,
        )
        eff.eff[i] = e.eff[0]
        eff.lambdas[i, mask] = e.lambdas[0]
        mask[i] = True

    return eff


def sdea(
    x: Union[List[List[float]], ArrayLike, NDArray[float]],
    y: Union[List[List[float]], ArrayLike, NDArray[float]],
    rts: Union[str, RTS] = RTS.vrs,
    orientation: Union[str, Orientation] = Orientation.input,
    direct: Optional[
        Union[List[List[float]], ArrayLike, NDArray[float]]
    ] = None,
    transpose: bool = False,
) -> Efficiency:
    rts = RTS.get(rts)
    orientation = Orientation.get(orientation)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if direct is not None:
        direct = np.asarray(direct, dtype=float)

    if transpose is True:
        x = x.transpose()
        y = y.transpose()

    validate_data(
        x=x, y=y, xref=x, yref=y, orientation=orientation, direct=direct
    )

    eff = _solve_sdea(
        x=x,
        y=y,
        rts=rts,
        orientation=orientation,
        direct=direct,
        transpose=transpose,
    )

    return eff
