__all__ = ["sdea"]

from typing import Optional

import numpy as np

from .._options import RTS, Orientation
from .._types import DIRECTION, MATRIX, ORIENTATION_T, RTS_T
from .._utils import validate_data
from .._wrappers import Efficiency
from ._dea import dea


def sdea(
    x: MATRIX,
    y: MATRIX,
    rts: RTS_T = RTS.vrs,
    orientation: ORIENTATION_T = Orientation.input,
    direct: Optional[DIRECTION] = None,
    transpose: bool = False,
) -> Efficiency:
    rts = RTS.get(rts)
    orientation = Orientation.get(orientation)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if isinstance(direct, list) or isinstance(direct, np.ndarray):
        direct = np.asarray(direct, dtype=float)
        if direct.ndim > 1:
            direct_matrix = True
        else:
            direct_matrix = False
    else:
        direct_matrix = False

    if transpose is True:
        x = x.transpose()
        y = y.transpose()
        if direct_matrix is True:
            direct = direct.transpose()

    k = x.shape[0]

    validate_data(
        x=x, y=y, xref=x, yref=y, orientation=orientation, direct=direct
    )

    e = Efficiency(
        rts=rts,
        orientation=orientation,
        transpose=transpose,
        eff=np.zeros(k),
        lambdas=np.zeros((k, k)),
    )
    mask = np.ones(k, dtype=bool)

    for i in range(k):
        if direct_matrix:
            direct_ = direct[i]
        else:
            direct_ = direct

        mask[i] = False
        de = dea(
            x=x[i][np.newaxis],
            y=y[i][np.newaxis],
            rts=rts,
            orientation=orientation,
            xref=x[mask],
            yref=y[mask],
            direct=direct_,
            transpose=transpose,
        )
        e.eff[i] = de.eff[0]
        e.lambdas[i, mask] = de.lambdas[0]
        mask[i] = True

    return e
