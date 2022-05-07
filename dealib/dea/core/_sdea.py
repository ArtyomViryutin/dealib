__all__ = ["sdea"]

from typing import Optional

import numpy as np

from ..utils.options import RTS, Orientation
from ..utils.types import DIRECTION, MATRIX, ORIENTATION_T, RTS_T
from ..utils.utils import validate_data
from ..utils.wrappers import Efficiency
from ._dea import dea


def sdea(
    x: MATRIX,
    y: MATRIX,
    rts: RTS_T = RTS.vrs,
    orientation: ORIENTATION_T = Orientation.input,
    direct: Optional[DIRECTION] = None,
    transpose: bool = False,
) -> Efficiency:
    """
    Estimates a DEA frontier and calcilates super-efficiency measures.

    :param 2-d array x: Inputs of firm to be evaluated. `(k, m)` matrix of observations of `k` firms with `m` inputs.
        In case `transpose=True` the input matrix is transposed.

    :param 2-d array y: Outputs of firm to be evaluated. `(k, n)` matrix of observations of `k` firms with `n` outputs.
        In case `transpose=True` the output matrix is transposed.

    :param int, str, RTS rts:
        Returns to scale assumption.

        `0 vrs` - Variable returns to scale

        `1 crs` - Constant returns to scale

        `2 drs` - Decreasing returns to scale

        `3 irs` -  Increasing returns to scale

    :param int, str, Orientation orientation: Efficiency orientation.

        `0 input` - Input efficiency

        `1 output` - Output efficiency

    :param float, 1-d array, 2-d array direct: Directional efficiency, direct is either a scalar, an array, or a matrix
        with nonnegative elements.

    :param bool transpose: Flag determining if input and output matrix are transposed. See :mod:`x`, :mod:`y`.

    :return: Result efficiency object.
    :rtype: Efficiency


    **Example**

    >>> x = [[20], [40], [40], [40], [60], [70], [50]]
    >>> y = [[20], [30], [50], [40], [60], [20]]
    >>> eff = sdea(x, y, rts="vrs", orientation="input")
    >>> print(eff.eff)
    [2.         0.66666667 1.4375     0.55555556 0.66666667 0.4       ]
    """

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
