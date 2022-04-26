__all__ = ["direct"]

from typing import Optional

import numpy as np

from ..utils.options import RTS, Orientation
from ..utils.types import DIRECTION, MATRIX, ORIENTATION_T, RTS_T
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
    """
    Estimates a DEA frontier and calculates directional efficiency measures.

    :param 2-d array x: Inputs of firm to be evaluated. `(k, m)` matrix of observations of `k` firms with `m` inputs.
        In case `transpose=True` the input matrix is transposed.

    :param 2-d array y: Outputs of firm to be evaluated. `(k, n)` matrix of observations of `k` firms with `n` outputs.
        In case `transpose=True` the output matrix is transposed.

    :param float, 1-d array, 2-d array direct_: Directional efficiency, direct is either a scalar, an array, or a matrix
        with nonnegative elements.

    :param int, str, RTS rts:
        Returns to scale assumption.

        `0 vrs` - Variable returns to scale

        `1 crs` - Constant returns to scale

        `2 drs` - Decreasing returns to scale

        `3 irs` -  Increasing returns to scale

    :param int, str, Orientation orientation: Efficiency orientation.

        `0 input` - Input efficiency

        `1 output` - Output efficiency

    :param 2-d array xref: Inputs of the firms determining the technology, defaults to :mod:`x`.

    :param 2-d array yref: Outputs of the firms determining the technology, defaults to :mod:`y`.

    :param bool two_phase: Flag determining slack optimization either one or two phase.

    :param bool transpose: Flag determining if input and output matrix are transposed. See :mod:`x`, :mod:`y`.

    :return: Result efficiency object.
    :rtype: Efficiency


    **Example**

    >>> x = [[20], [40], [40], [40], [60], [70], [50]]
    >>> y = [[20], [30], [50], [40], [60], [20]]
    >>> eff = direct(x, y, 0.5, rts="vrs", orientation="input")
    >>> print(eff.eff)
    [1.         0.66666667 1.         0.55555556 1.         0.4       ]
    """

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
