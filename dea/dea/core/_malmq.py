__all__ = ["malmq"]

from typing import Optional

import numpy as np

from ..utils.options import RTS, Orientation
from ..utils.types import DIRECTION, MATRIX, ORIENTATION_T, RTS_T
from ..utils.wrappers import Malmquist
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
    """
    Estimates Malmquist indices between two periods.

    :param 2-d array x0: Inputs of firm to be evaluated. `(k, m)` matrix of observations of `k` firms with `m` inputs.
        In case `transpose=True` the input matrix is transposed.

    :param 2-d array y0: Outputs of firm to be evaluated. `(k, n)` matrix of observations of `k` firms with `n` outputs.
        In case `transpose=True` the output matrix is transposed.

    :param 2-d array x1: Inputs of firm to be evaluated. `(k, m)` matrix of observations of `k` firms with `m` inputs.
        In case `transpose=True` the input matrix is transposed.

    :param 2-d array y1: Outputs of firm to be evaluated. `(k, n)` matrix of observations of `k` firms with `n` outputs.
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

    :param bool transpose: Flag determining if input and output matrix are transposed. See :mod:`x0`, :mod:`y0`,
        :mod:`x1`, :mod:`y1`.

    :return: Result malmquist object.
    :rtype: Malmquist


    **Example**

    >>> x0 = [[10], [28], [30], [60]]
    >>> y0 = [[5], [7], [10], [15]]
    >>> x1 = [[12], [26], [16], [60]]
    >>> y1 = [[6], [8], [9], [15]]
    >>> ma = malmq(x0, y0, x1, y1)
    >>> print(ma.m)
    [0.9860133  1.24869426 1.44543617 1.        ]
    """

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
