__all__ = ["dea"]

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from dealib.linprog import LPP, simplex

from ..utils.options import RTS, Orientation
from ..utils.types import DIRECTION, MATRIX, ORIENTATION_T, RTS_T
from ..utils.utils import (
    apply_scaling,
    construct_lpp,
    prepare_data,
    process_result_efficiency,
    validate_data,
)
from ..utils.wrappers import Efficiency
from ._slack import slack


def _construct_lpp(
    *,
    xref: NDArray[float],
    yref: NDArray[float],
    rts: RTS,
    orientation: Orientation,
    direct: Optional[NDArray[float]],
) -> LPP:
    lpp = construct_lpp(xref, yref, rts)

    lpp.A_ub = np.hstack(
        (lpp.A_ub, np.zeros(lpp.A_ub.shape[0])[:, np.newaxis])
    )

    if orientation == Orientation.input and direct is None:
        lpp.c = np.append(lpp.c, -1)
    else:
        lpp.c = np.append(lpp.c, 1)

    if rts == RTS.vrs:
        lpp.A_eq = np.append(lpp.A_eq[0], 0)[np.newaxis]

    return lpp


def _min_direction(lpp: LPP, m: int, n: int, orientation: Orientation):
    if orientation == Orientation.input:
        md = m
        mn0 = 0
    else:
        md = n
        mn0 = m

    direct = np.zeros(md, dtype=float)
    lpp.A_ub[:, -1] = np.zeros(lpp.A_ub.shape[0], dtype=float)
    for h in range(md):
        lpp.A_ub[mn0 + h, -1] = 1
        lpp_result = simplex(lpp, opt_f=True, opt_slacks=False)
        lpp.A_ub[mn0 + h, -1] = 0
        direct[h] = lpp_result.f
    return direct


def _solve_dea(
    *,
    x: NDArray[float],
    y: NDArray[float],
    rts: RTS,
    orientation: Orientation,
    xref: NDArray[float],
    yref: NDArray[float],
    direct: Optional[NDArray[float]],
    two_phase: bool,
    transpose: bool,
) -> Efficiency:
    m = x.shape[1]
    n = y.shape[1]
    k = x.shape[0]
    kr = xref.shape[0]

    if direct is not None and isinstance(direct, np.ndarray):
        if direct.ndim > 1:
            kd = direct.shape[0]
        else:
            kd = 0
    else:
        kd = 0

    lpp = _construct_lpp(
        xref=xref, yref=yref, rts=rts, orientation=orientation, direct=direct
    )

    if direct is not None and kd <= 1 and not isinstance(direct, str):
        if orientation == Orientation.input:
            lpp.A_ub[:m, -1] = direct
        else:
            lpp.A_ub[m : m + n, -1] = direct

    if direct is not None and isinstance(direct, str) and direct == "min":
        direct_min = True
        if orientation == Orientation.input:
            direct_matrix = np.zeros((k, m))
        else:
            direct_matrix = np.zeros((k, n))
    else:
        direct_min = False
        direct_matrix = None

    e = Efficiency(
        rts=rts,
        orientation=orientation,
        eff=np.zeros(k),
        objval=np.zeros(k),
        lambdas=np.zeros((k, kr)),
        slack=np.zeros((k, m + n)),
        sx=np.zeros((k, m)),
        sy=np.zeros((k, n)),
        ux=np.zeros((k, m)),
        vy=np.zeros((k, n)),
        transpose=transpose,
    )
    for i in range(k):
        if direct_min is True:
            lpp.b_ub[:m] = x[i]
            lpp.b_ub[m : m + n] = -y[i]
            direct = _min_direction(lpp, m, n, orientation)
            if orientation == Orientation.input:
                lpp.A_ub[:m, -1] = direct
            else:
                lpp.A_ub[m : m + n, -1] = direct
            direct_matrix[i] = direct
            if np.max(direct) < 1e-6:
                e.objval[i] = 0
                e.lambdas[i, i] = 1
                continue

        if direct is None:
            if orientation == Orientation.input:
                lpp.A_ub[:m, -1] = -x[i]
                lpp.b_ub[m : m + n] = -y[i]
            else:
                lpp.A_ub[m : m + n, -1] = y[i]
                lpp.b_ub[:m] = x[i]
        else:
            lpp.b_ub[:m] = x[i]
            lpp.b_ub[m : m + n] = -y[i]
            if kd > 1:
                if orientation == Orientation.input:
                    lpp.A_ub[:m, -1] = direct[i]
                    lpp.A_ub[m : m + n, -1] = 0
                else:
                    lpp.A_ub[:m, -1] = 0
                    lpp.A_ub[m : m + n, -1] = direct[i]

        if two_phase:
            lpp_result = simplex(lpp, opt_f=True, opt_slacks=False)
        else:
            lpp_result = simplex(lpp, opt_f=True, opt_slacks=True)

        e.objval[i] = lpp_result.x[-1]

        if orientation == Orientation.input:
            sign = 1
        else:
            sign = -1
        e.ux[i] = sign * lpp_result.dual[:m]
        e.vy[i] = sign * lpp_result.dual[m : m + n]

        e.lambdas[i] = lpp_result.x[:-1]

        e.sx[i] = lpp_result.slack[:m]
        e.sy[i] = lpp_result.slack[m : m + n]

    if direct_min is True:
        direct = direct

    e.eff = e.objval.copy()
    e.direct = direct
    e.slack = np.hstack((e.sx, e.sy))
    return e


def dea(
    x: MATRIX,
    y: MATRIX,
    *,
    rts: RTS_T = RTS.vrs,
    orientation: ORIENTATION_T = Orientation.input,
    xref: Optional[MATRIX] = None,
    yref: Optional[MATRIX] = None,
    direct: Optional[DIRECTION] = None,
    two_phase: bool = False,
    transpose: bool = False,
) -> Efficiency:
    """
    Estimates a DEA frontier and calculates Farrell efficiency measures.

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

    :param 2-d array xref: Inputs of the firms determining the technology, defaults to :mod:`x`.

    :param 2-d array yref: Outputs of the firms determining the technology, defaults to :mod:`y`.

    :param float, 1-d array, 2-d array direct: Directional efficiency, direct is either a scalar, an array, or a matrix
        with nonnegative elements.

    :param bool two_phase: Flag determining slack optimization either one or two phase.

    :param bool transpose: Flag determining if input and output matrix are transposed. See :mod:`x`, :mod:`y`.

    :return: Result efficiency object.
    :rtype: Efficiency


    **Example**

    >>> x = [[20], [40], [40], [40], [60], [70], [50]]
    >>> y = [[20], [30], [50], [40], [60], [20]]
    >>> eff = dea(x, y, rts="vrs", orientation="input")
    >>> print(eff.eff)
    [1.         0.66666667 1.         0.55555556 1.         0.4       ]
    """

    rts = RTS.get(rts)
    orientation = Orientation.get(orientation)

    x, y, xref, yref, direct = prepare_data(
        x=x,
        y=y,
        xref=xref,
        yref=yref,
        transpose=transpose,
        orientation=orientation,
        direct=direct,
    )

    validate_data(
        x=x, y=y, xref=xref, yref=yref, orientation=orientation, direct=direct
    )

    scaling, xref_s, yref_s = apply_scaling(
        x=x, y=y, xref=xref, yref=yref, orientation=orientation, direct=direct
    )

    e = _solve_dea(
        x=x,
        y=y,
        rts=rts,
        orientation=orientation,
        xref=xref,
        yref=yref,
        direct=direct,
        two_phase=two_phase,
        transpose=transpose,
    )

    if two_phase:
        se = slack(x=x, y=y, e=e, xref=xref, yref=yref)
        e.sx = se.sx
        e.sy = se.sy
        e.slack = se.slack
        e.lambdas = se.lambdas

    if scaling is True:
        e.sx = np.multiply(e.sx, xref_s)
        e.sy = np.multiply(e.sy, yref_s)
        e.slack = np.multiply(e.slack, np.hstack((xref_s, yref_s)))
        e.ux = np.divide(e.ux, xref_s)
        e.vy = np.divide(e.vy, yref_s)

        if isinstance(e.direct, np.ndarray):
            if orientation == Orientation.input:
                e.direct = np.multiply(e.direct, xref_s)
            else:
                e.direct = np.multiply(e.direct, yref_s)

    process_result_efficiency(e)

    return e
