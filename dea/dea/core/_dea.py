__all__ = ["dea"]

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from dea.dea._options import RTS, Orientation
from dea.dea._types import DIRECTION, MATRIX, ORIENTATION_T, RTS_T
from dea.dea._wrappers import Efficiency
from dea.linprog import LPP, simplex

from .._utils import (
    apply_scaling,
    construct_lpp,
    prepare_data,
    process_result_efficiency,
    validate_data,
)
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

    :param x: Inputs of firm to be evaluated. (k, m) matrix of observations of k firms with m inputs. In case
        transpose=True the input matrix is transposed
    :param y: Outputs of firm to be evaluated. (k, n) matrix of observations of k firms with n outputs. In case
        transpose=True the output matrix is transposed
    :param rts:
        Text string or number defining the underlying DEA technology / returns to scale assumption
        // 0 vrs - Variable returns to scale
        // 1 crs - Constant returns to scale
        // 2 drs - Decreasing returns to scale
        // 3 irs -  Increasing returns to scale
    :param orientation: Text string or number defining the DEA efficiency
        // 0 input - Input efficiency
        // 1 output - Output efficiency
    :param xref: Inputs of the firms determining the technology, defaults to `x`
    :param yref: Outputs of the firms determining the technology, defaults to `y`
    :param direct: Directional efficiency, direct is either a scalar, an array, or a matrix with nonnegative elements.
    :param two_phase: Flag determining slack optimization either one or two phase
    :param transpose: Flag determining if input and output matrix are transposed. See `x`, `y`.
    :return: Efficiency object

    Example
    --------
    >>> x = [[], [], [], []]
    >>> y = [[], [], [], []]
    >>> eff = dea(x, y, rts="vrs", orientation="input")
    >>> print(eff.eff)
    [1, 2, 3]
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
        se = slack(x=x, y=y, e=e, rts=rts, xref=xref, yref=yref)
        e.sx = se.sx
        e.sy = se.sy
        e.slack = se.slack
        e.lambdas = se.lambdas

    if scaling is True:
        e.sx = np.multiply(e.sx, xref_s)
        e.sy = np.multiply(e.sy, yref_s)
        e.slack = np.multiply(e.slack, np.hstack((xref_s, yref_s)))

        if isinstance(e.direct, np.ndarray):
            if orientation == Orientation.input:
                e.direct = np.multiply(e.direct, xref_s)
            else:
                e.direct = np.multiply(e.direct, yref_s)

    process_result_efficiency(e)

    return e
