__all__ = ["dea"]

from typing import List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from python_dea.dea._options import RTS, Orientation
from python_dea.dea._wrappers import Efficiency
from python_dea.linprog import LPP, simplex

from .._utils import construct_lpp, process_result_efficiency, validate_data
from ._slack import slack


def _construct_dea_lpp(
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
        lpp.A_eq = np.append(lpp.A_eq[0], 0)[np.newaxis, :]

    return lpp


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
) -> Efficiency:
    m = x.shape[0]
    k = x.shape[1]
    n = y.shape[0]

    lpp = _construct_dea_lpp(
        xref=xref, yref=yref, rts=rts, orientation=orientation, direct=direct
    )

    eff = Efficiency(rts, orientation, k, m, n)

    for i in range(k):
        if direct is not None:
            lpp.b_ub[:m] = x[:, i]
            lpp.b_ub[m : m + n] = -y[:, i]
            if orientation == Orientation.input:
                lpp.A_ub[:m, -1] = direct
                lpp.A_ub[m : m + n, -1] = 0
            else:
                lpp.A_ub[:m, -1] = 0
                lpp.A_ub[m : m + n, -1] = direct
        else:
            if orientation == Orientation.input:
                lpp.A_ub[:m, -1] = -x[:, i]
                lpp.b_ub[m : m + n] = -y[:, i]
            else:
                lpp.A_ub[m : m + n, -1] = y[:, i]
                lpp.b_ub[:m] = x[:, i]
        if two_phase:
            lpp_result = simplex(lpp, opt_f=True, opt_slacks=False)
        else:
            lpp_result = simplex(lpp, opt_f=True, opt_slacks=True)
        eff.objval[i] = lpp_result.x[-1]
        eff.lambdas[i] = lpp_result.x[:-1]
        eff.slack[i] = lpp_result.slack[: m + n]

    if direct is not None:
        mm = np.outer(eff.objval, direct)
        if orientation == Orientation.input:
            x_t = xref.transpose()
            not_nulls = x_t != 0
            mm[not_nulls] /= x_t[not_nulls]
            mm[np.logical_not(not_nulls)] = np.inf
            eff.eff = 1 - mm
        else:
            y_t = yref.transpose()
            not_nulls = y_t != 0
            mm[not_nulls] /= y_t[not_nulls]
            mm[np.logical_not(not_nulls)] = np.inf
            eff.eff = 1 + mm
    else:
        eff.eff = eff.objval.copy()
    if eff.eff.ndim == 2 and eff.eff.shape[1] == 1:
        eff.eff = eff.eff.flatten()

    return eff


def dea(
    x: Union[List[List[float]], ArrayLike, NDArray[float]],
    y: Union[List[List[float]], ArrayLike, NDArray[float]],
    *,
    rts: Union[str, RTS] = RTS.vrs,
    orientation: Union[str, Orientation] = Orientation.input,
    xref: Optional[Union[List[List[float]], ArrayLike, NDArray[float]]] = None,
    yref: Optional[Union[List[List[float]], ArrayLike, NDArray[float]]] = None,
    direct: Optional[
        Union[List[List[float]], ArrayLike, NDArray[float]]
    ] = None,
    two_phase: bool = False,
    transpose: bool = False,
) -> Efficiency:
    rts = RTS.get(rts)
    orientation = Orientation.get(orientation)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if xref is None:
        xref = x.copy()
    else:
        xref = np.asarray(xref, dtype=float)

    if yref is None:
        yref = y.copy()
    else:
        yref = np.asarray(yref, dtype=float)

    if transpose:
        x = x.transpose()
        y = y.transpose()
        xref = xref.transpose()
        yref = yref.transpose()

    validate_data(
        x=x, y=y, xref=xref, yref=yref, orientation=orientation, direct=direct
    )

    xref_m, yref_m = xref.mean(axis=0), yref.mean(axis=0)
    if (
        np.min(xref_m) < 1e-4
        or np.max(xref_m) > 10000
        or np.min(yref_m) < 1e-4
        or np.max(yref_m) > 10000
    ):
        scaling = True
        xref_s, yref_s = xref.std(axis=0), yref.std(axis=0)
        xref_s[xref_s < 1e-9] = 1
        yref_s[yref_s < 1e-9] = 1
        x = np.divide(x, xref_s)
        y = np.divide(y, yref_s)
        xref = np.divide(xref, xref_s)
        yref = np.divide(yref, yref_s)
        if direct is not None:
            if orientation == Orientation.input:
                direct = np.divide(direct, xref_s)
            else:
                direct = np.divide(direct, yref_s)
    else:
        scaling = False
        xref_s = yref_s = None

    x = x.transpose()
    y = y.transpose()
    xref = xref.transpose()
    yref = yref.transpose()

    eff = _solve_dea(
        x=x,
        y=y,
        rts=rts,
        orientation=orientation,
        xref=xref,
        yref=yref,
        direct=direct,
        two_phase=two_phase,
    )

    if two_phase:
        eff = slack(
            x=x, y=y, eff=eff, rts=rts, xref=xref, yref=yref, transpose=True
        )

    if scaling is True:
        eff.slack = np.multiply(eff.slack, np.hstack((xref_s, yref_s)))

    process_result_efficiency(eff)

    return eff
