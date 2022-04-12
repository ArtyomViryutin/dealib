__all__ = ["dea"]

from typing import List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from python_dea.dea._options import RTS, Orientation
from python_dea.dea._wrappers import Efficiency
from python_dea.linprog import LPP, simplex

from .._utils import construct_lpp
from ._slack import slack


def construct_dea_lpp(
    xref: NDArray[float],
    yref: NDArray[float],
    rts: RTS,
    orientation: Orientation,
    direct: Optional[NDArray[float]] = None,
) -> LPP:
    lpp = construct_lpp(xref, yref, rts)

    lpp.A_ub = np.hstack(
        (lpp.A_ub, np.zeros(lpp.A_ub.shape[0])[:, np.newaxis])
    )

    if orientation == Orientation.input and not direct:
        lpp.c = np.append(lpp.c, -1)
    else:
        lpp.c = np.append(lpp.c, 1)

    if rts == RTS.vrs:
        lpp.A_eq = np.array([np.append(lpp.A_eq[0], 0)])

    return lpp


def find_inefficient_dmu(x: NDArray[float], y: NDArray[float]) -> List[int]:
    k = x.shape[1]
    inefficient_dmu = []
    for i in range(k):
        for j in range(k):
            if (x[:, i] > x[:, j]).all() and (y[:, i] < y[:, j]).all():
                inefficient_dmu.append(i)
                break
    return inefficient_dmu


def solve_dea(
    x: NDArray[float],
    y: NDArray[float],
    rts: RTS,
    orientation: Orientation,
    xref: NDArray[float],
    yref: NDArray[float],
    two_phase: bool,
) -> Efficiency:
    m, k = x.shape
    n = y.shape[0]

    lpp = construct_dea_lpp(xref, yref, rts, orientation)
    inefficient_dmu = find_inefficient_dmu(xref, yref)

    eff_dmu = np.ones(k, dtype=bool)
    eff_dmu[inefficient_dmu] = False
    lpp.c = np.delete(lpp.c, inefficient_dmu)
    lpp.A_ub = np.delete(lpp.A_ub, inefficient_dmu, axis=1)
    if lpp.A_eq is not None:
        lpp.A_eq = np.delete(lpp.A_eq, inefficient_dmu, axis=1)

    eff_dmu_count = 0
    eff = Efficiency(rts, orientation, k, m, n)

    for i in range(k):
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
        eff.eff[i] = lpp_result.x[-1]
        eff.lambdas[i, eff_dmu] = lpp_result.x[:-1]
        eff.slack[i] = lpp_result.slack[: m + n]

        # TODO можно ли так делать при том чот технологическое множество другое
        # if eff_dmu[i]:
        #     if np.abs(1 - eff.eff[i]) > 1e-9:
        #         eff_dmu[i] = False
        #         lpp.c = np.delete(lpp.c, eff_dmu_count)
        #         lpp.A_ub = np.delete(lpp.A_ub, eff_dmu_count, axis=1)
        #         if lpp.A_eq is not None:
        #             lpp.A_eq = np.delete(lpp.A_eq, eff_dmu_count, axis=1)
        #     else:
        #         eff_dmu_count += 1
    if two_phase:
        eff = slack(x, y, eff, transpose=True)
    eff.objval = eff.eff
    return eff


def dea(
    x: Union[ArrayLike, NDArray[float]],
    y: Union[ArrayLike, NDArray[float]],
    *,
    rts: Union[str, RTS] = RTS.vrs,
    orientation: Union[str, Orientation] = Orientation.input,
    xref: Optional[Union[ArrayLike, NDArray[float]]] = None,
    yref: Optional[Union[ArrayLike, NDArray[float]]] = None,
    direct: Optional[Union[ArrayLike, NDArray[float]]] = None,
    slacks: bool = False,
    two_phase: bool = False,
    transpose: bool = False,
) -> Efficiency:
    rts = RTS.get(rts)
    orientation = Orientation.get(orientation)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if xref is None:
        xref = x.copy()
    if yref is None:
        yref = y.copy()

    if transpose:
        x = x.transpose()
        y = y.transpose()
        xref = xref.transpose()
        yref = yref.transpose()

    m = x.shape[1]
    n = y.shape[1]
    k = x.shape[0]

    if m != xref.shape[1]:
        raise ValueError("Number of inputs must be the same in 'x' and 'xref'")
    if n != yref.shape[1]:
        raise ValueError(
            "Number of outputs must be the same in 'y' and 'yref'"
        )
    if k != y.shape[0]:
        raise ValueError("Number of units must be the same in 'x' and 'y'")
    if k != xref.shape[0]:
        raise ValueError("Number of units must be the same in 'x' and 'xref'")
    if k != yref.shape[0]:
        raise ValueError("Number of units must be the same in 'x' and 'yref'")

    if direct:
        direct = np.asarray(direct)
        if orientation == Orientation.input and m != direct.shape[0]:
            raise ValueError(
                "Length of 'direct'' must be the number of inputs"
            )
        elif orientation == Orientation.output and n != direct.shape[0]:
            raise ValueError(
                "Length of 'direct'' must be the number of outputs"
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
        if direct:
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

    eff = solve_dea(x, y, rts, orientation, xref, yref, two_phase)

    if scaling:
        ss = np.hstack((xref_s, yref_s))
        eff.slack = np.multiply(eff.slack, ss)
        if direct:
            direct = np.multiply(direct, ss)

    eps = 1e-6
    eff.lambdas[np.abs(eff.lambdas) < eps] = 0
    eff.lambdas[np.abs(eff.lambdas - 1) < eps] = 1

    eff.slack[np.abs(eff.slack) < eps] = 0

    eff.eff = eff.objval.copy()
    eff.eff[np.abs(eff.eff) < eps] = 0
    eff.eff[np.abs(eff.eff - 1) < eps] = 1

    return eff
