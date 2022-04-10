__all__ = ["solve_envelopment"]

from typing import List

import numpy as np
from numpy.typing import NDArray
from tqdm import trange

from python_dea.dea._options import RTS, Model, Orientation
from python_dea.dea._wrappers import Efficiency
from python_dea.linprog import simplex
from python_dea.linprog.wrappers import LPP

from .._slack import slack


def construct_lpp(
    x: NDArray[float],
    y: NDArray[float],
    orientation: Orientation,
    rts: RTS,
) -> LPP:
    lpp = LPP()
    m, k = x.shape
    n = y.shape[0]

    lpp.c = np.zeros(k + 1)
    lpp.c[-1] = 1
    lpp.b_ub = np.zeros(m + n)
    lpp.A_ub = np.hstack((np.vstack((x, y)), np.zeros(m + n)[:, np.newaxis]))

    if orientation == Orientation.input:
        lpp.c[-1] = -1
        lpp.A_ub[m:] *= -1
    else:
        lpp.A_ub[m:, :-1] *= -1
    if rts != RTS.crs:
        rts_constraint = np.ones(k + 1)
        rts_constraint[-1] = 0
        if rts == RTS.vrs:
            lpp.A_eq = np.array([rts_constraint])
            lpp.b_eq = np.ones(1)
        elif rts == RTS.drs:
            lpp.A_ub = np.vstack((lpp.A_ub, rts_constraint))
            lpp.b_ub = np.append(lpp.b_ub, [1])
        elif rts == RTS.irs:
            lpp.A_ub = np.vstack((lpp.A_ub, -rts_constraint))
            lpp.b_ub = np.append(lpp.b_ub, [-1])

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


def solve_envelopment(
    x: NDArray[float],
    y: NDArray[float],
    orientation: Orientation,
    rts: RTS,
    two_phase: bool,
    eps: float,
    tol: float,
) -> Efficiency:
    m, k = x.shape
    n = y.shape[0]

    lpp = construct_lpp(x, y, orientation, rts)
    inefficient_dmu = find_inefficient_dmu(x, y)

    eff_dmu = np.ones(k, dtype=bool)
    eff_dmu[inefficient_dmu] = False
    lpp.c = np.delete(lpp.c, inefficient_dmu)
    lpp.A_ub = np.delete(lpp.A_ub, inefficient_dmu, axis=1)
    if lpp.A_eq is not None:
        lpp.A_eq = np.delete(lpp.A_eq, inefficient_dmu, axis=1)

    eff_dmu_count = 0
    eff = Efficiency(Model.envelopment, orientation, rts, k, m, n)

    for i in trange(
        k, desc=f"Computing {orientation}-{rts} envelopment model"
    ):
        if orientation == Orientation.input:
            lpp.A_ub[:m, -1] = -x[:, i]
            lpp.A_ub[m : m + n, -1] = 0
            lpp.b_ub[:m] = 0
            lpp.b_ub[m : m + n] = -y[:, i]
        else:
            lpp.A_ub[:m, -1] = 0
            lpp.A_ub[m : m + n, -1] = y[:, i]
            lpp.b_ub[:m] = x[:, i]
            lpp.b_ub[m : m + n] = 0
        if two_phase:
            lpp_result = simplex(lpp, opt_f=True, opt_slacks=False, tol=tol)
        else:
            lpp_result = simplex(
                lpp, opt_f=True, opt_slacks=True, eps=eps, tol=tol
            )
        eff.eff[i] = lpp_result.x[-1]
        eff.lambdas[i, eff_dmu] = lpp_result.x[:-1]
        eff.slack[i] = lpp_result.slack[: m + n]

        if eff_dmu[i]:
            if np.abs(1 - eff.eff[i]) > tol:
                eff_dmu[i] = False
                lpp.c = np.delete(lpp.c, eff_dmu_count)
                lpp.A_ub = np.delete(lpp.A_ub, eff_dmu_count, axis=1)
                if lpp.A_eq is not None:
                    lpp.A_eq = np.delete(lpp.A_eq, eff_dmu_count, axis=1)
            else:
                eff_dmu_count += 1
    if two_phase:
        eff = slack(x, y, eff, transpose=True, tol=tol)
    eff.objval = eff.eff
    return eff
