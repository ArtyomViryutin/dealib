__all__ = ["slack"]

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from python_dea.dea._options import RTS, Orientation
from python_dea.dea._wrappers import Efficiency
from python_dea.linprog import LPP, simplex

from .._utils import construct_lpp


def construct_slack_lpp(x: NDArray[float], y: NDArray[float], rts: RTS) -> LPP:
    return construct_lpp(x, y, rts)


def solve_slack(
    x: NDArray[float], y: NDArray[float], eff: Efficiency
) -> Efficiency:
    m, k = x.shape
    n = y.shape[0]

    lpp = construct_slack_lpp(x, y, eff.rts)

    for i in range(k):
        lpp.b_ub[:m] = x[:, i]
        lpp.b_ub[m : m + n] = -y[:, i]
        if eff.orientation == Orientation.input:
            lpp.b_ub[:m] *= eff.eff[i]
        else:
            lpp.b_ub[m : m + n] *= eff.eff[i]

        lpp_result = simplex(lpp, opt_f=False, opt_slacks=True)

        eff.objval[i] = lpp_result.f
        eff.lambdas[i] = lpp_result.x
        eff.slack[i] = lpp_result.slack[: m + n]

    return eff


def slack(
    x: Union[ArrayLike, NDArray[float]],
    y: Union[ArrayLike, NDArray[float]],
    eff: Efficiency,
    transpose: bool = False,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if transpose:
        x = x.transpose()
        y = y.transpose()

    if x.shape[0] != y.shape[0]:
        raise ValueError("Number of units must be the same in 'x' and 'y'")

    xm, ym = x.mean(axis=0), y.mean(axis=0)
    if (
        np.min(xm) < 1e-4
        or np.max(xm) > 10000
        or np.min(ym) < 1e-4
        or np.max(ym) > 10000
    ):
        scaling = True
        xx, yy = x.std(axis=0), y.std(axis=0)
        xx[xx < 1e-9] = 1
        yy[yy < 1e-9] = 1
        x = np.divide(x, xx)
        y = np.divide(y, yy)
    else:
        scaling = False
        xx = yy = None

    x = x.transpose()
    y = y.transpose()

    eff = solve_slack(x, y, eff)

    if scaling:
        eff.slack = np.multiply(eff.slack, np.hstack((xx, yy)))

    eps = 1e-6
    eff.lambdas[np.abs(eff.lambdas) < eps] = 0
    eff.lambdas[np.abs(eff.lambdas - 1) < eps] = 1
    eff.slack[np.abs(eff.slack) < eps] = 0
    return eff
