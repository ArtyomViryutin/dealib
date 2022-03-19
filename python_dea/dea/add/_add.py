__all__ = ["add"]

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import trange

from python_dea.linprog import simplex

from .._options import RTS
from .._utils import rescale_data
from .._validator import validate_data
from .._wrappers import DEAResult


def construct_lpp(
    x: NDArray[float], y: NDArray[float], rts: RTS
) -> Tuple[NDArray, ...]:
    m, k = x.shape
    n = y.shape[0]

    c = np.zeros(k)

    b_ub = np.zeros(m + n)
    A_ub = np.vstack((x, -y))

    A_eq = None
    b_eq = None

    if rts != RTS.crs:
        rts_constraint = np.ones(k)
        if rts == RTS.vrs:
            A_eq = np.array([rts_constraint])
            b_eq = np.ones(1)
        elif rts == RTS.drs:
            A_ub = np.vstack((A_ub, rts_constraint))
            b_ub = np.append(b_ub, [1])
        elif rts == RTS.irs:
            A_ub = np.vstack((A_ub, -rts_constraint))
            b_ub = np.append(b_ub, [-1])
    return c, A_ub, b_ub, A_eq, b_eq


def solve_add(x: NDArray, y: NDArray, rts: RTS, tol: float) -> DEAResult:
    m, k = x.shape
    n = y.shape[0]

    c, A_ub, b_ub, A_eq, b_eq = construct_lpp(x, y, rts)

    efficiency = np.zeros(k)
    lambdas = np.zeros((k, k))
    slack = np.zeros((k, m + n))

    for i in trange(k, desc=f"Computing {rts} additive model"):
        b_ub[:m] = x[:, i]
        b_ub[m : m + n] = -y[:, i]
        lpp_result = simplex(
            c, A_ub, b_ub, A_eq, b_eq, opt_f=False, opt_slacks=True, tol=tol
        )
        efficiency[i] = lpp_result.f
        lambdas[i] = lpp_result.x
        slack[i] = lpp_result.slack[: m + n]

    return DEAResult(efficiency, lambdas, slack)


def add(
    inputs: NDArray,
    outputs: NDArray,
    rts: RTS,
    tol: float = 1e-9,
) -> DEAResult:
    rts = RTS.get(rts)

    x = np.asarray(inputs)
    y = np.asarray(outputs)

    validate_data(x, y)

    # x, x_rescale = rescale_data(x, tol=tol)
    # y, y_rescale = rescale_data(y, tol=tol)

    x = x.transpose()
    y = y.transpose()

    dea_result = solve_add(x, y, rts, tol)

    dea_result.lambdas[dea_result.lambdas < tol] = 0
    dea_result.slack[dea_result.slack < tol] = 0

    return dea_result
