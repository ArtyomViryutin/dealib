__all__ = ["slack"]

import copy
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from dea.dea._options import RTS, Orientation
from dea.dea._wrappers import Efficiency
from dea.linprog import LPP, simplex

from .._types import MATRIX, RTS_T
from .._utils import construct_lpp, process_result_efficiency, validate_data


def _construct_slack_lpp(
    *, xref: NDArray[float], yref: NDArray[float], rts: RTS
) -> LPP:
    return construct_lpp(xref, yref, rts)


def _solve_slack(
    *,
    x: NDArray[float],
    y: NDArray[float],
    e: Efficiency,
    rts: RTS,
    xref: NDArray[float],
    yref: NDArray[float],
) -> Efficiency:
    m = xref.shape[1]
    n = yref.shape[1]
    k = xref.shape[0]

    lpp = _construct_slack_lpp(xref=xref, yref=yref, rts=rts)

    for i in range(k):
        lpp.b_ub[:m] = x[i]
        lpp.b_ub[m : m + n] = -y[i]
        if e.orientation == Orientation.input:
            lpp.b_ub[:m] *= e.eff[i]
        else:
            lpp.b_ub[m : m + n] *= e.eff[i]

        lpp_result = simplex(lpp, opt_f=False, opt_slacks=True)

        e.objval[i] = lpp_result.f
        e.lambdas[i] = lpp_result.x
        e.sx[i] = lpp_result.slack[:m]
        e.sy[i] = lpp_result.slack[m : m + n]

    e.slack = np.hstack((e.sx, e.sy))
    return e


def slack(
    x: MATRIX,
    y: MATRIX,
    e: Efficiency,
    *,
    rts: RTS_T = RTS.vrs,
    xref: Optional[MATRIX] = None,
    yref: Optional[MATRIX] = None,
    transpose: Optional[bool] = False,
) -> Efficiency:
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

    validate_data(x=x, y=y, xref=xref, yref=yref)

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
    else:
        scaling = False
        xref_s = yref_s = None

    se = _solve_slack(
        x=x, y=y, e=copy.deepcopy(e), xref=xref, yref=yref, rts=rts
    )

    if scaling is True:
        se.sx = np.multiply(se.sx, xref_s)
        se.sy = np.multiply(se.sy, yref_s)
        se.slack = np.multiply(se.slack, np.hstack((xref_s, yref_s)))

    process_result_efficiency(se)
    return se
