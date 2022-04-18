__all__ = ["slack"]

import copy
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from dea.dea._options import RTS, Orientation
from dea.dea._wrappers import Efficiency
from dea.linprog import LPP, simplex

from .._types import MATRIX, RTS_T
from .._utils import (
    apply_scaling,
    construct_lpp,
    prepare_data,
    process_result_efficiency,
    validate_data,
)


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
    x, y, xref, yref, _ = prepare_data(
        x=x, y=y, xref=xref, yref=yref, transpose=transpose
    )

    validate_data(x=x, y=y, xref=xref, yref=yref)

    scaling, xref_s, yref_s = apply_scaling(x=x, y=y, xref=xref, yref=yref)
    se = _solve_slack(
        x=x, y=y, e=copy.deepcopy(e), xref=xref, yref=yref, rts=rts
    )

    if scaling is True:
        se.sx = np.multiply(se.sx, xref_s)
        se.sy = np.multiply(se.sy, yref_s)
        se.slack = np.multiply(se.slack, np.hstack((xref_s, yref_s)))

    process_result_efficiency(se)
    return se
