__all__ = ["slack"]

from typing import List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from python_dea.dea._options import RTS, Orientation
from python_dea.dea._wrappers import Efficiency
from python_dea.linprog import LPP, simplex

from .._utils import construct_lpp, process_result_efficiency, validate_data


def _construct_slack_lpp(
    *, xref: NDArray[float], yref: NDArray[float], rts: RTS
) -> LPP:
    return construct_lpp(xref, yref, rts)


def _solve_slack(
    *,
    x: NDArray[float],
    y: NDArray[float],
    eff: Efficiency,
    rts: RTS,
    xref: NDArray[float],
    yref: NDArray[float],
) -> Efficiency:
    m = xref.shape[0]
    k = xref.shape[1]
    n = yref.shape[0]

    lpp = _construct_slack_lpp(xref=xref, yref=yref, rts=rts)

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
    x: Union[List[List[float]], ArrayLike, NDArray[float]],
    y: Union[List[List[float]], ArrayLike, NDArray[float]],
    eff: Efficiency,
    *,
    rts: Optional[RTS] = RTS.vrs,
    xref: Optional[Union[List[List[float]], ArrayLike, NDArray[float]]] = None,
    yref: Optional[Union[List[List[float]], ArrayLike, NDArray[float]]] = None,
    transpose: Optional[bool] = False,
):
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

    x = x.transpose()
    y = y.transpose()
    xref = xref.transpose()
    yref = yref.transpose()

    eff = _solve_slack(x=x, y=y, eff=eff, xref=xref, yref=yref, rts=rts)

    if scaling is True:
        eff.slack = np.multiply(eff.slack, np.hstack((xref_s, yref_s)))

    process_result_efficiency(eff)
    return eff
