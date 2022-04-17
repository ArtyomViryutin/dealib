__all__ = ["mult"]

from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from dea.dea._options import RTS, Orientation
from dea.dea._wrappers import Efficiency
from dea.linprog import simplex
from dea.linprog.wrappers import LPP

from .._types import MATRIX, ORIENTATION_T, RTS_T
from .._utils import process_result_efficiency, validate_data


def _construct_lpp(
    xref: NDArray[float],
    yref: NDArray[float],
    rts: RTS,
) -> LPP:
    lpp = LPP()
    k = xref.shape[0]
    m = xref.shape[1]
    n = yref.shape[1]

    lpp.c = np.zeros(m + n)
    lpp.A_ub = np.vstack(
        (
            np.hstack((-xref, yref)),
            -np.eye(m + n),
        )
    )
    lpp.b_ub = np.hstack((np.zeros(k), -np.full(m + n, 1e-6)))
    lpp.A_eq = np.array([np.zeros(m + n)])
    lpp.b_eq = np.ones(1)
    if rts == RTS.irs:
        lpp.c = np.append(lpp.c, 1)
        lpp.A_ub = np.hstack(
            (lpp.A_ub, np.hstack((np.ones(k), np.zeros(m + n)))[:, np.newaxis])
        )
        lpp.A_eq = np.array([np.append(lpp.A_eq[0], 0)])
    elif rts == RTS.drs:
        lpp.c = np.append(lpp.c, -1)
        lpp.A_ub = np.hstack(
            (
                lpp.A_ub,
                np.hstack((-np.ones(k), np.zeros(m + n)))[:, np.newaxis],
            )
        )
        lpp.A_eq = np.array([np.append(lpp.A_eq[0], 0)])
    elif rts == RTS.vrs:
        lpp.c = np.append(lpp.c, [1, -1])
        lpp.A_ub = np.hstack(
            (
                lpp.A_ub,
                np.hstack((np.ones(k), np.zeros(m + n)))[:, np.newaxis],
                np.hstack((-np.ones(k), np.zeros(m + n)))[:, np.newaxis],
            )
        )
        lpp.A_eq = np.array([np.append(lpp.A_eq[0], np.zeros(2))])
    return lpp


def _solve_mult(
    x: NDArray[float],
    y: NDArray[float],
    rts: RTS,
    orientation: Orientation,
    xref: NDArray[float],
    yref: NDArray[float],
) -> Efficiency:
    m = x.shape[1]
    n = y.shape[1]
    k = x.shape[0]
    kr = xref.shape[0]

    lpp = _construct_lpp(xref, yref, rts)

    e = Efficiency(
        rts=rts,
        orientation=orientation,
        eff=np.zeros(k),
        objval=np.zeros(k),
        ux=np.zeros((kr, m)),
        vy=np.zeros((kr, n)),
        slack=np.zeros((k, kr)),
    )

    for i in range(k):
        if orientation == Orientation.input:
            lpp.c[m : m + n] = y[i]
            lpp.A_eq[0][:m] = x[i]
        else:
            lpp.c[:m] = -x[i]
            lpp.A_eq[0][m : m + n] = y[i]
        lpp_result = simplex(
            lpp,
            opt_f=True,
            opt_slacks=False,
        )
        e.objval[i] = abs(lpp_result.f)
        e.ux[i] = lpp_result.x[:m]
        e.vy[i] = lpp_result.x[m : m + n]
        e.slack[i] = lpp_result.slack[:k]

    e.eff = e.objval.copy()
    return e


def mult(
    x: Union[MATRIX],
    y: Union[MATRIX],
    *,
    rts: RTS_T = RTS.vrs,
    orientation: ORIENTATION_T = Orientation.input,
    xref: Optional[MATRIX] = None,
    yref: Optional[MATRIX] = None,
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

    validate_data(x=x, y=y, xref=xref, yref=yref)

    xref_m, yref_m = x.mean(axis=0), y.mean(axis=0)
    if (
        min(xref_m) < 1e-4
        or max(xref_m) > 10000
        or min(yref_m) < 1e-4
        or max(yref_m) > 10000
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

    e = _solve_mult(
        x=x, y=y, rts=rts, orientation=orientation, xref=xref, yref=yref
    )

    if scaling is True:
        e.ux = np.divide(e.ux, xref_s)
        e.vy = np.divide(e.vy, yref_s)

    process_result_efficiency(e)
    return e
