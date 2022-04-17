__all__ = [
    "construct_lpp",
    "process_result_efficiency",
    "validate_data",
]

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from dea.linprog import LPP

from ._options import RTS, Orientation
from ._wrappers import Efficiency


def construct_lpp(
    xref: NDArray[float],
    yref: NDArray[float],
    rts: RTS,
) -> LPP:
    lpp = LPP()

    m = xref.shape[1]
    n = yref.shape[1]
    k = xref.shape[0]

    lpp.c = np.zeros(k)
    lpp.b_ub = np.zeros(m + n)
    lpp.A_ub = np.vstack((xref.transpose(), -yref.transpose()))

    if rts != RTS.crs:
        rts_constraint = np.ones(k)
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


def process_result_efficiency(e: Efficiency, eps: float = 1e-5) -> None:
    if e.lambdas is not None:
        e.lambdas[e.lambdas < 0] = np.nan
        e.lambdas[np.abs(e.lambdas) < eps] = 0
        e.lambdas[np.abs(e.lambdas - 1) < eps] = 1

    e.slack[e.slack < 0] = np.nan
    e.slack[np.abs(e.slack) < eps] = 0

    e.eff[e.eff < 0] = np.nan
    e.eff[np.abs(e.eff) < eps] = 0
    e.eff[np.abs(e.eff - 1) < eps] = 1


def validate_data(
    x: NDArray[float],
    y: NDArray[float],
    xref: NDArray[float],
    yref: NDArray[float],
    orientation: Optional[Orientation] = None,
    direct: Optional[NDArray[float]] = None,
) -> None:
    if x.ndim != 2:
        raise ValueError("'x' must be two-dimensional array")
    if y.ndim != 2:
        raise ValueError("'y' must be two-dimensional array")
    if xref.ndim != 2:
        raise ValueError("'xref' must be two-dimensional array")
    if yref.ndim != 2:
        raise ValueError("'yref' must be two-dimensional array")

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
    if xref.shape[0] != yref.shape[0]:
        raise ValueError(
            "Number of units must be the same in 'xref' and 'yref'"
        )

    if direct is not None and not isinstance(direct, str):
        if direct.ndim > 1:
            kd, md = direct.shape
        else:
            md = direct.shape[0]
            kd = 0

        if orientation == Orientation.input and m != md:
            raise ValueError("Length of 'direct' must be the number of inputs")
        elif orientation == Orientation.output and n != md:
            raise ValueError(
                "Length of 'direct'' must be the number of outputs"
            )
        if kd > 0 and kd != k:
            raise ValueError(
                "Number of units in 'direct' must equal units in 'x' and y'"
            )
