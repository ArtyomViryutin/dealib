__all__ = [
    "prepare_data",
    "apply_scaling",
    "construct_lpp",
    "process_result_efficiency",
    "validate_data",
]

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from dealib.linprog import LPP

from .options import RTS, Orientation
from .types import DIRECTION, MATRIX
from .wrappers import Efficiency


def prepare_data(
    x: MATRIX,
    y: MATRIX,
    xref: MATRIX,
    yref: MATRIX,
    transpose: bool = False,
    orientation: Optional[Orientation] = None,
    direct: Optional[DIRECTION] = None,
) -> Tuple[NDArray[float], ...]:
    x = np.array(x, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)

    if xref is None:
        xref = x.copy()
    else:
        xref = np.array(xref, dtype=float, copy=True)

    if yref is None:
        yref = y.copy()
    else:
        yref = np.array(yref, dtype=float, copy=True)

    if direct is not None and not isinstance(direct, str):
        if isinstance(direct, list) or isinstance(direct, np.ndarray):
            direct = np.array(direct, dtype=float, copy=True)
        else:
            if orientation == Orientation.input:
                direct = np.full(x.shape[1], direct)
            else:
                direct = np.full(y.shape[1], direct)

    if transpose is True:
        x = x.transpose()
        y = y.transpose()
        xref = xref.transpose()
        yref = yref.transpose()

        if (
            direct is not None
            and isinstance(direct, np.ndarray)
            and direct.ndim > 1
        ):
            direct = direct.transpose()
    return x, y, xref, yref, direct


def apply_scaling(
    x: NDArray[float],
    y: NDArray[float],
    xref: NDArray[float],
    yref: NDArray[float],
    orientation: Optional[Orientation] = None,
    direct: Optional[NDArray[float]] = None,
) -> Tuple[bool, NDArray[float], NDArray[float]]:
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
        np.divide(x, xref_s, out=x)
        np.divide(y, yref_s, out=y)
        np.divide(xref, xref_s, out=xref)
        np.divide(yref, yref_s, out=yref)
        if isinstance(direct, np.ndarray):
            if orientation == Orientation.input:
                np.divide(direct, xref_s, out=direct)
            else:
                np.divide(direct, yref_s, out=direct)
    else:
        scaling = False
        xref_s = yref_s = None
    return scaling, xref_s, yref_s


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


def process_result_efficiency(e: Efficiency, tol: float = 1e-5) -> None:
    if e.lambdas is not None:
        e.lambdas[e.lambdas < 0] = np.nan
        e.lambdas[np.abs(e.lambdas) < tol] = 0
        e.lambdas[np.abs(e.lambdas - 1) < tol] = 1

    e.slack[e.slack < 0] = np.nan
    e.slack[np.abs(e.slack) < tol] = 0

    e.eff[e.eff < 0] = np.nan
    e.eff[np.abs(e.eff) < tol] = 0
    e.eff[np.abs(e.eff - 1) < tol] = 1


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
