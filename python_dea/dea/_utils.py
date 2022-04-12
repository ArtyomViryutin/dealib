__all__ = ["construct_lpp", "scale_data"]

import numpy as np
from numpy.typing import NDArray

from python_dea.linprog import LPP

from ._options import RTS


def construct_lpp(
    xref: NDArray[float],
    yref: NDArray[float],
    rts: RTS,
) -> LPP:
    lpp = LPP()

    m, k = xref.shape
    n = yref.shape[0]

    lpp.c = np.zeros(k)
    lpp.b_ub = np.zeros(m + n)
    lpp.A_ub = np.vstack((xref, -yref))

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


def scale_data(x: NDArray[float], y: NDArray[float]):
    pass
