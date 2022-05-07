__all__ = ["slack"]

import copy
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from dealib.linprog import LPP, simplex

from ..utils.options import RTS, Orientation
from ..utils.types import MATRIX
from ..utils.utils import (
    apply_scaling,
    construct_lpp,
    prepare_data,
    process_result_efficiency,
    validate_data,
)
from ..utils.wrappers import Efficiency


def _construct_slack_lpp(
    *, xref: NDArray[float], yref: NDArray[float], rts: RTS
) -> LPP:
    return construct_lpp(xref, yref, rts)


def _solve_slack(
    *,
    x: NDArray[float],
    y: NDArray[float],
    e: Efficiency,
    xref: NDArray[float],
    yref: NDArray[float],
) -> Efficiency:
    m = xref.shape[1]
    n = yref.shape[1]
    k = xref.shape[0]

    lpp = _construct_slack_lpp(xref=xref, yref=yref, rts=e.rts)

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
    xref: Optional[MATRIX] = None,
    yref: Optional[MATRIX] = None,
    transpose: Optional[bool] = False,
) -> Efficiency:
    """
    Estimats a DEA frontier and optimize slacks.

    :param 2-d array x: Inputs of firm to be evaluated. `(k, m)` matrix of observations of `k` firms with `m` inputs.
        In case `transpose=True` the input matrix is transposed.

    :param 2-d array y: Outputs of firm to be evaluated. `(k, n)` matrix of observations of `k` firms with `n` outputs.
        In case `transpose=True` the output matrix is transposed.

    :param Efficiency e: Efficiency object returned by :ref:`dea<dealib.dea.core._dea.dea>`,
        :ref:`add<dealib.dea.core._add.add>`, :ref:`direct<dealib.dea.core._direct.direct>`,
        :ref:`mea<dealib.dea.core._mea.mea>`.

    :param 2-d array xref: Inputs of the firms determining the technology, defaults to :mod:`x`.

    :param 2-d array yref: Outputs of the firms determining the technology, defaults to :mod:`y`.

    :param bool transpose: Flag determining if input and output matrix are transposed. See :mod:`x`, :mod:`y`.

    :return: Result efficiency object.
    :rtype: Efficiency


    **Example**

    >>> x = [[1, 5], [2, 2], [4, 1], [6, 1], [4, 4]]
    >>> y = [[2], [2], [2], [2], [2]]
    >>> e = dea(x, y, rts="vrs", orientation="input")
    >>> eff = slack(x, y, e)
    >>> print(eff.slack)
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]
         [2. 0. 0.]
         [0. 0. 0.]]
    """

    x, y, xref, yref, _ = prepare_data(
        x=x, y=y, xref=xref, yref=yref, transpose=transpose
    )

    validate_data(x=x, y=y, xref=xref, yref=yref)

    scaling, xref_s, yref_s = apply_scaling(x=x, y=y, xref=xref, yref=yref)
    se = _solve_slack(x=x, y=y, e=copy.deepcopy(e), xref=xref, yref=yref)

    if scaling is True:
        se.sx = np.multiply(se.sx, xref_s)
        se.sy = np.multiply(se.sy, yref_s)
        se.slack = np.multiply(se.slack, np.hstack((xref_s, yref_s)))

    process_result_efficiency(se)
    return se
