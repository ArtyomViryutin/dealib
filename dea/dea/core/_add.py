__all__ = ["add"]

from typing import Optional

import numpy as np

from ..utils.options import RTS
from ..utils.types import MATRIX, RTS_T
from ..utils.utils import prepare_data, validate_data
from ..utils.wrappers import Efficiency
from ._slack import slack


def add(
    x: MATRIX,
    y: MATRIX,
    *,
    rts: RTS_T = RTS.vrs,
    xref: Optional[MATRIX] = None,
    yref: Optional[MATRIX] = None,
    transpose: Optional[bool] = False,
) -> Efficiency:
    """
    Calculates additive efficiency as sum of input and output slacks.

    :param 2-d array x: Inputs of firm to be evaluated. `(k, m)` matrix of observations of `k` firms with `m` inputs.
       In case `transpose=True` the input matrix is transposed.

    :param 2-d array y: Outputs of firm to be evaluated. `(k, n)` matrix of observations of `k` firms with `n` outputs.
       In case `transpose=True` the output matrix is transposed.

    :param int, str, RTS rts:
       Returns to scale assumption.

       `0 vrs` - Variable returns to scale

       `1 crs` - Constant returns to scale

       `2 drs` - Decreasing returns to scale

       `3 irs` -  Increasing returns to scale

    :param 2-d array xref: Inputs of the firms determining the technology, defaults to :mod:`x`.

    :param 2-d array yref: Outputs of the firms determining the technology, defaults to :mod:`y`.

    :param bool transpose: Flag determining if input and output matrix are transposed. See :mod:`x`, :mod:`y`.

    :return: Result efficiency object.
    :rtype: Efficiency


    **Example**

    >>> x = [[20], [40], [40], [40], [60], [70], [50]]
    >>> y = [[20], [30], [50], [40], [60], [20]]
    >>> eff = add(x, y, rts="vrs")
    >>> print(eff.objval)
    [ 5.  20.   0.  35.  27.5 42.5]
    """

    x, y, xref, yref, _ = prepare_data(
        x=x, y=y, xref=xref, yref=yref, transpose=transpose
    )

    validate_data(x=x, y=y, xref=xref, yref=yref)

    if transpose is True:
        m = x.shape[0]
        n = y.shape[0]
        k = x.shape[1]
    else:
        m = x.shape[1]
        n = y.shape[1]
        k = x.shape[0]

    if xref is not None:
        xref = np.asarray(xref, dtype=float)
        if transpose is True:
            kr = xref.shape[1]
        else:
            kr = xref.shape[0]
    else:
        kr = k

    e = Efficiency(
        rts=rts,
        transpose=transpose,
        eff=np.ones(k),
        objval=np.zeros(k),
        lambdas=np.zeros((k, kr)),
        sx=np.zeros((k, m)),
        sy=np.zeros((k, n)),
    )

    return slack(x, y, e, xref=xref, yref=yref, transpose=transpose)
