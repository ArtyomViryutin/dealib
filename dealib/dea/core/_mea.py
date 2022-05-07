__all__ = ["mea"]
from typing import Optional, Union

from ..utils.options import RTS, Orientation
from ..utils.types import MATRIX, ORIENTATION_T, RTS_T
from ..utils.wrappers import Efficiency
from ._dea import dea


def mea(
    x: Union[MATRIX],
    y: Union[MATRIX],
    rts: RTS_T = RTS.vrs,
    orientation: ORIENTATION_T = Orientation.input,
    xref: Optional[MATRIX] = None,
    yref: Optional[MATRIX] = None,
    transpose: bool = False,
) -> Efficiency:
    """
    Estimates a DEA frontier and calculates multi-directional Farrell measures.

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

    :param int, str, Orientation orientation: Efficiency orientation.

        `0 input` - Input efficiency

        `1 output` - Output efficiency

    :param 2-d array xref: Inputs of the firms determining the technology, defaults to :mod:`x`.

    :param 2-d array yref: Outputs of the firms determining the technology, defaults to :mod:`y`.

    :param bool transpose: Flag determining if input and output matrix are transposed. See :mod:`x`, :mod:`y`.

    :return: Result efficiency object.
    :rtype: Efficiency


    **Example**

    >>> x = [[20], [40], [40], [40], [60], [70], [50]]
    >>> y = [[20], [30], [50], [40], [60], [20]]
    >>> eff = mea(x, y, rts="vrs", orientation="input")
    >>> print(eff.eff)
    [0. 1. 0. 1. 0. 1.]
    """

    return dea(
        x=x,
        y=y,
        rts=rts,
        orientation=orientation,
        xref=xref,
        yref=yref,
        direct="min",
        transpose=transpose,
    )
