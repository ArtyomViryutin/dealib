__all__ = ["mea"]
from typing import Optional, Union

from .._options import RTS, Orientation
from .._types import MATRIX, ORIENTATION_T, RTS_T
from .._wrappers import Efficiency
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
