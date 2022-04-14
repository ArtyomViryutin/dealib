__all__ = ["direct"]

from typing import List, Optional, Union

from numpy.typing import ArrayLike, NDArray

from .._options import RTS, Orientation
from ._dea import dea


def direct(
    x: Union[List[List[float]], ArrayLike, NDArray[float]],
    y: Union[List[List[float]], ArrayLike, NDArray[float]],
    direct_: NDArray[float],
    *,
    rts: Union[str, RTS] = RTS.vrs,
    orientation: Union[str, Orientation] = Orientation.input,
    xref: Optional[Union[List[List[float]], ArrayLike, NDArray[float]]] = None,
    yref: Optional[Union[List[List[float]], ArrayLike, NDArray[float]]] = None,
    two_phase: bool = False,
    transpose: bool = False,
):
    return dea(
        x=x,
        y=y,
        rts=rts,
        orientation=orientation,
        direct=direct_,
        xref=xref,
        yref=yref,
        two_phase=two_phase,
        transpose=transpose,
    )
