__all__ = ["direct"]

from typing import Union

from numpy.typing import ArrayLike, NDArray

from .._options import RTS, Orientation
from ._dea import dea


def direct(
    x: Union[ArrayLike, NDArray[float]],
    y: Union[ArrayLike, NDArray[float]],
    direct_: NDArray[float],
    *,
    rts: Union[str, RTS] = RTS.vrs,
    orientation: Union[str, Orientation] = Orientation.input,
    two_phase: bool = False,
    transpose: bool = False,
):
    eff = dea(
        x,
        y,
        rts=rts,
        orientation=orientation,
        direct=direct_,
        two_phase=two_phase,
        transpose=transpose,
    )
    return eff
