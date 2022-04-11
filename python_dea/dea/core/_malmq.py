__all__ = ["malmq"]

from numpy.typing import NDArray

from .._options import RTS, Orientation


def malmq(
    x0: NDArray[float],
    y0: NDArray[float],
    x1: NDArray[float],
    y1: NDArray[float],
    orientation: Orientation = Orientation.input,
    rts: RTS = RTS.vrs,
    transpose: bool = False,
):
    pass
