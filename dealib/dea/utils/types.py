__all__ = ["MATRIX", "DIRECTION", "RTS_T", "ORIENTATION_T"]

from typing import List, Union

from numpy.typing import NDArray

from dealib.dea.utils.options import RTS, Orientation

RTS_T = Union[int, str, RTS]

ORIENTATION_T = Union[int, str, Orientation]

MATRIX = Union[List[List[float]], NDArray[float]]

DIRECTION = Union[str, float, List[float], MATRIX]
