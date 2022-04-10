__all__ = [
    "validate_data",
    "validate_options",
]

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ._exceptions import OptionsError, WrongDataFormatError
from ._options import Model


def validate_data(x: NDArray[float], y: NDArray[float]):
    x_shape, y_shape = x.shape, y.shape

    if x_shape[0] != y_shape[0]:
        raise ValueError(
            f"Inputs and Outputs contain different number of DNU. "
            f"Inputs: {x_shape[0]}, Outputs: {y_shape[0]}"
        )
    if len(x_shape) != 2:
        raise WrongDataFormatError("Inputs array must be two dimensional")
    if len(y_shape) != 2:
        raise WrongDataFormatError("Outputs array must be two dimensional")
    if np.any(np.less(x, 0)):
        raise WrongDataFormatError("All inputs must be non-negative")
    if np.any(np.less(y, 0)):
        raise WrongDataFormatError("All outputs must be non-negative")


def validate_options(
    model: Optional[Model] = None,
    slacks: Optional[bool] = None,
    two_phase: Optional[bool] = None,
) -> None:
    if model is not None:
        if two_phase is not None and slacks is not None:
            if model == Model.envelopment and not two_phase and slacks:
                raise OptionsError(
                    "Incompatible options for envelopment model: two_phase=False, slacks=True. One phase envelopment "
                    "model optimizes slacks by default"
                )
        if two_phase is not None:
            if model == Model.multiplier and two_phase:
                raise OptionsError(
                    "Incompatible options for multiplier model: two_phase=True. Multiplier model is one phase by "
                    "default. To optimize slacks use slacks=True"
                )
        if slacks is not None:
            if model == Model.multiplier and slacks:
                raise OptionsError(
                    "Incompatible options for multiplier model: slacks=True. Multiplier has not slacks to optimize"
                )
