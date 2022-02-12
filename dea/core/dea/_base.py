__all__ = ["dea"]
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from ._envelopment import solve_envelopment
from ._exceptions import OptionsError, WrongDataFormatError
from ._multiplier import solve_multiplier
from ._options import RTS, Model, Orientation
from ._wrappers import DEAResult


def rescale_data(data: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    data_std = data.std(axis=0)
    data_std[data_std < tol] = 1
    data = np.divide(data, data_std)
    return data, data_std


def check_options(
    model: Model,
    slacks: bool,
    two_phase: bool,
) -> None:
    if model == Model.envelopment and not two_phase and slacks:
        raise OptionsError(
            "Incompatible options for envelopment model: two_phase=False, slacks=True. One phase envelopment model "
            "optimizes slacks by default"
        )

    if model == Model.multiplier and two_phase:
        raise OptionsError(
            "Incompatible options for multiplier model: two_phase=True. Multiplier model is one phase by default"
            "To optimize slacks use slacks=True"
        )

    if model == Model.multiplier and slacks:
        raise OptionsError(
            "Incompatible options for multiplier model: slacks=True. Multiplier has not slacks to optimize"
        )


def validate_data(x: ArrayLike, y: ArrayLike):
    x_shape, y_shape = x.shape, y.shape
    if x_shape[0] != y_shape[0]:
        raise ValueError(
            f"Inputs and Outputs contain different number of DNU. Inputs: {x_shape[0]}, Outputs: {y_shape[0]}"
        )
    if len(x_shape) != 2:
        raise WrongDataFormatError("Inputs array must be two dimensional")
    if len(y_shape) != 2:
        raise WrongDataFormatError("Inputs array must be two dimensional")
    if np.any(np.less(x, 0)):
        raise WrongDataFormatError("All inputs must be non-negative")
    if np.any(np.less(y, 0)):
        raise WrongDataFormatError("All outputs must be non-negative")


def dea(
    inputs: ArrayLike,
    outputs: ArrayLike,
    *,
    orientation: Orientation = Orientation.input,
    rts: RTS = RTS.vrs,
    model: Model = Model.envelopment,
    slacks: bool = False,
    two_phase: bool = False,
    eps: float = 1e-6,
    tol: float = 1e-9,
) -> DEAResult:
    """
    Solve DEA
    """
    orientation = Orientation.get(orientation)
    rts = RTS.get(rts)
    model = Model.get(model)

    check_options(model, slacks, two_phase)

    x = np.asarray(inputs)
    y = np.asarray(outputs)
    validate_data(x, y)

    x, x_rescale = rescale_data(x, tol=tol)
    y, y_rescale = rescale_data(y, tol=tol)

    x = x.transpose()
    y = y.transpose()

    if model == Model.envelopment:
        dea_result = solve_envelopment(
            x, y, orientation, rts, two_phase, eps=eps, tol=tol
        )
    else:
        dea_result = solve_multiplier(x, y, orientation, rts, eps=eps, tol=tol)

    if model == Model.envelopment:
        dea_result.slack = np.multiply(
            dea_result.slack, np.hstack((x_rescale, y_rescale))
        )
    else:
        dea_result.lambdas = np.divide(
            dea_result.lambdas, np.hstack((x_rescale, y_rescale))
        )
    dea_result.lambdas[dea_result.lambdas < tol] = 0
    dea_result.slack[dea_result.slack < tol] = 0
    return dea_result
