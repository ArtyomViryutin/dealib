__all__ = ["dea"]

import numpy as np
from numpy.typing import ArrayLike

from .._options import RTS, Model, Orientation
from .._utils import rescale_data
from .._validator import validate_data, validate_options
from .._wrappers import DEAResult
from ._envelopment import solve_envelopment
from ._multiplier import solve_multiplier


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

    validate_options(model, slacks, two_phase)

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
