__all__ = ["dea"]

from numpy.typing import ArrayLike

from python_dea.dea._options import RTS, Model, Orientation
from python_dea.dea._utils import post_process_data, pre_process_data
from python_dea.dea._validators import validate_options
from python_dea.dea._wrappers import Efficiency

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
) -> Efficiency:
    """
    Solve DEA
    """
    model = Model.get(model)
    orientation = Orientation.get(orientation)
    rts = RTS.get(rts)

    validate_options(model, slacks, two_phase)

    x, y, x_std, y_std = pre_process_data(inputs, outputs, tol)

    if model == Model.envelopment:
        efficiency = solve_envelopment(
            x, y, orientation, rts, two_phase, eps=eps, tol=tol
        )
    else:
        efficiency = solve_multiplier(x, y, orientation, rts, eps=eps, tol=tol)

    post_process_data(efficiency, x_std, y_std, model)
    return efficiency
