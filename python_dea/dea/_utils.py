__all__ = ["pre_process_data", "post_process_data"]

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._options import Model
from ._validators import validate_data
from ._wrappers import Efficiency


def pre_process_data(
    inputs: ArrayLike, outputs: ArrayLike, tol: float
) -> Tuple[NDArray[float], ...]:
    x = np.asarray(inputs)
    y = np.asarray(outputs)
    validate_data(x, y)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    if (
        min(x_mean) < 1e-4
        or max(x_mean) > 10000
        or min(y_mean) < 1e-4
        or max(y_mean) > 1000
    ):
        x_std = x.std(axis=0)
        x_std[x_std < tol] = 1
        x = np.divide(x, x_std)

        y_std = y.std(axis=0)
        y_std[y_std < tol] = 1
        y = np.divide(y, y_std)
    else:
        x_std = np.ones(x.shape[1])
        y_std = np.ones(y.shape[1])

    return x.transpose(), y.transpose(), x_std, y_std


def post_process_data(
    eff: Efficiency, x_std: NDArray[float], y_std: NDArray[float], model: Model
) -> None:
    tol = 1e-5
    if model == Model.envelopment:
        eff.slack = np.multiply(eff.slack, np.hstack((x_std, y_std)))
    else:
        eff.lambdas = np.divide(eff.lambdas, np.hstack((x_std, y_std)))

    eff.lambdas[eff.lambdas < tol] = 0
    eff.slack[eff.slack < tol] = 0

    eff.objval[np.abs(eff.objval) < tol] = 0
    eff.objval[np.abs(eff.objval - 1) < tol] = 1

    eff.eff[np.abs(eff.eff) < tol] = 0
    eff.eff[np.abs(eff.eff - 1) < tol] = 1
