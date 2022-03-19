__all__ = ["rescale_data"]

from typing import Tuple

import numpy as np


def rescale_data(
    data: np.ndarray, tol: float
) -> Tuple[np.ndarray, np.ndarray]:
    data_std = data.std(axis=0)
    data_std[data_std < tol] = 1
    data = np.divide(data, data_std)
    return data, data_std
