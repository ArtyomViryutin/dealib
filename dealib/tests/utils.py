__all__ = [
    "get_data",
    "get_reference",
    "parametrize_options",
    "compare_valid_values",
]

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray


def parametrize_options(options, name):
    return pytest.mark.parametrize(name, [o for o in options])


def get_data(folder_name: str) -> Tuple[NDArray[float], NDArray[float]]:
    directory = Path(__file__).parent / "data" / f"{folder_name}"
    x = pd.read_csv(directory / "inputs.csv")
    y = pd.read_csv(directory / "outputs.csv")
    return x, y


def get_reference(folder_name: str, filename: str):
    with open(
        Path(__file__).parent / "reference" / folder_name / filename
    ) as f:
        return json.load(f)


def compare_valid_values(
    actual: NDArray[float],
    desired: NDArray[float],
    mismatches: int = 0,
    tol: float = 1e-6,
) -> None:
    valid_values = np.logical_and(desired != np.nan, np.abs(desired) != np.inf)
    assert (
        np.count_nonzero(
            np.abs(actual[valid_values] - desired[valid_values]) > tol
        )
        <= mismatches
    )
