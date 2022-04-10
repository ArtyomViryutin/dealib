__all__ = ["get_data", "get_reference", "parametrize_options"]

import json
from pathlib import Path

import pandas as pd
import pytest


def parametrize_options(options, name):
    return pytest.mark.parametrize(name, [o for o in options])


def get_data(folder_name: str):
    directory = Path(__file__).parent / "data" / f"{folder_name}"
    inputs = pd.read_csv(directory / "inputs.csv")
    outputs = pd.read_csv(directory / "outputs.csv")
    return inputs, outputs


def get_reference(folder_name: str, filename: str):
    with open(
        Path(__file__).parent / "reference" / folder_name / filename
    ) as f:
        return json.load(f)
