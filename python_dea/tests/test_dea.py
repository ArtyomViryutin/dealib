import json
from pathlib import Path

import pandas as pd
import pytest

from python_dea.dea import Model, dea

from .utils import benchmark

BASE_DIR = Path(__file__).parent


@pytest.fixture(scope="function")
def data(request):
    directory = BASE_DIR / "data" / f"{request.param}"
    inputs = pd.read_csv(directory / "inputs.csv")
    outputs = pd.read_csv(directory / "outputs.csv")
    return inputs, outputs


@pytest.fixture(scope="function")
def reference(request):
    with open(BASE_DIR / "reference" / "dea" / request.param) as f:
        return json.load(f)


@pytest.mark.parametrize(
    "data, reference, mismatches",
    [
        ["charnes", "charnes.json", 0],
        ["banks1", "banks1.json", 0],
        ["banks2", "banks2.json", 0],
        ["banks3", "banks3.json", 1],
    ],
    indirect=["data", "reference"],
)
def test_dea_envelopment(data, reference, mismatches):
    inputs, outputs = data
    benchmark(
        dea, inputs, outputs, Model.envelopment, reference, mismatches, 1e-6
    )


@pytest.mark.parametrize(
    "data, reference, mismatches",
    [
        ["charnes", "charnes.json", 0],
        ["banks1", "banks1.json", 4],
    ],
    indirect=["data", "reference"],
)
def test_dea_multiplier(data, reference, mismatches):
    inputs, outputs = data
    benchmark(
        dea, inputs, outputs, Model.multiplier, reference, mismatches, 1e-4
    )
