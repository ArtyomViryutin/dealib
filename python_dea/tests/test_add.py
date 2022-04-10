import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from python_dea.dea import RTS, add

BASE_DIR = Path(__file__).parent


@pytest.fixture(scope="function")
def data(request):
    directory = BASE_DIR / "data" / f"{request.param}"
    inputs = pd.read_csv(directory / "inputs.csv")
    outputs = pd.read_csv(directory / "outputs.csv")
    return inputs, outputs


@pytest.fixture(scope="function")
def reference(request):
    with open(BASE_DIR / "reference" / "add" / request.param) as f:
        return json.load(f)


@pytest.mark.parametrize(
    "data, reference, mismatches",
    [
        ["charnes", "charnes.json", 2],
    ],
    indirect=["data", "reference"],
)
def test_add(data, reference, mismatches):
    inputs, outputs = data
    for r in RTS:
        eff = add(inputs, outputs, rts=r)
        ref = np.asarray(reference[str(r)])
        assert (
            np.count_nonzero(np.abs(ref - eff.objval) > eff.objval * 0.05)
            <= mismatches
        )
