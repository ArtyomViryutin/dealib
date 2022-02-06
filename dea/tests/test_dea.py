import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dea import RTS, Model, Orientation, dea

BASE_DIR = Path(__file__).parent


@pytest.fixture(scope="function")
def banks(request):
    data = pd.read_csv(BASE_DIR / "data" / request.param)
    idata = np.nan_to_num(data[["STEXP", "FASUM", "INPSUM"]])
    odata = np.nan_to_num(data[["LNSUM", "DBSUM", "INCSUM"]])
    return idata, odata


@pytest.fixture(scope="function")
def reference(request):
    with open(BASE_DIR / "reference" / "dea" / request.param) as f:
        return json.load(f)


@pytest.mark.parametrize(
    "banks, reference, mismatches",
    [
        ["banks1.csv", "banks1.json", 0],
        ["banks2.csv", "banks2.json", 0],
        ["banks3.csv", "banks3.json", 1],
    ],
    indirect=["banks", "reference"],
)
def test_dea_envelopment(banks, reference, mismatches):
    inputs, outputs = banks
    tol = 1e-6
    for o in Orientation:
        for r in RTS:
            res = dea(inputs, outputs, model=Model.envelopment, orientation=o, rts=r)
            if o == Orientation.input:
                assert np.all(res.efficiency <= 1 + tol)
            else:
                assert np.all(res.efficiency >= 1 - tol)
            ref = np.asarray(reference[str(o)][str(r)])
            assert np.count_nonzero(np.abs(ref - res.efficiency) > tol) <= mismatches
