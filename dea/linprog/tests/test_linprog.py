import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from .. import LPP, simplex

BASE_DIR = Path(__file__).parent / "reference"


@pytest.mark.parametrize(
    "ref_data",
    BASE_DIR.iterdir(),
    ids=[file.stem for file in BASE_DIR.iterdir()],
)
def test_simplex(ref_data: str) -> None:
    with open(ref_data) as file:
        data = json.load(file)
    lpp = LPP(
        np.asarray(data["c"]),
        np.asarray(data["A_ub"]),
        np.asarray(data["b_ub"]),
        np.asarray(data.get("A_eq")),
        np.asarray(data.get("b_eq")),
    )
    simplex_result = simplex(lpp)
    ref_result = data["result"]
    assert_allclose(simplex_result.f, ref_result["f"], rtol=1e-10, atol=1e-6)
    assert_allclose(simplex_result.x, ref_result["x"], rtol=1e-10, atol=1e-6)
    assert_allclose(
        simplex_result.slack, ref_result["slack"], rtol=1e-10, atol=1e-6
    )
