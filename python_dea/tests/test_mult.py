import numpy as np
import pytest

from python_dea.dea import RTS, Orientation, mult

from .utils import get_data, get_reference, parametrize_options


@parametrize_options(RTS, "rts")
@parametrize_options(Orientation, "orientation")
@pytest.mark.parametrize(
    "folder_name, mismatches",
    [
        ["simple", 0],
        ["charnes", 9],
        # ["banks1", 4],
        # ["banks2", 1]
    ],
    ids=[
        "simple-max-0-mismatches",
        "charnes-max-2-mismatches",
        # "banks1-max-4-mismatches",
        # "banks2-max-0-mismatches",
    ],
)
def test_mult(rts, orientation, folder_name, mismatches):
    x, y = get_data(folder_name)
    reference = get_reference("dea", f"{folder_name}.json")
    eff = mult(
        x,
        y,
        orientation=orientation,
        rts=rts,
    )
    ref_eff = np.asarray(reference[orientation.name][rts.name]["eff"])
    assert np.count_nonzero(np.abs(ref_eff - eff.eff) > 1e-4) <= mismatches
