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
        ["banks1", 3],
        ["banks2", 10],
        ["banks3", 24],
    ],
    ids=["simple", "charnes", "banks1", "banks2", "banks3"],
)
def test_mult(rts, orientation, folder_name, mismatches):
    x, y = get_data(folder_name)
    reference = get_reference("dea", f"{folder_name}.json")
    eff = mult(
        x,
        y,
        orientation=orientation,
        rts=rts,
        xref=x.copy(),
        yref=y.copy(),
    )
    ref_eff = np.asarray(reference[orientation.name][rts.name]["eff"])
    print(ref_eff[np.abs(ref_eff - eff.eff) > 1e-2])
    print(eff.eff[np.abs(ref_eff - eff.eff) > 1e-2])
    assert np.count_nonzero(np.abs(ref_eff - eff.eff) > 1e-2) <= mismatches
