import tarfile

import numpy as np
import pytest

from python_dea.dea import RTS, Orientation, direct

from .utils import get_data, get_reference, parametrize_options


@parametrize_options(RTS, "rts")
@parametrize_options(Orientation, "orientation")
@pytest.mark.parametrize(
    "folder_name, mismatches",
    [
        ["simple", 0],
        ["charnes", 3],
        ["banks1", 0],
        ["banks2", 0],
        ["banks3", 0],
    ],
    ids=["simple", "charnes", "banks1", "banks2", "banks3"],
)
def test_direct(orientation, rts, folder_name, mismatches):
    x, y = get_data(folder_name)
    reference = get_reference("direct", f"{folder_name}.json")
    vector = reference[orientation.name]["vector"]
    eff = direct(
        x,
        y,
        vector,
        orientation=orientation,
        rts=rts,
        xref=x.copy(),
        yref=y.copy(),
    )

    ref_objval = np.asarray(reference[orientation.name][rts.name]["objval"])
    ref_eff = np.asarray(reference[orientation.name][rts.name]["eff"])
    ref_eff[ref_eff == np.nan] = 0
    ref_eff[ref_eff == np.inf] = 0
    eff.eff[eff.eff == np.nan] = 0
    eff.eff[eff.eff == np.inf] = 0
    assert (
        np.count_nonzero(np.abs(ref_objval - eff.objval) > 1e-6) <= mismatches
    )
    assert np.count_nonzero(
        np.abs(ref_eff - eff.eff) > 1e-6
    ) <= mismatches * len(vector)
