import numpy as np
import pytest

from dealib.dea import RTS, Orientation, mea

from .utils import get_data, get_reference, parametrize_options


@parametrize_options(RTS, "rts")
@parametrize_options(Orientation, "orientation")
@pytest.mark.parametrize(
    "folder_name, mismatches",
    [["simple", 0], ["charnes", 4]],
)
def test_mea(rts, orientation, folder_name, mismatches):
    x, y = get_data(folder_name)
    reference = get_reference("mea", f"{folder_name}.json")
    eff = mea(x, y, rts=rts, orientation=orientation)
    ref_eff = np.asarray(reference[orientation.name][rts.name], dtype=float)
    assert np.count_nonzero(np.abs(ref_eff - eff.eff) > 1e-6) <= mismatches
