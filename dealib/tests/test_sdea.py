import numpy as np
import pytest

from dealib.dea import RTS, Orientation, sdea

from .utils import (
    compare_valid_values,
    get_data,
    get_reference,
    parametrize_options,
)


@parametrize_options(RTS, "rts")
@parametrize_options(Orientation, "orientation")
@pytest.mark.parametrize(
    "folder_name, mismatches",
    [
        ["charnes", 0],
        ["banks1", 0],
        ["banks2", 1],
        ["banks3", 1],
    ],
)
def test_sdea(rts, orientation, folder_name, mismatches):
    x, y = get_data(folder_name)
    reference = get_reference("sdea", f"{folder_name}.json")
    eff = sdea(x, y, rts=rts, orientation=orientation)
    ref_eff = np.asarray(reference[orientation.name][rts.name], dtype=float)
    compare_valid_values(eff.eff, ref_eff, mismatches)
