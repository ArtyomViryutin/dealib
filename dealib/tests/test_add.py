import numpy as np
import pytest

from dealib.dea import RTS, add

from .utils import get_data, get_reference, parametrize_options


@parametrize_options(RTS, "rts")
@pytest.mark.parametrize(
    "folder_name, mismatches",
    [
        ["simple", 0],
        ["charnes", 2],
    ],
    ids=["simple", "charnes"],
)
def test_add(rts, folder_name, mismatches):
    x, y = get_data(folder_name)
    reference = get_reference("add", f"{folder_name}.json")
    eff = add(x, y, rts=rts)
    ref_objval = np.asarray(reference[rts.name], dtype=float)
    assert (
        np.count_nonzero(np.abs(ref_objval - eff.objval) > ref_objval * 0.05)
        <= mismatches
    )
