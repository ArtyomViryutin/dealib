import numpy as np
import pytest

from python_dea.dea import RTS, add

from .utils import get_data, get_reference, parametrize_options


@parametrize_options(RTS, "rts")
@pytest.mark.parametrize(
    "folder_name, mismatches",
    [
        ["simple", 0],
        ["charnes", 2],
    ],
    ids=[
        "simple-max-0-mismatches",
        "charnes-max-2-mismatches",
    ],
)
def test_add(rts, folder_name, mismatches):
    x, y = get_data(folder_name)
    reference = get_reference("add", f"{folder_name}.json")
    eff = add(x, y, rts=rts)
    ref_objval = np.asarray(reference[rts.name])
    print(ref_objval[np.abs(ref_objval - eff.objval) > ref_objval * 0.05])
    print(eff.objval[np.abs(ref_objval - eff.objval) > ref_objval * 0.05])
    assert (
        np.count_nonzero(np.abs(ref_objval - eff.objval) > ref_objval * 0.05)
        <= mismatches
    )
