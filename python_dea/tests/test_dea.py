import numpy as np
import pytest

from python_dea.dea import RTS, Orientation, dea

from .utils import get_data, get_reference, parametrize_options


@parametrize_options(RTS, "rts")
@parametrize_options(Orientation, "orientation")
@pytest.mark.parametrize(
    "folder_name, mismatches",
    [
        ["simple", 0],
        ["charnes", 1],
        ["banks1", 1],
        ["banks2", 1],
        ["banks3", 1],
    ],
    ids=["simple", "charnes", "banks1", "banks2", "banks3"],
)
@pytest.mark.parametrize(
    "two_phase", [False, True], ids=["one_phase", "two_phase"]
)
def test_dea(orientation, rts, folder_name, mismatches, two_phase):
    x, y = get_data(folder_name)
    reference = get_reference("dea", f"{folder_name}.json")
    eff = dea(
        x,
        y,
        orientation=orientation,
        rts=rts,
        xref=x.copy(),
        yref=y.copy(),
        two_phase=two_phase,
    )
    ref = reference[orientation.name][rts.name]
    ref_eff = np.asarray(ref["eff"], dtype=float)
    ref_slack = np.asarray(ref["slack"], dtype=float)

    assert np.count_nonzero(np.abs(ref_eff - eff.eff) > 1e-4) <= mismatches

    threshold = np.mean(ref_slack) * 1e-3
    m, n = x.shape[1], y.shape[1]
    assert np.count_nonzero(
        np.abs(ref_slack - eff.slack) > threshold
    ) <= mismatches * (m + n)
