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
    ids=[
        "simple-max-0-mismatches",
        "charnes-max-1-mismatches",
        "banks1-max-0-mismatches",
        "banks2-max-0-mismatches",
        "banks3-max-1-mismatches",
    ],
)
@pytest.mark.parametrize(
    "two_phase", [False, True], ids=["one_phase", "two_phase"]
)
def test_dea(orientation, rts, folder_name, mismatches, two_phase):
    inputs, outputs = get_data(folder_name)
    reference = get_reference("dea", f"{folder_name}.json")
    eff = dea(
        inputs,
        outputs,
        orientation=orientation,
        rts=rts,
        two_phase=two_phase,
    )
    ref = reference[orientation.name][rts.name]
    ref_eff = np.asarray(ref["eff"])
    ref_slack = np.asarray(ref["slack"])

    assert np.count_nonzero(np.abs(ref_eff - eff.eff) > 1e-4) <= mismatches

    threshold = np.mean(ref_slack) * 1e-3
    m, n = inputs.shape[1], outputs.shape[1]
    assert np.count_nonzero(
        np.abs(ref_slack - eff.slack) > threshold
    ) <= mismatches * (m + n)
