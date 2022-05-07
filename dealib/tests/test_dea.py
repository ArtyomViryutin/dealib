import numpy as np
import pytest

from dealib.dea import RTS, Orientation, dea

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
def test_dea(rts, orientation, folder_name, mismatches, two_phase):
    x, y = get_data(folder_name)
    reference = get_reference("dea", f"{folder_name}.json")
    eff = dea(
        x,
        y,
        rts=rts,
        orientation=orientation,
        two_phase=two_phase,
    )
    ref = reference[orientation.name][rts.name]
    ref_eff = np.asarray(ref["eff"], dtype=float)
    ref_slack = np.asarray(ref["slack"], dtype=float)

    assert np.count_nonzero(np.abs(ref_eff - eff.eff) > 1e-4) <= mismatches

    threshold = np.mean(ref_slack) * 1e-3
    m = x.shape[1]
    n = y.shape[1]
    assert np.count_nonzero(
        np.abs(ref_slack - eff.slack) > threshold
    ) <= mismatches * (m + n)


# @parametrize_options(RTS, "rts")
# @parametrize_options(Orientation, "orientation")
# @pytest.mark.parametrize(
#     "folder_name, mismatches",
#     [["simple", 0], ["charnes", 1]],
#     ids=["simple", "charnes"],
# )
# def test_dea_ux_yv(rts, orientation, folder_name, mismatches):
#     x, y = get_data(folder_name)
#     reference = get_reference("dea_ux_yv", f"{folder_name}.json")
#     eff = dea(
#         x,
#         y,
#         rts=rts,
#         orientation=orientation,
#     )
#     ref = reference[orientation.name][rts.name]
#     ref_ux = np.asarray(ref["ux"], dtype=float)
#     ref_vy = np.asarray(ref["yv"], dtype=float)
#
#     assert np.count_nonzero(np.abs(eff.ux - ref_ux) > 1e-4) <= mismatches
#     assert np.count_nonzero(np.abs(eff.vy - ref_vy) > 1e-4) <= mismatches
