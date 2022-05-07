import numpy as np
import pytest

from dealib.dea import RTS, Orientation, direct

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
        ["simple", 0],
        ["charnes", 3],
        ["banks1", 0],
        ["banks2", 0],
        ["banks3", 0],
    ],
)
def test_direct_vector(rts, orientation, folder_name, mismatches):
    x, y = get_data(folder_name)
    reference = get_reference("direct", f"{folder_name}.json")
    vector = reference[orientation.name]["vector"]
    eff = direct(
        x,
        y,
        vector,
        rts=rts,
        orientation=orientation,
    )

    ref_objval = np.asarray(
        reference[orientation.name][rts.name]["objval"], dtype=float
    )
    ref_eff = np.asarray(
        reference[orientation.name][rts.name]["eff"], dtype=float
    )
    assert (
        np.count_nonzero(np.abs(ref_objval - eff.objval) > 1e-6) <= mismatches
    )
    compare_valid_values(eff.eff, ref_eff, mismatches * len(vector))


# @parametrize_options(RTS, "rts")
# @parametrize_options(Orientation, "orientation")
# @pytest.mark.parametrize(
#     "folder_name, mismatches",
#     [
#         ["simple", 0],
#         ["charnes", 3],
#         ["banks1", 0],
#         ["banks2", 0],
#         ["banks3", 0],
#     ],
# )
# def test_direct_scalar(orientation, rts, folder_name, mismatches):
#     x, y = get_data(folder_name)
#     reference = get_reference("direct", f"{folder_name}.json")
#     vector = reference[orientation.name]["vector"]
#     eff = direct(
#         x,
#         y,
#         vector,
#         orientation=orientation,
#         rts=rts,
#         xref=x.copy(),
#         yref=y.copy(),
#     )
#
#     ref_objval = np.asarray(
#         reference[orientation.name][rts.name]["objval"], dtype=float
#     )
#     ref_eff = np.asarray(
#         reference[orientation.name][rts.name]["eff"], dtype=float
#     )
#     assert (
#         np.count_nonzero(np.abs(ref_objval - eff.objval) > 1e-6) <= mismatches
#     )
#     compare_valid_values(eff.eff, ref_eff, mismatches * len(vector))
