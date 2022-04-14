import numpy as np
import pytest

from python_dea.dea import RTS, Efficiency, Orientation, slack

from .utils import get_data, get_reference, parametrize_options


@parametrize_options(RTS, "rts")
@parametrize_options(Orientation, "orientation")
@pytest.mark.parametrize(
    "folder_name",
    ["simple", "charnes", "banks1", "banks2", "banks3"],
)
def test_slack(orientation, rts, folder_name):
    x, y = get_data(folder_name)

    reference = get_reference("dea", f"{folder_name}.json")
    ref = reference[orientation.name][rts.name]
    ref_slack = np.asarray(ref["slack"])
    ref_eff = np.asarray(ref["eff"])

    k = x.shape[0]
    m = x.shape[1]
    n = y.shape[1]
    eff = Efficiency(rts, orientation, k, m, n)
    eff.eff = ref_eff
    eff = slack(x, y, eff, rts=rts, xref=x.copy(), yref=y.copy())

    threshold = np.mean(ref_slack) * 1e-4
    assert np.count_nonzero(np.abs(ref_slack - eff.slack) > threshold) == 0
