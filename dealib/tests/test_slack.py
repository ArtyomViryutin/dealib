import numpy as np
import pytest

from dealib.dea import RTS, Efficiency, Orientation, slack

from .utils import get_data, get_reference, parametrize_options


@parametrize_options(RTS, "rts")
@parametrize_options(Orientation, "orientation")
@pytest.mark.parametrize(
    "folder_name",
    ["simple", "charnes", "banks1", "banks2", "banks3"],
)
def test_slack(rts, orientation, folder_name):
    x, y = get_data(folder_name)

    reference = get_reference("dea", f"{folder_name}.json")
    ref = reference[orientation.name][rts.name]
    ref_slack = np.asarray(ref["slack"], dtype=float)
    ref_eff = np.asarray(ref["eff"], dtype=float)

    m = x.shape[1]
    n = y.shape[1]
    k = x.shape[0]
    e = Efficiency(
        rts=rts,
        orientation=orientation,
        transpose=False,
        eff=ref_eff,
        objval=np.zeros(k),
        lambdas=np.zeros((k, k)),
        sx=np.zeros((k, m)),
        sy=np.zeros((k, n)),
    )
    sl_eff = slack(x, y, e)
    threshold = np.mean(ref_slack) * 1e-4
    assert np.count_nonzero(np.abs(ref_slack - sl_eff.slack) > threshold) == 0
