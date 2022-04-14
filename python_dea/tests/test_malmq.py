import numpy as np
import pytest

from python_dea.dea import RTS, Orientation, malmq

from .utils import get_data, get_reference, parametrize_options


@parametrize_options(RTS, "rts")
@parametrize_options(Orientation, "orientation")
@pytest.mark.parametrize(
    "data0, data1, mismatches",
    [
        ["banks1", "banks2", 3],
        ["banks1", "banks3", 2],
        ["banks2", "banks3", 2],
    ],
    ids=["banks1-banks2", "banks1-banks3", "banks2-banks3"],
)
def test_malmq(rts, orientation, data0, data1, mismatches):
    x0, y0 = get_data(data0)
    x1, y1 = get_data(data1)
    ref = get_reference("malmq", f"{data0}_{data1}.json")[orientation.name][
        rts.name
    ]
    eff = malmq(x0, y0, x1, y1, rts=rts, orientation=orientation)
    for index in ("m", "tc", "ec", "mq", "e00", "e10", "e11", "e01"):
        ref_index = np.asarray(ref[index])
        valid_values = np.logical_and(
            ref_index != np.nan, np.abs(ref_index) != np.inf
        )
        eff_index = getattr(eff, index)
        assert (
            np.count_nonzero(
                np.abs(ref_index[valid_values] - eff_index[valid_values])
                > 1e-6
            )
            <= mismatches
        )
