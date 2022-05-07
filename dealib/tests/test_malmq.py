import numpy as np
import pytest

from dealib.dea import RTS, Orientation, malmq

from .utils import (
    compare_valid_values,
    get_data,
    get_reference,
    parametrize_options,
)


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
        eff_index = getattr(eff, index)
        ref_index = np.asarray(ref[index], dtype=float)
        compare_valid_values(eff_index, ref_index, mismatches)
