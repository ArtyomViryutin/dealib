from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dealib.dea import *


def benchmark(inputs: NDArray[float], outputs: NDArray[float], n: int) -> None:
    for o in Orientation:
        for r in RTS:
            total = 0
            for i in range(n):
                start = datetime.now()
                dea(
                    inputs,
                    outputs,
                    orientation=o,
                    rts=r,
                )
                end = datetime.now()
                total += (end - start).total_seconds()
            print(f"{o}-{r}: {total / n:.3f}s")


def get_data(name: str):
    inputs = np.asarray(
        pd.read_csv(
            f"/home/artyomviryutin/PycharmProjects/DEA/dea/tests/data/{name}/inputs.csv"
        )
    )
    outputs = np.asarray(
        pd.read_csv(
            f"/home/artyomviryutin/PycharmProjects/DEA/dea/tests/data/{name}/outputs.csv"
        )
    )
    return inputs, outputs


if __name__ == "__main__":
    # x, y = get_data("banks1")
    x0 = [[10], [28], [30], [60]]
    y0 = [[5], [7], [10], [15]]
    x1 = [[12], [26], [16], [60]]
    y1 = [[6], [8], [9], [15]]
    eff = malmq(x0, y0, x1, y1)
    print(eff.m)
