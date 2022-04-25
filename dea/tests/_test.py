from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dea.dea import *


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
    x, y = get_data("banks1")
    # x = [[20], [40], [40], [60], [70], [50]]
    # y = [[20], [30], [50], [40], [60], [20]]
    eff = dea(x, y, rts="vrs")
    print(eff.ux)
    print(eff.vy)
