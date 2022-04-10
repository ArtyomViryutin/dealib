from datetime import datetime

import numpy as np
import numpy.linalg
import pandas as pd
from numpy.typing import NDArray

from python_dea.dea import (
    RTS,
    Efficiency,
    Model,
    Orientation,
    add,
    dea,
    direct,
    slack,
)


def benchmark(inputs: NDArray[float], outputs: NDArray[float], n: int) -> None:
    for o in Orientation:
        for r in RTS:
            total = 0
            for i in range(n):
                start = datetime.now()
                dea(
                    inputs,
                    outputs,
                    model=Model.envelopment,
                    orientation=o,
                    rts=r,
                )
                end = datetime.now()
                total += (end - start).total_seconds()
            print(f"{o}-{r}: {total / n:.3f}s")


def get_data(name: str):
    inputs = pd.read_csv(
        f"/home/artyomviryutin/PycharmProjects/DEA/python_dea/tests/data/{name}/inputs.csv"
    )
    outputs = pd.read_csv(
        f"/home/artyomviryutin/PycharmProjects/DEA/python_dea/tests/data/{name}/outputs.csv"
    )
    return inputs, outputs


if __name__ == "__main__":
    # x = np.asarray([
    #     [0.5, 4],
    #     [1, 2],
    #     [2, 1],
    #     [4, 0.5],
    #     [3, 2],
    #     [1, 4]
    # ])
    # y = np.asarray([[1], [1], [1], [1], [1], [1]])

    x = [[1, 5], [2, 2], [4, 1], [6, 1], [4, 4]]
    y = [[2], [2], [2], [2], [2]]
    x, y = get_data("banks1")
    print(x.mean())
    print(y.mean())
    d = np.asarray([100000, 100000, 100000])
    eff = direct(
        np.nan_to_num(x), y, d, orientation=Orientation.input, rts=RTS.vrs
    )
    print(eff.objval)
    # print(eff.objval)
    # print(eff.eff)
