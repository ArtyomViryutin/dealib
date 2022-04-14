from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_dea.dea import (
    RTS,
    Efficiency,
    Orientation,
    add,
    dea,
    direct,
    malmq,
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
    x0, y0 = get_data("banks1")
    x1, y1 = get_data("banks2")
    # x0 = [[10], [28], [30], [60]]
    # y0 = [[5], [7], [10], [15]]
    # x1 = [[12], [26], [16], [60]]
    # y1 = [[6], [8], [9], [15]]
    res = malmq(x0, y0, x1, y1)
    print(f"m: {res.m}")
    print(f"tc: {res.tc}")
    print(f"ec: {res.ec}")
    print(f"mq: {res.mq}")
    print(f"e00: {res.e00}")
    print(f"e10: {res.e10}")
    print(f"e11: {res.e11}")
    print(f"e01: {res.e01}")

    dea(x0, y0)
