from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_dea.dea import RTS, Efficiency, Model, Orientation, add, dea, slack


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
    # TODO надо найти данные чтоб проверять оптимизацию слаков
    # x, y = get_data("banks3")
    # eff1 = dea(x, y, two_phase=True)
    # eff2 = dea(x, y)
    x = [[1, 5], [2, 2], [4, 1], [6, 1], [4, 4]]
    y = [[2], [2], [2], [2], [2]]
    eff = Efficiency(
        model=Model.envelopment,
        orientation=Orientation.input,
        rts=RTS.vrs,
        k=5,
        m=2,
        n=1,
    )
    eff.eff = np.array([1, 1, 1, 1, 0.5])
    eff1 = slack(x, y, eff)
    # print(eff1.slack)
    eff2 = dea(x, y, two_phase=True)
    print(eff2.slack)
    # print(eff1.slack[np.abs(eff1.slack - eff2.slack) > 0.0001])
    # print(eff2.slack[np.abs(eff1.slack - eff2.slack) > 0.0001])
