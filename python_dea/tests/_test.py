from datetime import datetime

import pandas as pd
from numpy.typing import NDArray

from python_dea.dea import RTS, Model, Orientation, add, dea


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
    x, y = get_data("charnes")
    eff = add(x, y, rts=RTS.vrs)
    print(eff.objval)
