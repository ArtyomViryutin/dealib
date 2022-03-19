from datetime import datetime

import numpy as np
import pandas as pd

from python_dea.dea import RTS, Model, Orientation, add, dea


def get_charnes():
    data = pd.read_csv("../tests/data/charnes1981.csv", delimiter=";")
    idata = np.nan_to_num(data.iloc[:, 1:6])
    odata = np.nan_to_num(data.iloc[:, 6:9])
    return idata, odata


def get_banks_i_o():
    data = pd.read_csv("../tests/data/banks_i.csv")
    idata = np.nan_to_num(data.iloc[:375, 3:])
    data = pd.read_csv("../tests/data/banks_o.csv")
    odata = np.nan_to_num(data.iloc[:375, 3:])
    return idata, odata


def get_banks(name: str = "banks"):
    data = pd.read_csv(f"../tests/data/{name}.csv")
    idata = np.nan_to_num(data[["STEXP", "FASUM", "INPSUM"]])
    odata = np.nan_to_num(data[["LNSUM", "DBSUM", "INCSUM"]])
    return idata, odata


def benchmark(inputs: np.ndarray, outputs: np.ndarray, n: int) -> None:
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


if __name__ == "__main__":
    # idata, odata = get_charnes()
    idata = np.array([[4, 2], [2, 4], [4, 6]])
    odata = np.array([[4], [4], [2]])
    # idata = np.array([[2], [3], [2], [4], [6], [5], [6], [8]])
    # odata = np.array([[1], [3], [2], [3], [5], [2], [3], [5]])
    for rts in [RTS.drs]:
        res = add(
            idata,
            odata,
            rts=rts,
        )
        print()
        print(f"{rts}: {res.efficiency}")
