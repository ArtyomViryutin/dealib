__all__ = ["benchmark"]

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from python_dea.dea import RTS, Model, Orientation


def benchmark(
    f: Callable,
    inputs: NDArray[float],
    outputs: NDArray[float],
    model: Model,
    reference: dict,
    mismatches: int,
    tol: float,
    **kwargs,
) -> None:
    for o in Orientation:
        for r in RTS:
            eff = f(
                inputs, outputs, model=model, orientation=o, rts=r, **kwargs
            )
            ref = np.asarray(reference[str(o)][str(r)])
            assert np.count_nonzero(np.abs(ref - eff.eff) > tol) <= mismatches
