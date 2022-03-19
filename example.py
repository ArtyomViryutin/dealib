import numpy as np

from python_dea import RTS, Model, Orientation, dea

# Data from Benchmarking 4.6.1 Numerical example in R
inputs = np.array([[20], [40], [40], [60], [70], [50]])
outputs = np.array([[20], [30], [50], [40], [60], [20]])

result = dea(
    inputs,
    outputs,
    model=Model.envelopment,
    orientation=Orientation.input,
    rts=RTS.vrs,
)

print(result.efficiency)
print(result.slack)
print(result.lambdas)
