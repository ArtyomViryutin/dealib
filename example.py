import numpy as np

from python_dea import RTS, Orientation, dea

# Data from Benchmarking 4.6.1 Numerical example in R
x = np.array([[20], [40], [40], [60], [70], [50]])
y = np.array([[20], [30], [50], [40], [60], [20]])

result = dea(
    x,
    y,
    orientation=Orientation.input,
    rts=RTS.vrs,
)

print(result.eff)
print(result.slack)
print(result.lambdas)
