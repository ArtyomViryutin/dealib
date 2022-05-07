__all__ = ["Efficiency", "Malmquist"]

from dataclasses import dataclass

from numpy.typing import NDArray

from .options import RTS, Orientation


@dataclass
class Efficiency:
    """
    Object that is returned by :ref:`dea<dealib.dea.core._dea.dea>`, :ref:`add<dealib.dea.core._add.add>`,
    :ref:`direct<dealib.dea.core._direct.direct>`, :ref:`mea<dealib.dea.core._slack.slack>`,
    :ref:`mea<dealib.dea.core._mea.mea>`, :ref:`mea<dealib.dea.core._malmq.malmq>`.

    :param RTS rts: The return to scale assumption as in the option RTS in the call.

    :param Orientation orientation: The efficiency orientation as in the call.

    :param bool transpose: As in the call.

    :param 1-d array, 2-d array direct: Direction used for an estimating of efficiencies.

    :param 1-d array, 2-d array eff: The efficiencies. Note when DIRECT is used then the efficiencies are not Farrell
        efficiencies but rather excess values in DIRECT units of measurement.

    :param 1-d array objval: The objective value as returned from the LP program; normally the same as eff,
        but for slack it is the sum of the slacks.

    :param 2-d array lambdas: The lambdas, i.e. the weight of the peers, for each firm.

    :param 2-d array sx: A matrix for input slacks for each firm.

    :param 2-d array sy: A matrix for output slacks for each firm.

    :param 2-d array slack: A matrix of slacks for each firm.

    :param 2-d array ux: Dual variable for input.

    :param 2-d array vy: Dual variable for output.
    """

    rts: RTS = None
    orientation: Orientation = None
    transpose: bool = None
    direct: NDArray[float] = None
    eff: NDArray[float] = None
    objval: NDArray[float] = None
    lambdas: NDArray[float] = None
    sx: NDArray[float] = None
    sy: NDArray[float] = None
    slack: NDArray[float] = None
    ux: NDArray[float] = None
    vy: NDArray[float] = None


@dataclass
class Malmquist:
    """
    Object that is returned by *malmq*.

    :param 1-d array m: Malmquist index for productivity.

    :param 1-d array tc: Index for technology change.

    :param 1-d array ec: Index for efficiency change.

    :param 1-d array mq: Malmquist index for productivity; same as m.

    :param 1-d array e00: The efficiencies for period 0 with reference technology from period 0.

    :param 1-d array e10: The efficiencies for period 1 with reference technology from period 0.

    :param 1-d array e11: The efficiencies for period 1 with reference technology from period 1.

    :param 1-d array e01: The efficiencies for period 0 with reference technology from period 1.
    """

    m: NDArray[float]
    tc: NDArray[float]
    ec: NDArray[float]
    mq: NDArray[float]
    e00: NDArray[float]
    e10: NDArray[float]
    e11: NDArray[float]
    e01: NDArray[float]
