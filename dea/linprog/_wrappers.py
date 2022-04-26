__all__ = ["LPPResult", "LPP"]

from dataclasses import dataclass
from typing import Optional

from numpy.typing import NDArray


@dataclass
class LPPResult:
    """
    Result of linear programming problem.

    :param bool f: A value of objective function.

    :param 1-d array x: Solution vector at optima.

    :param 2-d array slack: Matrix of slack variables.

    :param 2-d array dual: Matrix of variables of dual LPP.
    """

    f: float
    x: NDArray[float]
    slack: NDArray[float]
    dual: NDArray[float]


@dataclass
class LPP:
    """
    Linear programming problem.

    :param 1-d array c: Objective function to optimize.

    :param 2-d array A_ub: Lhs matrix of inequalities constraints.

    :param 1-d array b_ub: Rhs vector of inequalities constraints.

    :param 1-d array A_eq: Lhs matrix of equalities constraints.

    :param 1-d array b_eq: Rhs vector of equalities constraints
    """

    c: Optional[NDArray[float]] = None
    A_ub: Optional[NDArray[float]] = None
    b_ub: Optional[NDArray[float]] = None
    A_eq: Optional[NDArray[float]] = None
    b_eq: Optional[NDArray[float]] = None
