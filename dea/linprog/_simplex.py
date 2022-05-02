__all__ = ["simplex"]

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ._wrappers import LPP, LPPResult


def _pivot_col(
    *,
    T: NDArray[float],
    tol: float,
) -> Tuple[bool, int]:
    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return False, np.nan
    return True, np.ma.nonzero(ma == ma.min())[0][0]


def _pivot_row(
    *,
    T: NDArray[float],
    pivot_col: int,
    phase: int,
    tol: float,
) -> Tuple[bool, int]:
    if phase == 1:
        k = 2
    else:
        k = 1
    ma = np.ma.masked_where(
        T[:-k, pivot_col] <= tol, T[:-k, pivot_col], copy=False
    )
    if ma.count() == 0:
        return False, np.nan
    mb = np.ma.masked_where(T[:-k, pivot_col] <= tol, T[:-k, -1], copy=False)
    x = mb / ma
    return True, np.ma.nonzero(x == x.min())[0][0]


def _apply_pivot(
    *,
    T: NDArray[float],
    basis: NDArray[float],
    pivot_row: int,
    pivot_col: int,
) -> None:
    basis[pivot_row] = pivot_col
    pivot_value = T[pivot_row, pivot_col]
    T[pivot_row] = T[pivot_row] / pivot_value
    for iter_row in range(T.shape[0]):
        if iter_row != pivot_row:
            T[iter_row] = T[iter_row] - T[pivot_row] * T[iter_row, pivot_col]


def _solve_simplex(
    *,
    T: NDArray[float],
    basis: NDArray[float],
    maxiter: int,
    phase: int,
    tol: float,
    nit0: Optional[int] = 0,
):
    nit = nit0
    complete = False
    if phase == 2:
        for pivot_row in [
            row for row in range(basis.size) if basis[row] > T.shape[1] - 2
        ]:
            non_zero_row = [
                col
                for col in range(T.shape[1] - 1)
                if abs(T[pivot_row, col]) > tol
            ]
            if len(non_zero_row) > 0:
                pivot_col = non_zero_row[0]
                _apply_pivot(
                    T=T, basis=basis, pivot_row=pivot_row, pivot_col=pivot_col
                )
                nit += 1

    while not complete:
        pivot_col_found, pivot_col = _pivot_col(T=T, tol=tol)
        if not pivot_col_found:
            complete = True
        else:
            pivot_row_found, pivot_row = _pivot_row(
                T=T, pivot_col=pivot_col, phase=phase, tol=tol
            )
            if not pivot_row_found:
                complete = True
        if not complete:
            if nit >= maxiter:
                complete = True
            else:
                # noinspection PyUnboundLocalVariable
                _apply_pivot(
                    T=T, basis=basis, pivot_row=pivot_row, pivot_col=pivot_col
                )
                nit += 1
    return nit


def _get_canonical_form(
    *,
    lpp: LPP,
    opt_f: bool,
    opt_slacks: bool,
    eps: float,
) -> Tuple[NDArray[float], ...]:
    m_ub, n_ub = lpp.A_ub.shape
    if np.any(lpp.A_eq):
        m_eq = lpp.A_eq.shape[0]
    else:
        m_eq = 0

    if m_eq > 0:
        A = np.vstack(
            (
                np.hstack((lpp.A_ub, np.eye(m_ub))),
                np.hstack((lpp.A_eq, np.zeros((m_eq, m_ub)))),
            )
        )
        b = np.hstack((lpp.b_ub, lpp.b_eq))
    else:
        A = np.hstack((lpp.A_ub, np.eye(m_ub)))
        b = lpp.b_ub.copy()
    if opt_f and opt_slacks:
        c = np.hstack((lpp.c, eps * np.ones(m_ub)))
    elif not opt_f and opt_slacks:
        c = np.hstack((np.zeros(n_ub), np.ones(m_ub)))
    elif opt_f and not opt_slacks:
        c = np.hstack((lpp.c, np.zeros(m_ub)))
    else:
        raise NotImplementedError
    return A, b, c


def simplex(
    lpp: LPP,
    *,
    opt_f: Optional[bool] = True,
    opt_slacks: Optional[bool] = False,
    maxiter: Optional[int] = 1000,
    eps: Optional[float] = 1e-6,
    tol: Optional[float] = 1e-9,
) -> LPPResult:
    """
    Solves the given linear programming problem.

    :param LPP lpp: Linear programming problem.

    :param bool opt_f: Flag that determines whether to optimize the objective function.

    :param bool opt_slacks: Flag that determines whether to optimize the slack variables.

    :param int maxiter: Maximal number of iteration.

    :param float eps: Float number for slack optimization.

    :param float tol: Tolerance for LPP solving.

    :return: LPP solution.
    :rtype: LPPResult


    **Example**

    >>> c = np.array([3, 2])
    >>> A_ub = np.array([[2, 1], [-1, -3]])
    >>> b_ub = np.array([50, -15])
    >>> A_eq = np.array([[5, 6]])
    >>> b_eq = np.array([60])
    >>> lpp = LPP(c, A_ub, b_ub, A_eq, b_eq)
    >>> result = simplex(lpp)
    >>> print(result.f)
    33.333333
    >>> print(result.x)
    [10, 1.666667]
    >>> print(result.slack)
    [28.333333, 0]
    """

    init_m, init_n = lpp.A_ub.shape
    A, b, c = _get_canonical_form(
        lpp=lpp, opt_f=opt_f, opt_slacks=opt_slacks, eps=eps
    )
    m, n = A.shape

    is_negative_constraint = np.less(b, 0)
    A[is_negative_constraint] *= -1
    b[is_negative_constraint] *= -1
    av = np.arange(m) + n
    basis = av.copy()

    row_constraints = np.hstack((A, np.eye(m), b[:, np.newaxis]))
    row_objective = np.hstack((-c, np.zeros(m + 1)))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective[av] = 0
    T = np.vstack((row_constraints, row_objective, row_pseudo_objective))

    # Phase 1
    nit1 = _solve_simplex(T=T, basis=basis, maxiter=maxiter, phase=1, tol=tol)
    T = np.delete(T, -1, axis=0)
    T = np.delete(T, av, axis=1)

    # Phase 2
    _solve_simplex(
        T=T, basis=basis, maxiter=maxiter, phase=2, tol=tol, nit0=nit1
    )
    solution = np.zeros(n)
    solution[basis[:m]] = T[:m, -1]
    dual = T[-1, -(init_m + 1) : -1]
    x = solution[:init_n]
    slack = solution[init_n:]
    f = T[-1, -1]
    return LPPResult(f=f, x=x, slack=slack, dual=dual)
