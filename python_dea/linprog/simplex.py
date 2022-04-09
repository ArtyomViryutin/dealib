__all__ = ["simplex"]

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .wrappers import LPP, LPPResult


def _pivot_col(
    T: NDArray[float],
    tol: float,
) -> Tuple[bool, int]:
    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return False, np.nan
    return True, np.ma.nonzero(ma == ma.min())[0][0]


def _pivot_row(
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
    T: NDArray[float],
    basis: NDArray[float],
    maxiter: int,
    phase: int,
    tol: float,
    nit0: int = 0,
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
                _apply_pivot(T, basis, pivot_row, pivot_col)
                nit += 1

    while not complete:
        pivot_col_found, pivot_col = _pivot_col(T, tol)
        if not pivot_col_found:
            complete = True
        else:
            pivot_row_found, pivot_row = _pivot_row(T, pivot_col, phase, tol)
            if not pivot_row_found:
                complete = True
        if not complete:
            if nit >= maxiter:
                complete = True
            else:
                _apply_pivot(T, basis, pivot_row, pivot_col)  # noqa
                nit += 1
    return nit


def _get_canonical_form(
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
    opt_f: bool = True,
    opt_slacks: bool = False,
    maxiter: int = 1000,
    eps: float = 1e-6,
    tol: float = 1e-9,
) -> LPPResult:
    init_m, init_n = lpp.A_ub.shape
    A, b, c = _get_canonical_form(lpp, opt_f, opt_slacks, eps=eps)
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
    nit1 = _solve_simplex(T, basis, maxiter=maxiter, phase=1, tol=tol)
    T = np.delete(T, -1, axis=0)
    T = np.delete(T, av, axis=1)

    # Phase 2
    _ = _solve_simplex(T, basis, maxiter=maxiter, phase=2, tol=tol, nit0=nit1)
    solution = np.zeros(n)
    solution[basis[:m]] = T[:m, -1]
    x = solution[:init_n]
    slack = solution[init_n:]
    f = T[-1, -1]
    return LPPResult(f, x, slack)
