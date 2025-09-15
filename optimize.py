# optimize.py
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class Constraints:
    # Upper bound per asset weight (e.g., 0.4 means each weight <= 40%).
    # If None, the bound defaults to 1.0 (no per-asset cap; shorting is still disallowed).
    max_weight_per_asset: float | None = None


def _as_np(x) -> np.ndarray:
    """Convert pandas objects or arrays to a float numpy array."""
    if hasattr(x, "values"):
        return np.asarray(x.values, dtype=float)
    return np.asarray(x, dtype=float)


def equal_weight(n: int) -> np.ndarray:
    """Equal-weight vector of length n."""
    if n <= 0:
        raise ValueError("n must be positive.")
    return np.ones(n, dtype=float) / n


def _bounds(n: int, cons: Constraints | None) -> list[tuple[float, float]]:
    """
    Box bounds for SLSQP: 0 <= w_i <= hi (no shorting).
    If max_weight_per_asset is None, hi = 1.0.
    """
    hi = 1.0 if cons is None or cons.max_weight_per_asset is None else float(cons.max_weight_per_asset)
    hi = min(hi, 1.0)
    if hi <= 0:
        raise ValueError("max_weight_per_asset must be > 0.")
    return [(0.0, hi) for _ in range(n)]


def _sum_to_one_constraint():
    """Linear equality constraint: sum(w) = 1."""
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _cleanup_weights(w: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Numerical cleanup after optimization: zero-out tiny values and renormalize to sum to 1.
    (SLSQP already enforces sum(w)=1, but this guards against tiny drift.)
    """
    w = np.asarray(w, dtype=float)
    w[np.abs(w) < tol] = 0.0
    s = w.sum()
    return (w / s) if s > 0 else w


def _random_feasible_start(n: int, cap: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random feasible starting point under:
      sum(w)=1, 0<=w_i<=cap.
    Simple approach: Dirichlet -> clip by cap -> renormalize.
    """
    w = rng.dirichlet(np.ones(n))
    if cap < 1.0:
        w = np.minimum(w, cap)
        s = w.sum()
        w = (w / s) if s > 0 else equal_weight(n)
    return w

def _ensure_cap_feasible(n: int, cons: Constraints | None):
    if cons is None or cons.max_weight_per_asset is None:
        return
    cap = float(cons.max_weight_per_asset)
    if cap <= 0:
        raise ValueError("max_weight_per_asset must be > 0.")
    if cap * n < 1.0:
        raise ValueError(
            f"Infeasible: n={n}, max_weight_per_asset={cap} → n*cap={cap*n:.3f} < 1. "
            f"Increase cap or add more assets."
        )


def solve_min_variance(Sigma, cons: Constraints | None = None) -> np.ndarray:
    """
    Minimum variance portfolio:
        minimize   w^T Σ w
        subject to sum(w) = 1,  0 <= w_i <= max_weight_per_asset
    - Sigma: covariance matrix (preferably annualized), np.ndarray or pd.DataFrame.
    """
    _ensure_cap_feasible(_as_np(Sigma).shape[0], cons)
    S = _as_np(Sigma)
    n = S.shape[0]
    w0 = equal_weight(n)
    obj = lambda w: float(w @ S @ w)
    bounds = _bounds(n, cons)
    result = minimize(obj, w0, bounds=bounds, constraints=[_sum_to_one_constraint()], method="SLSQP")
    w = result.x
    return _cleanup_weights(w)


def solve_max_return(mu, cons: Constraints | None = None) -> np.ndarray:
    """
    Maximum expected return portfolio:
        maximize   μ^T w    (equivalently minimize -μ^T w)
        subject to sum(w) = 1,  0 <= w_i <= max_weight_per_asset
    - mu: expected returns vector (use the same scale as used elsewhere, e.g., annualized).
    """
    _ensure_cap_feasible(_as_np(mu).reshape(-1).shape[0], cons)
    m = _as_np(mu).reshape(-1)
    n = m.shape[0]
    w0 = equal_weight(n)
    obj = lambda w: float(-(m @ w))
    bounds = _bounds(n, cons)
    result = minimize(obj, w0, bounds=bounds, constraints=[_sum_to_one_constraint()], method="SLSQP")
    w = result.x
    return _cleanup_weights(w)


def solve_max_sharpe(
    mu,
    Sigma,
    rf: float = 0.0,
    cons: Constraints | None = None,
    eps: float = 1e-12,
    restarts: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """
    Maximum Sharpe ratio portfolio:
        maximize   ((μ - r_f)^T w) / sqrt(w^T Σ w)
        subject to sum(w) = 1,  0 <= w_i <= max_weight_per_asset
    Implemented by minimizing the negative Sharpe. A small eps is added to avoid division by zero.
    Supports multi-start (restarts > 0) to improve stability on non-convex landscapes.
    - mu: expected returns vector (same scale as Sigma, e.g., annualized if Sigma is annualized).
    - Sigma: covariance matrix (preferably annualized).
    - rf: risk-free rate (same scale as mu).
    - restarts: number of additional random restarts (0 disables multi-start).
    - seed: random seed for reproducibility of starting points.
    """
    _ensure_cap_feasible(_as_np(mu).reshape(-1).shape[0], cons)
    m = _as_np(mu).reshape(-1)
    S = _as_np(Sigma)
    n = m.shape[0]

    def neg_sharpe(w: np.ndarray) -> float:
        ret_excess = float(m @ w - rf)
        vol = float(np.sqrt(max(w @ S @ w, 0.0)) + eps)
        return -ret_excess / vol

    bounds = _bounds(n, cons)
    eq_cons = [_sum_to_one_constraint()]

    # Determine per-asset cap for feasible random starts
    cap = 1.0 if (cons is None or cons.max_weight_per_asset is None) else float(cons.max_weight_per_asset)
    cap = min(cap, 1.0)

    # Candidate starting points: equal weight + random feasible starts
    rng = np.random.default_rng(seed)
    starts = [equal_weight(n)]
    for _ in range(max(0, int(restarts))):
        starts.append(_random_feasible_start(n, cap, rng))

    # Run SLSQP from each start, keep the best solution
    best_fun = np.inf
    best_w = None
    for w0 in starts:
        res = minimize(neg_sharpe, w0, bounds=bounds, constraints=eq_cons, method="SLSQP")
        if res.fun < best_fun:
            best_fun = res.fun
            best_w = res.x

    return _cleanup_weights(best_w)
