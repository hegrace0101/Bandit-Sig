"""Feature transforms for linear-UCB baselines.

Each function takes a context window `ctx` of shape (L, d) (or a 1D vector,
treated as (L, 1)) and returns a 1D numpy feature vector.

`make_rff` is a factory: it returns a callable with a *fixed* random projection,
so the same features are produced on every call (as required by online UCB).
"""
from __future__ import annotations

from typing import Callable
import numpy as np


def flatten_features(ctx: np.ndarray) -> np.ndarray:
    """Row-major flatten of the window. Shape (L, d) -> (L*d,)."""
    x = np.asarray(ctx, dtype=float)
    return x.ravel()


def flatten_dim(L: int, d: int) -> int:
    return int(L * d)


_TEMPORAL_STATS = 7  # mean, std, min, max, first, last, slope


def temporal_stats_features(ctx: np.ndarray) -> np.ndarray:
    """Per-channel [mean, std, min, max, first, last, slope].

    Shape (L, d) -> (d * 7,). A 1D input is treated as (L, 1).
    Slope is the finite difference (last - first) / max(L - 1, 1).
    """
    x = np.asarray(ctx, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    L = x.shape[0]

    mean_ = x.mean(axis=0)
    std_ = x.std(axis=0)
    min_ = x.min(axis=0)
    max_ = x.max(axis=0)
    first_ = x[0]
    last_ = x[-1]
    slope_ = (last_ - first_) / max(L - 1, 1)

    return np.concatenate([mean_, std_, min_, max_, first_, last_, slope_])


def temporal_stats_dim(d: int) -> int:
    return int(d * _TEMPORAL_STATS)


def polynomial_features(ctx: np.ndarray, degree: int = 2) -> np.ndarray:
    """Degree-2 polynomial features (no bias) on the temporal average.

    Given a (L, d) window, first average over time to a (d,) vector `x`, then
    return [x, x**2, x_i * x_j for i < j]. Length = 2d + d*(d-1)/2.
    """
    if degree != 2:
        raise ValueError(f"Only degree=2 is supported, got degree={degree}.")

    x = np.asarray(ctx, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=0)

    d = x.shape[0]
    if d > 1:
        iu, ju = np.triu_indices(d, k=1)
        cross = x[iu] * x[ju]
    else:
        cross = np.zeros(0, dtype=float)

    return np.concatenate([x, x * x, cross])


def polynomial_dim(d: int, degree: int = 2) -> int:
    if degree != 2:
        raise ValueError(f"Only degree=2 is supported, got degree={degree}.")
    return int(2 * d + (d * (d - 1)) // 2)


def make_rff(
    d: int,
    D: int = 64,
    sigma: float = 1.0,
    seed: int = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Random Fourier Features approximating the RBF kernel

        k(x, z) = exp(-||x - z||^2 / (2 * sigma^2)).

    Returns a callable that maps a (L, d) window to a (D,) feature vector via
    phi(x) = sqrt(2/D) * cos(W x + b), with W ~ N(0, 1/sigma^2 I_d) and
    b ~ Uniform[0, 2 pi] fixed at construction time.

    2D input is first averaged over time to a (d,) vector.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(loc=0.0, scale=1.0 / float(sigma), size=(int(D), int(d)))
    b = rng.uniform(low=0.0, high=2.0 * np.pi, size=int(D))
    scale = float(np.sqrt(2.0 / int(D)))

    def rff(ctx: np.ndarray) -> np.ndarray:
        x = np.asarray(ctx, dtype=float)
        if x.ndim == 2:
            x = x.mean(axis=0)
        return scale * np.cos(W @ x + b)

    return rff
