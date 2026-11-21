from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import numpy as np


_TM_CACHE: dict[tuple, float] = {}
_TM_RNG_SEED = 2024


def sample_f_composite_gain(size: int, m: float, m_s: float, rng: np.random.Generator) -> np.ndarray:
    short_term = rng.gamma(shape=m, scale=1.0 / m, size=size)
    shadowing = 1.0 / rng.gamma(shape=m_s, scale=1.0 / m_s, size=size)
    return short_term * shadowing


def local_energy_gamma(
    N: int,
    M: int,
    snr_db: float,
    hypothesis: str,
    h: np.ndarray,
    unc_db: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact conditional Gamma law for complex Gaussian energy averaging."""
    sigma_u2 = 10.0 ** (np.asarray(unc_db, dtype=float) / 10.0)
    alpha = 10.0 ** (float(snr_db) / 10.0)
    if hypothesis == "H0":
        mean_energy = sigma_u2
    elif hypothesis == "H1":
        mean_energy = sigma_u2 + np.asarray(h, dtype=float) * (alpha * sigma_u2)
    else:
        raise ValueError("hypothesis must be H0 or H1")
    return rng.gamma(shape=M, scale=mean_energy / M)


def reporting_channel(
    T: np.ndarray,
    g: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    return np.asarray(g) * np.asarray(T) + np.asarray(eta)


def apply_attack(
    T: np.ndarray,
    malicious_idx: np.ndarray,
    attack: str,
    rng_attack: np.random.Generator,
    shift_db: float = 0.5,
    extreme_db: float = 6.5,
    mild_db: float = 0.5,
) -> np.ndarray:
    out = np.asarray(T, dtype=float).copy()
    idx = np.asarray(malicious_idx, dtype=int)
    if idx.size == 0 or attack == "none":
        return out
    if attack == "always_yes":
        out[idx] *= 10.0 ** (abs(shift_db) / 10.0)
    elif attack == "always_no":
        out[idx] /= 10.0 ** (abs(shift_db) / 10.0)
    elif attack == "masking":
        half = idx.size // 2
        out[idx[:half]] *= 10.0 ** (extreme_db / 10.0)
        out[idx[half:]] *= 10.0 ** (mild_db / 10.0)
    elif attack == "shift":
        out[idx] *= 10.0 ** (abs(shift_db) / 10.0)
    else:
        raise ValueError(f"unknown attack {attack}")
    return out


def choose_malicious_idx(N: int, rho: float, seed: int, condition_id: int) -> np.ndarray:
    L = int(round(N * float(rho)))
    if L <= 0:
        return np.empty(0, dtype=int)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 500_000 + int(condition_id)]))
    return np.sort(rng.choice(N, size=L, replace=False).astype(int))


def tietjen_moore_statistic(x: np.ndarray, t: int, side: str = "upper") -> float:
    x = np.sort(np.asarray(x, dtype=float))
    N = len(x)
    if not 1 <= t < N:
        return 1.0
    xbar = x.mean()
    if side == "upper":
        trimmed = x[: N - t]
    elif side == "lower":
        trimmed = x[t:]
    else:
        raise ValueError(side)
    xbar_t = trimmed.mean()
    num = np.sum((trimmed - xbar_t) ** 2)
    den = np.sum((x - xbar) ** 2)
    return float(num / den) if den > 0 else 1.0


def tm_critical_value(N: int, t: int, alpha: float, n_sim: int, side: str = "upper") -> float:
    key = (N, t, alpha, n_sim, side)
    if key not in _TM_CACHE:
        rng = np.random.default_rng(np.random.SeedSequence([_TM_RNG_SEED, N, t, int(alpha * 10_000), n_sim]))
        sims = rng.standard_normal((n_sim, N))
        vals = np.empty(n_sim)
        for i in range(n_sim):
            vals[i] = tietjen_moore_statistic(sims[i], t, side=side)
        _TM_CACHE[key] = float(np.quantile(vals, alpha))
    return _TM_CACHE[key]


def warm_tm_cache(N: int, alpha: float, n_sim: int) -> None:
    for t in range(1, N - 1):
        tm_critical_value(N, t, alpha, n_sim, "upper")


def save_tm_cache(path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(_TM_CACHE, f)


def load_tm_cache(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        with open(p, "rb") as f:
            _TM_CACHE.update(pickle.load(f))


def modified_largest_gap_tm(x: np.ndarray, alpha: float, n_sim: int) -> np.ndarray:
    """Recursive upper-tail MLG+TM; preserves the cited baseline's one-sided behavior."""
    order = np.argsort(np.asarray(x, dtype=float))
    D = np.asarray(x, dtype=float)[order]
    idx_map = order.copy()
    while True:
        n = len(D)
        if n < 4:
            break
        half = n // 2
        D_u = D[half:]
        idx_u = idx_map[half:]
        if len(D_u) < 2:
            break
        gaps = np.diff(D_u)
        g = int(np.argmax(gaps))
        s1, s2 = D_u[: g + 1], D_u[g + 1:]
        i1, i2 = idx_u[: g + 1], idx_u[g + 1:]
        if len(s2) == 0:
            break
        S = np.concatenate([D[:half], s1])
        S_idx = np.concatenate([idx_map[:half], i1])
        full = np.concatenate([S, s2])
        t = len(s2)
        if tm_critical_value(len(full), t, alpha, n_sim, "upper") > tietjen_moore_statistic(full, t, "upper"):
            D, idx_map = S, S_idx
        else:
            break
    return np.sort(np.setdiff1d(np.arange(len(x)), idx_map))


def mad_scale(x: np.ndarray, c: float = 1.4826) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(c * np.median(np.abs(x - med)))


def sn_estimator(x: np.ndarray, c: float = 1.1926) -> float:
    x = np.asarray(x, dtype=float)
    med_j = np.array([np.median(np.abs(x - xi)) for xi in x])
    return float(c * np.median(med_j))


def qn_estimator(x: np.ndarray, c: float = 2.219144465985076) -> float:
    x = np.asarray(x, dtype=float)
    diffs = np.array([abs(x[i] - x[j]) for i in range(len(x)) for j in range(i + 1, len(x))])
    h = len(x) // 2 + 1
    k = h * (h - 1) // 2
    return float(c * np.sort(diffs)[k - 1])


def robust_iterative_prune(
    x: np.ndarray,
    scale_fn: Callable[[np.ndarray], float],
    c: float,
    direction: str = "both",
    max_iter: int = 50,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    remaining = np.arange(len(x), dtype=int)
    for _ in range(max_iter):
        vals = x[remaining]
        med = np.median(vals)
        scale = max(float(scale_fn(vals)), 1e-12)
        low, high = med - c * scale, med + c * scale
        if direction == "both":
            keep = (vals >= low) & (vals <= high)
        elif direction == "low":
            keep = vals >= low
        elif direction == "high":
            keep = vals <= high
        else:
            raise ValueError(direction)
        if keep.all():
            break
        new_remaining = remaining[keep]
        if len(new_remaining) == len(remaining):
            break
        remaining = new_remaining
        if len(remaining) <= 2:
            break
    return np.sort(np.setdiff1d(np.arange(len(x)), remaining))


def detector_flags(x: np.ndarray, name: str, alpha: float, n_sim: int, mad_c: float, qn_constant: float) -> np.ndarray:
    if name == "MLG+TM":
        return modified_largest_gap_tm(x, alpha, n_sim)
    if name == "MAD":
        return robust_iterative_prune(x, mad_scale, mad_c, direction="both")
    if name == "Sn":
        return robust_iterative_prune(x, sn_estimator, mad_c, direction="both")
    if name == "Qn":
        return robust_iterative_prune(x, lambda z: qn_estimator(z, c=qn_constant), mad_c, direction="both")
    raise ValueError(name)


def fusion_statistic(x: np.ndarray, flagged: np.ndarray, fusion: str = "mean") -> float:
    keep = np.ones(len(x), dtype=bool)
    flagged = np.asarray(flagged, dtype=int)
    if flagged.size:
        keep[flagged] = False
    if not keep.any():
        keep[:] = True
    vals = x[keep]
    return float(np.mean(vals) if fusion == "mean" else np.median(vals))
