from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class TTFConfig:
    lam: float = 0.20
    cusum_k: float = 0.50
    W: int = 20
    kappa_sign: float = 0.80
    kappa_p: float = 5.0
    q: float = 0.05
    rho_max: float = 0.40
    ph_delta: float = 0.05
    ph_threshold: float = 5.0
    ph_beta: float = 0.05
    ph_warmup: int = 5
    fusion: str = "mean"
    use_coalition: bool = True
    domain: str = "linear"
    age_cap: int = 39


class ConformalBank:
    def __init__(self, age_cap: int = 39):
        self.age_cap = int(age_cap)
        self._raw = [[] for _ in range(self.age_cap + 1)]
        self._sorted = None

    def add(self, age: int, scores: np.ndarray) -> None:
        a = min(max(int(age), 0), self.age_cap)
        self._raw[a].extend(np.asarray(scores, float).tolist())

    def finalize(self) -> "ConformalBank":
        self._sorted = [np.sort(np.asarray(v, float)) for v in self._raw]
        return self

    def pvalues(self, age: int, s: np.ndarray) -> np.ndarray:
        if self._sorted is None:
            raise RuntimeError("ConformalBank must be finalized")
        arr = self._sorted[min(max(int(age), 0), self.age_cap)]
        n = len(arr)
        if n == 0:
            return np.ones_like(s, dtype=float)
        ge = n - np.searchsorted(arr, s - 1e-12, side="left")
        return (1.0 + ge) / (n + 1.0)

    def sizes(self) -> list[int]:
        return [len(v) for v in self._raw]


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    m = len(p)
    if m == 0:
        return np.empty(0, dtype=int)
    order = np.argsort(p, kind="mergesort")
    thresh = q * np.arange(1, m + 1) / m
    good = np.where(p[order] <= thresh)[0]
    if good.size == 0:
        return np.empty(0, dtype=int)
    return np.sort(order[: good.max() + 1]).astype(int)


class TTFCF:
    def __init__(self, N: int, cfg: TTFConfig, bank: Optional[ConformalBank]):
        self.N = int(N)
        self.cfg = cfg
        self.bank = bank
        self.e = np.zeros(self.N)
        self.s = np.zeros(self.N)
        self.sign_history: list[np.ndarray] = []
        self.age = 0
        self.prev_d: Optional[np.ndarray] = None
        self.ph_mu: Optional[float] = None
        self.ph_sd = 0.25
        self.ph_M = 0.0
        self.ph_m = 0.0
        self.ph_count = 0
        self.reset_count = 0

    def _residual(self, x: np.ndarray) -> np.ndarray:
        z = np.log10(np.maximum(x, 1e-6)) if self.cfg.domain == "log" else x
        d = np.empty(self.N)
        for i in range(self.N):
            others = np.delete(z, i)
            med = np.median(others)
            mad = np.median(np.abs(others - med))
            scale = max(1.4826 * mad, 1e-6)
            d[i] = (z[i] - med) / scale
        return d

    def _coalition_flags(self, F_bh: np.ndarray, p: np.ndarray) -> np.ndarray:
        if not F_bh.size or len(self.sign_history) < min(self.cfg.W, 5):
            return F_bh
        pers = np.abs(np.mean(np.stack(self.sign_history[-self.cfg.W:]), axis=0))
        anchor = np.sign(np.sum(self.e[F_bh]))
        if anchor == 0:
            return F_bh
        cand = np.ones(self.N, dtype=bool)
        cand[F_bh] = False
        cand &= pers >= self.cfg.kappa_sign
        cand &= p <= min(1.0, self.cfg.kappa_p * self.cfg.q)
        cand &= np.sign(self.e) == anchor
        return np.sort(np.union1d(F_bh, np.where(cand)[0]).astype(int))

    def _flags(self, p: np.ndarray) -> np.ndarray:
        F_bh = benjamini_hochberg(p, self.cfg.q)
        F = self._coalition_flags(F_bh, p) if self.cfg.use_coalition else F_bh
        Kmax = int(np.floor(self.cfg.rho_max * self.N))
        if F.size > Kmax:
            # Keep the most extreme evidence first: smallest p-values.
            keep_order = np.argsort(p[F], kind="mergesort")[:Kmax]
            F = np.sort(F[keep_order])
        return F.astype(int)

    def _fuse(self, x: np.ndarray, flagged: np.ndarray) -> float:
        keep = np.ones(self.N, dtype=bool)
        if flagged.size:
            keep[flagged] = False
        if not keep.any():
            keep[:] = True
        vals = x[keep]
        return float(np.mean(vals) if self.cfg.fusion == "mean" else np.median(vals))

    def _ph_step(self, d: np.ndarray) -> bool:
        if self.prev_d is None:
            self.prev_d = d.copy()
            return False
        a_t = float(np.mean(np.abs(d - self.prev_d)))
        if self.ph_mu is None:
            self.ph_mu = a_t
        z = (a_t - self.ph_mu) / max(self.ph_sd, 0.05)
        self.ph_m += z - self.cfg.ph_delta
        self.ph_M = min(self.ph_M, self.ph_m)
        self.ph_count += 1
        alarm = (
            self.ph_count >= self.cfg.ph_warmup
            and (self.ph_m - self.ph_M) > self.cfg.ph_threshold
        )
        if not alarm:
            self.ph_mu = (1 - self.cfg.ph_beta) * self.ph_mu + self.cfg.ph_beta * a_t
            self.ph_sd = (1 - self.cfg.ph_beta) * self.ph_sd + self.cfg.ph_beta * abs(a_t - self.ph_mu)
        self.prev_d = d.copy()
        return alarm

    def reset(self) -> None:
        self.e.fill(0.0)
        self.s.fill(0.0)
        self.sign_history = []
        self.age = 0
        self.prev_d = None
        self.ph_mu = None
        self.ph_sd = 0.25
        self.ph_M = 0.0
        self.ph_m = 0.0
        self.ph_count = 0
        self.reset_count += 1

    def step(self, x: np.ndarray) -> dict:
        x = np.asarray(x, float)
        if x.shape != (self.N,):
            raise ValueError(f"expected {self.N} reports, got {x.shape}")
        d = self._residual(x)
        self.e = self.cfg.lam * d + (1 - self.cfg.lam) * self.e
        self.s = np.maximum(0.0, self.s + np.abs(d) - self.cfg.cusum_k)
        self.age += 1
        self.sign_history.append(np.sign(d))
        self.sign_history = self.sign_history[-self.cfg.W:]
        p = self.bank.pvalues(self.age, self.s) if self.bank is not None else np.ones(self.N)
        flags = self._flags(p)
        stat = self._fuse(x, flags)
        alarm = self._ph_step(d)
        out = {
            "flagged": flags,
            "stat": stat,
            "pvalues": p,
            "residual": d,
            "ewma": self.e.copy(),
            "cusum": self.s.copy(),
            "age": min(self.age, self.cfg.age_cap),
            "ph_alarm": int(alarm),
            "reset_count": self.reset_count,
        }
        if alarm:
            self.reset()
        return out
