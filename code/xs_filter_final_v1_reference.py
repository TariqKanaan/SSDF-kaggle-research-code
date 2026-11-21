"""XS: cross-session persistent-residual filter.

Idea (interpretable). Within a session the reporting gain g_i, sensing gain h_i and noise offset are
frozen, so a user's standardized residual is almost constant from round to round and extra rounds inside
one session add little evidence. Across sessions those honest nuisance factors are redrawn, whereas a
malicious user keeps falsifying. So evidence should be accumulated ACROSS blocks, not within a block.

  d_i(t)   : leave-one-out robust residual (identical to TTF-CF Eq. 8)
  block    : maximal run of rounds between detected boundaries (Page-Hinkley on the residual field)
  m_ik     : mean of d_i over block k (blocks with >= min_block rounds)
  S_i      : (1/B) sum_k m_ik        score_i = |S_i| * sqrt(B)          (B = number of blocks)
  p_i      : conformal p-value of score_i against an H0 bank bucketed by min(B, b_cap)
  flags    : Benjamini-Hochberg at q, capped at floor(rho_max*N); ties broken by larger score (NOT index)
  fusion   : mean of unflagged reports

No coalition stage, no CUSUM. Assumption stated openly: the malicious set is fixed across blocks.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from ttf_cf_final import ConformalBank, benjamini_hochberg
from xs_config import XSConfig


def loo_residual(x: np.ndarray) -> np.ndarray:
    N = len(x); d = np.empty(N)
    for i in range(N):
        others = np.delete(x, i)
        med = np.median(others)
        mad = np.median(np.abs(others - med))
        d[i] = (x[i] - med) / max(1.4826 * mad, 1e-6)
    return d


class _PHSegmenter:
    """Same Page-Hinkley recursion as ttf_cf_final.TTFCF._ph_step, used only to place block boundaries."""
    def __init__(self, cfg: XSConfig):
        self.cfg = cfg; self.reset()
    def reset(self):
        self.prev_d = None; self.mu = None; self.sd = 0.25; self.M = 0.0; self.m = 0.0; self.count = 0
    def step(self, d: np.ndarray) -> bool:
        if self.prev_d is None:
            self.prev_d = d.copy(); return False
        a = float(np.mean(np.abs(d - self.prev_d)))
        if self.mu is None: self.mu = a
        z = (a - self.mu) / max(self.sd, 0.05)
        self.m += z - self.cfg.ph_delta
        self.M = min(self.M, self.m)
        self.count += 1
        alarm = self.count >= self.cfg.ph_warmup and (self.m - self.M) > self.cfg.ph_threshold
        if not alarm:
            self.mu = (1 - self.cfg.ph_beta) * self.mu + self.cfg.ph_beta * a
            self.sd = (1 - self.cfg.ph_beta) * self.sd + self.cfg.ph_beta * abs(a - self.mu)
        self.prev_d = d.copy()
        return alarm


class XSFilter:
    def __init__(self, N: int, cfg: XSConfig, bank: Optional[ConformalBank]):
        self.N = int(N); self.cfg = cfg; self.bank = bank
        self.seg = _PHSegmenter(cfg)
        self.closed: list[np.ndarray] = []
        self.cur_sum = np.zeros(self.N); self.cur_n = 0

    def _close_block(self):
        if self.cur_n >= self.cfg.min_block:
            self.closed.append(self.cur_sum / self.cur_n)
        self.cur_sum = np.zeros(self.N); self.cur_n = 0

    def _scores(self):
        blocks = list(self.closed)
        if self.cur_n >= self.cfg.min_block:
            blocks.append(self.cur_sum / self.cur_n)
        B = len(blocks)
        if B == 0:
            return B, np.zeros(self.N)
        S = np.mean(np.stack(blocks), axis=0)
        return B, np.abs(S) * np.sqrt(B)

    def step(self, x: np.ndarray) -> dict:
        x = np.asarray(x, float)
        d = loo_residual(x)
        boundary = False
        if self.cfg.block_mode == "ph":
            boundary = self.seg.step(d)
            if boundary:
                self.seg.reset()
        else:  # fixed length
            boundary = self.cur_n >= self.cfg.fixed_len
        if boundary:
            self._close_block()
        self.cur_sum += d; self.cur_n += 1

        B, score = self._scores()
        age = min(B, self.cfg.b_cap)
        if B == 0 or self.bank is None:
            p = np.ones(self.N)
        else:
            p = self.bank.pvalues(age, score)
        F = benjamini_hochberg(p, self.cfg.q)
        Kmax = int(np.floor(self.cfg.rho_max * self.N))
        if F.size > Kmax:
            order = np.lexsort((-score[F], p[F]))[:Kmax]   # smallest p first; larger score breaks ties
            F = np.sort(F[order])
        keep = np.ones(self.N, bool)
        if F.size: keep[F] = False
        if not keep.any(): keep[:] = True
        stat = float(np.mean(x[keep]) if self.cfg.fusion == "mean" else np.median(x[keep]))
        return {"flagged": F.astype(int), "stat": stat, "pvalues": p, "score": score,
                "blocks": B, "boundary": int(boundary)}


def build_xs_bank(c, snr_db: float, cfg: XSConfig, seeds, make_session, rng_fn) -> ConformalBank:
    """H0/no-attack score bank, built with the same machinery. Uses xs_score_cal seeds only."""
    import system_model_final as sm
    bank = ConformalBank(age_cap=cfg.b_cap)
    for seed in seeds:
        rr = rng_fn(seed, 601_000 + int(round((snr_db + 50) * 10)))
        f = XSFilter(c.N, cfg, bank=None)
        for t in range(c.SCORE_CAL_ROUNDS):
            if t % c.SESSION_LENGTH == 0:
                h, g, unc = make_session(c, rr)
            T = sm.local_energy_gamma(c.N, c.M_SAMPLES, snr_db, "H0", h, unc, rr)
            eta = rr.normal(0.0, c.REPORT_NOISE_STD, size=c.N)
            z = f.step(sm.reporting_channel(T, g, eta))
            B = z["blocks"]
            if B >= 1:
                bank.add(min(B, cfg.b_cap), z["score"])
    return bank.finalize()
