"""Instrumented XS filter for the V2 reviewer-response experiments.

The numerical XS calculations intentionally preserve the V1 recursion and decision
logic. Additional return fields are telemetry only. In particular, the PH reset
behavior is unchanged so that instrumentation does not silently alter V1 results.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from ttf_cf_final import ConformalBank, benjamini_hochberg
from xs_config import XSConfig


def loo_residual(x: np.ndarray) -> np.ndarray:
    """Leave-one-out median/MAD residuals.

    For the even-N setting used in this study (N=20), compute every leave-one-out
    median and MAD with vectorized order statistics. This is algebraically
    equivalent to the original per-user loop but avoids N separate median calls.
    For odd N, retain the simple reference implementation.
    """
    x = np.asarray(x, float)
    N = len(x)
    if N < 3 or N % 2 == 1:
        d = np.empty(N, dtype=float)
        for i in range(N):
            others = np.delete(x, i)
            med = np.median(others)
            mad = np.median(np.abs(others - med))
            d[i] = (x[i] - med) / max(1.4826 * mad, 1e-6)
        return d

    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    rank = np.empty(N, dtype=int)
    rank[order] = np.arange(N)
    # N-1 is odd. Its middle order statistic is at index (N-2)//2.
    mid = (N - 2) // 2
    med = np.where(rank <= mid, sorted_x[mid + 1], sorted_x[mid])

    # Row i contains |x_j - median(x excluding i)|. Exclude the diagonal,
    # then the middle of the remaining N-1 deviations is the LOO MAD.
    deviations = np.abs(x[np.newaxis, :] - med[:, np.newaxis])
    np.fill_diagonal(deviations, np.inf)
    mad = np.sort(deviations, axis=1)[:, mid]
    return (x - med) / np.maximum(1.4826 * mad, 1e-6)


class _PHSegmenter:
    """Page-Hinkley recursion retained from the V1 implementation."""
    def __init__(self, cfg: XSConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.prev_d = None
        self.mu = None
        self.sd = 0.25
        self.M = 0.0
        self.m = 0.0
        self.count = 0

    def step(self, d: np.ndarray) -> bool:
        if self.prev_d is None:
            self.prev_d = d.copy()
            return False
        a = float(np.mean(np.abs(d - self.prev_d)))
        if self.mu is None:
            self.mu = a
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
    """XS with explicit per-round and block-boundary telemetry."""
    def __init__(self, N: int, cfg: XSConfig, bank: Optional[ConformalBank]):
        self.N = int(N)
        self.cfg = cfg
        self.bank = bank
        self.seg = _PHSegmenter(cfg)
        self.closed: list[np.ndarray] = []
        self.closed_meta: list[dict] = []
        self.cur_sum = np.zeros(self.N)
        self.cur_n = 0
        self.round_index = 0
        self.block_index = 0
        self.block_start_round = 0

    def _close_block(self, end_round: int):
        meta = {
            "block_index": int(self.block_index),
            "start_round": int(self.block_start_round),
            "end_round": int(end_round),
            "length": int(self.cur_n),
            "qualified": bool(self.cur_n >= self.cfg.min_block),
        }
        if self.cur_n >= self.cfg.min_block:
            self.closed.append(self.cur_sum / self.cur_n)
        self.closed_meta.append(meta)
        self.cur_sum = np.zeros(self.N)
        self.cur_n = 0

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
        if x.shape != (self.N,):
            raise ValueError(f"expected {self.N} reports, got {x.shape}")
        t = self.round_index
        d = loo_residual(x)
        boundary = False
        reason = "none"
        old_block_length = 0
        old_block_start = self.block_start_round

        if self.cfg.block_mode == "ph":
            boundary = self.seg.step(d)
            if boundary:
                reason = "page_hinkley"
                old_block_length = int(self.cur_n)
                self._close_block(end_round=t - 1)
                self.block_index += 1
                self.block_start_round = t
                # Preserve the V1 behavior: the PH monitor resets after an alarm.
                self.seg.reset()
        else:
            boundary = self.cur_n >= self.cfg.fixed_len
            if boundary:
                reason = "fixed_length"
                old_block_length = int(self.cur_n)
                self._close_block(end_round=t - 1)
                self.block_index += 1
                self.block_start_round = t

        self.cur_sum += d
        self.cur_n += 1
        B, score = self._scores()
        age = min(B, self.cfg.b_cap)
        if B == 0 or self.bank is None:
            p = np.ones(self.N)
        else:
            p = self.bank.pvalues(age, score)
        F_bh = benjamini_hochberg(p, self.cfg.q)
        F = F_bh.copy()
        Kmax = int(np.floor(self.cfg.rho_max * self.N))
        if F.size > Kmax:
            order = np.lexsort((-score[F], p[F]))[:Kmax]
            F = np.sort(F[order])
        keep = np.ones(self.N, bool)
        if F.size:
            keep[F] = False
        if not keep.any():
            keep[:] = True
        stat = float(np.mean(x[keep]) if self.cfg.fusion == "mean" else np.median(x[keep]))

        out = {
            "flagged": F.astype(int),
            "bh_flagged": F_bh.astype(int),
            "stat": stat,
            "pvalues": p.copy(),
            "score": score.copy(),
            "residual": d.copy(),
            "blocks": int(B),
            "boundary": int(boundary),
            "boundary_reason": reason,
            "closed_block_index": int(self.block_index - 1) if boundary else -1,
            "closed_block_start_round": int(old_block_start) if boundary else -1,
            "closed_block_end_round": int(t - 1) if boundary else -1,
            "closed_block_length": int(old_block_length) if boundary else 0,
            "block_index": int(self.block_index),
            "block_start_round": int(self.block_start_round),
            "block_length_so_far": int(self.cur_n),
            "round_index": int(t),
            "n_closed_blocks": int(len(self.closed_meta)),
        }
        self.round_index += 1
        return out

    def finalize_blocks(self) -> list[dict]:
        """Return metadata for closed blocks plus the final open block (without changing scores)."""
        out = list(self.closed_meta)
        if self.cur_n:
            out.append({
                "block_index": int(self.block_index),
                "start_round": int(self.block_start_round),
                "end_round": int(self.round_index - 1),
                "length": int(self.cur_n),
                "qualified": bool(self.cur_n >= self.cfg.min_block),
                "final_open_block": True,
            })
        return out


def build_xs_bank_from_sequence(sequence, cfg: XSConfig) -> ConformalBank:
    """Build an XS H0 bank from already-generated clean H0 trajectories.

    Each trajectory is processed with a fresh XSFilter. Calibration must use an
    independent seed pool and a session schedule matching the evaluation cell.
    """
    bank = ConformalBank(age_cap=cfg.b_cap)
    for seq in sequence:
        f = XSFilter(len(seq[0]), cfg, bank=None)
        for x in seq:
            z = f.step(x)
            if z["blocks"] >= 1:
                bank.add(min(z["blocks"], cfg.b_cap), z["score"])
    return bank.finalize()
