"""Instrumented XS B1-V3 experiment runner with development/calibration/test separation.

This is a new V2 runner. It does not overwrite or modify V1 files/results.

Key outputs (CSV.GZ):
  v2_seed_level.csv.gz             one row per seed/condition/pipeline
  v2_pipeline_round.csv.gz         round-level fused statistics and decisions
  v2_user_observations.csv.gz      raw reports, identities, subgroups and ranks
  v2_user_decisions.csv.gz         residuals, p-values, BH flags and exclusions
  v2_session_boundaries.csv.gz     simulator ground-truth session boundaries
  v2_block_boundaries.csv.gz       detected XS/TTF block boundaries
  v2_thresholds.csv.gz             calibration thresholds, including matched Oracle
  v2_calibration_scores.csv.gz     calibration observations used to set thresholds
  v2_manifest.json, v2_seed_ledger.json

Example smoke test:
  python run_xs_study_v2.py --out v2_smoke --snrs=-9 --rhos=0.2 --attacks=masking \
      --session-specs='fixed:20' --extreme-fractions=0.5 --caps=8 --fixed-blocks=20 \
      --test-count=1 --score-cal-count=2 --tau-cal-count=2 --rounds=30 \
      --pipelines='NoDefense,XS-PH,XS-fixed20,Huber,BetaHistory,OracleRemoval'

Example B1:
  python run_xs_study_v2.py --out v2_B1 --snrs=-9,-6 --rhos=0.2 --attacks=masking \
      --session-specs='fixed:13;fixed:20;fixed:30;fixed:50;random:13,20,30,50' \
      --extreme-fractions=0.5 --caps=8 --fixed-blocks=10,13,20,30,50 --test-count=400

Example B2:
  python run_xs_study_v2.py --out v2_B2 --snrs=-9,-6 --rhos=0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 \
      --attacks=masking --session-specs='fixed:20' --extreme-fractions=0,0.25,0.5,0.75,1 \
      --caps=4,8,12 --fixed-blocks=20 --test-count=100

Notes:
- Fixed:20 preserves the V1 generator's random-stream ordering. Other session modes are new V2 conditions.
- BetaHistory is an explicitly documented study-adapted reputation/history baseline, not a claim of exact reproduction of [7].
- OracleRemoval is calibrated on clean H0 after removing a fixed random subset of size L for each rho.
- All report-level records are telemetry; they do not change the evaluated detector decisions.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, trim_mean
from sklearn.metrics import roc_auc_score

import run_final_experiment as R
import system_model_final as sm
from config_final import Config, cfg_from_env
from paradigm_a_final import ParadigmA
from ttf_cf_final import ConformalBank, TTFCF, TTFConfig
from xs_config import XSConfig
from xs_experiment_v2 import generate_paired_condition_v2
from xs_filter_v2 import XSFilter, loo_residual


DEFAULT_PIPELINES = [
    "NoDefense", "MLG+TM", "MAD", "Sn", "Qn", "ParadigmA", "TTF-CF",
    "Trim10", "Trim20", "Trim40", "Median", "Huber", "BetaHistory", "BDFHistory", "OracleRemoval",
]
SUPPORTED_PIPELINES = set(DEFAULT_PIPELINES + ["XS-PH"])


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_semicolon_list(s: str) -> list[str]:
    vals = [x.strip() for x in s.split(";") if x.strip()]
    return vals or ["fixed:20"]


def safe_name(value: str) -> str:
    return value.replace(":", "_").replace(",", "-").replace("/", "-").replace(" ", "")


def append_gzip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, index=False, mode="wt" if header else "at", header=header,
              compression="gzip", encoding="utf-8")


def ensure_empty_output(out: Path, overwrite: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "v2_manifest.json"
    if marker.exists() and not overwrite:
        raise SystemExit(f"Output directory already contains {marker.name}; choose a new --out or pass --overwrite.")
    if overwrite:
        for pattern in ("v2_*", "v3_*"):
            for p in out.glob(pattern):
                if p.is_file():
                    p.unlink()


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    kind: str
    xs_cfg: XSConfig | None = None
    base_name: str | None = None


def make_pipeline_specs(pipeline_names: list[str], fixed_blocks: list[int], caps: list[int], N: int,
                        ph_delta: float = 0.05, ph_threshold: float = 5.0,
                        ph_beta: float = 0.05, ph_warmup: int = 5,
                        min_block: int = 5) -> list[PipelineSpec]:
    out: list[PipelineSpec] = []
    multiple_caps = len(caps) > 1
    for base in pipeline_names:
        if base == "XS-PH":
            for cap in caps:
                name = f"XS-PH_cap{cap}" if multiple_caps else "XS-PH"
                cfg = XSConfig(block_mode="ph", rho_max=float(cap) / N,
                               ph_delta=float(ph_delta), ph_threshold=float(ph_threshold),
                               ph_beta=float(ph_beta), ph_warmup=int(ph_warmup),
                               min_block=int(min_block))
                out.append(PipelineSpec(name, "xs", cfg, "XS-PH"))
        elif base == "XS-fixed":
            for k in fixed_blocks:
                for cap in caps:
                    name = f"XS-fixed{k}_cap{cap}" if multiple_caps else f"XS-fixed{k}"
                    cfg = XSConfig(block_mode="fixed", fixed_len=k, rho_max=float(cap) / N,
                                   min_block=int(min_block))
                    out.append(PipelineSpec(name, "xs", cfg, f"XS-fixed{k}"))
        elif base == "XS-fixed20":
            for cap in caps:
                name = f"XS-fixed20_cap{cap}" if multiple_caps else "XS-fixed20"
                cfg = XSConfig(block_mode="fixed", fixed_len=20, rho_max=float(cap) / N,
                               min_block=int(min_block))
                out.append(PipelineSpec(name, "xs", cfg, "XS-fixed20"))
        elif base in {"XS-PH", "XS-fixed20"}:
            continue
        elif base in SUPPORTED_PIPELINES:
            out.append(PipelineSpec(base, base.lower(), None, base))
        else:
            raise ValueError(f"unsupported pipeline {base!r}")
    names = [p.name for p in out]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate pipeline names generated: {names}")
    return out


def huber_location(x: np.ndarray, c_huber: float = 1.345, iterations: int = 30) -> tuple[float, np.ndarray]:
    """Huber M-estimator of location, with report-specific IRLS weights."""
    x = np.asarray(x, float)
    mu = float(np.median(x))
    scale = max(1.4826 * float(np.median(np.abs(x - mu))), 1e-9)
    weights = np.ones_like(x)
    for _ in range(iterations):
        u = (x - mu) / scale
        abs_u = np.abs(u)
        weights = np.ones_like(x)
        mask = abs_u > c_huber
        weights[mask] = c_huber / np.maximum(abs_u[mask], 1e-12)
        new_mu = float(np.sum(weights * x) / max(np.sum(weights), 1e-12))
        if abs(new_mu - mu) <= 1e-10 * max(1.0, abs(mu)):
            mu = new_mu
            break
        mu = new_mu
    return mu, weights


class BetaHistoryState:
    """A transparent Beta-Bernoulli history baseline (adapted study implementation).

    Per-round evidence is 1 when |LOO residual| > evidence_z. Each user's anomaly
    probability is the posterior mean of Beta(alpha,beta). Flags are selected by
    posterior evidence and capped at cap reports. This is not presented as a
    faithful reproduction of any single cited paper; exact reference replication
    should replace this adapter if the cited algorithm's full specification differs.
    """
    def __init__(self, N: int, cap: int, alpha0: float = 1.0, beta0: float = 19.0,
                 evidence_z: float = 3.0, posterior_cut: float = 0.20):
        self.alpha = np.full(N, alpha0, dtype=float)
        self.beta = np.full(N, beta0, dtype=float)
        self.cap = int(cap)
        self.evidence_z = float(evidence_z)
        self.posterior_cut = float(posterior_cut)

    def step(self, x: np.ndarray) -> dict:
        d = loo_residual(x)
        evidence = np.abs(d) > self.evidence_z
        self.alpha += evidence.astype(float)
        self.beta += (~evidence).astype(float)
        reputation = self.alpha / (self.alpha + self.beta)
        candidates = np.where(reputation >= self.posterior_cut)[0]
        if len(candidates) > self.cap:
            candidates = candidates[np.argsort(reputation[candidates], kind="mergesort")[-self.cap:]]
        return {"flagged": np.sort(candidates.astype(int)), "residual": d,
                "reputation": reputation.copy(), "evidence": evidence.astype(int)}


class BDFWindowHistoryState:
    """Windowed Beta peer-consistency baseline inspired by the reputation component of Wu et al. (2024).

    Each round, a user's leave-one-out median/MAD residual is converted to a
    peer-consistency indicator (1 if |residual| <= evidence_z, else 0). A Beta
    posterior estimates the user's recent reliability over a rolling window.
    Users below posterior_cut are excluded, up to a fixed cap.

    This is an explicit adaptation to continuous energy reports and the present
    per-round fusion task. It is NOT a faithful reproduction of the paper's full
    data-delivery, window-weighted positive/negative evaluation and sequential
    likelihood-ratio procedure. The distinction is recorded in the manifest and
    must be preserved in the manuscript.
    """
    def __init__(self, N: int, cap: int, window: int = 20, alpha0: float = 1.0,
                 beta0: float = 1.0, evidence_z: float = 3.0,
                 posterior_cut: float = 0.65):
        from collections import deque
        self.N = int(N)
        self.cap = int(cap)
        self.window = int(window)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.evidence_z = float(evidence_z)
        self.posterior_cut = float(posterior_cut)
        self.history = deque(maxlen=self.window)
        self.positive = np.zeros(self.N, dtype=float)

    def step(self, x: np.ndarray) -> dict:
        d = loo_residual(x)
        evidence = (np.abs(d) <= self.evidence_z).astype(float)
        if len(self.history) == self.window:
            oldest = self.history[0]
            self.positive -= oldest
        self.history.append(evidence.copy())
        self.positive += evidence
        n_obs = float(len(self.history))
        negative = n_obs - self.positive
        reliability = (self.alpha0 + self.positive) / (
            self.alpha0 + self.beta0 + self.positive + negative)
        candidates = np.where(reliability < self.posterior_cut)[0]
        if len(candidates) > self.cap:
            candidates = candidates[np.argsort(reliability[candidates], kind="mergesort")[:self.cap]]
        return {"flagged": np.sort(candidates.astype(int)), "residual": d,
                "reputation": reliability.copy(), "anomaly_score": (1.0 - reliability).copy(),
                "evidence": evidence.astype(int), "window_observations": int(n_obs)}


def pipeline_state(spec: PipelineSpec, c: Config, ctx: dict, oracle_indices: np.ndarray | None = None):
    if spec.kind == "xs":
        return XSFilter(c.N, spec.xs_cfg, ctx["xs_banks"][spec.base_name])
    if spec.name == "TTF-CF":
        return TTFCF(c.N, TTFConfig(
            lam=c.TTF_LAMBDA, cusum_k=c.TTF_CUSUM_K, W=c.TTF_W,
            kappa_sign=c.TTF_KAPPA_SIGN, kappa_p=c.TTF_KAPPA_P,
            q=c.TTF_Q, rho_max=c.TTF_RHO_MAX,
            ph_delta=c.TTF_PH_DELTA, ph_threshold=c.TTF_PH_THRESHOLD,
            ph_beta=c.TTF_PH_BETA, ph_warmup=c.TTF_PH_WARMUP,
            fusion="mean", use_coalition=True, domain="linear"), ctx.get("ttf_bank"))
    if spec.name == "BetaHistory":
        cap = int(math.floor(c.TTF_RHO_MAX * c.N))
        return BetaHistoryState(c.N, cap)
    if spec.name == "BDFHistory":
        cap = int(math.floor(c.TTF_RHO_MAX * c.N))
        return BDFWindowHistoryState(c.N, cap, window=20, alpha0=1.0, beta0=1.0,
                                     evidence_z=3.0, posterior_cut=0.65)
    return None


def _trim_flags(x: np.ndarray, proportion: float) -> np.ndarray:
    k = int(proportion * len(x))
    if k <= 0:
        return np.empty(0, dtype=int)
    order = np.argsort(np.asarray(x), kind="mergesort")
    return np.sort(np.concatenate([order[:k], order[-k:]]).astype(int))


def evaluate_sequence(c: Config, spec: PipelineSpec, sequence: list[np.ndarray], malicious_idx: np.ndarray,
                      ctx: dict, tau: float, beta_cap: int | None = None,
                      oracle_indices: np.ndarray | None = None, collect_telemetry: bool = True,
                      collect_user_telemetry: bool = True) -> tuple[list[dict], list[dict], list[dict]]:
    """Run one pipeline on one sequence and materialize per-round and per-user records."""
    true_set = set(np.asarray(malicious_idx, int).tolist())
    n_honest = max(c.N - len(true_set), 1)
    state = pipeline_state(spec, c, ctx)
    round_rows: list[dict] = []
    user_rows: list[dict] = []
    block_rows: list[dict] = []

    for t, x0 in enumerate(sequence):
        x = np.asarray(x0, float)
        residual = loo_residual(x)
        residual_definition = "diagnostic_LOO_median_MAD"
        pvalue_source = "not_applicable"
        score_definition = "not_applicable"
        pvalues = np.full(c.N, np.nan)
        scores = np.full(c.N, np.nan)
        bh_flagged = np.empty(0, dtype=int)
        robust_weights = np.full(c.N, np.nan)
        auxiliary = {}
        block_index = -1
        block_start = -1
        block_length = 0
        block_boundary = 0
        boundary_reason = "none"
        boundary_closed_start = -1
        boundary_closed_end = -1
        boundary_closed_length = 0

        if spec.kind == "xs":
            z = state.step(x)
            flags = np.asarray(z["flagged"], int)
            bh_flagged = np.asarray(z["bh_flagged"], int)
            residual = np.asarray(z["residual"], float)
            residual_definition = "XS_LOO_median_MAD"
            pvalue_source = "XS_empirical_conformal_bank_bucketed_by_block_count"
            score_definition = "abs(mean_block_residual)*sqrt(number_of_qualified_blocks)"
            pvalues = np.asarray(z["pvalues"], float)
            scores = np.asarray(z["score"], float)
            stat = float(z["stat"])
            block_index = int(z["block_index"])
            block_start = int(z["block_start_round"])
            block_length = int(z["block_length_so_far"])
            block_boundary = int(z["boundary"])
            boundary_reason = z["boundary_reason"]
            boundary_closed_start = int(z["closed_block_start_round"])
            boundary_closed_end = int(z["closed_block_end_round"])
            boundary_closed_length = int(z["closed_block_length"])
            auxiliary["blocks"] = int(z["blocks"])
            auxiliary["n_closed_blocks"] = int(z["n_closed_blocks"])
        elif spec.name == "TTF-CF":
            z = state.step(x)
            flags = np.asarray(z["flagged"], int)
            residual = np.asarray(z["residual"], float)
            residual_definition = "TTF_CF_internal_residual"
            pvalue_source = "TTF_CF_empirical_conformal_bank"
            score_definition = "TTF_CF_CUSUM"
            pvalues = np.asarray(z["pvalues"], float)
            scores = np.asarray(z["cusum"], float)
            stat = float(z["stat"])
            block_boundary = int(z["ph_alarm"])
            boundary_reason = "ttf_page_hinkley" if block_boundary else "none"
        elif spec.name == "NoDefense":
            flags = np.empty(0, dtype=int)
            stat = float(np.mean(x))
        elif spec.name == "OracleRemoval":
            # Oracle uses the known attacker identities even on the paired clean counterfactual;
            # the separate malicious_idx argument still controls FFR/Catch accounting.
            flags = np.asarray(malicious_idx if oracle_indices is None else oracle_indices, int)
            keep = np.ones(c.N, dtype=bool)
            keep[flags] = False
            if not keep.any():
                keep[:] = True
            stat = float(np.mean(x[keep]))
        elif spec.name in {"MLG+TM", "MAD", "Sn", "Qn"}:
            flags = sm.detector_flags(x, spec.name, c.TM_ALPHA, c.TM_N_SIM, c.MAD_C, c.QN_CONSTANT)
            stat = sm.fusion_statistic(x, flags, "mean")
        elif spec.name == "ParadigmA":
            flags = np.asarray(ctx["pa_model"].detect(x), int)
            stat = sm.fusion_statistic(x, flags, "mean")
        elif spec.name in {"Trim10", "Trim20", "Trim40"}:
            prop = int(spec.name[4:]) / 100.0
            flags = _trim_flags(x, prop)
            stat = float(trim_mean(x, prop))
        elif spec.name == "Median":
            flags = np.empty(0, dtype=int)
            stat = float(np.median(x))
        elif spec.name == "Huber":
            stat, robust_weights = huber_location(x)
            flags = np.empty(0, dtype=int)
        elif spec.name == "BetaHistory":
            z = state.step(x)
            flags = np.asarray(z["flagged"], int)
            residual = np.asarray(z["residual"], float)
            residual_definition = "diagnostic_LOO_median_MAD_for_BetaHistory_evidence"
            score_definition = "Beta_Bernoulli_posterior_mean_anomaly_rate"
            scores = np.asarray(z["reputation"], float)
            auxiliary["beta_evidence"] = np.asarray(z["evidence"], int)
            stat = sm.fusion_statistic(x, flags, "mean")
        elif spec.name == "BDFHistory":
            z = state.step(x)
            flags = np.asarray(z["flagged"], int)
            residual = np.asarray(z["residual"], float)
            residual_definition = "LOO_median_MAD_peer_consistency_for_BDF_inspired_history"
            pvalue_source = "not_applicable_Beta_window_posterior"
            score_definition = "one_minus_rolling_Beta_posterior_peer_reliability"
            scores = np.asarray(z["anomaly_score"], float)
            auxiliary["beta_peer_consistency"] = np.asarray(z["evidence"], int)
            auxiliary["history_window_observations"] = int(z["window_observations"])
            stat = sm.fusion_statistic(x, flags, "mean")
        else:
            raise ValueError(f"No evaluator for pipeline {spec.name}")

        flags = np.unique(np.asarray(flags, int))
        keep_mask = np.ones(c.N, dtype=bool)
        if flags.size:
            keep_mask[flags] = False
        if not keep_mask.any():
            keep_mask[:] = True
        decision = int(stat > float(tau))
        fs = set(flags.tolist())
        catch_all = len(fs & true_set) / len(true_set) if true_set else np.nan
        ffr_attack = len(fs - true_set) / n_honest
        if collect_telemetry:
            round_rows.append({
                "pipeline": spec.name, "round": int(t), "fused_statistic": stat,
                "decision": decision, "threshold": float(tau), "flagged_count": int(len(flags)),
                "bh_flagged_count": int(len(bh_flagged)), "catch": catch_all,
                "ffr_attack": float(ffr_attack), "block_boundary": int(block_boundary),
                "boundary_reason": boundary_reason, "block_index": block_index,
                "block_start_round": block_start, "block_length_so_far": block_length,
                "closed_block_start_round": boundary_closed_start,
                "closed_block_end_round": boundary_closed_end,
                "closed_block_length": boundary_closed_length,
                **auxiliary,
            })

            if collect_user_telemetry:
                rank_asc = rankdata(x, method="average")
                rank_desc = rankdata(-x, method="average")
                flagged_set = set(flags.tolist())
                bh_set = set(bh_flagged.tolist())
                for i in range(c.N):
                    user_rows.append({
                        "pipeline": spec.name, "round": int(t), "user_id": int(i),
                        "raw_report": float(x[i]), "rank_ascending": float(rank_asc[i]),
                        "rank_descending": float(rank_desc[i]),
                        "residual": float(residual[i]), "residual_definition": residual_definition,
                        "conformal_p_value_source": pvalue_source, "score_definition": score_definition,
                        "conformal_p_value": float(pvalues[i]) if np.isfinite(pvalues[i]) else np.nan,
                        "evidence_score": float(scores[i]) if np.isfinite(scores[i]) else np.nan,
                        "bh_flag": int(i in bh_set),
                        "excluded": int(i in flagged_set), "retained_for_fusion": int(bool(keep_mask[i])),
                        "robust_weight": float(robust_weights[i]) if np.isfinite(robust_weights[i]) else np.nan,
                    })

            if block_boundary and boundary_closed_start >= 0:
                block_rows.append({
                    "pipeline": spec.name, "block_index": int(block_index - 1),
                    "start_round": boundary_closed_start, "end_round": boundary_closed_end,
                    "length": boundary_closed_length, "boundary_round": int(t),
                    "boundary_reason": boundary_reason,
                    "qualified": bool(boundary_closed_length >= spec.xs_cfg.min_block)
                    if spec.kind == "xs" and spec.xs_cfg is not None else np.nan,
                })
        else:
            # Keep only the scalar statistic required for threshold calibration.
            # Avoid constructing per-user telemetry during calibration runs.
            round_rows.append({"pipeline": spec.name, "round": int(t),
                               "fused_statistic": float(stat)})

    if collect_telemetry and spec.kind == "xs":
        # IMPORTANT: finalize_blocks() returns metadata for ALL closed blocks
        # plus the terminal open block. Closed blocks have already been recorded
        # at the exact PH/fixed-length boundary above. Append ONLY the terminal
        # open block here; otherwise every closed block is duplicated and
        # mislabeled as final_open_block in the telemetry CSV.
        for b in state.finalize_blocks():
            if not b.get("final_open_block", False):
                continue
            block_rows.append({
                "pipeline": spec.name, "block_index": b["block_index"],
                "start_round": b["start_round"], "end_round": b["end_round"],
                "length": b["length"], "boundary_round": np.nan,
                "boundary_reason": "final_open_block", "qualified": b["qualified"],
            })
    return round_rows, user_rows, block_rows


def build_context(c: Config, snr: float, session_spec: str, persistence_w: float,
                  pipeline_specs: list[PipelineSpec], score_seeds: list[int], tau_seeds: list[int],
                  rounds: int, tau_quantile: float, rhos: list[float], condition_id: int) -> tuple[dict, dict, list[dict]]:
    """Build w=0 conformal banks and clean-H0 thresholds recalibrated at persistence_w."""
    ctx: dict[str, Any] = {"xs_banks": {}, "xs_bank_sizes": {}}
    need_pa = any(p.name == "ParadigmA" for p in pipeline_specs)
    need_ttf = any(p.name == "TTF-CF" for p in pipeline_specs)
    xs_specs = [p for p in pipeline_specs if p.kind == "xs"]

    if need_pa:
        ctx["pa_model"] = R.make_pa_model(c, snr)

    score_sequences = []
    for seed in score_seeds:
        pair = generate_paired_condition_v2(c, seed, 710_000 + condition_id, snr, 0.0, "none",
                                            c.SCORE_CAL_ROUNDS, session_spec=session_spec, persistence_w=0.0)
        score_sequences.append(pair["H0_clean"])

    for base_name in sorted({p.base_name for p in xs_specs}):
        representative = next(p for p in xs_specs if p.base_name == base_name)
        bank = ConformalBank(age_cap=representative.xs_cfg.b_cap)
        for seq in score_sequences:
            f = XSFilter(c.N, representative.xs_cfg, bank=None)
            for x in seq:
                z = f.step(x)
                if z["blocks"] >= 1:
                    bank.add(min(z["blocks"], representative.xs_cfg.b_cap), z["score"])
        ctx["xs_banks"][base_name] = bank.finalize()
        ctx["xs_bank_sizes"][base_name] = bank.sizes()

    if need_ttf:
        ttf_bank = ConformalBank(age_cap=39)
        cfg = TTFConfig(
            lam=c.TTF_LAMBDA, cusum_k=c.TTF_CUSUM_K, W=c.TTF_W,
            kappa_sign=c.TTF_KAPPA_SIGN, kappa_p=c.TTF_KAPPA_P,
            q=c.TTF_Q, rho_max=c.TTF_RHO_MAX,
            ph_delta=c.TTF_PH_DELTA, ph_threshold=c.TTF_PH_THRESHOLD,
            ph_beta=c.TTF_PH_BETA, ph_warmup=c.TTF_PH_WARMUP,
            fusion="mean", use_coalition=True, domain="linear")
        for seq in score_sequences:
            st = TTFCF(c.N, cfg, bank=None)
            for x in seq:
                z = st.step(x)
                ttf_bank.add(z["age"], z["cusum"])
        ctx["ttf_bank"] = ttf_bank.finalize()
        ctx["ttf_bank_sizes"] = ttf_bank.sizes()

    # Calibrate all ordinary pipelines at clean H0. Oracle is calibrated separately below
    # after removing the same number L of clean reports as the corresponding rho.
    ordinary = [p for p in pipeline_specs if p.name != "OracleRemoval"]
    calibration_values: dict[str, list[float]] = {p.name: [] for p in ordinary}
    cal_score_rows: list[dict] = []
    for seed in tau_seeds:
        pair = generate_paired_condition_v2(c, seed, 720_000 + condition_id, snr, 0.0, "none",
                                            c.TAU_CAL_ROUNDS, session_spec=session_spec,
                                            persistence_w=persistence_w)
        for p in ordinary:
            rr, _, _ = evaluate_sequence(c, p, pair["H0_clean"], np.empty(0, int), ctx,
                                         tau=math.inf, collect_telemetry=False)
            vals = [r["fused_statistic"] for r in rr]
            calibration_values[p.name].extend(vals)
            for t, value in enumerate(vals):
                cal_score_rows.append({"snr_db": snr, "session_spec": session_spec,
                                       "persistence_w": float(persistence_w),
                                       "calibration_seed": seed, "round": t,
                                       "pipeline": p.name, "rho_oracle_removed": np.nan,
                                       "calibration_statistic": float(value)})

    thresholds: dict[str, float] = {}
    threshold_rows: list[dict] = []
    for name, vals in calibration_values.items():
        tau = float(np.quantile(vals, tau_quantile))
        thresholds[name] = tau
        threshold_rows.append({"snr_db": snr, "session_spec": session_spec,
                               "persistence_w": float(persistence_w), "pipeline": name,
                               "rho_oracle_removed": np.nan, "quantile": tau_quantile,
                               "threshold": tau, "n_calibration_rounds": len(vals),
                               "n_calibration_seeds": len(tau_seeds),
                               "oracle_removed_reports": np.nan})

    return ctx, thresholds, threshold_rows + cal_score_rows


def condition_metadata(pair: dict, c: Config, snr: float, rho: float, attack: str,
                        seed: int, session_spec: str, extreme_fraction: float, cap_label: str) -> dict:
    return {
        "seed": int(seed), "snr_db": float(snr), "persistence_w": float(pair.get("persistence_w", 0.0)),
        "rho_requested": float(rho),
        "rho_actual": float(pair["rho_actual"]), "attack": attack,
        "extreme_fraction": float(extreme_fraction), "extreme_count": int(pair["extreme_count"]),
        "mild_count": int(pair["mild_count"]), "session_spec": session_spec,
        "cap_design": cap_label, "rounds": int(len(pair["H0_clean"])),
        "subgroup_assignment": pair["subgroup_assignment"],
    }


def make_observation_rows(pair: dict, meta: dict, sequence_type: str) -> list[dict]:
    seq = pair[sequence_type]
    session = pair["session"]
    attacked = sequence_type.endswith("_att") and pair["attack"] != "none"
    rows = []
    for t, x in enumerate(seq):
        ra = rankdata(np.asarray(x, float), method="average")
        rd = rankdata(-np.asarray(x, float), method="average")
        for i, value in enumerate(x):
            rows.append({**meta, "sequence_type": sequence_type, "round": int(t),
                         "session_id": int(session.session_id[t]),
                         "session_start": int(session.session_start[t]),
                         "session_end": int(session.session_end[t]),
                         "session_length": int(session.session_length[t]),
                         "user_id": int(i), "user_type": str(pair["user_type"][i]),
                         "attacker_subtype": str(pair["attacker_subtype"][i]),
                         "is_malicious_identity": int(pair["user_type"][i] == "malicious"),
                         "attack_applied_to_user": int(bool(attacked and pair["user_type"][i] == "malicious")),
                         "raw_report": float(value), "rank_ascending": float(ra[i]),
                         "rank_descending": float(rd[i])})
    return rows


def run_one_seed(c: Config, pair: dict, meta: dict, pipeline_specs: list[PipelineSpec],
                 ctx: dict, thresholds: dict, session_rows: list[dict],
                 collect_user_telemetry: bool = True) -> tuple[list, list, list, list, list, list]:
    seed_rows: list[dict] = []
    round_rows_out: list[dict] = []
    user_decision_rows: list[dict] = []
    block_rows_out: list[dict] = []
    displacement_rows: list[dict] = []
    observations: list[dict] = []
    sequences = ["H0_clean", "H0_att", "H1_clean", "H1_att"]
    if collect_user_telemetry:
        for seq_type in sequences:
            observations.extend(make_observation_rows(pair, meta, seq_type))

    mal = np.asarray(pair["malicious_idx"], int)
    extreme = set(np.asarray(pair["extreme_idx"], int).tolist())
    mild = set(np.asarray(pair["mild_idx"], int).tolist())
    for spec in pipeline_specs:
        oracle_rho_key = 0.0 if meta["attack"] == "none" else float(meta["rho_requested"])
        tau = thresholds[f"OracleRemoval|rho={oracle_rho_key:g}"] if spec.name == "OracleRemoval" else thresholds[spec.name]
        evaluated = {}
        user_rows_by_sequence = {}
        for seq_type in sequences:
            # Under the clean counterfactual, every excluded report is a false exclusion,
            # including a user who is designated malicious in the paired attack arm.
            true_for_seq = mal if seq_type.endswith("_att") and meta["attack"] != "none" else np.empty(0, int)
            oracle_indices = mal if (spec.name == "OracleRemoval" and meta["attack"] != "none") else np.empty(0, int)
            rr, ur, br = evaluate_sequence(c, spec, pair[seq_type], true_for_seq, ctx, tau,
                                           oracle_indices=oracle_indices,
                                           collect_user_telemetry=collect_user_telemetry)
            evaluated[seq_type] = rr
            user_rows_by_sequence[seq_type] = ur
            for row in rr:
                session = pair["session"]
                t = row["round"]
                round_rows_out.append({**meta, "pipeline": spec.name,
                                       "sequence_type": seq_type,
                                       "session_id": int(session.session_id[t]),
                                       "session_start": int(session.session_start[t]),
                                       "session_end": int(session.session_end[t]),
                                       "session_length": int(session.session_length[t]), **row})
            for row in ur:
                t = row["round"]
                i = row["user_id"]
                is_attack_seq = seq_type.endswith("_att") and meta["attack"] != "none"
                user_decision_rows.append({**meta, "pipeline": spec.name, "sequence_type": seq_type,
                    "session_id": int(session.session_id[t]), "round": int(t), "user_id": int(i),
                    "user_type": str(pair["user_type"][i]), "attacker_subtype": str(pair["attacker_subtype"][i]),
                    "attack_applied_to_user": int(bool(is_attack_seq and pair["user_type"][i] == "malicious")),
                    **row})
            for row in br:
                block_rows_out.append({**meta, "sequence_type": seq_type, "pipeline": spec.name, **row})

        # Metrics are computed over the four paired sequences at the same threshold.
        seq_to_values = {}
        for seq_type in sequences:
            d = np.asarray([x["decision"] for x in evaluated[seq_type]], int)
            seq_to_values[seq_type] = d
        pd_attack = float(np.mean(seq_to_values["H1_att"]))
        pfa_attack = float(np.mean(seq_to_values["H0_att"]))
        pd_clean = float(np.mean(seq_to_values["H1_clean"]))
        pfa_clean = float(np.mean(seq_to_values["H0_clean"]))
        j_attack = pd_attack - pfa_attack
        j_clean = pd_clean - pfa_clean
        fused_by_seq = {
            name: np.asarray([float(r["fused_statistic"]) for r in evaluated[name]], float)
            for name in sequences
        }
        auc_clean = float(roc_auc_score(
            np.r_[np.zeros(len(fused_by_seq["H0_clean"])), np.ones(len(fused_by_seq["H1_clean"]))],
            np.r_[fused_by_seq["H0_clean"], fused_by_seq["H1_clean"]]))
        auc_attack = float(roc_auc_score(
            np.r_[np.zeros(len(fused_by_seq["H0_att"])), np.ones(len(fused_by_seq["H1_att"]))],
            np.r_[fused_by_seq["H0_att"], fused_by_seq["H1_att"]]))

        # Report-level metrics can be recovered from per-round flagged counts and
        # attacker-aware catch/FFR diagnostics; user-level telemetry is optional.
        h1att_rows = evaluated["H1_att"]
        h0att_rows = evaluated["H0_att"]
        h0clean_rows = evaluated["H0_clean"]
        catches_all = [float(r.get("catch", np.nan)) for r in h1att_rows]
        ffr_attack_round = [float(r.get("ffr_attack", np.nan)) for r in h0att_rows]
        ffr_clean_round = [float(r.get("flagged_count", 0)) / c.N for r in h0clean_rows]
        catches_ext: list[float] = []
        catches_mild: list[float] = []
        if collect_user_telemetry and (extreme or mild):
            h1_rows = user_rows_by_sequence["H1_att"]
            for t in range(meta["rounds"]):
                round_flags = {int(r["user_id"]) for r in h1_rows
                               if int(r["round"]) == t and int(r["excluded"]) == 1}
                if extreme:
                    catches_ext.append(len(round_flags & extreme) / len(extreme))
                if mild:
                    catches_mild.append(len(round_flags & mild) / len(mild))

        # Paired H0 displacement decomposition requires user-level kept/excluded
        # identities. Skip it in compact B1 runs; keep full logging available for B5.
        if collect_user_telemetry:
            h0_clean = np.asarray(pair["H0_clean"], float)
            h0_att = np.asarray(pair["H0_att"], float)
            ur_clean_by_round = {int(r["round"]): r for r in []}
            clean_user_rows = user_rows_by_sequence["H0_clean"]
            att_user_rows = user_rows_by_sequence["H0_att"]
            clean_lookup = {(int(r["round"]), int(r["user_id"])): r for r in clean_user_rows}
            att_lookup = {(int(r["round"]), int(r["user_id"])): r for r in att_user_rows}
            benign_ids = np.asarray([i for i in range(c.N) if i not in set(mal.tolist())], dtype=int)
            for t in range(meta["rounds"]):
                clean_vals = h0_clean[t]
                attacked_vals = h0_att[t]
                if len(benign_ids):
                    n_high = max(1, int(np.ceil(0.25 * len(benign_ids))))
                    high_honest = set(benign_ids[np.argsort(clean_vals[benign_ids], kind="mergesort")[-n_high:]].tolist())
                else:
                    high_honest = set()
                keep_att = np.asarray([bool(att_lookup[(t, i)]["retained_for_fusion"]) for i in range(c.N)], bool)
                keep_clean = np.asarray([bool(clean_lookup[(t, i)]["retained_for_fusion"]) for i in range(c.N)], bool)
                n_att_keep = max(int(keep_att.sum()), 1)
                n_clean_keep = max(int(keep_clean.sum()), 1)
                direct = attacked_vals - clean_vals
                direct_extreme = float(np.sum(direct[list(extreme & set(np.where(keep_att)[0]))]) / n_att_keep) if extreme else 0.0
                direct_mild = float(np.sum(direct[list(mild & set(np.where(keep_att)[0]))]) / n_att_keep) if mild else 0.0
                # Exact selection/composition shift on clean counterfactual reports.
                selection_contrib = {"mild": 0.0, "extreme": 0.0, "high_honest": 0.0, "other": 0.0}
                for i in range(c.N):
                    weight_delta = (float(keep_att[i]) / n_att_keep) - (float(keep_clean[i]) / n_clean_keep)
                    if i in mild:
                        group = "mild"
                    elif i in extreme:
                        group = "extreme"
                    elif i in high_honest:
                        group = "high_honest"
                    else:
                        group = "other"
                    selection_contrib[group] += float(clean_vals[i]) * weight_delta
                attacked_fused = float(np.mean(attacked_vals[keep_att])) if keep_att.any() else float(np.mean(attacked_vals))
                clean_fused = float(np.mean(clean_vals[keep_clean])) if keep_clean.any() else float(np.mean(clean_vals))
                direct_total = direct_extreme + direct_mild
                selection_total = float(sum(selection_contrib.values()))
                total_shift = attacked_fused - clean_fused
                displacement_rows.append({**meta, "pipeline": spec.name, "round": int(t),
                    "attacked_h0_fused_stat": attacked_fused, "clean_h0_counterfactual_fused_stat": clean_fused,
                    "total_shift": total_shift, "direct_effect_extreme_retained": direct_extreme,
                    "direct_effect_mild_retained": direct_mild, "direct_effect_total": direct_total,
                    "selection_effect_mild": selection_contrib["mild"],
                    "selection_effect_extreme": selection_contrib["extreme"],
                    "selection_effect_high_honest": selection_contrib["high_honest"],
                    "selection_effect_other": selection_contrib["other"],
                    "selection_effect_total": selection_total,
                    "decomposition_error": total_shift - (direct_total + selection_total),
                    "n_retained_attacked": int(keep_att.sum()), "n_retained_clean": int(keep_clean.sum()),
                    "high_honest_definition": "top quartile of benign H0_clean reports within round"})

        seed_rows.append({**meta, "pipeline": spec.name,
            "threshold": float(tau), "pd_attack": pd_attack, "pfa_attack": pfa_attack,
            "J_attack": j_attack, "pd_clean": pd_clean, "pfa_clean": pfa_clean,
            "J_clean": j_clean, "auc_clean": auc_clean, "auc_attack": auc_attack,
            "tau_quantile": float(meta.get("tau_quantile", np.nan)),
            "ffr_attack": float(np.mean(ffr_attack_round)) if ffr_attack_round else np.nan,
            "ffr_clean": float(np.mean(ffr_clean_round)) if ffr_clean_round else np.nan,
            "catch_all": float(np.mean(catches_all)) if catches_all else np.nan,
            "catch_extreme": float(np.mean(catches_ext)) if catches_ext else np.nan,
            "catch_mild": float(np.mean(catches_mild)) if catches_mild else np.nan,
            "mean_flagged_fraction_clean": float(np.mean(ffr_clean_round)) if ffr_clean_round else np.nan,
        })
    return seed_rows, round_rows_out, user_decision_rows, block_rows_out, observations, displacement_rows


def session_boundary_rows(pair: dict, meta: dict) -> list[dict]:
    session = pair["session"]
    rows = []
    for sid, (start, length) in enumerate(zip(session.boundaries, session.lengths)):
        rows.append({**meta, "session_id": int(sid), "start_round": int(start),
                     "end_round": int(start + length - 1), "length": int(length),
                     "session_mode": session.mode})
    return rows


def write_manifest(out: Path, args, c: Config, seeds: dict, pipeline_specs: list[PipelineSpec]) -> None:
    manifest = {
        "study_version": "B1-B10-supplement-instrumented",
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.__dict__.items()},
        "arguments": vars(args), "seed_pools": seeds,
        "pipelines": [{"name": p.name, "kind": p.kind,
                       "xs_config": p.xs_cfg.__dict__ if p.xs_cfg else None} for p in pipeline_specs],
        "random_stream_note": ("B1-V3 uses the same condition id across SNRs so seed-specific session schedules, malicious identities, and subgroup identities are paired across SNRs. Within each condition, all pipelines are evaluated on the exact same generated H0/H1 report sequences. Session schedules are pre-generated by the simulator and are not exposed to the detectors."),
        "persistence_note": ("For persistence_w>0, h and g use log-domain sqrt(w) persistent plus sqrt(1-w) session components; uncertainty uses the corresponding linear mixture clipped to +/-NOISE_UNC_DB. Conformal score banks remain trained at w=0; tau thresholds are recalibrated at the specified persistence_w."),
        "beta_history_note": "BetaHistory is a study-adapted Beta-Bernoulli anomaly-history baseline, not an exact reproduction of a cited paper.",
        "bdf_history_note": ("BDFHistory is a study-adapted windowed Beta peer-consistency baseline inspired by the reputation component in Wu et al. (2024), Electronics 13, 3386, DOI 10.3390/electronics13173386. It adapts continuous reports to LOO median/MAD peer-consistency labels and does not reproduce the paper's complete data-delivery, weighted positive/negative evaluation, and sequential likelihood-ratio process."),
        "supplement_protocol_note": ("This is a reviewer-requested supplemental run, not a rerun or modification of the frozen B1-V3 primary experiment. Supplemental hypotheses and condition grids are defined in supplement_protocol.json. B1-V3 remains the only predeclared primary test family; all new sensitivity, baseline, telemetry, and persistence analyses are reported as supplemental."),
        "baseline_parameters": {
            "Huber": {"c_huber": 1.345, "iterations": 30, "scale": "1.4826*MAD"},
            "BetaHistory": {"alpha0": 1.0, "beta0": 19.0, "evidence_rule": "abs(LOO residual)>3", "posterior_cut": 0.20,
                            "flag_cap": int(math.floor(c.TTF_RHO_MAX * c.N))},
            "BDFHistory": {"window": 20, "alpha0": 1.0, "beta0": 1.0,
                           "positive_evidence": "abs(LOO residual)<=3", "posterior_reliability_cut": 0.65,
                           "flag_cap": int(math.floor(c.TTF_RHO_MAX * c.N)),
                           "status": "BDF-inspired study adaptation; not exact reproduction"},
        },
        "subgroup_assignment": args.subgroup_assignment,
        "rank_definition": "rank_ascending=1 is the smallest raw report; rank_descending=1 is the largest raw report; computed before any pipeline exclusion.",
        "residual_definition": "XS uses its LOO median/MAD residual; TTF-CF uses its internal residual; other baselines store the same LOO median/MAD residual as a diagnostic, not as their internal decision variable.",
        "oracle_note": "OracleRemoval threshold is calibrated on clean H0 after removing a fixed random subset of size L for each rho.",
        "telemetry_note": "Telemetry is observational and is not fed back into detector decisions.",
        "source_sha256_at_run_start": {
            name: sha256(Path(__file__).resolve().parent / name)
            for name in ["run_xs_study_v2.py", "xs_experiment_v2.py", "xs_filter_v2.py",
                         "xs_config.py", "run_final_experiment.py", "system_model_final.py",
                         "ttf_cf_final.py", "paradigm_a_final.py", "config_final.py",
                         "analyze_v2.py", "validate_v2.py", "xs_filter_final_v1_reference.py",
                         "xs_scenarios_v1_reference.py", "run_supplement_b1_b10.py",
                         "analyze_supplement_b1_b10.py", "validate_supplement.py",
                         "supplement_protocol.json"]
        },
    }
    (out / "v2_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Instrumented XS B1-V3 experiment runner")
    ap.add_argument("--out", default="xs_v2_results")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--snrs", default="-9,-6")
    ap.add_argument("--rhos", default="0.2,0.4")
    ap.add_argument("--attacks", default="none,masking")
    ap.add_argument("--session-specs", default="fixed:20", help="semicolon-separated: fixed:13;fixed:20;cycle:13,30,50;random:13,20,30,50;random_balanced:13,20,30,50")
    ap.add_argument("--extreme-fractions", default="0.5")
    ap.add_argument("--persistence-ws", default="0.0", help="comma-separated persistent heterogeneity levels; 0 means independent sessions")
    ap.add_argument("--subgroup-assignment", choices=["random", "legacy_sorted"], default="random")
    ap.add_argument("--caps", default="8", help="max exclusions as counts, e.g. 4,8,12")
    ap.add_argument("--fixed-blocks", default="10,13,20,30,50")
    ap.add_argument("--ph-delta", type=float, default=0.05)
    ap.add_argument("--ph-threshold", type=float, default=5.0)
    ap.add_argument("--ph-beta", type=float, default=0.05)
    ap.add_argument("--ph-warmup", type=int, default=5)
    ap.add_argument("--min-block", type=int, default=5)
    ap.add_argument("--pipelines", default=None, help="comma-separated names; default includes core baselines, Huber and BetaHistory")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--test-start", type=int, default=63000)
    ap.add_argument("--test-count", type=int, default=400,
                    help="default 400 is a conservative primary-RQ1 planning target using V1 variance and an 8-test family")
    ap.add_argument("--score-cal-start", type=int, default=62100)
    ap.add_argument("--score-cal-count", type=int, default=100)
    ap.add_argument("--tau-cal-start", type=int, default=62200)
    ap.add_argument("--tau-cal-count", type=int, default=100)
    ap.add_argument("--tau-quantile", type=float, default=0.90)
    ap.add_argument("--oracle-rho-calibration", action="store_true", default=True)
    ap.add_argument("--skip-user-telemetry", action="store_true", help="skip very detailed user-level CSVs (not recommended for B5)")
    ap.add_argument("--skip-round-telemetry", action="store_true", help="do not write the large round-level fused-statistics CSV; seed-level metrics and block/session boundaries are still retained")
    a = ap.parse_args()

    if a.rounds < 1 or a.test_count < 1 or a.score_cal_count < 1 or a.tau_cal_count < 1:
        raise SystemExit("rounds and all seed counts must be positive")
    if not (0.0 < a.tau_quantile < 1.0):
        raise SystemExit("tau-quantile must lie in (0,1)")
    out = Path(a.out)
    ensure_empty_output(out, a.overwrite)
    c = cfg_from_env()

    snrs = parse_float_list(a.snrs)
    rhos = parse_float_list(a.rhos)
    attacks = [x.strip() for x in a.attacks.split(",") if x.strip()]
    session_specs = parse_semicolon_list(a.session_specs)
    extreme_fractions = parse_float_list(a.extreme_fractions)
    persistence_ws = parse_float_list(a.persistence_ws)
    if any(not (0.0 <= w < 1.0) for w in persistence_ws):
        raise SystemExit("persistence-ws values must lie in [0,1)")
    caps = parse_int_list(a.caps)
    fixed_blocks = parse_int_list(a.fixed_blocks)
    if any(cap < 0 or cap >= c.N for cap in caps):
        raise SystemExit(f"caps must be >=0 and < N={c.N}")
    if any(not (0 <= x <= 1) for x in extreme_fractions):
        raise SystemExit("extreme fractions must be in [0,1]")
    if a.pipelines:
        pipeline_names = [x.strip() for x in a.pipelines.split(",") if x.strip()]
    else:
        pipeline_names = DEFAULT_PIPELINES.copy() + ["XS-PH", "XS-fixed"]
    if not (a.ph_delta >= 0 and a.ph_threshold > 0 and 0 < a.ph_beta <= 1 and a.ph_warmup >= 1 and a.min_block >= 1):
        raise SystemExit("invalid PH/min-block parameters")
    pipeline_specs = make_pipeline_specs(
        pipeline_names, fixed_blocks, caps, c.N,
        ph_delta=a.ph_delta, ph_threshold=a.ph_threshold,
        ph_beta=a.ph_beta, ph_warmup=a.ph_warmup, min_block=a.min_block)

    seeds = {
        "score_cal": list(range(a.score_cal_start, a.score_cal_start + a.score_cal_count)),
        "tau_cal": list(range(a.tau_cal_start, a.tau_cal_start + a.tau_cal_count)),
        "test": list(range(a.test_start, a.test_start + a.test_count)),
    }
    pools = [set(v) for v in seeds.values()]
    if any(pools[i] & pools[j] for i in range(len(pools)) for j in range(i + 1, len(pools))):
        raise SystemExit("score-calibration, tau-calibration and test seed pools overlap")

    write_manifest(out, a, c, seeds, pipeline_specs)
    ledger = [{"stage": "started", "time_utc": utc_now(), "seed_pools": seeds,
               "note": "V2 instrumented run; V1 source and outputs remain unchanged."}]
    (out / "v2_seed_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    seed_path = out / "v2_seed_level.csv.gz"
    round_path = out / "v2_pipeline_round.csv.gz"
    user_obs_path = out / "v2_user_observations.csv.gz"
    user_dec_path = out / "v2_user_decisions.csv.gz"
    session_path = out / "v2_session_boundaries.csv.gz"
    block_path = out / "v2_block_boundaries.csv.gz"
    displacement_path = out / "v2_displacement.csv.gz"
    threshold_path = out / "v2_thresholds.csv.gz"
    score_bank_path = out / "v2_score_bank_sizes.csv.gz"
    calibration_path = out / "v2_calibration_scores.csv.gz"
    oracle_subset_path = out / "v2_oracle_calibration_subsets.csv.gz"
    identity_path = out / "v3_identity_ledger.csv.gz"

    context_cache = {}
    tau_cache = {}
    # The no-attack reference is run once per SNR/session schedule, not repeated
    # under every requested rho or extreme/mild split (it is the same clean cell).
    cells_per_snr_session = sum(
        (len(extreme_fractions) if attack != "none" else (1 if rho == rhos[0] else 0))
        for rho in rhos for attack in attacks
    )
    total_cells = len(snrs) * len(session_specs) * len(persistence_ws) * cells_per_snr_session
    cell_num = 0
    for snr in snrs:
        for session_spec in session_specs:
            for persistence_w in persistence_ws:
                # Same base seed/condition id across w values; the persistent component
                # changes the nuisance process while preserving paired sensing streams.
                # B1-V3: common condition stream across SNRs preserves the
                # same session schedule, attacker identities and subgroup labels
                # for each seed; physical report values still reflect the SNR.
                condition_id = 810_000
                ctx_key = (snr, session_spec, persistence_w)
                if ctx_key not in context_cache:
                    print(f"[V2] calibrating score banks (w=0) and clean-H0 thresholds (w={persistence_w:g}): "
                          f"SNR={snr:g}, sessions={session_spec}", flush=True)
                    ctx, thresholds_base, calibration_rows = build_context(
                        c, snr, session_spec, persistence_w, pipeline_specs, seeds["score_cal"], seeds["tau_cal"],
                        a.rounds, a.tau_quantile, rhos, condition_id)
                    context_cache[ctx_key] = (ctx, thresholds_base)
                    bank_size_rows = []
                    for bank_name, bucket_sizes in ctx.get("xs_bank_sizes", {}).items():
                        for age_bucket, n_scores in enumerate(bucket_sizes):
                            bank_size_rows.append({"snr_db": snr, "session_spec": session_spec,
                                "evaluation_persistence_w": float(persistence_w), "bank_training_persistence_w": 0.0,
                                "bank": bank_name, "age_bucket": age_bucket, "n_scores": int(n_scores),
                                "n_calibration_seeds": len(seeds["score_cal"]),
                                "rounds_per_seed": c.SCORE_CAL_ROUNDS})
                    for age_bucket, n_scores in enumerate(ctx.get("ttf_bank_sizes", [])):
                        bank_size_rows.append({"snr_db": snr, "session_spec": session_spec,
                            "evaluation_persistence_w": float(persistence_w), "bank_training_persistence_w": 0.0,
                            "bank": "TTF-CF", "age_bucket": age_bucket, "n_scores": int(n_scores),
                            "n_calibration_seeds": len(seeds["score_cal"]),
                            "rounds_per_seed": c.SCORE_CAL_ROUNDS})
                    append_gzip_csv(score_bank_path, bank_size_rows)
                    if calibration_rows:
                        cal_score_rows = [r for r in calibration_rows if "calibration_statistic" in r]
                        threshold_rows = [r for r in calibration_rows if "threshold" in r]
                        append_gzip_csv(calibration_path, cal_score_rows)
                        append_gzip_csv(threshold_path, threshold_rows)
                ctx, thresholds_base = context_cache[ctx_key]

                # Calibrate Oracle at every requested rho and also at rho=0 for the
                # no-attack reference. Calibration is clean H0 with a random subset
                # of L reports removed, under the same persistence_w as evaluation.
                oracle_rhos = sorted(set(rhos) | ({0.0} if "none" in attacks else set()))
                for rho_cal in oracle_rhos:
                    oracle_key = (snr, session_spec, persistence_w, rho_cal)
                    if oracle_key not in tau_cache:
                        L = int(round(c.N * rho_cal))
                        vals = []
                        oracle_cal_rows = []
                        oracle_subset_rows = []
                        for seed in seeds["tau_cal"]:
                            pair = generate_paired_condition_v2(c, seed, 735_000 + condition_id, snr, 0.0, "none",
                                                                c.TAU_CAL_ROUNDS, session_spec=session_spec,
                                                                persistence_w=persistence_w)
                            rrng = R.rng(seed, 745_000 + condition_id + int(round(rho_cal * 10_000)))
                            removed = np.sort(rrng.choice(c.N, size=L, replace=False)) if L else np.empty(0, int)
                            removed_set = set(removed.tolist())
                            for user_id in range(c.N):
                                oracle_subset_rows.append({"snr_db": snr, "session_spec": session_spec,
                                    "persistence_w": float(persistence_w), "calibration_seed": seed,
                                    "rho_oracle_removed": rho_cal, "oracle_removed_reports": L,
                                    "user_id": int(user_id), "removed_for_oracle_calibration": int(user_id in removed_set)})
                            for t, x in enumerate(pair["H0_clean"]):
                                keep = np.ones(c.N, dtype=bool)
                                keep[removed] = False
                                value = float(np.mean(x[keep]))
                                vals.append(value)
                                oracle_cal_rows.append({"snr_db": snr, "session_spec": session_spec,
                                    "persistence_w": float(persistence_w),
                                    "calibration_seed": seed, "round": t, "pipeline": "OracleRemoval",
                                    "rho_oracle_removed": rho_cal, "calibration_statistic": value})
                        oracle_tau = float(np.quantile(vals, a.tau_quantile))
                        tau_cache[oracle_key] = oracle_tau
                        append_gzip_csv(calibration_path, oracle_cal_rows)
                        append_gzip_csv(oracle_subset_path, oracle_subset_rows)
                        append_gzip_csv(threshold_path, [{"snr_db": snr, "session_spec": session_spec,
                            "persistence_w": float(persistence_w), "pipeline": "OracleRemoval",
                            "rho_oracle_removed": rho_cal, "quantile": a.tau_quantile,
                            "threshold": oracle_tau, "n_calibration_rounds": len(vals),
                            "n_calibration_seeds": len(seeds["tau_cal"]), "oracle_removed_reports": L}])

                for rho in rhos:
                    thresholds = dict(thresholds_base)
                    for attack in attacks:
                        is_clean_reference = attack == "none"
                        # One explicit no-attack cell is sufficient; label it rho=0 and
                        # extreme_fraction=0 to avoid duplicate clean cells.
                        if is_clean_reference and rho != rhos[0]:
                            continue
                        rho_run = 0.0 if is_clean_reference else float(rho)
                        fractions_run = [0.0] if is_clean_reference else extreme_fractions
                        oracle_rho_key = 0.0 if is_clean_reference else float(rho)
                        thresholds[f"OracleRemoval|rho={oracle_rho_key:g}"] = tau_cache[(snr, session_spec, persistence_w, oracle_rho_key)]
                        for extreme_fraction in fractions_run:
                            cell_num += 1
                            print(f"[V2] cell {cell_num}/{total_cells}: SNR={snr:g}, rho={rho_run:g}, attack={attack}, "
                                  f"extreme_fraction={extreme_fraction:g}, sessions={session_spec}, w={persistence_w:g}", flush=True)
                            for seed in seeds["test"]:
                                pair = generate_paired_condition_v2(
                                    c, seed, condition_id + int(round(rho_run * 1000)) + 7 * attacks.index(attack),
                                    snr, rho_run, attack, a.rounds, session_spec=session_spec,
                                    extreme_fraction=extreme_fraction,
                                    subgroup_assignment=a.subgroup_assignment,
                                    persistence_w=persistence_w)
                                cap_label = ",".join(str(x) for x in caps)
                                meta = condition_metadata(pair, c, snr, rho_run, attack, seed,
                                                           session_spec, extreme_fraction, cap_label)
                                meta["tau_quantile"] = float(a.tau_quantile)
                                append_gzip_csv(identity_path, [{
                                    "seed": int(seed), "snr_db": float(snr),
                                    "session_spec": session_spec, "attack": attack,
                                    "rho_requested": float(rho_run),
                                    "malicious_ids": json.dumps(sorted(map(int, pair["malicious_idx"]))),
                                    "extreme_ids": json.dumps(sorted(map(int, pair["extreme_idx"]))),
                                    "mild_ids": json.dumps(sorted(map(int, pair["mild_idx"]))),
                                    "session_lengths": json.dumps(list(map(int, pair["session"].lengths))),
                                    "subgroup_assignment": pair["subgroup_assignment"],
                                }])
                                seed_rows, round_rows, user_rows, block_rows, obs_rows, displacement_rows = run_one_seed(
                                    c, pair, meta, pipeline_specs, ctx, thresholds, [],
                                    collect_user_telemetry=not a.skip_user_telemetry)
                                append_gzip_csv(seed_path, seed_rows)
                                if not a.skip_round_telemetry:
                                    append_gzip_csv(round_path, round_rows)
                                append_gzip_csv(block_path, block_rows)
                                append_gzip_csv(displacement_path, displacement_rows)
                                append_gzip_csv(session_path, session_boundary_rows(pair, meta))
                                if not a.skip_user_telemetry:
                                    append_gzip_csv(user_obs_path, obs_rows)
                                    append_gzip_csv(user_dec_path, user_rows)

    # Simple descriptive aggregate by condition and pipeline (seed is the independent unit).
    if seed_path.exists():
        df = pd.read_csv(seed_path)
        keys = ["snr_db", "rho_requested", "rho_actual", "attack", "extreme_fraction", "session_spec", "persistence_w", "pipeline", "cap_design"]
        metrics = ["pd_attack", "pfa_attack", "J_attack", "pd_clean", "pfa_clean", "J_clean",
                   "ffr_attack", "ffr_clean", "catch_all", "catch_extreme", "catch_mild"]
        agg = df.groupby(keys, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["_".join([str(v) for v in col if str(v) != ""]).rstrip("_") if isinstance(col, tuple) else col for col in agg.columns]
        agg.to_csv(out / "v2_aggregate.csv.gz", index=False, compression="gzip")

    ledger.append({"stage": "completed", "completed_utc": utc_now(), "cells_completed": cell_num,
                   "output_files": sorted(p.name for p in out.glob("v2_*"))})
    (out / "v2_seed_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    output_hashes = {p.name: sha256(p) for p in sorted(out.glob("v2_*")) if p.is_file() and p.name != "v2_output_sha256.json"}
    (out / "v2_output_sha256.json").write_text(json.dumps(output_hashes, indent=2), encoding="utf-8")
    print("V2 run completed. Outputs:", out.resolve())


if __name__ == "__main__":
    main()
