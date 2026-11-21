from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config_final import Config, all_seed_pools, cfg_from_env
from paradigm_a_final import ParadigmA, ParadigmAConfig
import system_model_final as sm
from ttf_cf_final import ConformalBank, TTFConfig, TTFCF

try:
    import scipy
    import sklearn
except Exception:
    scipy = None
    sklearn = None


PRIMARY_BASELINES = ["NoDefense", "MLG+TM", "MAD", "Sn", "Qn", "ParadigmA"]
TTF_PIPELINES = ["TTF-CF", "TTF-CF/median", "TTF-CF/noCoal", "TTF-CF/noPH"]
ALL_PIPELINES = PRIMARY_BASELINES + TTF_PIPELINES + ["OracleRemoval"]


class RunState:
    def __init__(self, c: Config):
        self.c = c
        self.out = Path(c.OUTPUT_DIR)
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": getattr(scipy, "__version__", None),
            "scikit_learn": getattr(sklearn, "__version__", None),
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.__dict__.items()},
            "seed_pools": all_seed_pools(c),
            "primary_pipelines": PRIMARY_BASELINES,
            "ttf_pipelines": TTF_PIPELINES,
            "nondefense_reference": "OracleRemoval",
        }


def rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def make_session(c: Config, r: np.random.Generator):
    h = sm.sample_f_composite_gain(c.N, c.F_M, c.F_MS, r)
    g = r.rayleigh(scale=c.RAYLEIGH_SCALE, size=c.N)
    unc = r.uniform(-c.NOISE_UNC_DB, c.NOISE_UNC_DB, size=c.N)
    return h, g, unc


def generate_paired_condition(
    c: Config,
    seed: int,
    condition_id: int,
    snr_db: float,
    rho: float,
    attack: str,
    rounds: int,
    attack_shift_db: float | None = None,
):
    """Generate H0-clean, H0-attacked, H1-clean, H1-attacked using common session states.

    All detector pipelines receive exactly the same arrays. H0/H1 use separate sensing/reporting
    random streams but share the physical session state (h, g, persistent uncertainty).
    """
    mal = sm.choose_malicious_idx(c.N, rho if attack != "none" else 0.0, seed, condition_id)
    env = rng(seed, 100_000 + condition_id)
    h1_rng = rng(seed, 110_000 + condition_id)
    h0_rng = rng(seed, 120_000 + condition_id)
    atk_rng = rng(seed, 130_000 + condition_id)
    h_sessions = []
    g_sessions = []
    unc_sessions = []
    H1_clean, H1_att, H0_clean, H0_att = [], [], [], []

    for r in range(rounds):
        if r % c.SESSION_LENGTH == 0:
            h, g, unc = make_session(c, env)
            h_sessions.append(h.copy())
            g_sessions.append(g.copy())
            unc_sessions.append(unc.copy())
        h, g, unc = h_sessions[-1], g_sessions[-1], unc_sessions[-1]

        T1 = sm.local_energy_gamma(c.N, c.M_SAMPLES, snr_db, "H1", h, unc, h1_rng)
        T0 = sm.local_energy_gamma(c.N, c.M_SAMPLES, snr_db, "H0", h, unc, h0_rng)
        eta1 = env.normal(0.0, c.REPORT_NOISE_STD, size=c.N)
        eta0 = env.normal(0.0, c.REPORT_NOISE_STD, size=c.N)
        H1_clean.append(sm.reporting_channel(T1, g, eta1))
        H0_clean.append(sm.reporting_channel(T0, g, eta0))
        shift_db = c.SHIFT_DB if attack_shift_db is None else float(attack_shift_db)
        T1a = sm.apply_attack(T1, mal, attack, atk_rng, shift_db, c.MASK_EXTREME_DB, c.MASK_MILD_DB)
        T0a = sm.apply_attack(T0, mal, attack, atk_rng, shift_db, c.MASK_EXTREME_DB, c.MASK_MILD_DB)
        # Reuse the same reporting noise for clean vs attacked copies: only the attack changes.
        H1_att.append(sm.reporting_channel(T1a, g, eta1))
        H0_att.append(sm.reporting_channel(T0a, g, eta0))

    return {
        "malicious_idx": mal,
        "H1_clean": H1_clean,
        "H1_att": H1_att,
        "H0_clean": H0_clean,
        "H0_att": H0_att,
    }


def make_pa_model(c: Config, snr_db: float) -> ParadigmA:
    pools = all_seed_pools(c)
    rounds = c.DEV_ROUNDS
    honest_rounds = []
    for seed in pools["development"]:
        r = rng(seed, 201_000 + int(round((snr_db + 50) * 10)))
        for t in range(rounds):
            if t % c.SESSION_LENGTH == 0:
                h, g, unc = make_session(c, r)
            # Balance honest H0/H1 during training so high-energy honest reports are not treated as anomalies.
            hyp = "H0" if (t % 2 == 0) else "H1"
            T = sm.local_energy_gamma(c.N, c.M_SAMPLES, snr_db, hyp, h, unc, r)
            eta = r.normal(0.0, c.REPORT_NOISE_STD, size=c.N)
            honest_rounds.append(sm.reporting_channel(T, g, eta))
    model = ParadigmA(
        seed=777 + int(round((snr_db + 50) * 10)),
        cfg=ParadigmAConfig(
            n_estimators=c.PA_IF_ESTIMATORS,
            ocsvm_nu=c.PA_OCSVM_NU,
            lof_neighbors=c.PA_LOF_N_NEIGHBORS,
            ae_hidden=c.PA_AE_HIDDEN,
            ae_max_iter=c.PA_AE_MAX_ITER,
            vote_fraction=c.PA_ENSEMBLE_VOTE_FRACTION,
        ),
    )
    return model.fit(honest_rounds)


def classical_flag(c: Config, x: np.ndarray, pipeline: str) -> np.ndarray:
    if pipeline == "NoDefense":
        return np.empty(0, dtype=int)
    return sm.detector_flags(x, pipeline, c.TM_ALPHA, c.TM_N_SIM, c.MAD_C, c.QN_CONSTANT)


def eval_one_pipeline(
    c: Config,
    pipeline: str,
    sequence: list[np.ndarray],
    malicious_idx: np.ndarray,
    tau: float | None,
    pa_model: ParadigmA | None,
    bank: ConformalBank | None,
):
    true_set = set(np.asarray(malicious_idx, dtype=int).tolist())
    n_honest = max(c.N - len(true_set), 1)
    decisions, flags_catch, flags_ffr, flag_count = [], [], [], []
    ttf = None
    if pipeline.startswith("TTF-CF"):
        variant = pipeline
        cfg = TTFConfig(
            lam=c.TTF_LAMBDA,
            cusum_k=c.TTF_CUSUM_K,
            W=c.TTF_W,
            kappa_sign=c.TTF_KAPPA_SIGN,
            kappa_p=c.TTF_KAPPA_P,
            q=c.TTF_Q,
            rho_max=c.TTF_RHO_MAX,
            ph_delta=c.TTF_PH_DELTA,
            ph_threshold=c.TTF_PH_THRESHOLD,
            ph_beta=c.TTF_PH_BETA,
            ph_warmup=c.TTF_PH_WARMUP,
            fusion="median" if variant == "TTF-CF/median" else "mean",
            use_coalition=(variant != "TTF-CF/noCoal"),
            domain="linear",
        )
        ttf = TTFCF(c.N, cfg, bank)

    prev_stat = None
    for x in sequence:
        if pipeline == "OracleRemoval":
            fl = np.asarray(sorted(true_set), dtype=int)
            stat = sm.fusion_statistic(x, fl, "mean")
        elif pipeline in PRIMARY_BASELINES:
            if pipeline == "ParadigmA":
                fl = pa_model.detect(x)  # type: ignore[union-attr]
            else:
                fl = classical_flag(c, x, pipeline)
            stat = sm.fusion_statistic(x, fl, "mean")
        else:
            z = ttf.step(x)  # type: ignore[union-attr]
            fl = z["flagged"]
            # TTFConfig's primary and variants are all label-free.
            stat = float(z["stat"])
        decisions.append(int(stat > float(tau)))
        fs = set(np.asarray(fl, dtype=int).tolist())
        flags_catch.append(len(fs & true_set) / max(len(true_set), 1) if true_set else np.nan)
        flags_ffr.append(len(fs - true_set) / n_honest)
        flag_count.append(len(fs) / c.N)
        prev_stat = stat

    return {
        "pd": float(np.mean(decisions)),
        "catch": (float(np.mean([v for v in flags_catch if np.isfinite(v)])) if true_set else np.nan),
        "ffr": float(np.mean(flags_ffr)),
        "flag_fraction": float(np.mean(flag_count)),
        "n_rounds": len(sequence),
    }


def build_conformal_bank(c: Config, snr_db: float) -> ConformalBank:
    bank = ConformalBank(age_cap=39)
    # Score-calibration uses an independent pool and the same honest environment model.
    for seed in all_seed_pools(c)["score_calibration"]:
        rr = rng(seed, 301_000 + int(round((snr_db + 50) * 10)))
        ttf = TTFCF(c.N, TTFConfig(
            lam=c.TTF_LAMBDA, cusum_k=c.TTF_CUSUM_K, W=c.TTF_W,
            kappa_sign=c.TTF_KAPPA_SIGN, kappa_p=c.TTF_KAPPA_P, q=c.TTF_Q,
            rho_max=c.TTF_RHO_MAX, ph_delta=c.TTF_PH_DELTA,
            ph_threshold=c.TTF_PH_THRESHOLD, ph_beta=c.TTF_PH_BETA,
            ph_warmup=c.TTF_PH_WARMUP, fusion="mean"), bank=None)
        for t in range(c.SCORE_CAL_ROUNDS):
            if t % c.SESSION_LENGTH == 0:
                h, g, unc = make_session(c, rr)
            T = sm.local_energy_gamma(c.N, c.M_SAMPLES, snr_db, "H0", h, unc, rr)
            eta = rr.normal(0.0, c.REPORT_NOISE_STD, size=c.N)
            x = sm.reporting_channel(T, g, eta)
            z = ttf.step(x)
            bank.add(z["age"], z["cusum"])
    return bank.finalize()


def calibrate_thresholds(c: Config, snr_db: float, pa_model: ParadigmA, bank: ConformalBank) -> dict[str, float]:
    """Calibrate final decision threshold for each pipeline on disjoint H0-no-attack observations."""
    pools = all_seed_pools(c)["tau_calibration"]
    stats = {p: [] for p in PRIMARY_BASELINES + TTF_PIPELINES + ["OracleRemoval"]}

    for seed in pools:
        condition_id = int(round((snr_db + 50) * 10))
        pair = generate_paired_condition(c, seed, 9000 + condition_id, snr_db, 0.0, "none", c.TAU_CAL_ROUNDS)
        for pipeline in PRIMARY_BASELINES + TTF_PIPELINES:
            seq = pair["H0_clean"]
            if pipeline.startswith("TTF-CF"):
                variant = pipeline
                cfg = TTFConfig(
                    lam=c.TTF_LAMBDA, cusum_k=c.TTF_CUSUM_K, W=c.TTF_W,
                    kappa_sign=c.TTF_KAPPA_SIGN, kappa_p=c.TTF_KAPPA_P, q=c.TTF_Q,
                    rho_max=c.TTF_RHO_MAX, ph_delta=c.TTF_PH_DELTA,
                    ph_threshold=(1e12 if variant == "TTF-CF/noPH" else c.TTF_PH_THRESHOLD),
                    ph_beta=c.TTF_PH_BETA, ph_warmup=c.TTF_PH_WARMUP,
                    fusion=("median" if variant == "TTF-CF/median" else "mean"),
                    use_coalition=(variant != "TTF-CF/noCoal"),
                )
                st = TTFCF(c.N, cfg, bank)
                vals = [float(st.step(x)["stat"]) for x in seq]
                stats[pipeline].extend(vals)
            else:
                for x in seq:
                    if pipeline == "ParadigmA":
                        fl = pa_model.detect(x)
                    else:
                        fl = classical_flag(c, x, pipeline)
                    stats[pipeline].append(sm.fusion_statistic(x, fl, "mean"))

    # Oracle threshold is informational only. It reuses the NoDefense threshold
    # as an upper-bound reference and has no independent calibration samples.
    thresholds = {p: (float(np.quantile(v, 0.90)) if len(v) else np.nan)
                  for p, v in stats.items() if p != "OracleRemoval"}
    thresholds["OracleRemoval"] = thresholds["NoDefense"]
    calibration_rows = []
    for p, tau in thresholds.items():
        v = np.asarray(stats.get(p, []), dtype=float)
        calibration_rows.append({
            "snr_db": snr_db,
            "pipeline": p,
            "threshold": tau,
            "calibration_pfa_empirical": (float(np.mean(v > tau)) if v.size else np.nan),
            "n": int(v.size),
        })
    pd.DataFrame(calibration_rows).to_csv(
        Path(c.OUTPUT_DIR) / f"threshold_calibration_{snr_db:g}.csv", index=False
    )
    return thresholds


def run_main(c: Config, state: RunState) -> None:
    out = state.out
    all_rows = []
    threshold_rows = []
    pa_models = {}
    banks = {}
    thresholds = {}

    for snr in c.SNR_GRID:
        print(f"[calibration] SNR={snr:g}", flush=True)
        sm.warm_tm_cache(c.N, c.TM_ALPHA, c.TM_N_SIM)
        pa_models[snr] = make_pa_model(c, snr)
        banks[snr] = build_conformal_bank(c, snr)
        thresholds[snr] = calibrate_thresholds(c, snr, pa_models[snr], banks[snr])
        threshold_rows.extend([{"snr_db": snr, "pipeline": p, "threshold": thresholds[snr][p]} for p in thresholds[snr]])

    Path(out / "all_thresholds.csv").write_text(pd.DataFrame(threshold_rows).to_csv(index=False), encoding="utf-8")

    conditions = []
    cid = 0
    for snr in c.SNR_GRID:
        for rho in list(c.MAIN_RHOS) + [c.STRESS_RHO]:
            for attack in c.ATTACKS:
                conditions.append((cid, snr, rho, attack))
                cid += 1

    for condition_id, snr, rho, attack in conditions:
        print(f"[main] condition={condition_id} SNR={snr:g} rho={rho:.0%} attack={attack}", flush=True)
        for seed in all_seed_pools(c)["evaluation"]:
            pair = generate_paired_condition(c, seed, condition_id, snr, rho, attack, c.EVAL_ROUNDS)
            # For "none", use clean sequence; for attacks, evaluate attacked H1/H0 and separately clean H0.
            h1 = pair["H1_clean"] if attack == "none" else pair["H1_att"]
            h0_att = pair["H0_clean"] if attack == "none" else pair["H0_att"]
            h0_clean = pair["H0_clean"]
            for pipeline in PRIMARY_BASELINES + TTF_PIPELINES:
                # Ensure each stateful pipeline is reset between H1 and H0 evaluations.
                r1 = eval_one_pipeline(c, pipeline, h1, pair["malicious_idx"], thresholds[snr][pipeline], pa_models[snr], banks[snr])
                r0a = eval_one_pipeline(c, pipeline, h0_att, pair["malicious_idx"], thresholds[snr][pipeline], pa_models[snr], banks[snr])
                r0c = eval_one_pipeline(c, pipeline, h0_clean, np.empty(0, dtype=int), thresholds[snr][pipeline], pa_models[snr], banks[snr])
                all_rows.append({
                    "study": "MAIN",
                    "condition_id": condition_id,
                    "seed": seed,
                    "snr_db": snr,
                    "rho": rho,
                    "attack": attack,
                    "pipeline": pipeline,
                    "pd": r1["pd"],
                    "pfa_attack": r0a["pd"],
                    "pfa_clean": r0c["pd"],
                    "J_attack": r1["pd"] - r0a["pd"],
                    "J_clean": r1["pd"] - r0c["pd"],
                    "TER_attack": 1.0 - (r1["pd"] - r0a["pd"]),
                    "catch": r1["catch"],
                    "ffr": r1["ffr"],
                    "flag_fraction": r1["flag_fraction"],
                    "n_malicious": int(len(pair["malicious_idx"])),
                })
            # Oracle upper bound is separate; no defense claim.
            r1o = eval_one_pipeline(c, "OracleRemoval", h1, pair["malicious_idx"], thresholds[snr]["OracleRemoval"], pa_models[snr], banks[snr])
            r0o = eval_one_pipeline(c, "OracleRemoval", h0_att, pair["malicious_idx"], thresholds[snr]["OracleRemoval"], pa_models[snr], banks[snr])
            all_rows.append({
                "study": "MAIN", "condition_id": condition_id, "seed": seed,
                "snr_db": snr, "rho": rho, "attack": attack,
                "pipeline": "OracleRemoval", "role": "upper_bound",
                "pd": r1o["pd"], "pfa_attack": r0o["pd"], "pfa_clean": np.nan,
                "J_attack": r1o["pd"] - r0o["pd"], "J_clean": np.nan,
                "TER_attack": 1.0 - (r1o["pd"] - r0o["pd"]), "catch": 1.0,
                "ffr": 0.0, "flag_fraction": len(pair["malicious_idx"]) / c.N,
                "n_malicious": int(len(pair["malicious_idx"])),
            })
        pd.DataFrame(all_rows).to_csv(out / "main_seed_level.csv", index=False)

    write_analysis_files(c, out)


def write_analysis_files(c: Config, out: Path) -> None:
    df = pd.read_csv(out / "main_seed_level.csv")
    rows = []
    group_cols = ["snr_db", "rho", "attack", "pipeline"]
    for key, g in df.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, key))
        rec["n_seeds"] = int(g["seed"].nunique())
        for m in ["pd", "pfa_attack", "pfa_clean", "J_attack", "J_clean", "TER_attack", "catch", "ffr"]:
            x = g[m].dropna().to_numpy(float)
            if len(x) == 0:
                rec[f"{m}_mean"] = np.nan; rec[f"{m}_ci95"] = np.nan
            elif len(x) == 1:
                rec[f"{m}_mean"] = float(x[0]); rec[f"{m}_ci95"] = 0.0
            else:
                from scipy.stats import t
                rec[f"{m}_mean"] = float(np.mean(x))
                rec[f"{m}_ci95"] = float(t.ppf(0.975, len(x)-1) * np.std(x, ddof=1) / np.sqrt(len(x)))
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out / "main_aggregate.csv", index=False)

    # Primary paired bootstrap: TTF-CF vs classical baselines on J_attack.
    boot = []
    rng_b = np.random.default_rng(20261001)
    for (snr, rho, attack), g in df[df["pipeline"] != "OracleRemoval"].groupby(["snr_db", "rho", "attack"]):
        ttf = g[g.pipeline == "TTF-CF"].set_index("seed")
        for base in ["MLG+TM", "MAD", "Sn", "Qn", "ParadigmA"]:
            b = g[g.pipeline == base].set_index("seed")
            common = ttf.index.intersection(b.index)
            if len(common) < 2:
                continue
            for metric in ["J_attack", "catch", "ffr", "pd", "pfa_attack"]:
                d = ttf.loc[common, metric].to_numpy(float) - b.loc[common, metric].to_numpy(float)
                d = d[np.isfinite(d)]
                idx = rng_b.integers(0, len(d), size=(5000, len(d)))
                bm = np.mean(d[idx], axis=1)
                boot.append({
                    "snr_db": snr, "rho": rho, "attack": attack,
                    "comparison": f"TTF-CF - {base}", "metric": metric,
                    "mean_diff": float(np.mean(d)),
                    "ci_low": float(np.quantile(bm, 0.025)),
                    "ci_high": float(np.quantile(bm, 0.975)),
                    "n_seeds": len(d),
                })
    pd.DataFrame(boot).to_csv(out / "paired_bootstrap.csv", index=False)


def generate_regime_sequence(c: Config, seed: int):
    env = rng(seed, 400_000)
    h1_rng = rng(seed, 401_000)
    atk_rng = rng(seed, 402_000)
    mal = sm.choose_malicious_idx(c.N, 0.20, seed, 990_001)
    records = []
    session_h = session_g = session_unc = None
    for r in range(c.REGIME_ROUNDS):
        if r % c.SESSION_LENGTH == 0:
            session_h, session_g, session_unc = make_session(c, env)
        phase = next((p for p in c.REGIME_PHASES if p[0] <= r < p[1]), c.REGIME_PHASES[-1])
        _, _, attack, magnitude = phase
        T = sm.local_energy_gamma(c.N, c.M_SAMPLES, -12.0, "H1", session_h, session_unc, h1_rng)
        eta = env.normal(0.0, c.REPORT_NOISE_STD, size=c.N)
        T_att = sm.apply_attack(T, mal, attack, atk_rng, magnitude, c.MASK_EXTREME_DB, c.MASK_MILD_DB)
        x = sm.reporting_channel(T_att, session_g, eta)
        records.append({"round": r, "attack": attack, "magnitude_db": magnitude, "x": x})
    return records, mal


def run_regime(c: Config, state: RunState) -> None:
    out = state.out
    # Calibration at -12 dB.
    pa_model = make_pa_model(c, -12.0)
    bank = build_conformal_bank(c, -12.0)
    thresholds = calibrate_thresholds(c, -12.0, pa_model, bank)
    rows = []
    for seed in all_seed_pools(c)["regime"]:
        seq, mal = generate_regime_sequence(c, seed)
        ttf_cfgs = {
            "TTF-CF": TTFConfig(lam=c.TTF_LAMBDA, cusum_k=c.TTF_CUSUM_K, W=c.TTF_W, kappa_sign=c.TTF_KAPPA_SIGN, kappa_p=c.TTF_KAPPA_P, q=c.TTF_Q, rho_max=c.TTF_RHO_MAX, ph_delta=c.TTF_PH_DELTA, ph_threshold=c.TTF_PH_THRESHOLD, ph_beta=c.TTF_PH_BETA, ph_warmup=c.TTF_PH_WARMUP, fusion="mean"),
            "TTF-CF/median": TTFConfig(lam=c.TTF_LAMBDA, cusum_k=c.TTF_CUSUM_K, W=c.TTF_W, kappa_sign=c.TTF_KAPPA_SIGN, kappa_p=c.TTF_KAPPA_P, q=c.TTF_Q, rho_max=c.TTF_RHO_MAX, ph_delta=c.TTF_PH_DELTA, ph_threshold=c.TTF_PH_THRESHOLD, ph_beta=c.TTF_PH_BETA, ph_warmup=c.TTF_PH_WARMUP, fusion="median"),
            "TTF-CF/noCoal": TTFConfig(lam=c.TTF_LAMBDA, cusum_k=c.TTF_CUSUM_K, W=c.TTF_W, kappa_sign=c.TTF_KAPPA_SIGN, kappa_p=c.TTF_KAPPA_P, q=c.TTF_Q, rho_max=c.TTF_RHO_MAX, ph_delta=c.TTF_PH_DELTA, ph_threshold=c.TTF_PH_THRESHOLD, ph_beta=c.TTF_PH_BETA, ph_warmup=c.TTF_PH_WARMUP, fusion="mean", use_coalition=False),
            "TTF-CF/noPH": TTFConfig(lam=c.TTF_LAMBDA, cusum_k=c.TTF_CUSUM_K, W=c.TTF_W, kappa_sign=c.TTF_KAPPA_SIGN, kappa_p=c.TTF_KAPPA_P, q=c.TTF_Q, rho_max=c.TTF_RHO_MAX, ph_delta=c.TTF_PH_DELTA, ph_threshold=1e12, ph_beta=c.TTF_PH_BETA, ph_warmup=c.TTF_PH_WARMUP, fusion="mean"),
        }
        ttfs = {k: TTFCF(c.N, v, bank) for k, v in ttf_cfgs.items()}
        for rrec in seq:
            x = rrec["x"]
            true_set = set(mal.tolist()); nh = c.N - len(true_set)
            for pipe in ["MLG+TM", "MAD"]:
                fl = classical_flag(c, x, pipe)
                fs = set(fl.tolist())
                rows.append({"seed":seed,"round":rrec["round"],"attack":rrec["attack"],"pipeline":pipe,"catch":len(fs & true_set)/len(true_set),"ffr":len(fs-true_set)/nh,"ph_alarm":0})
            for pipe, st in ttfs.items():
                z = st.step(x); fs = set(np.asarray(z["flagged"], dtype=int).tolist())
                rows.append({"seed":seed,"round":rrec["round"],"attack":rrec["attack"],"pipeline":pipe,"catch":len(fs & true_set)/len(true_set),"ffr":len(fs-true_set)/nh,"ph_alarm":int(z["ph_alarm"])})
        # checkpoint each seed
        pd.DataFrame(rows).to_csv(out / "regime_rounds.csv", index=False)

    df = pd.DataFrame(rows)
    phase_rows = []
    for p0,p1,attack,_mag in c.REGIME_PHASES:
        for seed in all_seed_pools(c)["regime"]:
            for pipe in df.pipeline.unique():
                z = df[(df.seed==seed)&(df.pipeline==pipe)&(df["round"]>=p0)&(df["round"]<p1)]
                if z.empty:
                    continue
                last50 = z[z["round"] >= p1-50]
                phase_rows.append({"seed":seed,"phase_start":p0,"phase_end":p1,"attack":attack,"pipeline":pipe,
                                   "catch_mean":float(z.catch.mean()),"ffr_mean":float(z.ffr.mean()),
                                   "catch_final50":float(last50.catch.mean()),"ffr_final50":float(last50.ffr.mean()),
                                   "ph_alarm_count":int(z.ph_alarm.sum())})
    pd.DataFrame(phase_rows).to_csv(out / "regime_phase_seed_level.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all","validate","main","regime"], default="all")
    args = ap.parse_args()
    c = cfg_from_env()
    state = RunState(c)
    # Source hashes make reruns traceable.
    root = Path(__file__).resolve().parent
    state.manifest["source_hashes"] = {p.name: sha256_file(p) for p in root.glob("*.py")}
    write_json(state.out / "run_manifest.json", state.manifest)

    if args.stage in {"all", "validate"}:
        # Warm deterministic TM cache and write self-test marker.
        sm.warm_tm_cache(c.N, c.TM_ALPHA, c.TM_N_SIM)
        sm.save_tm_cache(state.out / c.TM_CACHE_FILE)
        print("Validation prerequisites passed: deterministic model imports and TM cache created.")
        if args.stage == "validate":
            return
    if args.stage in {"all", "main"}:
        run_main(c, state)
    if args.stage in {"all", "regime"}:
        run_regime(c, state)
    write_json(state.out / "run_manifest.json", state.manifest)
    print(f"Completed. Results: {state.out.resolve()}")


if __name__ == "__main__":
    main()
