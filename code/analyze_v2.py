"""Post-processing for the instrumented XS V2 study.

Creates:
  v2_paired_tests.csv.gz        paired differences, Wins/Losses/Ties, raw/Holm p, power
  v2_roc_seed_level.csv.gz      AUC and detection at a calibration-matched PFA target
  v2_boundary_metrics.csv.gz    PH/session boundary precision, recall and timing error
  v2_displacement_aggregate.csv.gz  summary of the per-round displacement decomposition

Usage:
  python analyze_v2.py --results v2_B1
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nct, t as student_t, wilcoxon
from sklearn.metrics import roc_auc_score


def append_holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    if not m:
        return []
    order = np.argsort(p)
    adj = np.empty(m, float)
    running = 0.0
    for rank0, idx in enumerate(order):
        rank = rank0 + 1
        value = min(1.0, (m - rank + 1) * p[idx])
        running = max(running, value)
        adj[idx] = running
    return adj.tolist()


def paired_bootstrap_ci(d, n_boot=10000, seed=20261109):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def paired_power(d, delta=0.05, alpha=0.05):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return np.nan
    sd = float(np.std(d, ddof=1))
    if sd <= 1e-15:
        return 1.0 if abs(delta) > 0 else alpha
    df = n - 1
    ncp = abs(delta) / (sd / np.sqrt(n))
    crit = student_t.ppf(1 - alpha / 2, df)
    return float(nct.sf(crit, df, ncp) + nct.cdf(-crit, df, ncp))


def select_xs_ph(names):
    exact = [n for n in names if n == "XS-PH"]
    if exact:
        return exact[0]
    cap8 = [n for n in names if n.startswith("XS-PH_cap8")]
    if cap8:
        return cap8[0]
    candidates = [n for n in names if n.startswith("XS-PH")]
    return candidates[0] if candidates else None


def analyze_paired(seed_df, out):
    """Paired inference for RQ1, PH-vs-fixed-K (B1), and secondary baselines.

    Holm families are defined separately for each experimental condition and
    each XS-PH cap variant. The primary RQ1 family contains four robust-fusion
    contrasts at each primary SNR (eight contrasts for -9/-6 dB together).
    """
    primary_baselines = ["Trim10", "Trim20", "Trim40", "Median"]
    secondary_baselines = ["MLG+TM", "MAD", "Sn", "Qn", "ParadigmA", "TTF-CF"]
    group_cols = ["rho_requested", "attack", "extreme_fraction", "session_spec", "persistence_w", "cap_design"]
    results = []

    for gkey, g in seed_df.groupby(group_cols, dropna=False):
        names = sorted(g.pipeline.unique().tolist())
        xs_names = [n for n in names if n.startswith("XS-PH")]
        fixed_names = [n for n in names if n.startswith("XS-fixed")]
        if not xs_names:
            continue

        for xs_name in xs_names:
            primary_rows = []
            secondary_rows = []
            b1_rows = []
            fixed20_rows = []

            def make_test_row(snr, base, metric, family):
                b = g[(g.pipeline == base) & (g.snr_db == snr)].set_index("seed")
                a = g[(g.pipeline == xs_name) & (g.snr_db == snr)].set_index("seed")
                if b.empty or a.empty:
                    return None
                common = sorted(set(a.index) & set(b.index))
                if len(common) < 2:
                    return None
                d = a.loc[common, metric].to_numpy(float) - b.loc[common, metric].to_numpy(float)
                d = d[np.isfinite(d)]
                if len(d) < 2:
                    return None
                wins = int(np.sum(d > 1e-12)); losses = int(np.sum(d < -1e-12)); ties = int(np.sum(np.abs(d) <= 1e-12))
                try:
                    if np.all(np.abs(d) <= 1e-12):
                        p = 1.0
                    else:
                        p = float(wilcoxon(d, zero_method="pratt", alternative="two-sided", method="auto").pvalue)
                        if not np.isfinite(p):
                            p = 1.0
                except (ValueError, FloatingPointError):
                    p = 1.0
                lo, hi = paired_bootstrap_ci(d, seed=20261109 + int(abs(snr) * 10) + len(results) + len(primary_rows) + len(secondary_rows) + len(b1_rows))
                return {
                    **dict(zip(group_cols, gkey if isinstance(gkey, tuple) else (gkey,))),
                    "family": family, "snr_db": float(snr), "xs_pipeline": xs_name,
                    "baseline": base, "metric": metric,
                    "mean_difference": float(np.mean(d)), "median_difference": float(np.median(d)),
                    "ci95_low": lo, "ci95_high": hi, "wins": wins, "losses": losses,
                    "ties": ties, "p_raw_wilcoxon": p, "n_test_seeds": int(len(d)),
                    "estimated_power_deltaJ_0p05": paired_power(d, delta=0.05) if metric == "J_attack" else np.nan,
                }

            for snr in sorted(g.snr_db.unique()):
                for base in primary_baselines:
                    row = make_test_row(snr, base, "J_attack", "primary_RQ1")
                    if row is not None:
                        primary_rows.append(row)
                for base in secondary_baselines:
                    for metric in ("J_attack", "catch_all", "ffr_attack"):
                        row = make_test_row(snr, base, metric, "secondary")
                        if row is not None:
                            secondary_rows.append(row)
                if "_cap" in xs_name:
                    cap_suffix = xs_name.split("_cap", 1)[1]
                    fixed_for_cap = [n for n in fixed_names if n.endswith(f"_cap{cap_suffix}")]
                else:
                    fixed_for_cap = fixed_names
                for base in fixed_for_cap:
                    row = make_test_row(snr, base, "J_attack", "B1_PH_vs_fixedK")
                    if row is not None:
                        b1_rows.append(row)

                fixed20_name = next((n for n in fixed_for_cap if n.startswith("XS-fixed20")), None)
                if fixed20_name is not None:
                    a20 = g[(g.pipeline == fixed20_name) & (g.snr_db == snr)].set_index("seed")
                    for other_name in fixed_for_cap:
                        if other_name == fixed20_name:
                            continue
                        bK = g[(g.pipeline == other_name) & (g.snr_db == snr)].set_index("seed")
                        common = sorted(set(a20.index) & set(bK.index))
                        if len(common) < 2:
                            continue
                        d = a20.loc[common, "J_attack"].to_numpy(float) - bK.loc[common, "J_attack"].to_numpy(float)
                        d = d[np.isfinite(d)]
                        if len(d) < 2:
                            continue
                        wins = int(np.sum(d > 1e-12)); losses = int(np.sum(d < -1e-12)); ties = int(np.sum(np.abs(d) <= 1e-12))
                        if np.all(np.abs(d) <= 1e-12):
                            p = 1.0
                        else:
                            try:
                                p = float(wilcoxon(d, zero_method="pratt", alternative="two-sided", method="auto").pvalue)
                                if not np.isfinite(p): p = 1.0
                            except (ValueError, FloatingPointError):
                                p = 1.0
                        lo, hi = paired_bootstrap_ci(d, seed=20261109 + int(abs(snr) * 10) + len(results) + len(fixed20_rows))
                        fixed20_rows.append({
                            **dict(zip(group_cols, gkey if isinstance(gkey, tuple) else (gkey,))),
                            "family": "B1_fixed20_vs_fixedK", "snr_db": float(snr),
                            "xs_pipeline": fixed20_name, "baseline": other_name, "metric": "J_attack",
                            "mean_difference": float(np.mean(d)), "median_difference": float(np.median(d)),
                            "ci95_low": lo, "ci95_high": hi, "wins": wins, "losses": losses, "ties": ties,
                            "p_raw_wilcoxon": p, "n_test_seeds": int(len(d)),
                            "estimated_power_deltaJ_0p05": paired_power(d, delta=0.05)})

            # Holm across both primary SNRs: 4 baselines x 2 SNRs = 8 tests.
            for family_rows in (primary_rows, secondary_rows, b1_rows, fixed20_rows):
                if family_rows:
                    adjusted = append_holm([r["p_raw_wilcoxon"] for r in family_rows])
                    for r, p_adj in zip(family_rows, adjusted):
                        r["p_holm_family"] = p_adj
                        results.append(r)

    columns = ["rho_requested", "attack", "extreme_fraction", "session_spec", "persistence_w", "cap_design", "family", "snr_db",
               "xs_pipeline", "baseline", "metric", "mean_difference", "median_difference", "ci95_low",
               "ci95_high", "wins", "losses", "ties", "p_raw_wilcoxon", "p_holm_family", "n_test_seeds",
               "estimated_power_deltaJ_0p05"]
    pd.DataFrame(results, columns=columns).to_csv(out / "v2_paired_tests.csv.gz", index=False, compression="gzip")


def analyze_roc(seed_df, round_df, cal_df, out, target_pfa=0.05):
    results = []
    key_cols = ["seed", "snr_db", "rho_requested", "attack", "extreme_fraction", "session_spec", "persistence_w", "pipeline"]
    for key, g in round_df.groupby(key_cols, dropna=False):
        d = dict(zip(key_cols, key if isinstance(key, tuple) else (key,)))
        h0c = g[g.sequence_type == "H0_clean"].fused_statistic.to_numpy(float)
        h1c = g[g.sequence_type == "H1_clean"].fused_statistic.to_numpy(float)
        h0a = g[g.sequence_type == "H0_att"].fused_statistic.to_numpy(float)
        h1a = g[g.sequence_type == "H1_att"].fused_statistic.to_numpy(float)
        scores = np.concatenate([h0c, h1c])
        labels = np.concatenate([np.zeros(len(h0c), int), np.ones(len(h1c), int)])
        auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else np.nan
        cal = cal_df[(cal_df.snr_db == d["snr_db"]) &
                     (cal_df.session_spec == d["session_spec"]) &
                     (cal_df.persistence_w == d["persistence_w"]) &
                     (cal_df.pipeline == d["pipeline"])]
        if d["pipeline"] == "OracleRemoval":
            rho_cal = 0.0 if d["attack"] == "none" else float(d["rho_requested"])
            cal = cal[cal.rho_oracle_removed == rho_cal]
        else:
            cal = cal[cal.rho_oracle_removed.isna()]
        vals = cal.calibration_statistic.to_numpy(float)
        if len(vals):
            tau05 = float(np.quantile(vals, 1 - target_pfa))
            pd_clean = float(np.mean(h1c > tau05)); pfa_clean = float(np.mean(h0c > tau05))
            pd_attack = float(np.mean(h1a > tau05)); pfa_attack = float(np.mean(h0a > tau05))
        else:
            tau05 = pd_clean = pfa_clean = pd_attack = pfa_attack = np.nan
        results.append({**d, "auc_clean": auc, "calibration_pfa_target": target_pfa,
                        "matched_pfa_threshold": tau05, "pd_clean_at_matched_pfa": pd_clean,
                        "achieved_pfa_clean_test": pfa_clean,
                        "pd_attack_at_clean_matched_pfa": pd_attack,
                        "pfa_attack_at_clean_matched_pfa": pfa_attack,
                        "n_calibration_scores": len(vals)})
    pd.DataFrame(results).to_csv(out / "v2_roc_seed_level.csv.gz", index=False, compression="gzip")




def analyze_threshold_transfer(round_df, cal_df, out, quantile=0.90):
    """Compare clean-H0 test PFA under w=0 vs w-matched calibration thresholds.

    This diagnostic helps identify whether NoDefense's persistence inflation is
    primarily a threshold-transfer problem or remains after scenario-matched tau.
    """
    keys = ["seed", "snr_db", "rho_requested", "attack", "session_spec", "persistence_w", "pipeline"]
    rows = []
    for key, g in round_df.groupby(keys, dropna=False):
        d = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        h0 = g[g.sequence_type == "H0_clean"].fused_statistic.to_numpy(float)
        if not len(h0):
            continue
        cal = cal_df[(cal_df.snr_db == d["snr_db"]) &
                     (cal_df.session_spec == d["session_spec"]) &
                     (cal_df.pipeline == d["pipeline"])]
        if d["pipeline"] == "OracleRemoval":
            rho_cal = 0.0 if d["attack"] == "none" else float(d["rho_requested"])
            cal = cal[cal.rho_oracle_removed == rho_cal]
        else:
            cal = cal[cal.rho_oracle_removed.isna()]
        cal0 = cal[cal.persistence_w == 0.0].calibration_statistic.to_numpy(float)
        calw = cal[cal.persistence_w == d["persistence_w"]].calibration_statistic.to_numpy(float)
        tau0 = float(np.quantile(cal0, quantile)) if len(cal0) else np.nan
        tauw = float(np.quantile(calw, quantile)) if len(calw) else np.nan
        pfa0 = float(np.mean(h0 > tau0)) if np.isfinite(tau0) else np.nan
        pfaw = float(np.mean(h0 > tauw)) if np.isfinite(tauw) else np.nan
        rows.append({**d, "tau_quantile": quantile, "tau_w0": tau0, "tau_w_matched": tauw,
                     "tau_shift_w_minus_0": tauw - tau0 if np.isfinite(tau0) and np.isfinite(tauw) else np.nan,
                     "pfa_clean_test_using_w0_tau": pfa0,
                     "pfa_clean_test_using_w_matched_tau": pfaw,
                     "n_h0_clean_test_rounds": int(len(h0)),
                     "n_tau_cal_scores_w0": int(len(cal0)),
                     "n_tau_cal_scores_w_matched": int(len(calw))})
    pd.DataFrame(rows).to_csv(out / "v2_threshold_transfer_diagnostics.csv.gz", index=False, compression="gzip")


def analyze_boundaries(session_df, block_df, out, tolerance=2):
    session_keys = ["seed", "snr_db", "rho_requested", "attack", "extreme_fraction", "session_spec", "persistence_w", "cap_design"]
    detect_keys = session_keys + ["sequence_type", "pipeline"]
    true_groups = {k: g for k, g in session_df.groupby(session_keys, dropna=False)}
    ph_all = block_df[block_df.pipeline.str.startswith("XS-PH")]
    results = []
    for key, g_all in ph_all.groupby(detect_keys, dropna=False):
        g = g_all[(g_all.boundary_reason == "page_hinkley") & g_all.boundary_round.notna()]
        skey = key[:len(session_keys)]
        tg = true_groups.get(skey)
        true_bounds = sorted([int(x) for x in tg.start_round.tolist() if int(x) > 0]) if tg is not None else []
        detected = sorted(g.boundary_round.astype(int).tolist())
        unmatched = set(true_bounds)
        errors = []
        tp = 0
        for det in detected:
            candidates = [x for x in unmatched if abs(x - det) <= tolerance]
            if candidates:
                hit = min(candidates, key=lambda x: (abs(x - det), x))
                unmatched.remove(hit); tp += 1; errors.append(det - hit)
        precision = tp / len(detected) if detected else (1.0 if not true_bounds else 0.0)
        recall = tp / len(true_bounds) if true_bounds else np.nan
        results.append({**dict(zip(detect_keys, key if isinstance(key, tuple) else (key,))),
                        "n_true_boundaries": len(true_bounds), "n_ph_boundaries": len(detected),
                        "matched_boundaries": tp, "tolerance_rounds": tolerance,
                        "precision": precision, "recall": recall,
                        "median_signed_timing_error": float(np.median(errors)) if errors else np.nan,
                        "median_absolute_timing_error": float(np.median(np.abs(errors))) if errors else np.nan})
    pd.DataFrame(results).to_csv(out / "v2_boundary_metrics.csv.gz", index=False, compression="gzip")




def analyze_attacker_ranks(obs_df, out):
    """Summarize raw-report rank distributions before any pipeline exclusion."""
    keys = ["snr_db", "rho_requested", "attack", "extreme_fraction", "session_spec",
            "persistence_w", "sequence_type", "attacker_subtype"]
    rows = []
    for key, g in obs_df.groupby(keys, dropna=False):
        rd = g["rank_descending"].to_numpy(float)
        ra = g["rank_ascending"].to_numpy(float)
        N = int(g["user_id"].nunique())
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                     "n_user_round_observations": int(len(g)),
                     "rank_desc_mean": float(np.nanmean(rd)),
                     "rank_desc_median": float(np.nanmedian(rd)),
                     "rank_desc_q05": float(np.nanquantile(rd, 0.05)),
                     "rank_desc_q25": float(np.nanquantile(rd, 0.25)),
                     "rank_desc_q75": float(np.nanquantile(rd, 0.75)),
                     "rank_desc_q95": float(np.nanquantile(rd, 0.95)),
                     "top_decile_fraction": float(np.nanmean(rd <= max(1.0, 0.10 * N))),
                     "bottom_decile_fraction": float(np.nanmean(ra <= max(1.0, 0.10 * N))),
                     "rank_definition": "raw reports ranked within each round before any exclusion; descending rank 1 is largest"})
    pd.DataFrame(rows).to_csv(out / "v2_attacker_rank_distribution.csv.gz", index=False, compression="gzip")


def analyze_displacement(displacement_df, out):
    keys = ["snr_db", "rho_requested", "attack", "extreme_fraction", "session_spec", "persistence_w", "pipeline"]
    metrics = ["total_shift", "direct_effect_extreme_retained", "direct_effect_mild_retained",
               "direct_effect_total", "selection_effect_mild", "selection_effect_extreme",
               "selection_effect_high_honest", "selection_effect_other", "selection_effect_total",
               "decomposition_error"]
    outdf = displacement_df.groupby(keys, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    outdf.columns = ["_".join([str(v) for v in col if str(v) != ""]).rstrip("_") if isinstance(col, tuple) else col for col in outdf.columns]
    outdf.to_csv(out / "v2_displacement_aggregate.csv.gz", index=False, compression="gzip")


def analyze_calibration_precision(cal_df, out, quantile=0.90, n_boot=1000, seed=20261109):
    """Cluster bootstrap calibration trajectories (seeds), never individual rounds."""
    keys = ["snr_db", "session_spec", "persistence_w", "pipeline", "rho_oracle_removed"]
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in cal_df.groupby(keys, dropna=False):
        pivot = g.pivot_table(index="calibration_seed", columns="round", values="calibration_statistic", aggfunc="first")
        matrix = pivot.to_numpy(float)
        matrix = matrix[np.isfinite(matrix).any(axis=1)]
        if matrix.size == 0:
            continue
        n_seeds = matrix.shape[0]
        observed = float(np.nanquantile(matrix, quantile))
        boot = np.empty(n_boot, float)
        for b in range(n_boot):
            ix = rng.integers(0, n_seeds, size=n_seeds)
            boot[b] = float(np.nanquantile(matrix[ix, :], quantile))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                     "quantile": quantile, "threshold_estimate": observed,
                     "cluster_bootstrap_ci95_low": float(lo), "cluster_bootstrap_ci95_high": float(hi),
                     "cluster_bootstrap_sd": float(np.std(boot, ddof=1)),
                     "n_calibration_seeds": int(n_seeds),
                     "rounds_per_seed": int(matrix.shape[1]),
                     "total_rounds_recorded": int(np.isfinite(matrix).sum()),
                     "bootstrap_replicates": int(n_boot)})
    pd.DataFrame(rows).to_csv(out / "v2_calibration_precision.csv.gz", index=False, compression="gzip")


def analyze_condition_intervals(seed_df, out):
    """Mean and 95% t intervals across independent test seeds for B8/B9 metrics."""
    from scipy.stats import t as student_t
    keys = ["snr_db", "rho_requested", "rho_actual", "attack", "extreme_fraction",
            "session_spec", "persistence_w", "pipeline", "cap_design"]
    metrics = ["pd_attack", "pfa_attack", "J_attack", "pd_clean", "pfa_clean", "J_clean",
               "ffr_attack", "ffr_clean", "catch_all", "catch_extreme", "catch_mild"]
    rows = []
    for key, g in seed_df.groupby(keys, dropna=False):
        base = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row = {**base, "n_test_seeds": int(len(g))}
        for metric in metrics:
            x = g[metric].to_numpy(float)
            x = x[np.isfinite(x)]
            n = len(x)
            mean = float(np.mean(x)) if n else np.nan
            sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
            half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = mean - half if n > 1 else np.nan
            row[f"{metric}_ci95_high"] = mean + half if n > 1 else np.nan
            row[f"{metric}_n"] = n
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "v2_condition_summary_ci.csv.gz", index=False, compression="gzip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--matched-pfa", type=float, default=0.05)
    ap.add_argument("--boundary-tolerance", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.results)
    seed_df = pd.read_csv(out / "v2_seed_level.csv.gz")
    round_df = pd.read_csv(out / "v2_pipeline_round.csv.gz")
    cal_df = pd.read_csv(out / "v2_calibration_scores.csv.gz")
    analyze_paired(seed_df, out)
    analyze_calibration_precision(cal_df, out)
    analyze_condition_intervals(seed_df, out)
    analyze_roc(seed_df, round_df, cal_df, out, a.matched_pfa)
    manifest_path = out / "v2_manifest.json"
    tau_quantile = 0.90
    if manifest_path.exists():
        try:
            tau_quantile = float(json.loads(manifest_path.read_text(encoding="utf-8"))["arguments"].get("tau_quantile", 0.90))
        except Exception:
            pass
    analyze_threshold_transfer(round_df, cal_df, out, tau_quantile)
    session_df = pd.read_csv(out / "v2_session_boundaries.csv.gz")
    block_df = pd.read_csv(out / "v2_block_boundaries.csv.gz")
    analyze_boundaries(session_df, block_df, out, a.boundary_tolerance)
    disp_df = pd.read_csv(out / "v2_displacement.csv.gz")
    analyze_displacement(disp_df, out)
    obs_path = out / "v2_user_observations.csv.gz"
    if obs_path.exists():
        analyze_attacker_ranks(pd.read_csv(obs_path), out)
    print("V2 post-analysis completed:", out.resolve())


if __name__ == "__main__":
    main()
