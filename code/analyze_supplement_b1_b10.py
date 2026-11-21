#!/usr/bin/env python3
"""Analyze the B2/B5/B6/B7/B8/B9/B10 supplemental runs.

All supplemental analyses are reported as supplementary/reviewer-requested.
The script never edits B1-V3 outputs. Run on a root containing the named run
subdirectories created by run_supplement_b1_b10.py.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
from sklearn.metrics import roc_auc_score, roc_curve

METRICS = [
    "pd_attack", "pfa_attack", "J_attack", "pd_clean", "pfa_clean", "J_clean",
    "auc_clean", "auc_attack", "ffr_attack", "ffr_clean", "catch_all",
    "catch_extreme", "catch_mild", "mean_flagged_fraction_clean",
]


def read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path, compression="gzip" if path.suffix == ".gz" else "infer")


def mean_ci(x: Iterable[float]) -> dict:
    a = pd.to_numeric(pd.Series(list(x)), errors="coerce").dropna().to_numpy(float)
    n = len(a)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    m = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if n > 1 else np.nan
    half = 1.96 * sd / math.sqrt(n) if n > 1 else np.nan
    return {"n": n, "mean": m, "sd": sd,
            "ci95_low": m - half if n > 1 else np.nan,
            "ci95_high": m + half if n > 1 else np.nan}


def summarize(df: pd.DataFrame, keys: list[str], metrics: list[str] = METRICS) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keys = [k for k in keys if k in df.columns]
    metrics = [m for m in metrics if m in df.columns]
    rows = []
    for vals, g in df.groupby(keys, dropna=False, sort=True):
        if not isinstance(vals, tuple):
            vals = (vals,)
        row = dict(zip(keys, vals))
        for metric in metrics:
            s = mean_ci(g[metric])
            for k, v in s.items():
                row[f"{metric}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(df: pd.DataFrame, metric: str, reference: str = "XS-PH") -> pd.DataFrame:
    if df.empty or metric not in df.columns or "pipeline" not in df.columns:
        return pd.DataFrame()
    key_candidates = ["seed", "snr_db", "rho_requested", "attack", "extreme_fraction", "persistence_w"]
    keys = [k for k in key_candidates if k in df.columns]
    piv = df.pivot_table(index=keys, columns="pipeline", values=metric, aggfunc="first")
    if reference not in piv.columns:
        return pd.DataFrame()
    rows = []
    for baseline in [c for c in piv.columns if c != reference]:
        comp = piv[[reference, baseline]].dropna()
        d = (comp[reference] - comp[baseline]).to_numpy(float)
        n = len(d)
        if n == 0:
            continue
        s = mean_ci(d)
        if np.allclose(d, 0):
            p = 1.0
        else:
            try:
                p = float(wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="approx").pvalue)
            except TypeError:
                p = float(wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
        row = {"metric": metric, "reference": reference, "baseline": baseline,
               "n_pairs": n, "mean_difference_reference_minus_baseline": s["mean"],
               "ci95_low": s["ci95_low"], "ci95_high": s["ci95_high"],
               "wins": int(np.sum(d > 0)), "losses": int(np.sum(d < 0)),
               "ties": int(np.sum(d == 0)), "p_wilcoxon_raw": p}
        # Store conditions to make the comparisons auditable when more than one is present.
        for k in ["snr_db", "rho_requested", "attack", "extreme_fraction", "persistence_w"]:
            if k in df.columns and df[k].nunique(dropna=False) == 1:
                row[k] = df[k].iloc[0]
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        # Holm correction within each metric x condition table.
        order = np.argsort(out["p_wilcoxon_raw"].to_numpy())
        p = out["p_wilcoxon_raw"].to_numpy()
        adj = np.empty(len(p), float)
        running = 0.0
        m = len(p)
        for rank, idx in enumerate(order):
            val = min(1.0, (m - rank) * p[idx])
            running = max(running, val)
            adj[idx] = running
        out["p_holm_within_exported_comparison_set"] = adj
    return out


def b5_telemetry(telemetry_dir: Path, outdir: Path) -> dict:
    obs = read_csv(telemetry_dir / "v2_user_observations.csv.gz")
    dec = read_csv(telemetry_dir / "v2_user_decisions.csv.gz")
    disp = read_csv(telemetry_dir / "v2_displacement.csv.gz")
    result = {}
    if not obs.empty:
        sub = obs[(obs["attack"] == "masking") & obs["sequence_type"].isin(["H1_att", "H0_att"]) &
                  (obs["attacker_subtype"].isin(["extreme", "mild"]))].copy()
        keys = [k for k in ["sequence_type", "snr_db", "rho_requested", "extreme_fraction", "attacker_subtype"] if k in sub.columns]
        rows = []
        for vals, g in sub.groupby(keys, dropna=False):
            if not isinstance(vals, tuple): vals = (vals,)
            ranks = pd.to_numeric(g["rank_descending"], errors="coerce").dropna()
            row = dict(zip(keys, vals))
            row.update({"n_user_round_observations": len(ranks),
                        "mean_rank_descending": float(ranks.mean()) if len(ranks) else np.nan,
                        "median_rank_descending": float(ranks.median()) if len(ranks) else np.nan,
                        "fraction_top_quartile_rank_le_5": float((ranks <= 5).mean()) if len(ranks) else np.nan,
                        "fraction_bottom_quartile_rank_ge_16": float((ranks >= 16).mean()) if len(ranks) else np.nan})
            rows.append(row)
        result["B5_attacker_rank_summary.csv"] = pd.DataFrame(rows)
    if not dec.empty:
        sub = dec[(dec["attack"] == "masking") & (dec["sequence_type"] == "H1_att") &
                  (dec["user_type"] == "malicious")].copy()
        keys = [k for k in ["pipeline", "snr_db", "rho_requested", "extreme_fraction", "attacker_subtype"] if k in sub.columns]
        result["B5_catch_by_subtype.csv"] = sub.groupby(keys, dropna=False).agg(
            n_user_rounds=("excluded", "size"), catch_rate=("excluded", "mean"),
            mean_rank_descending=("rank_descending", "mean"),
            median_residual=("residual", "median")).reset_index()
        # Rank-conditioned screening performance, preserving attacker subtype.
        sub["rank_band"] = pd.cut(sub["rank_descending"],
                                  bins=[0, 5, 10, 15, 20.0001],
                                  labels=["top5", "6to10", "11to15", "16to20"],
                                  include_lowest=True)
        result["B5_catch_by_rank_band.csv"] = sub.groupby(
            [k for k in ["pipeline", "snr_db", "rho_requested", "extreme_fraction", "attacker_subtype", "rank_band"] if k in sub.columns],
            dropna=False, observed=False).agg(n=("excluded", "size"), catch_rate=("excluded", "mean")).reset_index()
    if not disp.empty:
        keys = [k for k in ["pipeline", "snr_db", "rho_requested", "extreme_fraction"] if k in disp.columns]
        metrics = [k for k in ["total_shift", "direct_effect_extreme_retained", "direct_effect_mild_retained",
                               "direct_effect_total", "selection_effect_mild", "selection_effect_extreme",
                               "selection_effect_high_honest", "selection_effect_other", "selection_effect_total",
                               "decomposition_error", "n_retained_attacked", "n_retained_clean"] if k in disp.columns]
        agg = disp.groupby(keys, dropna=False)[metrics].agg(["mean", "std", "max"]).reset_index()
        # Flatten pandas' hierarchical aggregation columns.
        agg.columns = ["_".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else str(c) for c in agg.columns]
        result["B5_displacement_summary.csv"] = agg
        err = pd.to_numeric(disp.get("decomposition_error", pd.Series(dtype=float)), errors="coerce").abs()
        result["B5_displacement_integrity.json"] = pd.DataFrame([{
            "n_rows": int(len(disp)), "max_abs_decomposition_error": float(err.max()) if len(err) else None,
            "mean_abs_decomposition_error": float(err.mean()) if len(err) else None,
            "all_decomposition_errors_within_1e-8": bool((err <= 1e-8).all()) if len(err) else None,
        }])
    for name, frame in result.items():
        if name.endswith(".json"):
            frame.iloc[0].to_json(outdir / name, orient="index", indent=2)
        else:
            frame.to_csv(outdir / name, index=False)
    return {"observations_rows": int(len(obs)), "decision_rows": int(len(dec)), "displacement_rows": int(len(disp)),
            "outputs": list(result)}


def roc_per_seed(rounds: pd.DataFrame) -> pd.DataFrame:
    if rounds.empty:
        return pd.DataFrame()
    req = {"seed", "pipeline", "sequence_type", "fused_statistic", "attack", "snr_db", "rho_requested"}
    if not req.issubset(rounds.columns):
        return pd.DataFrame()
    sub = rounds[rounds["attack"] == "masking"].copy()
    group_cols = [k for k in ["seed", "snr_db", "rho_requested", "extreme_fraction", "persistence_w", "pipeline"] if k in sub.columns]
    rows = []
    for vals, g in sub.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(vals, tuple): vals = (vals,)
        meta = dict(zip(group_cols, vals))
        g0 = g[g["sequence_type"] == "H0_att"]["fused_statistic"].to_numpy(float)
        g1 = g[g["sequence_type"] == "H1_att"]["fused_statistic"].to_numpy(float)
        if len(g0) == 0 or len(g1) == 0 or not np.isfinite(g0).all() or not np.isfinite(g1).all():
            continue
        y = np.r_[np.zeros(len(g0), int), np.ones(len(g1), int)]
        scores = np.r_[g0, g1]
        try:
            auc = float(roc_auc_score(y, scores))
            fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
            pd_at_05 = float(np.interp(0.05, fpr, tpr))
        except ValueError:
            continue
        rows.append({**meta, "auc_attack_from_round_scores": auc,
                     "pd_at_fpr_0.05_from_test_roc": pd_at_05,
                     "n_h0_rounds": len(g0), "n_h1_rounds": len(g1)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="output root from run_supplement_b1_b10.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    outdir = Path(args.out).resolve() if args.out else root / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    status = {"root": str(root), "outputs": [], "missing_runs": [], "notes": [
        "Supplemental analyses are secondary/reviewer-requested and do not replace or pool with the frozen B1-V3 primary test.",
        "B5 telemetry replay duplicates a subset of B2 split seed observations by design and must not be counted as additional independent seeds.",
        "ROC PD@FPR=0.05 is estimated from each seed's held-out test round scores; it is a threshold-free test-ROC interpolation, not a preselected operating-point test.",
        "BDFHistory is an adapted comparator, not an exact reproduction of the full BDF sequential method in Wu et al. (2024)."
    ]}

    for run, output, keys in [
        ("B2_rho_cap", "B2_rho_cap_summary.csv", ["snr_db", "rho_requested", "extreme_fraction", "cap_design", "pipeline"]),
        ("B2_extreme_mild_split", "B2_extreme_mild_summary.csv", ["snr_db", "rho_requested", "extreme_fraction", "cap_design", "pipeline"]),
    ]:
        df = read_csv(root / run / "v2_seed_level.csv.gz")
        if df.empty:
            status["missing_runs"].append(run)
            continue
        attacked = df[df["attack"] == "masking"]
        summary = summarize(attacked, keys)
        summary.to_csv(outdir / output, index=False)
        status["outputs"].append(output)

    b5dir = root / "B5_telemetry_replay"
    if (b5dir / "v2_manifest.json").exists():
        status["B5_telemetry"] = b5_telemetry(b5dir, outdir)
        status["outputs"].extend(status["B5_telemetry"]["outputs"])
    else:
        status["missing_runs"].append("B5_telemetry_replay")

    compdir = root / "B6_B7_B8_B9_comparators"
    seed = read_csv(compdir / "v2_seed_level.csv.gz")
    if not seed.empty:
        masking = seed[seed["attack"] == "masking"]
        b6 = summarize(masking, ["snr_db", "rho_requested", "extreme_fraction", "pipeline"])
        b6.to_csv(outdir / "B6_baseline_performance_summary.csv", index=False)
        status["outputs"].append("B6_baseline_performance_summary.csv")
        comparisons = []
        for (snr, rho), g in masking.groupby(["snr_db", "rho_requested"], dropna=False):
            for metric in ["J_attack", "auc_attack"]:
                z = paired_summary(g, metric, reference="XS-PH")
                if not z.empty:
                    z["snr_db"] = snr; z["rho_requested"] = rho
                    comparisons.append(z)
        if comparisons:
            pd.concat(comparisons, ignore_index=True).to_csv(outdir / "B6_paired_baseline_contrasts.csv", index=False)
            status["outputs"].append("B6_paired_baseline_contrasts.csv")

        thresholds = read_csv(compdir / "v2_thresholds.csv.gz")
        subsets = read_csv(compdir / "v2_oracle_calibration_subsets.csv.gz")
        oracle_test = masking[masking["pipeline"] == "OracleRemoval"]
        if not thresholds.empty:
            o = thresholds[thresholds["pipeline"] == "OracleRemoval"].copy()
            if not o.empty:
                o.to_csv(outdir / "B7_oracle_calibration_thresholds.csv", index=False)
                status["outputs"].append("B7_oracle_calibration_thresholds.csv")
        if not subsets.empty:
            check_cols = [k for k in ["snr_db", "session_spec", "persistence_w", "calibration_seed", "rho_oracle_removed"] if k in subsets.columns]
            subcheck = subsets.groupby(check_cols, dropna=False).agg(
                removed_rows=("removed_for_oracle_calibration", "sum"),
                declared_L=("oracle_removed_reports", "first"), n_users=("user_id", "nunique")).reset_index()
            subcheck["exact_L_match"] = subcheck["removed_rows"] == subcheck["declared_L"]
            subcheck.to_csv(outdir / "B7_oracle_calibration_subset_checks.csv", index=False)
            status["outputs"].append("B7_oracle_calibration_subset_checks.csv")
        if not oracle_test.empty:
            summarize(oracle_test, ["snr_db", "rho_requested", "pipeline"],
                      ["pd_attack", "pfa_attack", "J_attack", "pd_clean", "pfa_clean", "J_clean", "ffr_clean", "auc_attack"]).to_csv(
                          outdir / "B7_oracle_test_summary.csv", index=False)
            status["outputs"].append("B7_oracle_test_summary.csv")

        # B8: preserve the separate clean operating-point summaries and a test-ROC matched-PFA analysis.
        rounds = read_csv(compdir / "v2_pipeline_round.csv.gz")
        roc = roc_per_seed(rounds)
        if not roc.empty:
            roc.to_csv(outdir / "B8_roc_matched_pfa_per_seed.csv", index=False)
            roc_keys = [k for k in ["snr_db", "rho_requested", "extreme_fraction", "pipeline"] if k in roc.columns]
            roc_summary = summarize(roc, roc_keys, ["auc_attack_from_round_scores", "pd_at_fpr_0.05_from_test_roc"])
            roc_summary.to_csv(outdir / "B8_roc_matched_pfa_summary.csv", index=False)
            status["outputs"].extend(["B8_roc_matched_pfa_per_seed.csv", "B8_roc_matched_pfa_summary.csv"])
        clean_guard = summarize(seed[seed["attack"] == "masking"],
                                ["snr_db", "rho_requested", "pipeline"],
                                ["pd_clean", "pfa_clean", "J_clean", "ffr_clean"])
        clean_guard.to_csv(outdir / "B8_clean_counterfactual_operating_summary.csv", index=False)
        status["outputs"].append("B8_clean_counterfactual_operating_summary.csv")

        # B9: explicit no-attack arm, distinct from fixed:200/no-change boundary control.
        no_attack = seed[seed["attack"] == "none"]
        if not no_attack.empty:
            summarize(no_attack, ["snr_db", "pipeline"],
                      ["pd_attack", "pfa_attack", "J_attack", "pd_clean", "pfa_clean", "J_clean", "auc_clean", "ffr_clean"]).to_csv(
                          outdir / "B9_attack_none_summary.csv", index=False)
            status["outputs"].append("B9_attack_none_summary.csv")
    else:
        status["missing_runs"].append("B6_B7_B8_B9_comparators")

    b10dir = root / "B10_persistence"
    b10 = read_csv(b10dir / "v2_seed_level.csv.gz")
    if not b10.empty:
        b10mask = b10[b10["attack"] == "masking"]
        summarize(b10mask, ["snr_db", "rho_requested", "persistence_w", "pipeline"],
                  ["pd_attack", "pfa_attack", "J_attack", "pd_clean", "pfa_clean", "J_clean", "auc_attack", "ffr_attack", "ffr_clean"]).to_csv(
                      outdir / "B10_persistence_summary.csv", index=False)
        status["outputs"].append("B10_persistence_summary.csv")
        paired = []
        for (snr, rho, pipeline), g in b10mask.groupby(["snr_db", "rho_requested", "pipeline"], dropna=False):
            for w in [0.5, 0.9]:
                a = g[g["persistence_w"] == w].set_index("seed")
                b = g[g["persistence_w"] == 0.0].set_index("seed")
                common = a.index.intersection(b.index)
                for metric in ["J_attack", "pfa_clean", "ffr_clean", "auc_attack"]:
                    if metric not in a.columns or metric not in b.columns or len(common) == 0:
                        continue
                    d = (a.loc[common, metric] - b.loc[common, metric]).to_numpy(float)
                    s = mean_ci(d)
                    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
                    paired.append({"snr_db": snr, "rho_requested": rho, "pipeline": pipeline,
                                   "persistence_w_vs_zero": w, "metric": metric, "n_pairs": len(d),
                                   "mean_difference": s["mean"], "ci95_low": s["ci95_low"],
                                   "ci95_high": s["ci95_high"], "p_wilcoxon_raw": p})
        if paired:
            pd.DataFrame(paired).to_csv(outdir / "B10_paired_persistence_contrasts.csv", index=False)
            status["outputs"].append("B10_paired_persistence_contrasts.csv")
    else:
        status["missing_runs"].append("B10_persistence")

    (outdir / "SUPPLEMENT_ANALYSIS_STATUS.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    (outdir / "SUPPLEMENT_ANALYSIS_NOTES.md").write_text(
        "# Supplemental analysis notes\n\n"
        "These files are intended to address reviewer-requested secondary analyses. They do not alter or replace the frozen B1-V3 primary experiment. "
        "B5 telemetry replays a subset of B2 split seeds; it is used for mechanism/identity analysis only and is not pooled as extra independent sample size. "
        "The matched-PFA diagnostic interpolates held-out test ROC curves at FPR=0.05; operating-point PFA values from the independently calibrated threshold are reported separately. "
        "BDFHistory is a transparent adaptation inspired by the beta-reputation component of the cited BDF paper, not a faithful reproduction of its full sequential procedure.\n",
        encoding="utf-8")
    print(json.dumps(status, indent=2, default=str))

if __name__ == "__main__":
    main()
