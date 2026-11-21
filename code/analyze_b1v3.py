"""Reproducible B1-V3 analysis: integrity, boundary timing, paired tests, PFA/ROC.

Seed/trajectory is the independent unit. Rounds are used to calculate within-seed
PD/PFA/AUC only and are never counted as independent test replicates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t, wilcoxon, nct
from sklearn.metrics import roc_auc_score

RANDOM_SPEC = "random_balanced:13,20,30,50"


def mean_ci(x, alpha=0.05):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(np.mean(x))
    if n == 1:
        return m, np.nan, np.nan, n
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    h = float(student_t.ppf(1 - alpha / 2, n - 1) * se)
    return m, m - h, m + h, n


def paired_bootstrap_ci(d, n_boot=20000, seed=20261109):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, float)
    for i in range(n_boot):
        out[i] = np.mean(d[rng.integers(0, len(d), size=len(d))])
    return tuple(map(float, np.quantile(out, [0.025, 0.975])))


def paired_permutation_p(d, n_perm=100000, seed=20261109, chunk=2000):
    """Two-sided paired sign-flip randomization test; test seed is the unit."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return np.nan
    observed = abs(float(np.mean(d)))
    rng = np.random.default_rng(seed)
    extreme = 0
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
        stats = np.abs(np.mean(signs * d[None, :], axis=1))
        extreme += int(np.sum(stats >= observed - 1e-15))
        done += b
    return float((extreme + 1) / (n_perm + 1))


def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        out[idx] = running
    return out


def chronological_boundary_match(true_bounds, detected, max_delay):
    """One-to-one matching; a detection can match only a true boundary at or before it.

    A detection is eligible for a true boundary b only when b <= d <= b+max_delay.
    The assignment maximizes matches first and minimizes total delay second.
    """
    true_bounds = np.asarray(sorted(set(int(x) for x in true_bounds)), int)
    detected = np.asarray(sorted(int(x) for x in detected), int)
    nt, nd = len(true_bounds), len(detected)
    if nt == 0 or nd == 0:
        return [], list(range(nd)), list(range(nt))
    # Each true boundary gets nd detection columns plus nt private dummy columns.
    dummy_cost = float(max_delay + 1)
    invalid_cost = float((max_delay + 1) * (nt + nd + 10))
    cost = np.full((nt, nd + nt), invalid_cost, float)
    for i, b in enumerate(true_bounds):
        for j, d in enumerate(detected):
            delay = int(d - b)
            if 0 <= delay <= max_delay:
                cost[i, j] = delay
        cost[i, nd + i] = dummy_cost
    rr, cc = linear_sum_assignment(cost)
    matches = []
    used_det = set()
    used_true = set()
    for i, j in zip(rr, cc):
        if j < nd and cost[i, j] < invalid_cost:
            matches.append((int(true_bounds[i]), int(detected[j]), int(detected[j] - true_bounds[i])))
            used_det.add(int(j)); used_true.add(int(i))
    fp = [j for j in range(nd) if j not in used_det]
    fn = [i for i in range(nt) if i not in used_true]
    return matches, fp, fn


def load_inputs(root: Path):
    required = ["v2_seed_level.csv.gz", "v2_session_boundaries.csv.gz",
                "v2_block_boundaries.csv.gz", "v2_manifest.json"]
    missing = [x for x in required if not (root / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {root}: {missing}")
    seed = pd.read_csv(root / required[0])
    sessions = pd.read_csv(root / required[1])
    blocks = pd.read_csv(root / required[2])
    manifest = json.loads((root / required[3]).read_text(encoding="utf-8"))
    identity_path = root / "v3_identity_ledger.csv.gz"
    identities = pd.read_csv(identity_path) if identity_path.exists() else None
    return seed, sessions, blocks, manifest, identities

def integrity_checks(seed, sessions, blocks, manifest, identities, out: Path):
    problems = []
    key = ["seed", "snr_db", "rho_requested", "attack", "session_spec", "pipeline"]
    dup = int(seed.duplicated(key).sum())
    if dup:
        problems.append(f"duplicate seed-level pipeline rows: {dup}")
    j_err = float(np.nanmax(np.abs(seed.J_attack - (seed.pd_attack - seed.pfa_attack))))
    if j_err > 1e-10:
        problems.append(f"J_attack identity error: {j_err}")
    # Session intervals must be valid, contiguous, and cover the full sequence.
    sess_keys = ["seed", "snr_db", "session_spec"]
    session_rows = []
    for k, g in sessions.groupby(sess_keys, dropna=False):
        g = g.sort_values("start_round")
        starts = g.start_round.astype(int).tolist()
        ends = g.end_round.astype(int).tolist()
        lens = g.length.astype(int).tolist()
        expected_start = 0
        ok = True
        for st, en, le in zip(starts, ends, lens):
            ok &= (st == expected_start and en >= st and en - st + 1 == le)
            expected_start = en + 1
        rounds_n = int(g.rounds.iloc[0])
        ok &= expected_start == rounds_n
        session_rows.append({"seed": k[0], "snr_db": k[1], "session_spec": k[2],
                             "n_sessions": len(g), "rounds": rounds_n,
                             "schedule_lengths": json.dumps(lens), "valid": bool(ok)})
        if not ok:
            problems.append(f"invalid session schedule for {k}")
        if k[2] == RANDOM_SPEC and rounds_n == 200:
            nominal_counts = [sum(int(x) == v for x in lens) for v in [13, 20, 30, 50]]
            if min(nominal_counts) < 1 or max(nominal_counts) - min(nominal_counts) > 1:
                problems.append(f"random variable session lengths are not count-balanced for {k}: {nominal_counts}")
    pd.DataFrame(session_rows).to_csv(out / "B1V3_session_integrity.csv", index=False)

    # With the B1-V3 common condition id, each test seed's session schedule should
    # be identical across SNRs for the same session specification.
    sd = pd.DataFrame(session_rows)
    schedule_mismatch = 0
    if len(sd):
        schedule_mismatch = int((sd.groupby(["seed", "session_spec"]).schedule_lengths.nunique() > 1).sum())
        if schedule_mismatch:
            problems.append(f"session schedule differs across SNR for {schedule_mismatch} seed/spec groups")

    # Block segments for each XS pipeline/sequence must partition [0, rounds-1] exactly once.
    block_keys = ["seed", "snr_db", "session_spec", "pipeline", "sequence_type"]
    block_integrity_rows = []
    for k, g in blocks.groupby(block_keys, dropna=False):
        g = g.sort_values("start_round")
        starts = g.start_round.astype(int).tolist()
        ends = g.end_round.astype(int).tolist()
        lens = g.length.astype(int).tolist()
        rounds_n = int(g.rounds.iloc[0])
        expected = 0
        ok = True
        for st, en, le in zip(starts, ends, lens):
            ok &= (st == expected and en >= st and en - st + 1 == le)
            expected = en + 1
        ok &= expected == rounds_n
        final_count = int((g.boundary_reason == "final_open_block").sum())
        ok &= final_count == 1
        block_integrity_rows.append({"seed": k[0], "snr_db": k[1], "session_spec": k[2],
                                     "pipeline": k[3], "sequence_type": k[4],
                                     "n_blocks": len(g), "sum_lengths": int(sum(lens)),
                                     "final_open_count": final_count, "valid_partition": bool(ok)})
        if not ok:
            problems.append(f"invalid block partition for {k}")
    pd.DataFrame(block_integrity_rows).to_csv(out / "B1V3_block_integrity.csv", index=False)

    # Seed pools must be disjoint (development/test/calibration separation).
    pools = manifest.get("seed_pools", {})
    pool_names = [x for x in ["score_cal", "tau_cal", "test"] if x in pools]
    overlap_pairs = []
    for i, a in enumerate(pool_names):
        for b in pool_names[i + 1:]:
            inter = sorted(set(pools[a]) & set(pools[b]))
            if inter:
                overlap_pairs.append({"pool_a": a, "pool_b": b, "overlap": inter[:10]})
    if overlap_pairs:
        problems.append("seed pools overlap")

    # Development pools are stored in the frozen final-run metadata. Check that
    # neither final calibration nor final test seeds overlap any development pool.
    run_info_path = out / "B1V3_final_run_info.json"
    if run_info_path.exists():
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        development_ranges = [
            run_info.get("development_seed_range"),
            run_info.get("development_score_calibration_seed_range"),
            run_info.get("development_threshold_calibration_seed_range"),
        ]
        final_ranges = [
            [run_info.get("score_calibration_seed_start"),
             run_info.get("score_calibration_seed_start") + run_info.get("score_calibration_seed_count", 0) - 1],
            [run_info.get("threshold_calibration_seed_start"),
             run_info.get("threshold_calibration_seed_start") + run_info.get("threshold_calibration_seed_count", 0) - 1],
            [run_info.get("final_test_seed_start"),
             run_info.get("final_test_seed_start") + run_info.get("final_test_seed_count", 0) - 1],
        ]
        for fr in final_ranges:
            for dr in development_ranges:
                if fr[0] is not None and fr[1] is not None and dr and max(fr[0], dr[0]) <= min(fr[1], dr[1]):
                    problems.append(f"final seed pool {fr} overlaps development pool {dr}")

    identity_mismatch = 0
    if identities is not None and len(identities):
        ikeys = ["seed", "session_spec", "attack", "rho_requested"]
        for k, g in identities.groupby(ikeys, dropna=False):
            if g.malicious_ids.nunique() > 1 or g.extreme_ids.nunique() > 1 or g.mild_ids.nunique() > 1 or g.session_lengths.nunique() > 1:
                identity_mismatch += 1
        if identity_mismatch:
            problems.append(f"attacker identities/subgroups/session schedules differ across SNR for {identity_mismatch} seed-condition groups")

    report = {
        "seed_level_rows": int(len(seed)),
        "n_test_seeds": int(seed.seed.nunique()),
        "duplicate_seed_pipeline_rows": dup,
        "max_abs_J_identity_error": j_err,
        "session_schedule_groups": int(len(session_rows)),
        "session_schedule_mismatches_across_SNR": schedule_mismatch,
        "block_trajectory_groups": int(len(block_integrity_rows)),
        "invalid_block_partitions": int(sum(not r["valid_partition"] for r in block_integrity_rows)),
        "seed_pool_overlap_pairs": overlap_pairs,
        "identity_or_schedule_mismatch_groups_across_SNR": identity_mismatch,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    (out / "B1V3_integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        raise RuntimeError("B1-V3 integrity checks failed; inspect B1V3_integrity_report.json")
    return report


def performance_summary(seed: pd.DataFrame, out: Path):
    metrics = ["J_attack", "pd_attack", "pfa_attack", "pd_clean", "pfa_clean", "J_clean", "ffr_clean", "auc_attack", "auc_clean"]
    keys = ["snr_db", "session_spec", "pipeline"]
    rows = []
    for key, g in seed.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row["n_test_seeds"] = int(g.seed.nunique())
        for m in metrics:
            mean, lo, hi, n = mean_ci(g[m])
            row[f"{m}_mean"] = mean; row[f"{m}_ci95_low"] = lo; row[f"{m}_ci95_high"] = hi
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "B1V3_performance_summary.csv", index=False)


def calibration_target_pfa_per_seed(seed: pd.DataFrame, target_pfa: float, out: Path):
    """Summarize operating points calibrated at a nominal clean-PFA target.

    This is not exact test-set PFA matching. Each pipeline's threshold is selected
    on independent clean-H0 calibration trajectories at the same nominal quantile;
    achieved clean PFA is measured and reported on held-out test seeds.
    """
    if "tau_quantile" in seed:
        bad = seed[np.abs(seed.tau_quantile - (1 - target_pfa)) > 1e-9]
        if len(bad):
            raise ValueError("Run tau_quantile does not match target PFA requested for analysis")
    cols = ["seed", "snr_db", "session_spec", "pipeline", "tau_quantile",
            "pd_clean", "pfa_clean", "J_clean", "pd_attack", "pfa_attack", "J_attack",
            "auc_clean", "auc_attack", "ffr_clean"]
    d = seed[[c for c in cols if c in seed.columns]].copy()
    d["target_clean_pfa"] = float(target_pfa)
    d = d.rename(columns={
        "pd_clean": "pd_clean_at_calibration_target",
        "pfa_clean": "pfa_clean_achieved",
        "J_clean": "J_clean_at_calibration_target",
        "pd_attack": "pd_attack_at_calibration_target",
        "pfa_attack": "pfa_attack_at_calibration_target",
        "J_attack": "J_attack_at_calibration_target"
    })
    d.to_csv(out / "B1V3_calibration_target_PFA_per_seed.csv", index=False)
    keys = ["snr_db", "session_spec", "pipeline"]
    vals = ["pd_clean_at_calibration_target", "pfa_clean_achieved", "J_clean_at_calibration_target",
            "pd_attack_at_calibration_target", "pfa_attack_at_calibration_target",
            "J_attack_at_calibration_target", "auc_clean", "auc_attack", "ffr_clean"]
    rows = []
    for key, g in d.groupby(keys, dropna=False):
        r = dict(zip(keys, key)); r["n_test_seeds"] = int(g.seed.nunique())
        for m in vals:
            if m not in g:
                continue
            mu, lo, hi, _ = mean_ci(g[m])
            r[m + "_mean"] = mu; r[m + "_ci95_low"] = lo; r[m + "_ci95_high"] = hi
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "B1V3_calibration_target_PFA_summary.csv", index=False)
    return d

def boundary_analysis(sessions, blocks, out: Path, delays=(0, 1, 2, 3, 5, 10)):
    ph = blocks[blocks.pipeline == "XS-PH"].copy()
    det_keys = ["seed", "snr_db", "session_spec", "sequence_type"]
    sess_keys = ["seed", "snr_db", "session_spec"]
    sess_map = {k: g for k, g in sessions.groupby(sess_keys, dropna=False)}
    trajectory_rows = []
    for key, g in ph.groupby(det_keys, dropna=False):
        skey = key[:3]
        sg = sess_map.get(skey)
        true_bounds = [] if sg is None else sorted(int(x) for x in sg.start_round if int(x) > 0)
        detected = sorted(int(x) for x in g.loc[(g.boundary_reason == "page_hinkley") & g.boundary_round.notna(), "boundary_round"])
        rounds_n = int(g.rounds.iloc[0])
        for delay in delays:
            matches, fp_idx, fn_idx = chronological_boundary_match(true_bounds, detected, int(delay))
            tp = len(matches); fp = len(fp_idx); fn = len(fn_idx)
            precision = tp / (tp + fp) if (tp + fp) else (1.0 if len(true_bounds) == 0 else 0.0)
            recall = tp / (tp + fn) if (tp + fn) else np.nan
            f1 = (2 * precision * recall / (precision + recall)) if np.isfinite(recall) and precision + recall > 0 else np.nan
            delays_matched = [x[2] for x in matches]
            trajectory_rows.append({"seed": key[0], "snr_db": key[1], "session_spec": key[2],
                                    "sequence_type": key[3], "max_allowed_delay": int(delay),
                                    "rounds": rounds_n, "n_true_boundaries": len(true_bounds),
                                    "n_detected_boundaries": len(detected), "tp": tp, "fp": fp, "fn": fn,
                                    "precision": precision, "recall": recall, "f1": f1,
                                    "mean_detection_delay": float(np.mean(delays_matched)) if delays_matched else np.nan,
                                    "false_alarms_per_100_rounds": fp / rounds_n * 100.0})
    traj = pd.DataFrame(trajectory_rows)
    traj.to_csv(out / "B1V3_boundary_metrics_per_trajectory.csv", index=False)
    if len(traj):
        keys = ["snr_db", "session_spec", "sequence_type", "max_allowed_delay"]
        rows = []
        for key, g in traj.groupby(keys, dropna=False):
            r = dict(zip(keys, key)); r["n_seed_trajectories"] = int(g.seed.nunique())
            for m in ["precision", "recall", "f1", "mean_detection_delay", "false_alarms_per_100_rounds"]:
                mu, lo, hi, _ = mean_ci(g[m])
                r[m + "_mean"] = mu; r[m + "_ci95_low"] = lo; r[m + "_ci95_high"] = hi
            r["true_boundaries_mean"] = float(g.n_true_boundaries.mean())
            r["detected_boundaries_mean"] = float(g.n_detected_boundaries.mean())
            rows.append(r)
        pd.DataFrame(rows).to_csv(out / "B1V3_boundary_summary.csv", index=False)
    return traj


def paired_tests(seed, calibration_target, selection, out: Path, n_perm=100000, n_boot=20000):
    kbest = int(selection["selected_fixed_k"]["K"])
    baselines = ["XS-fixed20", f"XS-fixed{kbest}"]
    baselines = list(dict.fromkeys(baselines))
    primary_spec = selection["primary_test"]["session_spec"]
    primary_rows = []
    seed_base = seed[(seed.session_spec == primary_spec) & (seed.attack == "masking") & (seed.rho_requested == 0.2)]
    for snr in [-9.0, -6.0]:
        z = seed_base[seed_base.snr_db == snr].pivot(index="seed", columns="pipeline", values="J_attack")
        for baseline in baselines:
            if "XS-PH" not in z or baseline not in z:
                raise ValueError(f"Missing primary pipeline for comparison XS-PH vs {baseline}")
            paired = z[["XS-PH", baseline]].dropna()
            d = (paired["XS-PH"] - paired[baseline]).to_numpy(float)
            ci = paired_bootstrap_ci(d, n_boot=n_boot, seed=20261109 + int(abs(snr) * 10) + len(baseline))
            p_perm = paired_permutation_p(d, n_perm=n_perm, seed=20261109 + int(abs(snr) * 100) + len(baseline))
            try:
                p_w = float(wilcoxon(d, alternative="two-sided", method="auto").pvalue) if np.any(np.abs(d) > 1e-14) else 1.0
            except Exception:
                p_w = np.nan
            primary_rows.append({"family": "B1V3_primary_random_sessions", "session_spec": primary_spec,
                                 "snr_db": snr, "metric": "J_attack", "contrast": f"XS-PH - {baseline}",
                                 "baseline": baseline, "n_test_seeds": len(d),
                                 "mean_difference": float(np.mean(d)), "median_difference": float(np.median(d)),
                                 "ci95_low": ci[0], "ci95_high": ci[1],
                                 "wins": int(np.sum(d > 1e-12)), "losses": int(np.sum(d < -1e-12)),
                                 "ties": int(np.sum(np.abs(d) <= 1e-12)),
                                 "p_raw_paired_permutation": p_perm, "p_raw_wilcoxon_secondary": p_w})
    primary = pd.DataFrame(primary_rows)
    primary["p_holm_primary_family"] = holm_adjust(primary.p_raw_paired_permutation.to_numpy())
    primary["ci_excludes_zero"] = (primary.ci95_low > 0) | (primary.ci95_high < 0)
    primary["statistically_supported_improvement"] = (
        (primary.mean_difference > 0) & (primary.ci95_low > 0) & (primary.p_holm_primary_family < 0.05)
    )
    primary.to_csv(out / "B1V3_primary_paired_tests.csv", index=False)

    # Secondary fixed-session contrasts: PH against all fixed K within each SNR/session.
    # Use fewer Monte Carlo replicates for this exploratory family; the primary
    # random-duration family retains the full requested permutation/bootstrap counts.
    n_perm_secondary = min(int(n_perm), 20000)
    n_boot_secondary = min(int(n_boot), 5000)
    sec_rows = []
    sec = seed[(seed.attack == "masking") & (seed.rho_requested == 0.2) &
               (seed.pipeline.str.startswith("XS-fixed") | (seed.pipeline == "XS-PH"))]
    fixed_specs = ["fixed:13", "fixed:20", "fixed:30", "fixed:50"]
    for spec in fixed_specs:
        for snr in [-9.0, -6.0]:
            g = sec[(sec.session_spec == spec) & (sec.snr_db == snr)]
            z = g.pivot(index="seed", columns="pipeline", values="J_attack")
            for baseline in ["XS-fixed10", "XS-fixed13", "XS-fixed20", "XS-fixed30", "XS-fixed50"]:
                if baseline not in z or "XS-PH" not in z:
                    continue
                d = (z["XS-PH"] - z[baseline]).dropna().to_numpy(float)
                if len(d) < 2:
                    continue
                lo, hi = paired_bootstrap_ci(d, n_boot=n_boot_secondary, seed=20261200 + len(sec_rows))
                p = paired_permutation_p(d, n_perm=n_perm_secondary, seed=20261300 + len(sec_rows))
                sec_rows.append({"session_spec": spec, "snr_db": snr, "baseline": baseline,
                                 "n_test_seeds": len(d), "mean_difference": float(np.mean(d)),
                                 "ci95_low": lo, "ci95_high": hi,
                                 "wins": int(np.sum(d > 1e-12)), "losses": int(np.sum(d < -1e-12)),
                                 "ties": int(np.sum(np.abs(d) <= 1e-12)), "p_raw_permutation": p})
    secondary = pd.DataFrame(sec_rows)
    if len(secondary):
        secondary["p_holm_secondary_family"] = holm_adjust(secondary.p_raw_permutation.to_numpy())
    secondary.to_csv(out / "B1V3_secondary_fixed_session_tests.csv", index=False)

    # Calibration-target operating-point contrasts are supportive secondary evidence.
    # These are NOT exact matched-PFA test comparisons; achieved test clean PFA is reported.
    if len(calibration_target):
        mbase = calibration_target[(calibration_target.session_spec == primary_spec)]
        mrows = []
        for snr in [-9.0, -6.0]:
            z = mbase[mbase.snr_db == snr].pivot(index="seed", columns="pipeline", values="J_attack_at_calibration_target")
            for baseline in baselines:
                if "XS-PH" not in z or baseline not in z:
                    continue
                d = (z["XS-PH"] - z[baseline]).dropna().to_numpy(float)
                lo, hi = paired_bootstrap_ci(d, n_boot=n_boot_secondary, seed=20261400 + len(mrows))
                p = paired_permutation_p(d, n_perm=n_perm_secondary, seed=20261500 + len(mrows))
                mrows.append({"snr_db": snr, "baseline": baseline, "n_test_seeds": len(d),
                              "mean_difference_J_at_calibration_target": float(np.mean(d)),
                              "ci95_low": lo, "ci95_high": hi,
                              "wins": int(np.sum(d > 1e-12)), "losses": int(np.sum(d < -1e-12)),
                              "ties": int(np.sum(np.abs(d) <= 1e-12)), "p_raw_permutation": p})
        mdf = pd.DataFrame(mrows)
        if len(mdf):
            mdf["p_holm_calibration_target_family"] = holm_adjust(mdf.p_raw_permutation.to_numpy())
        mdf.to_csv(out / "B1V3_calibration_target_PFA_paired_tests.csv", index=False)

    # Threshold-free discrimination is a supportive secondary analysis.
    # AUC is calculated per independent test trajectory; the seed is the unit.
    auc_rows = []
    primary_seed = seed[(seed.session_spec == primary_spec) & (seed.attack == "masking") &
                        (seed.rho_requested == 0.2)]
    for snr in [-9.0, -6.0]:
        z = primary_seed[primary_seed.snr_db == snr].pivot(index="seed", columns="pipeline", values="auc_attack")
        for baseline in baselines:
            if "XS-PH" not in z or baseline not in z:
                continue
            d = (z["XS-PH"] - z[baseline]).dropna().to_numpy(float)
            lo, hi = paired_bootstrap_ci(d, n_boot=n_boot_secondary, seed=20261600 + len(auc_rows))
            p = paired_permutation_p(d, n_perm=n_perm_secondary, seed=20261700 + len(auc_rows))
            auc_rows.append({"snr_db": snr, "baseline": baseline, "n_test_seeds": len(d),
                             "mean_difference_AUC_attack": float(np.mean(d)),
                             "ci95_low": lo, "ci95_high": hi,
                             "wins": int(np.sum(d > 1e-12)), "losses": int(np.sum(d < -1e-12)),
                             "ties": int(np.sum(np.abs(d) <= 1e-12)), "p_raw_permutation": p})
    auc_df = pd.DataFrame(auc_rows)
    if len(auc_df):
        auc_df["p_holm_AUC_family"] = holm_adjust(auc_df.p_raw_permutation.to_numpy())
    auc_df.to_csv(out / "B1V3_AUC_attack_paired_tests.csv", index=False)

    return primary, secondary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--target-pfa", type=float, default=0.05)
    ap.add_argument("--n-permutations", type=int, default=100000)
    ap.add_argument("--n-bootstrap", type=int, default=20000)
    a = ap.parse_args()
    root = Path(a.results).resolve(); root.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(a.selection).read_text(encoding="utf-8"))
    seed, sessions, blocks, manifest, identities = load_inputs(root)
    integ = integrity_checks(seed, sessions, blocks, manifest, identities, root)
    performance_summary(seed, root)
    calibration_target = calibration_target_pfa_per_seed(seed, a.target_pfa, root)
    traj = boundary_analysis(sessions, blocks, root)
    primary, secondary = paired_tests(seed, calibration_target, selection, root,
                                      n_perm=a.n_permutations, n_boot=a.n_bootstrap)

    # PH alarms in the no-transition fixed:200 control.
    no_change = blocks[(blocks.pipeline == "XS-PH") & (blocks.session_spec == "fixed:200")]
    no_change_rows = []
    if len(no_change):
        for key, g in no_change.groupby(["snr_db", "sequence_type"]):
            rounds_n = int(g.rounds.iloc[0])
            alarms = int(((g.boundary_reason == "page_hinkley") & g.boundary_round.notna()).sum())
            ntraj = int(g[["seed", "snr_db", "sequence_type"]].drop_duplicates().shape[0])
            no_change_rows.append({"snr_db": key[0], "sequence_type": key[1],
                                   "n_trajectories": ntraj, "total_alarms": alarms,
                                   "alarms_per_100_rounds": alarms / (ntraj * rounds_n) * 100.0})
    pd.DataFrame(no_change_rows).to_csv(root / "B1V3_no_change_control.csv", index=False)

    primary_spec = selection["primary_test"]["session_spec"]
    kbest = int(selection["selected_fixed_k"]["K"])
    guardrail_pipelines = list(dict.fromkeys(["XS-PH", "XS-fixed20", f"XS-fixed{kbest}"]))
    clean_guardrail_rows = []
    primary_clean = seed[seed.session_spec == primary_spec]
    for pipeline_name in guardrail_pipelines:
        gpipe = primary_clean[primary_clean.pipeline == pipeline_name]
        for snr, g in gpipe.groupby("snr_db"):
            clean_guardrail_rows.append({
                "pipeline": pipeline_name,
                "snr_db": float(snr),
                "mean_achieved_clean_PFA": float(g.pfa_clean.mean()),
                "mean_FFRclean": float(g.ffr_clean.mean()),
                "n_seeds": int(g.seed.nunique()),
                "clean_PFA_guardrail_pass": bool(g.pfa_clean.mean() <= 0.075),
                "FFRclean_guardrail_pass": bool(g.ffr_clean.mean() <= 0.10),
            })
    pd.DataFrame(clean_guardrail_rows).to_csv(root / "B1V3_clean_operating_guardrails.csv", index=False)
    ph_guardrails = [x for x in clean_guardrail_rows if x["pipeline"] == "XS-PH"]
    comparator_guardrails = [x for x in clean_guardrail_rows if x["pipeline"] != "XS-PH"]
    clean_guardrails_pass = bool(ph_guardrails) and all(
        x["clean_PFA_guardrail_pass"] and x["FFRclean_guardrail_pass"] for x in ph_guardrails
    )
    comparators_guardrails_pass = bool(comparator_guardrails) and all(
        x["clean_PFA_guardrail_pass"] and x["FFRclean_guardrail_pass"] for x in comparator_guardrails
    )
    all_primary_supported = bool(primary.statistically_supported_improvement.all()) if len(primary) else False
    conclusion = {
        "integrity_status": integ["status"],
        "n_test_seeds": int(seed.seed.nunique()),
        "primary_spec": primary_spec,
        "selected_fixed_k": selection["selected_fixed_k"]["K"],
        "primary_comparisons": primary.to_dict(orient="records"),
        "clean_guardrails_by_pipeline_and_snr": clean_guardrail_rows,
        "XS_PH_clean_guardrails_pass": clean_guardrails_pass,
        "comparison_baselines_clean_guardrails_pass": comparators_guardrails_pass,
        "any_primary_improvement_supported": bool(primary.statistically_supported_improvement.any()) if len(primary) else False,
        "all_primary_comparisons_support_improvement": all_primary_supported,
        "claim_ph_performance_improvement_supported": bool(all_primary_supported and clean_guardrails_pass and comparators_guardrails_pass),
        "interpretation_rule": "A performance-improvement claim is supported only if every predeclared primary contrast is positive, its paired 95% CI excludes zero, its Holm-adjusted paired-permutation p-value is below 0.05, and XS-PH plus the primary fixed baselines meet per-SNR clean-PFA (<=0.075) and FFRclean (<=0.10) guardrails. Thresholds target nominal clean PFA 0.05 using independent calibration data; achieved test clean PFA is reported and is not forced to equal 0.05. Boundary-detection metrics are a separate outcome."
    }
    (root / "B1V3_final_conclusion_aid.json").write_text(json.dumps(conclusion, indent=2), encoding="utf-8")

    # Human-readable report, generated from recorded outcomes without manual edits.
    report_lines = [
        "# B1-V3 Final Evaluation Report",
        "",
        f"- Integrity: **{integ['status']}**",
        f"- Independent test seeds: **{conclusion['n_test_seeds']}**",
        f"- Primary variable-session condition: `{primary_spec}`",
        f"- Selected PH parameters: `{selection['selected_ph']}`",
        f"- Development-selected fixed K: **{kbest}**",
        f"- PH clean guardrails pass: **{clean_guardrails_pass}**",
        f"- Comparison baseline clean guardrails pass: **{comparators_guardrails_pass}**",
        "",
        "## Primary paired J_attack contrasts",
        "",
        "| SNR | Contrast | Mean difference | 95% CI | Wins/Losses/Ties | Raw p | Holm p | Supported improvement |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in primary.iterrows():
        report_lines.append(
            f"| {r.snr_db:g} | {r.contrast} | {r.mean_difference:.5f} | "
            f"[{r.ci95_low:.5f}, {r.ci95_high:.5f}] | "
            f"{int(r.wins)}/{int(r.losses)}/{int(r.ties)} | "
            f"{r.p_raw_paired_permutation:.5g} | {r.p_holm_primary_family:.5g} | "
            f"{bool(r.statistically_supported_improvement)} |"
        )
    report_lines.extend([
        "",
        "## Clean operating guardrails",
        "",
        "| Pipeline | SNR | Achieved clean PFA | FFRclean | PFA pass | FFR pass |",
        "|---|---:|---:|---:|---|---|",
    ])
    for r in clean_guardrail_rows:
        report_lines.append(
            f"| {r['pipeline']} | {r['snr_db']:g} | {r['mean_achieved_clean_PFA']:.5f} | "
            f"{r['mean_FFRclean']:.5f} | {r['clean_PFA_guardrail_pass']} | {r['FFRclean_guardrail_pass']} |"
        )
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        ("The predeclared criteria for a PH sensing-performance improvement were met in this evaluation."
         if conclusion["claim_ph_performance_improvement_supported"] else
         "The predeclared criteria for claiming a PH sensing-performance improvement were NOT met. Report the observed estimates and uncertainty; do not claim an incremental sensing-performance benefit. Boundary-detection performance is a separate result."),
        "",
        "Thresholds were calibrated on independent clean-H0 data at a nominal 0.05 PFA target. Achieved clean PFA is measured on final test seeds and is not forced to equal 0.05.",
        "",
        "Final-test results must not be used to retune PH parameters or choose K. Any changes after inspection require a new independent test-seed pool.",
    ])
    (root / "B1V3_interpretation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("B1-V3 final analysis completed.")
    print(json.dumps({"integrity": integ["status"],
                      "primary_results": primary.to_dict(orient="records"),
                      "any_primary_improvement_supported": conclusion["any_primary_improvement_supported"]}, indent=2))


if __name__ == "__main__":
    main()
