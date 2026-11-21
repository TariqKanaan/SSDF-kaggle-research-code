"""Select PH parameters and fixed K from independent B1-V3 development data.

Efficiency design: all fixed-K baselines are produced once in fixed_baselines/;
PH candidate runs contain XS-PH only. The script verifies common seed pools and
identical identity/session ledgers before selecting settings.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nct, t as student_t

RANDOM_SPEC = "random_balanced:13,20,30,50"
DEFAULT_CANDIDATE = "ph_default_d005_t5"
CANDIDATE_INFO = {
    "ph_d002_t5": {"delta": 0.02, "threshold": 5.0, "beta": 0.05, "warmup": 5},
    "ph_d005_t4": {"delta": 0.05, "threshold": 4.0, "beta": 0.05, "warmup": 5},
    "ph_default_d005_t5": {"delta": 0.05, "threshold": 5.0, "beta": 0.05, "warmup": 5},
    "ph_d005_t6": {"delta": 0.05, "threshold": 6.0, "beta": 0.05, "warmup": 5},
    "ph_d010_t5": {"delta": 0.10, "threshold": 5.0, "beta": 0.05, "warmup": 5},
}
FIXED_KS = [10, 13, 20, 30, 50]


def read_seed(path: Path) -> pd.DataFrame:
    f = path / "v2_seed_level.csv.gz"
    if not f.exists():
        raise FileNotFoundError(f)
    d = pd.read_csv(f)
    key = ["seed", "snr_db", "session_spec", "pipeline"]
    if d.duplicated(key).any():
        raise ValueError(f"Duplicate seed/SNR/session/pipeline rows in {f}")
    return d


def read_ledger(path: Path) -> pd.DataFrame:
    f = path / "v3_identity_ledger.csv.gz"
    if not f.exists():
        raise FileNotFoundError(f)
    d = pd.read_csv(f).sort_values(["seed", "snr_db", "session_spec"]).reset_index(drop=True)
    return d


def verify_same_generation(reference: pd.DataFrame, current: pd.DataFrame, label: str) -> None:
    cols = ["seed", "snr_db", "session_spec", "attack", "rho_requested",
            "malicious_ids", "extreme_ids", "mild_ids", "session_lengths", "subgroup_assignment"]
    a = reference[cols].sort_values(cols[:3]).reset_index(drop=True)
    b = current[cols].sort_values(cols[:3]).reset_index(drop=True)
    if not a.equals(b):
        raise ValueError(f"Generated schedules/attacker identities differ for {label}; common-random-number pairing failed.")


def summarize_ph(cid: str, d: pd.DataFrame) -> dict:
    ph_all = d[(d.attack == "masking") & (d.rho_requested == 0.2) & (d.pipeline == "XS-PH")]
    ph = ph_all[ph_all.session_spec == RANDOM_SPEC]
    if ph.empty:
        raise ValueError(f"Missing XS-PH random-session data for {cid}")
    by_snr = ph.groupby("snr_db").agg(
        J_attack_mean=("J_attack", "mean"),
        pfa_clean_mean=("pfa_clean", "mean"),
        ffr_clean_mean=("ffr_clean", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return {
        "candidate": cid, **CANDIDATE_INFO[cid],
        "J_attack_mean_random": float(by_snr.J_attack_mean.mean()),
        "pfa_clean_mean_random": float(by_snr.pfa_clean_mean.mean()),
        "ffr_clean_mean_random": float(by_snr.ffr_clean_mean.mean()),
        "n_dev_seeds_per_snr": int(by_snr.n_seeds.min()),
        "J_attack_mean_fixed20_session": float(ph_all[ph_all.session_spec == "fixed:20"].J_attack.mean()),
    }


def required_n(sd: float, delta: float, alpha: float, target_power: float) -> tuple[int, float, bool]:
    if not np.isfinite(sd) or sd <= 1e-12:
        raise ValueError("Cannot plan sample size: paired SD is unavailable or zero. Increase development seeds and inspect the contrast data.")
    effect = float(delta) / float(sd)
    for n in range(2, 10001):
        crit = student_t.ppf(1 - alpha / 2, n - 1)
        ncp = effect * np.sqrt(n)
        power = nct.cdf(-crit, n - 1, ncp) + nct.sf(crit, n - 1, ncp)
        if power >= target_power:
            return n, float(power), False
    n = 10000
    crit = student_t.ppf(1 - alpha / 2, n - 1)
    ncp = effect * np.sqrt(n)
    power = nct.cdf(-crit, n - 1, ncp) + nct.sf(crit, n - 1, ncp)
    return n, float(power), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--meaningful-delta", type=float, default=0.05)
    ap.add_argument("--alpha-family", type=float, default=0.05)
    ap.add_argument("--target-power", type=float, default=0.90)
    ap.add_argument("--pfa-guardrail", type=float, default=0.075)
    ap.add_argument("--ffr-guardrail", type=float, default=0.10)
    a = ap.parse_args()
    root = Path(a.dev_root).resolve()
    out = Path(a.out).resolve() if a.out else root
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "B1V3_development_protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(f"Missing development protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    candidate_dfs = {cid: read_seed(root / cid) for cid in CANDIDATE_INFO}
    fixed = read_seed(root / "fixed_baselines")
    candidate_ledgers = {cid: read_ledger(root / cid) for cid in CANDIDATE_INFO}
    fixed_ledger = read_ledger(root / "fixed_baselines")

    ref_seed_set = set(candidate_dfs[DEFAULT_CANDIDATE].seed.unique())
    expected_seeds = set(range(protocol["seed_pools"]["development"][0],
                               protocol["seed_pools"]["development"][1] + 1))
    if ref_seed_set != expected_seeds:
        raise ValueError("PH candidate seeds do not match the development protocol seed range")
    if set(fixed.seed.unique()) != expected_seeds:
        raise ValueError("Fixed-K baseline seed pool differs from protocol")

    for cid, d in candidate_dfs.items():
        if set(d.pipeline.unique()) != {"XS-PH"}:
            raise ValueError(f"Candidate {cid} must contain XS-PH only; found {sorted(d.pipeline.unique())}")
        if set(d.seed.unique()) != ref_seed_set or set(d.snr_db.unique()) != {-9.0, -6.0}:
            raise ValueError(f"Development seed/SNR pool differs for candidate {cid}")
        for spec in [RANDOM_SPEC, "fixed:20"]:
            if spec not in set(d.session_spec.unique()):
                raise ValueError(f"Candidate {cid} missing session condition {spec}")
        verify_same_generation(fixed_ledger, candidate_ledgers[cid], cid)
    expected_fixed = {f"XS-fixed{k}" for k in FIXED_KS}
    if not expected_fixed.issubset(set(fixed.pipeline.unique())):
        raise ValueError(f"Missing fixed-K baselines: {sorted(expected_fixed - set(fixed.pipeline.unique()))}")
    if not set(fixed.session_spec.unique()).issuperset({RANDOM_SPEC, "fixed:20"}):
        raise ValueError("Fixed baseline run missing required session specs")

    ph_summary = pd.DataFrame([summarize_ph(cid, d) for cid, d in candidate_dfs.items()])
    ph_summary["pfa_guardrail_pass"] = ph_summary.pfa_clean_mean_random <= a.pfa_guardrail
    ph_summary["ffr_guardrail_pass"] = ph_summary.ffr_clean_mean_random <= a.ffr_guardrail
    ph_summary["selection_eligible"] = ph_summary.pfa_guardrail_pass & ph_summary.ffr_guardrail_pass
    ph_summary.to_csv(out / "B1V3_PH_development_grid.csv", index=False)

    fixed_random = fixed[(fixed.session_spec == RANDOM_SPEC) & (fixed.attack == "masking") &
                         (fixed.rho_requested == 0.2)]
    fixed_rows = []
    for k in FIXED_KS:
        name = f"XS-fixed{k}"
        g = fixed_random[fixed_random.pipeline == name]
        by_snr = g.groupby("snr_db").agg(J=("J_attack", "mean"),
                                           pfa=("pfa_clean", "mean"),
                                           ffr=("ffr_clean", "mean"),
                                           n=("seed", "nunique")).reset_index()
        if set(by_snr.snr_db) != {-9.0, -6.0}:
            raise ValueError(f"Fixed-K {name} missing one of the two SNR values")
        fixed_rows.append({"K": k, "pipeline": name,
                           "J_attack_mean_random": float(by_snr.J.mean()),
                           "pfa_clean_mean_random": float(by_snr.pfa.mean()),
                           "ffr_clean_mean_random": float(by_snr.ffr.mean()),
                           "n_dev_seeds_per_snr": int(by_snr.n.min())})
    fixed_summary = pd.DataFrame(fixed_rows)
    fixed_summary["guardrail_pass"] = ((fixed_summary.pfa_clean_mean_random <= a.pfa_guardrail) &
                                       (fixed_summary.ffr_clean_mean_random <= a.ffr_guardrail))
    fixed_summary.to_csv(out / "B1V3_fixedK_development_grid.csv", index=False)

    eligible_ph = ph_summary[ph_summary.selection_eligible].copy()
    if len(eligible_ph):
        selected_ph = eligible_ph.sort_values(["J_attack_mean_random", "candidate"], ascending=[False, True]).iloc[0]
        ph_guardrail_pass = True
        ph_selection_note = "Selected the highest-development-J PH candidate among candidates passing clean PFA and FFR guardrails."
    else:
        selected_ph = ph_summary[ph_summary.candidate == DEFAULT_CANDIDATE].iloc[0]
        ph_guardrail_pass = False
        ph_selection_note = "No PH candidate passed both guardrails; retained the original frozen PH configuration. Do not claim tuning success."

    eligible_k = fixed_summary[fixed_summary.guardrail_pass].copy()
    k_pool = eligible_k if len(eligible_k) else fixed_summary
    max_j = float(k_pool.J_attack_mean_random.max())
    tied = k_pool[np.abs(k_pool.J_attack_mean_random - max_j) <= 1e-12].copy()
    tied["tie_rank"] = tied.K.map(lambda x: (0 if x == 20 else 1, x))
    selected_k_row = tied.sort_values("tie_rank").iloc[0]
    k_guardrail_pass = bool(len(eligible_k))
    kbest = int(selected_k_row.K)

    # Pair every PH candidate with each fixed method on the exact same development
    # seed/SNR/schedule. Use the maximum paired SD for a conservative test-size plan.
    sd_records = []
    merge_keys = ["seed", "snr_db", "session_spec"]
    fixed_index = fixed.set_index(merge_keys + ["pipeline"])
    for cid, d in candidate_dfs.items():
        ph = d[d.session_spec == RANDOM_SPEC].set_index(merge_keys)["J_attack"].rename("J_PH")
        for snr in [-9.0, -6.0]:
            ph_s = ph[ph.index.get_level_values("snr_db") == snr]
            for baseline in ["XS-fixed20", f"XS-fixed{kbest}"]:
                base_s = fixed_index.xs(baseline, level="pipeline")["J_attack"]
                base_s = base_s[base_s.index.get_level_values("snr_db") == snr]
                paired = pd.concat([ph_s, base_s.rename("J_fixed")], axis=1).dropna()
                if len(paired) > 1:
                    sd_records.append({"candidate": cid, "snr_db": snr, "baseline": baseline,
                                       "n_dev": len(paired), "paired_sd": float(paired.J_PH.sub(paired.J_fixed).std(ddof=1))})
    sd_df = pd.DataFrame(sd_records)
    if sd_df.empty:
        raise ValueError("No paired development contrasts available for sample-size planning")
    sd_df.to_csv(out / "B1V3_power_variance_inputs.csv", index=False)
    max_sd = float(sd_df.paired_sd.max())
    unique_primary_baselines = list(dict.fromkeys(["XS-fixed20", f"XS-fixed{kbest}"]))
    n_primary_tests = 2 * len(unique_primary_baselines)
    alpha_per = a.alpha_family / n_primary_tests
    n_req, estimated_power_at_n, cap_reached = required_n(max_sd, a.meaningful_delta, alpha_per, a.target_power)
    sample_plan = {
        "meaningful_delta_J": a.meaningful_delta,
        "family_alpha": a.alpha_family,
        "number_primary_tests": n_primary_tests,
        "bonferroni_alpha_for_planning": alpha_per,
        "target_power": a.target_power,
        "max_paired_sd_across_dev_candidates_and_primary_contrasts": max_sd,
        "recommended_independent_test_seeds_per_condition": n_req,
        "estimated_power_at_recommended_n_from_development_sd": estimated_power_at_n,
        "sample_size_search_cap_reached": cap_reached,
        "warning": "Planning estimate based on development paired SD, not achieved power. Freeze the test size before inspecting final-test outcomes."
    }
    (out / "B1V3_sample_size_plan.json").write_text(json.dumps(sample_plan, indent=2), encoding="utf-8")

    unique_baselines = list(dict.fromkeys(["XS-fixed20", f"XS-fixed{kbest}"]))
    selection = {
        "study": "B1-V3",
        "selected_ph": {
            "candidate": str(selected_ph.candidate), "delta": float(selected_ph.delta),
            "threshold": float(selected_ph.threshold), "beta": float(selected_ph.beta),
            "warmup": int(selected_ph.warmup), "min_block": 5,
            "development_mean_J_random": float(selected_ph.J_attack_mean_random),
            "development_mean_clean_PFA_random": float(selected_ph.pfa_clean_mean_random),
            "development_mean_FFRclean_random": float(selected_ph.ffr_clean_mean_random),
            "guardrail_pass": ph_guardrail_pass, "selection_note": ph_selection_note,
        },
        "selected_fixed_k": {
            "K": kbest, "pipeline": f"XS-fixed{kbest}",
            "development_mean_J_random": float(selected_k_row.J_attack_mean_random),
            "development_mean_clean_PFA_random": float(selected_k_row.pfa_clean_mean_random),
            "development_mean_FFRclean_random": float(selected_k_row.ffr_clean_mean_random),
            "guardrail_pass": k_guardrail_pass,
            "selection_note": "Selected globally from fixed-only development run; tie-break prefers K=20 then smaller K.",
        },
        "primary_test": {
            "session_spec": RANDOM_SPEC, "snrs_db": [-9.0, -6.0],
            "attack": "masking", "rho": 0.20, "primary_metric": "J_attack",
            "primary_contrasts": [f"XS-PH vs {name}" for name in unique_baselines],
            "n_primary_tests": 2 * len(unique_baselines), "clean_pfa_target": 0.05,
        },
        "sample_size_plan": sample_plan,
        "development_seed_range": protocol["seed_pools"]["development"],
        "score_calibration_seed_range": protocol["seed_pools"]["score_calibration"],
        "threshold_calibration_seed_range": protocol["seed_pools"]["threshold_calibration"],
        "final_test_seed_start": 73000,
        "separation_note": "PH settings and K-best selected using development pools only. Final test seeds are disjoint from development and calibration seeds.",
    }
    (out / "B1V3_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("B1-V3 development selection completed.")
    print(json.dumps({"selected_ph": selection["selected_ph"],
                      "selected_fixed_k": selection["selected_fixed_k"],
                      "recommended_test_seeds": n_req,
                      "sample_size_note": sample_plan["warning"]}, indent=2))


if __name__ == "__main__":
    main()
