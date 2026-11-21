"""Estimate a V2 test-seed count from the old V1 paired evaluation data.

This is a planning aid, not a prospective preregistration. Run it BEFORE locking
new V2 test seeds; report the resulting assumptions and retain the output.

Example:
  python plan_seed_count.py --v1 xs_test_seed_level_ALL.csv --out seed_count_plan.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from scipy.stats import nct, t as student_t


def power_for_n(n, sd, delta=0.05, alpha=0.05):
    if n < 2:
        return 0.0
    if not np.isfinite(sd) or sd <= 1e-15:
        return 1.0
    df = n - 1
    ncp = abs(delta) / (sd / np.sqrt(n))
    crit = student_t.ppf(1 - alpha / 2, df)
    return float(nct.sf(crit, df, ncp) + nct.cdf(-crit, df, ncp))


def required_n(sd, target, delta=0.05, alpha=0.05, max_n=10000):
    for n in range(2, max_n + 1):
        if power_for_n(n, sd, delta, alpha) >= target:
            return n
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="V1 xs_test_seed_level_ALL.csv")
    ap.add_argument("--out", default="v2_seed_count_plan.csv")
    ap.add_argument("--delta-j", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-powers", default="0.8,0.9")
    ap.add_argument("--family-size", type=int, default=8,
                    help="RQ1 primary family size; Bonferroni is used as a conservative planning approximation")
    a = ap.parse_args()
    df = pd.read_csv(a.v1)
    req = {"scenario", "rho", "attack", "snr_db", "seed", "pipeline", "J_attack"}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"V1 file is missing required columns: {sorted(missing)}")
    d = df[(df.scenario == "default") & np.isclose(df.rho, 0.20) &
           (df.attack == "masking") & df.snr_db.isin([-9.0, -6.0])]
    baselines = ["Trim10", "Trim20", "Trim40", "Median"]
    powers = [float(x) for x in a.target_powers.split(",") if x.strip()]
    alpha_family = a.alpha / max(a.family_size, 1)
    rows = []
    for snr in [-9.0, -6.0]:
        part = d[d.snr_db == snr]
        pivot = part.pivot(index="seed", columns="pipeline", values="J_attack")
        for base in baselines:
            if "XS-PH" not in pivot or base not in pivot:
                continue
            pair = pivot[["XS-PH", base]].dropna()
            diff = (pair["XS-PH"] - pair[base]).to_numpy(float)
            sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
            row = {"snr_db": snr, "contrast": f"XS-PH - {base}", "n_v1_pairs": len(diff),
                   "mean_diff_v1": float(np.mean(diff)) if len(diff) else np.nan,
                   "sd_paired_diff_v1": sd, "planning_delta_j": a.delta_j,
                   "alpha_two_sided_nominal": a.alpha, "rq1_family_size": a.family_size,
                   "alpha_bonferroni_planning": alpha_family,
                   "estimated_power_at_20_nominal": power_for_n(20, sd, a.delta_j, a.alpha),
                   "estimated_power_at_20_family_adjusted": power_for_n(20, sd, a.delta_j, alpha_family)}
            for target in powers:
                row[f"n_for_{int(target*100)}pct_nominal"] = required_n(sd, target, a.delta_j, a.alpha)
                row[f"n_for_{int(target*100)}pct_family_adjusted"] = required_n(sd, target, a.delta_j, alpha_family)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No primary-condition paired contrasts were found in the supplied V1 file.")
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))
    print("\nPlanning estimates use V1 paired-difference SDs and a two-sided paired-t approximation.")
    print("They are not proof of achieved power and should be documented as a design sensitivity analysis.")
    print("Calibration-seed adequacy is assessed separately by cluster-bootstrap threshold stability in V2.")


if __name__ == "__main__":
    main()
