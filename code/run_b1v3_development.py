"""Run efficient B1-V3 development: PH candidates separately, fixed-K baselines once.

The PH candidates share development and calibration seed pools. Fixed-K is
simulated once because its output is invariant to PH parameters. All runs use
identical seed/condition IDs so generated schedules, attacker identities and
report streams are paired across candidates and fixed baselines.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CANDIDATES = [
    {"id": "ph_d002_t5", "delta": 0.02, "threshold": 5.0, "beta": 0.05, "warmup": 5},
    {"id": "ph_d005_t4", "delta": 0.05, "threshold": 4.0, "beta": 0.05, "warmup": 5},
    {"id": "ph_default_d005_t5", "delta": 0.05, "threshold": 5.0, "beta": 0.05, "warmup": 5},
    {"id": "ph_d005_t6", "delta": 0.05, "threshold": 6.0, "beta": 0.05, "warmup": 5},
    {"id": "ph_d010_t5", "delta": 0.10, "threshold": 5.0, "beta": 0.05, "warmup": 5},
]


def run_one(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path, label: str) -> None:
    print(label, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log,
                              stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed. Inspect {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-count", type=int, default=40)
    ap.add_argument("--score-cal-count", type=int, default=60)
    ap.add_argument("--tau-cal-count", type=int, default=60)
    ap.add_argument("--score-cal-rounds", type=int, default=200)
    ap.add_argument("--tau-cal-rounds", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if min(a.dev_count, a.score_cal_count, a.tau_cal_count, a.rounds) <= 0:
        raise SystemExit("seed counts and rounds must be positive")
    if a.dev_count > 1000 or a.score_cal_count > 100 or a.tau_cal_count > 900:
        raise SystemExit("Development seed counts would overlap declared pools; use dev<=1000, score-cal<=100, tau-cal<=900.")
    if a.rounds != 200:
        raise SystemExit("B1-V3 protocol fixes the horizon at 200 rounds; use 200 for development and final evaluation.")

    root = Path(a.out).resolve()
    if root.exists() and any(root.iterdir()) and not a.overwrite:
        raise SystemExit(f"Development output is not empty: {root}. Use a new folder or --overwrite.")
    if root.exists() and a.overwrite:
        for p in root.iterdir():
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
            else:
                p.unlink()
    root.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().parent / "run_xs_study_v2.py"
    protocol = {
        "stage": "development_only",
        "primary_development_session": "random_balanced:13,20,30,50",
        "guardrail_session": "fixed:20",
        "snrs_db": [-9.0, -6.0],
        "rho": 0.20,
        "attack": "masking",
        "extreme_fraction": 0.50,
        "fixed_blocks": [10, 13, 20, 30, 50],
        "ph_grid": CANDIDATES,
        "ph_beta_fixed": 0.05,
        "ph_warmup_fixed": 5,
        "min_block_fixed": 5,
        "tau_quantile": 0.95,
        "target_clean_pfa": 0.05,
        "seed_pools": {
            "development": [64000, 64000 + a.dev_count - 1],
            "score_calibration": [65000, 65000 + a.score_cal_count - 1],
            "threshold_calibration": [65100, 65100 + a.tau_cal_count - 1],
        },
        "rounds": a.rounds,
        "efficiency_note": "Fixed-K baselines are run once because their outputs are invariant to PH parameters. Each PH candidate is run separately using identical report streams and seed pools.",
        "selection_rule": {
            "ph": "Maximize mean J_attack over both SNRs in count-balanced random variable-duration sessions, among candidates with mean clean PFA <= 0.075 and mean FFRclean <= 0.10. If none pass, retain the frozen default PH configuration and declare tuning guardrail failure.",
            "fixed_k": "Select one global K maximizing mean J_attack over both SNRs in count-balanced random variable-duration sessions among K values satisfying the same clean-PFA and FFRclean guardrails. Ties prefer K=20, then smaller K. If none pass, select the highest-J K but explicitly flag the guardrail failure.",
        },
    }
    (root / "B1V3_development_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (root / "candidate_grid.json").write_text(json.dumps(CANDIDATES, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["SSDF_SCORE_CAL_ROUNDS"] = str(a.score_cal_rounds)
    env["SSDF_TAU_CAL_ROUNDS"] = str(a.tau_cal_rounds)
    env["SSDF_EVAL_ROUNDS"] = str(a.rounds)
    common = [
        "--snrs=-9,-6", "--rhos=0.2", "--attacks=masking",
        "--session-specs=random_balanced:13,20,30,50;fixed:20",
        "--extreme-fractions=0.5", "--persistence-ws=0.0",
        "--subgroup-assignment=random", "--caps=8",
        "--fixed-blocks=10,13,20,30,50", f"--rounds={a.rounds}",
        "--test-start=64000", f"--test-count={a.dev_count}",
        "--score-cal-start=65000", f"--score-cal-count={a.score_cal_count}",
        "--tau-cal-start=65100", f"--tau-cal-count={a.tau_cal_count}",
        "--tau-quantile=0.95", "--min-block=5",
        "--skip-user-telemetry", "--skip-round-telemetry",
    ]

    # Compute all fixed-K baselines once, under the same paired development pools.
    fixed_out = root / "fixed_baselines"
    fixed_cmd = [sys.executable, str(runner), "--out", str(fixed_out),
                 *common, "--pipelines=XS-fixed", "--ph-delta=0.05",
                 "--ph-threshold=5", "--ph-beta=0.05", "--ph-warmup=5"]
    if a.overwrite:
        fixed_cmd.append("--overwrite")
    run_one(fixed_cmd, Path(__file__).resolve().parent, env,
            root / "fixed_baselines.log", "[B1-V3 DEV] fixed-K baselines (once)")

    # Evaluate PH candidates only. This avoids recalculating all fixed K baselines
    # and their calibration for every candidate.
    for ix, cand in enumerate(CANDIDATES, start=1):
        outdir = root / cand["id"]
        cmd = [sys.executable, str(runner), "--out", str(outdir),
               *common, "--pipelines=XS-PH",
               f"--ph-delta={cand['delta']}",
               f"--ph-threshold={cand['threshold']}",
               f"--ph-beta={cand['beta']}",
               f"--ph-warmup={cand['warmup']}"]
        if a.overwrite:
            cmd.append("--overwrite")
        run_one(cmd, Path(__file__).resolve().parent, env,
                root / f"{cand['id']}.log",
                f"[B1-V3 DEV] PH candidate {ix}/{len(CANDIDATES)} {cand['id']}")

    print("Development grid completed:", root)
    print("Next: run select_b1v3_development.py --dev-root", root)


if __name__ == "__main__":
    main()
