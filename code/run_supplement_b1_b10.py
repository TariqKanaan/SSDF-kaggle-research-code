#!/usr/bin/env python3
"""Launch the frozen supplemental B2/B5/B6/B7/B8/B9/B10 study runs.

The B1-V3 primary package/results are not modified. Each run writes to its own
subdirectory and uses seed pools disjoint from B1-V3 and from other inferential
supplemental runs. B5 is intentionally a telemetry replay of part of B2 split.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_xs_study_v2.py"
OUTROOT = Path("/kaggle/working/XS_PH_B1_B10_SUPPLEMENT")

COMMON = [
    "--session-specs=fixed:20", "--rounds=200", "--tau-quantile=0.95",
    "--ph-delta=0.02", "--ph-threshold=5.0", "--ph-beta=0.05",
    "--ph-warmup=5", "--min-block=5", "--subgroup-assignment=random",
]

RUNS = {
    "B2_rho_cap": [
        "--snrs=-9,-6", "--rhos=0.1,0.2,0.3,0.4,0.5", "--attacks=none,masking",
        "--extreme-fractions=0.5", "--caps=4,8,12", "--fixed-blocks=20,50",
        "--pipelines=XS-PH,XS-fixed", "--score-cal-start=84000", "--score-cal-count=150",
        "--tau-cal-start=84200", "--tau-cal-count=150", "--test-start=85000", "--test-count=96",
        "--skip-user-telemetry", "--skip-round-telemetry",
    ],
    "B2_extreme_mild_split": [
        "--snrs=-9,-6", "--rhos=0.2,0.4", "--attacks=none,masking",
        "--extreme-fractions=0,0.5,1", "--caps=8", "--fixed-blocks=20,50",
        "--pipelines=XS-PH,XS-fixed", "--score-cal-start=84400", "--score-cal-count=150",
        "--tau-cal-start=84600", "--tau-cal-count=150", "--test-start=85100", "--test-count=96",
        "--skip-user-telemetry", "--skip-round-telemetry",
    ],
    "B5_telemetry_replay": [
        "--snrs=-9", "--rhos=0.2,0.4", "--attacks=none,masking",
        "--extreme-fractions=0,0.5,1", "--caps=8", "--fixed-blocks=20,50",
        "--pipelines=XS-PH,XS-fixed", "--score-cal-start=84400", "--score-cal-count=150",
        "--tau-cal-start=84600", "--tau-cal-count=150", "--test-start=85100", "--test-count=48",
    ],
    "B6_B7_B8_B9_comparators": [
        "--snrs=-9,-6", "--rhos=0.2,0.4", "--attacks=none,masking",
        "--extreme-fractions=0.5", "--caps=8", "--fixed-blocks=20,50",
        "--pipelines=NoDefense,MLG+TM,MAD,Sn,Qn,TTF-CF,Trim10,Trim20,Trim40,Median,Huber,BetaHistory,BDFHistory,OracleRemoval,XS-PH,XS-fixed",
        "--score-cal-start=86000", "--score-cal-count=150", "--tau-cal-start=86200", "--tau-cal-count=150",
        "--test-start=87000", "--test-count=96", "--skip-user-telemetry",
    ],
    "B10_persistence": [
        "--snrs=-9,-6", "--rhos=0.2,0.4", "--attacks=none,masking",
        "--extreme-fractions=0.5", "--persistence-ws=0,0.5,0.9", "--caps=8", "--fixed-blocks=20,50",
        "--pipelines=NoDefense,Huber,BetaHistory,BDFHistory,TTF-CF,XS-PH,XS-fixed",
        "--score-cal-start=88000", "--score-cal-count=150", "--tau-cal-start=88200", "--tau-cal-count=150",
        "--test-start=89000", "--test-count=96", "--skip-user-telemetry", "--skip-round-telemetry",
    ],
}


def command_for(name: str, outroot: Path) -> list[str]:
    if name not in RUNS:
        raise ValueError(f"Unknown run: {name}")
    return [sys.executable, str(RUNNER), f"--out={outroot / name}", *COMMON, *RUNS[name]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="all", help="all or one run key: " + ", ".join(RUNS))
    ap.add_argument("--outroot", default=str(OUTROOT))
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()
    outroot = Path(args.outroot).resolve()
    names = list(RUNS) if args.run == "all" else [args.run]
    unknown = set(names) - set(RUNS)
    if unknown:
        ap.error(f"unknown run name(s): {sorted(unknown)}")
    print("B1-V3 is not modified. Supplemental runs use disjoint pools, except B5 which deliberately replays the first 48 B2-split test seeds for telemetry only.")
    for name in names:
        cmd = command_for(name, outroot)
        print("\n###", name)
        print(" ".join(cmd))
        if args.plan_only:
            continue
        outdir = outroot / name
        if outdir.exists() and any(outdir.iterdir()):
            raise SystemExit(f"Refusing to append to or overwrite existing output: {outdir}. Choose a fresh --outroot or inspect/remove it intentionally.")
        outdir.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Run {name} failed with exit code {exc.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                raise
    if not args.plan_only:
        print("\nAll requested supplemental runs finished. Next run analyze_supplement_b1_b10.py --root", outroot)

if __name__ == "__main__":
    main()
