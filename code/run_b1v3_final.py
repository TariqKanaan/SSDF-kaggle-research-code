"""Launch the frozen B1-V3 final test using development-selected PH and K-best settings."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-count", type=int, default=None,
                    help="Default: use the development-based sample-size recommendation. Override only with explicit justification.")
    ap.add_argument("--score-cal-count", type=int, default=150)
    ap.add_argument("--tau-cal-count", type=int, default=150)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-power-cap", action="store_true",
                    help="Explicitly allow running at the 10000-seed search cap even if development-based power target was not reached. Use only with a written justification.")
    ap.add_argument("--allow-underpowered-override", action="store_true",
                    help="Allow an explicit test-count smaller than the development-based recommendation. For debugging only; do not describe it as the planned full test.")
    a = ap.parse_args()

    selection_path = Path(a.selection).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    recommended = int(selection["sample_size_plan"]["recommended_independent_test_seeds_per_condition"])
    n_test = a.test_count if a.test_count is not None else recommended
    if n_test < 2:
        raise SystemExit("Final test requires at least two independent seeds")
    if a.test_count is not None and n_test < recommended and not a.allow_underpowered_override:
        raise SystemExit(f"Requested test-count {n_test} is below the development recommendation {recommended}. For debugging only, pass --allow-underpowered-override; otherwise use the recommended count.")
    if n_test > 10000:
        raise SystemExit("Recommended test count exceeds the declared maximum of 10000; inspect the variance/power plan.")
    if selection["sample_size_plan"].get("sample_size_search_cap_reached", False) and not a.allow_power_cap:
        raise SystemExit("The development-based power target was not reached within the 10000-seed planning cap. Do not call this adequately powered. Revisit the design or explicitly pass --allow-power-cap with a written justification.")

    # Explicitly guard the final test and its calibration pools against any
    # overlap with development or development-calibration seeds.
    dev_range = selection["development_seed_range"]
    dev_score_range = selection["score_calibration_seed_range"]
    dev_tau_range = selection["threshold_calibration_seed_range"]
    final_ranges = [(72000, 72000 + a.score_cal_count - 1),
                    (72200, 72200 + a.tau_cal_count - 1),
                    (73000, 73000 + n_test - 1)]
    prior_ranges = [dev_range, dev_score_range, dev_tau_range]
    for lo, hi in final_ranges:
        for plo, phi in prior_ranges:
            if max(lo, plo) <= min(hi, phi):
                raise SystemExit(f"Final seed pool [{lo},{hi}] overlaps development/calibration pool [{plo},{phi}].")
    for i, (lo, hi) in enumerate(final_ranges):
        for plo, phi in final_ranges[i+1:]:
            if max(lo, plo) <= min(hi, phi):
                raise SystemExit("Final score-calibration, threshold-calibration and test pools overlap.")

    ph = selection["selected_ph"]
    runner = Path(__file__).resolve().parent / "run_xs_study_v2.py"
    out = Path(a.out).resolve()
    cmd = [
        sys.executable, str(runner), "--out", str(out),
        "--snrs=-9,-6", "--rhos=0.2", "--attacks=masking",
        "--session-specs=fixed:13;fixed:20;fixed:30;fixed:50;random_balanced:13,20,30,50;fixed:200",
        "--extreme-fractions=0.5", "--persistence-ws=0.0",
        "--subgroup-assignment=random", "--caps=8",
        "--fixed-blocks=10,13,20,30,50", "--pipelines=XS-PH,XS-fixed",
        "--rounds=200", "--test-start=73000", f"--test-count={n_test}",
        "--score-cal-start=72000", f"--score-cal-count={a.score_cal_count}",
        "--tau-cal-start=72200", f"--tau-cal-count={a.tau_cal_count}",
        "--tau-quantile=0.95",
        f"--ph-delta={ph['delta']}", f"--ph-threshold={ph['threshold']}",
        f"--ph-beta={ph['beta']}", f"--ph-warmup={ph['warmup']}",
        "--min-block=5", "--skip-user-telemetry", "--skip-round-telemetry",
    ]
    if a.overwrite:
        cmd.append("--overwrite")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["SSDF_EVAL_ROUNDS"] = "200"
    env["SSDF_SCORE_CAL_ROUNDS"] = "200"
    env["SSDF_TAU_CAL_ROUNDS"] = "200"
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.parent / (out.name + "_run.log")
    print("B1-V3 frozen final run")
    print("PH config:", ph)
    print("Development-selected fixed K:", selection["selected_fixed_k"]["K"])
    print("Final test seeds:", n_test, "(recommended from development:", recommended, ")")
    print("Output:", out)
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), env=env,
                              stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Final run failed; inspect log: {log}")
    package_root = Path(__file__).resolve().parent
    package_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package_root.iterdir()) if p.is_file() and (p.suffix == ".py" or p.name in {"README_B1_V3_AR.md", "requirements.txt"})
    }
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    frozen_selection_copy = out / "B1V3_selection_frozen.json"
    shutil.copy2(selection_path, frozen_selection_copy)
    dev_protocol = selection_path.parent / "B1V3_development_protocol.json"
    if dev_protocol.exists():
        shutil.copy2(dev_protocol, out / "B1V3_development_protocol_frozen.json")

    run_info = {
        "selection_file": str(selection_path),
        "selection_sha256": selection_hash,
        "package_sha256": package_hashes,
        "selected_ph": ph,
        "selected_fixed_k": selection["selected_fixed_k"],
        "development_seed_range": dev_range,
        "development_score_calibration_seed_range": dev_score_range,
        "development_threshold_calibration_seed_range": dev_tau_range,
        "final_test_seed_start": 73000,
        "final_test_seed_count": n_test,
        "score_calibration_seed_start": 72000,
        "score_calibration_seed_count": a.score_cal_count,
        "threshold_calibration_seed_start": 72200,
        "threshold_calibration_seed_count": a.tau_cal_count,
        "rounds": 200,
        "tau_quantile": 0.95,
        "sample_size_search_cap_reached": bool(selection["sample_size_plan"].get("sample_size_search_cap_reached", False)),
        "power_cap_override_used": bool(a.allow_power_cap),
        "underpowered_test_count_override_used": bool(a.allow_underpowered_override),
        "session_specs": ["fixed:13", "fixed:20", "fixed:30", "fixed:50",
                          "random_balanced:13,20,30,50", "fixed:200"],
        "primary_test_session": "random_balanced:13,20,30,50",
        "status": "completed"
    }
    (out / "B1V3_final_run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print("B1-V3 final simulation completed:", out)
    print("Run log:", log)


if __name__ == "__main__":
    main()
