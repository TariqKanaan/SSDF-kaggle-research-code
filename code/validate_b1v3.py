"""Fast validation suite for B1-V3 before any costly experiment is launched."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent

REQUIRED = [
    "run_xs_study_v2.py", "xs_experiment_v2.py", "xs_filter_v2.py",
    "xs_config.py", "run_final_experiment.py", "system_model_final.py",
    "ttf_cf_final.py", "config_final.py", "analyze_v2.py",
    "analyze_b1v3.py", "select_b1v3_development.py",
    "run_b1v3_development.py", "run_b1v3_final.py",
]


def main():
    missing = [x for x in REQUIRED if not (ROOT / x).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing B1-V3 package files: {missing}")

    # Parse Python source without generating __pycache__ in the package input.
    py_files = list(ROOT.glob("*.py"))
    for p in py_files:
        compile(p.read_text(encoding="utf-8"), str(p), "exec")

    from xs_experiment_v2 import make_session_meta
    rounds = 200
    choices = [13, 20, 30, 50]
    schedules_seen = set()
    for seed in range(1000, 1020):
        m = make_session_meta(rounds, "random_balanced:13,20,30,50", 20, seed, 810000)
        assert m.boundaries[0] == 0
        assert sum(m.lengths) == rounds
        assert all(13 <= int(x) <= 50 for x in m.lengths)
        nominal_counts = [sum(int(x) == v for x in m.lengths) for v in choices]
        assert min(nominal_counts) >= 1
        assert max(nominal_counts) - min(nominal_counts) <= 1
        assert len(m.lengths) == len(m.boundaries)
        schedules_seen.add(tuple(m.lengths))
        assert np.array_equal(m.session_id, np.repeat(np.arange(len(m.lengths)), m.lengths))
        for i, (start, length) in enumerate(zip(m.boundaries, m.lengths)):
            assert int(start) == (0 if i == 0 else m.boundaries[i-1] + m.lengths[i-1])
            assert int(m.session_end[start + length - 1]) == 1

    assert len(schedules_seen) > 1, "Random variable schedules are not varying across seeds"

    # The declared calibration/test pools are disjoint.
    dev = set(range(64000, 64040))
    dev_score = set(range(65000, 65080))
    dev_tau = set(range(65100, 65180))
    final_score = set(range(72000, 72150))
    final_tau = set(range(72200, 72350))
    final_test_max = set(range(73000, 83000))  # up to the declared 10,000-seed planning cap
    pools = [dev, dev_score, dev_tau, final_score, final_tau, final_test_max]
    for i in range(len(pools)):
        for j in range(i + 1, len(pools)):
            assert pools[i].isdisjoint(pools[j])

    # Verify that the patched runner exposes all PH parameters.
    runner = (ROOT / "run_xs_study_v2.py").read_text(encoding="utf-8")
    for flag in ["--ph-delta", "--ph-threshold", "--ph-beta", "--ph-warmup", "--min-block"]:
        assert flag in runner, f"Runner missing {flag}"
    assert 'if not b.get("final_open_block", False)' in runner
    assert 'collect_user_telemetry=not a.skip_user_telemetry' in runner

    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob("*.py"))}
    report = {
        "status": "PASS",
        "python_files_parsed": len(py_files),
        "random_balanced_schedules_checked": 20,
        "random_schedule_count_balance_max_spread": 1,
        "distinct_random_schedules_in_validation": len(schedules_seen),
        "rounds_per_schedule": rounds,
        "session_lengths": choices,
        "seed_pools_disjoint": True,
        "final_test_seed_pool_max_size_checked": 10000,
        "ph_cli_parameters_present": True,
        "closed_block_double_logging_patch_present": True,
        "compact_b1_telemetry_supported": True,
        "source_sha256": hashes,
    }
    (ROOT / "b1v3_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))
    print("B1-V3 validation passed.")


if __name__ == "__main__":
    main()
