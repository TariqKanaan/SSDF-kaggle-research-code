#!/usr/bin/env python3
"""Integrity/seed-pool checks for supplemental B2/B5/B6/B7/B8/B9/B10 outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

INFERENTIAL_RUNS = ["B2_rho_cap", "B2_extreme_mild_split", "B6_B7_B8_B9_comparators", "B10_persistence"]
B1V3_POOLS = {
    "score_cal": set(range(72000, 72150)),
    "tau_cal": set(range(72200, 72350)),
    "test": set(range(73000, 73096)),
}
EXPECTED = {
    "B2_rho_cap": {"user": False, "round": False},
    "B2_extreme_mild_split": {"user": False, "round": False},
    "B5_telemetry_replay": {"user": True, "round": True},
    "B6_B7_B8_B9_comparators": {"user": False, "round": True},
    "B10_persistence": {"user": False, "round": False},
}


def read_manifest(path: Path):
    m = json.loads((path / "v2_manifest.json").read_text(encoding="utf-8"))
    led = json.loads((path / "v2_seed_ledger.json").read_text(encoding="utf-8"))
    return m, led


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    checks = []
    pools = {}
    run_reports = {}
    errors = []
    for name in INFERENTIAL_RUNS + ["B5_telemetry_replay"]:
        path = root / name
        if not (path / "v2_manifest.json").exists():
            run_reports[name] = {"present": False}
            continue
        m, ledger = read_manifest(path)
        final = next((x for x in reversed(ledger) if x.get("stage") == "completed"), None)
        p = m.get("seed_pools", {})
        sp = set(p.get("score_cal", [])); tp = set(p.get("tau_cal", [])); ep = set(p.get("test", []))
        internal_disjoint = not (sp & tp or sp & ep or tp & ep)
        b1_disjoint = not ((sp | tp | ep) & set.union(*B1V3_POOLS.values()))
        files = {"seed": (path / "v2_seed_level.csv.gz").exists(),
                 "round": (path / "v2_pipeline_round.csv.gz").exists(),
                 "user": (path / "v2_user_decisions.csv.gz").exists(),
                 "observations": (path / "v2_user_observations.csv.gz").exists(),
                 "displacement": (path / "v2_displacement.csv.gz").exists(),
                 "oracle_subsets": (path / "v2_oracle_calibration_subsets.csv.gz").exists()}
        expected = EXPECTED[name]
        files_ok = files["seed"] and (files["round"] if expected["round"] else True) and (files["user"] and files["observations"] and files["displacement"] if expected["user"] else True)
        df = pd.read_csv(path / "v2_seed_level.csv.gz") if files["seed"] else pd.DataFrame()
        keycols = [c for c in ["seed", "snr_db", "rho_requested", "attack", "extreme_fraction", "persistence_w", "pipeline"] if c in df.columns]
        dup = int(df.duplicated(keycols).sum()) if len(df) and keycols else None
        ok = bool(final and internal_disjoint and b1_disjoint and files_ok and dup == 0 and m.get("study_version") == "B1-B10-supplement-instrumented")
        if not ok:
            errors.append(name)
        pools[name] = {"score": sp, "tau": tp, "test": ep}
        run_reports[name] = {
            "present": True, "status": "PASS" if ok else "CHECK",
            "study_version": m.get("study_version"),
            "cells_completed": final.get("cells_completed") if final else None,
            "test_seed_count_declared": len(ep), "seed_level_rows": int(len(df)),
            "duplicate_seed_pipeline_rows": dup,
            "calibration_test_pools_disjoint": internal_disjoint,
            "disjoint_from_frozen_B1V3_pools": b1_disjoint,
            "required_files": files, "required_files_ok": files_ok,
            "seed_pools": {"score_cal": [min(sp), max(sp)] if sp else [],
                            "tau_cal": [min(tp), max(tp)] if tp else [],
                            "test": [min(ep), max(ep)] if ep else []},
        }
    # B5 must intentionally replay exactly the first 48 test seeds of B2 split.
    b2 = pools.get("B2_extreme_mild_split", {})
    b5 = pools.get("B5_telemetry_replay", {})
    replay_ok = False
    if b2 and b5:
        replay_ok = (b5["test"].issubset(b2["test"]) and
                     b5["score"] == b2["score"] and b5["tau"] == b2["tau"] and
                     len(b5["test"]) == 48)
    run_reports["B5_replay_relationship"] = {"status": "PASS" if replay_ok else "CHECK",
                                             "exact_48_seed_subset_of_B2_split": replay_ok,
                                             "note": "B5 rows are a telemetry replay, not new independent test data."}
    if not replay_ok and b2 and b5:
        errors.append("B5_replay_relationship")
    # Between inferential runs, test pools must be disjoint; B5 is excluded intentionally.
    seen = {}
    cross = []
    for name in INFERENTIAL_RUNS:
        if name not in pools: continue
        for other, old in seen.items():
            overlap = pools[name]["test"] & old
            if overlap: cross.append({"run_a": other, "run_b": name, "n_overlap": len(overlap)})
        seen[name] = pools[name]["test"]
    if cross: errors.append("cross_run_test_seed_overlap")
    report = {"root": str(root), "run_reports": run_reports,
              "cross_run_test_seed_overlaps": cross,
              "integrity_status": "PASS" if not errors else "CHECK",
              "problems": errors,
              "important_note": "B1-V3 seed pools remain frozen. B5 is an intentional replay of B2 split seeds for telemetry, not an independent inferential sample."}
    out = Path(args.out).resolve() if args.out else root / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUPPLEMENT_INTEGRITY_REPORT.json").write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")
    print(json.dumps(report, indent=2, default=list))
    if errors: raise SystemExit(2)

if __name__ == "__main__":
    main()
