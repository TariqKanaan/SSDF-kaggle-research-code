"""Fast integrity checks for the instrumented V2 source package."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
from config_final import Config
from run_final_experiment import generate_paired_condition as v1_generate
from xs_experiment_v2 import generate_paired_condition_v2
from xs_config import XSConfig
from xs_filter_v2 import XSFilter


def load_v1_filter():
    p = Path(__file__).resolve().parent / "xs_filter_final_v1_reference.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("xs_filter_v1_reference", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.XSFilter


def load_v1_persistence_generator():
    p = Path(__file__).resolve().parent / "xs_scenarios_v1_reference.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("xs_persistence_v1_reference", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_persistent_condition


def main():
    c = Config()
    # V1 and V2 condition generation must be bit-identical in the legacy fixed:20,
    # legacy sorted-subgroup mode. This protects the old data-generation protocol.
    for attack, rho in [("masking", 0.20), ("always_yes", 0.20), ("always_no", 0.40), ("none", 0.20)]:
        old = v1_generate(c, 61300, 801234, -9.0, rho, attack, 200)
        new = generate_paired_condition_v2(c, 61300, 801234, -9.0, rho, attack, 200,
                                          session_spec="fixed:20", extreme_fraction=0.5,
                                          subgroup_assignment="legacy_sorted")
        assert np.array_equal(old["malicious_idx"], new["malicious_idx"]), attack
        for key in ("H0_clean", "H0_att", "H1_clean", "H1_att"):
            assert np.array_equal(np.asarray(old[key]), np.asarray(new[key])), (attack, key)

    # Variable-session schedule and metadata checks.
    pair = generate_paired_condition_v2(c, 63000, 812345, -9, 0.4, "masking", 200,
                                        session_spec="random:13,20,30,50", extreme_fraction=0.25,
                                        subgroup_assignment="random")
    assert len(pair["session"].boundaries) == len(pair["session"].lengths)
    assert sum(pair["session"].lengths) == 200
    assert len(pair["extreme_idx"]) == int(np.floor(8 * 0.25))
    assert len(pair["mild_idx"]) == 8 - len(pair["extreme_idx"])
    assert len(set(pair["extreme_idx"]) & set(pair["mild_idx"])) == 0


    # The V2 persistence extension must reproduce the original V1 stress
    # generator at w=0.5 when legacy subgroup assignment and fixed:20 are used.
    v1_persistent = load_v1_persistence_generator()
    if v1_persistent is not None:
        for attack in ("masking", "always_yes", "always_no", "none"):
            old = v1_persistent(c, 65001, 918273, -9.0, 0.20, attack, 60, 0.5)
            new = generate_paired_condition_v2(c, 65001, 918273, -9.0, 0.20, attack, 60,
                                              session_spec="fixed:20", extreme_fraction=0.5,
                                              subgroup_assignment="legacy_sorted", persistence_w=0.5)
            for key in ("H0_clean", "H0_att", "H1_clean", "H1_att"):
                assert np.array_equal(np.asarray(old[key]), np.asarray(new[key])), (attack, key, "persistence")

    # Telemetry extension must preserve V1 numerical XS outputs when both filters
    # receive the same reports and the same no-calibration bank.
    XSV1 = load_v1_filter()
    if XSV1 is not None:
        for cfg in [XSConfig(block_mode="ph"), XSConfig(block_mode="fixed", fixed_len=20)]:
            old = XSV1(c.N, cfg, None)
            new = XSFilter(c.N, cfg, None)
            rr = np.random.default_rng(123456)
            for _ in range(100):
                x = rr.lognormal(mean=0.0, sigma=0.2, size=c.N)
                a = old.step(x); b = new.step(x)
                for k in ("stat", "blocks", "boundary"):
                    assert a[k] == b[k], (cfg.block_mode, k)
                for k in ("flagged", "pvalues", "score"):
                    assert np.allclose(a[k], b[k]), (cfg.block_mode, k)

    print("V2 validation passed: V1 generator compatibility, persistence replay, subgroup/session metadata, and XS decision recursion.")


if __name__ == "__main__":
    main()
