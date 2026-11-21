"""Frozen design for the cross-session residual (XS) statistic study.

Seed pools are disjoint from every pool in config_final.py (9000-9519), from the
Paradigm A model seeds (777+...), from validate_code.py seeds, and from the TM cache seed.
"""
from __future__ import annotations
from dataclasses import dataclass

XS_SEED_POOLS = {
    "xs_dev":        list(range(61000, 61010)),   # design smoke tests only; may be inspected
    "xs_score_cal":  list(range(61100, 61110)),   # conformal bank for XS
    "xs_tau_cal":    list(range(61200, 61210)),   # decision thresholds for ALL pipelines in this study
    "xs_test":       list(range(61300, 61320)),   # single-shot confirmatory test
}

OLD_POOLS_FORBIDDEN = (
    list(range(9000, 9010)) + list(range(9100, 9110)) + list(range(9200, 9210)) +
    list(range(9300, 9320)) + list(range(9400, 9420)) + list(range(9500, 9520))
)

@dataclass(frozen=True)
class XSConfig:
    q: float = 0.05            # BH level
    rho_max: float = 0.40      # cap on flagged fraction
    min_block: int = 5         # a block must contain >= this many rounds to count as evidence
    b_cap: int = 10            # block-count bucket cap for the conformal bank
    block_mode: str = "ph"     # "ph" = Page-Hinkley-detected blocks; "fixed" = fixed length
    fixed_len: int = 20
    ph_delta: float = 0.05
    ph_threshold: float = 5.0
    ph_beta: float = 0.05
    ph_warmup: int = 5
    fusion: str = "mean"

# Pre-registered confirmatory design (xs_test). Nothing here may change after freezing.
PREREG = {
    "primary_condition": {"snr_db": [-9.0, -6.0], "rho": [0.20], "attack": ["masking"]},
    "secondary_conditions": {"snr_db": [-18.0, -12.0, -9.0, -6.0], "rho": [0.20, 0.40],
                             "attack": ["always_yes", "masking", "always_no"]},
    "stress": {"w": [0.5, 0.9], "snr_db": [-9.0, -6.0], "rho": [0.20], "attack": ["masking", "none"],
               "note": ("Persistent-heterogeneity scenario (xs_scenarios.py). Conformal banks stay those built under the "
                        "independent-session assumption (deliberate mismatch); decision thresholds tau are re-calibrated "
                        "under the scenario on xs_tau_cal. Reported for every pipeline, including clean-H0 false-flag rate ffr_clean."),
               "rule": ("XS-PH's advantage is called robust to persistent heterogeneity only if the main decision rule is also met at "
                        "w=0.5 in the primary conditions. Results at w=0.9 and any ffr_clean(XS-PH) > 0.10 are reported as limitations "
                        "whether or not they favour XS-PH.")},
    "rounds": 200,
    "pipelines": ["NoDefense", "MLG+TM", "MAD", "Sn", "Qn", "ParadigmA", "TTF-CF",
                  "XS-PH", "XS-fixed20", "Trim10", "Trim20", "Trim40", "Median", "OracleRemoval"],
    "primary_metric": "J_attack",
    "decision_rule": ("XS-PH is reported as better than a baseline only if, in a primary condition, "
                      "the 95% paired-bootstrap CI of the J_attack difference excludes 0 AND the "
                      "Holm-adjusted exact sign-test p < 0.05 (Holm over 6 baselines x {J_attack, catch, ffr} within each condition). Otherwise it is reported as 'no detectable "
                      "difference'. Fusion-only baselines (Trim10/20/40, Median) are included in this rule; if any of them "
                      "is not beaten, XS-PH is reported as adding nothing beyond robust fusion in that condition. Catch and FFR are secondary and are never used to claim superiority."),
    "also_report": ["fraction_of_oracle_gain = (J - J_NoDefense)/(J_Oracle - J_NoDefense)", "catch", "ffr", "pd", "pfa_attack"],
}

DESIGN_LOG = [
    "Added persistent-heterogeneity stress scenario (w=0.5, 0.9) before freezing, because the base simulator redraws all nuisance factors independently per session, which is exactly the structure XS exploits.",
    "After inspecting xs_dev only (SNR -6, masking, rho=0.2): Trim20 (no filtering) reached J~0.70 vs XS-PH ~0.73 vs NoDefense ~0.22; "
    "XS-PH exceeded OracleRemoval at -9 dB because Oracle reuses the NoDefense threshold. Therefore fusion-only baselines "
    "(Trim10/20/40, Median) were added to the pre-registered comparison, and Oracle is reported but is NOT treated as an upper bound.",
]
