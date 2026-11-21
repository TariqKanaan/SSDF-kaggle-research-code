from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import os


@dataclass(frozen=True)
class Config:
    # Physical/sensing model
    N: int = 20
    M_SAMPLES: int = 10_000
    F_M: float = 1.3
    F_MS: float = 2.7
    NOISE_UNC_DB: float = 2.0
    RAYLEIGH_SCALE: float = 1.0
    REPORT_NOISE_STD: float = 0.05
    SESSION_LENGTH: int = 20

    # Classical detector constants
    TM_ALPHA: float = 0.05
    TM_N_SIM: int = 10_000
    MAD_C: float = 3.0
    QN_CONSTANT: float = 2.219144465985076  # corrected normal-consistency constant

    # Paradigm A
    PA_IF_ESTIMATORS: int = 200
    PA_OCSVM_NU: float = 0.10
    PA_LOF_N_NEIGHBORS: int = 20
    PA_AE_HIDDEN: int = 8
    PA_AE_MAX_ITER: int = 1000
    PA_FEATURES: tuple = ("raw", "median_dev", "mad_z")
    PA_ENSEMBLE_VOTE_FRACTION: float = 0.5

    # TTF-CF primary configuration; fixed before final evaluation
    TTF_LAMBDA: float = 0.20
    TTF_CUSUM_K: float = 0.50
    TTF_W: int = 20
    TTF_KAPPA_SIGN: float = 0.80
    TTF_KAPPA_P: float = 5.0
    TTF_Q: float = 0.05
    TTF_RHO_MAX: float = 0.40
    TTF_PH_DELTA: float = 0.05
    TTF_PH_THRESHOLD: float = 5.0
    TTF_PH_BETA: float = 0.05
    TTF_PH_WARMUP: int = 5
    TTF_DOMAIN: str = "linear"
    TTF_FUSION: str = "mean"

    # Attack model
    SHIFT_DB: float = 0.5
    MASK_EXTREME_DB: float = 6.5
    MASK_MILD_DB: float = 0.5

    # Main final evaluation design
    SNR_GRID: tuple = (-18.0, -12.0, -9.0, -6.0)
    MAIN_RHOS: tuple = (0.20, 0.40)
    STRESS_RHO: float = 0.50
    ATTACKS: tuple = ("none", "always_yes", "masking", "always_no")
    EVAL_ROUNDS: int = 200

    # Temporal regime study
    REGIME_ROUNDS: int = 600
    REGIME_PHASES: tuple = (
        (0, 200, "always_yes", 0.5),
        (200, 400, "masking", 6.5),
        (400, 600, "always_no", 0.5),
    )

    # LEB study
    LEB_ROUNDS: int = 600
    LEB_ALPHA: float = 0.85
    LEB_MIN_CLASS: int = 20
    LEB_PERTURBATION_STEP_DB: float = 0.25
    LEB_MAX_PERTURB_DB: float = 1.5

    # Seed pools: explicitly disjoint
    DEV_SEED_START: int = 9000
    DEV_SEEDS_N: int = 10
    SCORE_CAL_SEED_START: int = 9100
    SCORE_CAL_SEEDS_N: int = 10
    TAU_CAL_SEED_START: int = 9200
    TAU_CAL_SEEDS_N: int = 10
    EVAL_SEED_START: int = 9300
    EVAL_SEEDS_N: int = 20
    REGIME_SEED_START: int = 9400
    REGIME_SEEDS_N: int = 20
    LEB_SEED_START: int = 9500
    LEB_SEEDS_N: int = 20

    DEV_ROUNDS: int = 200
    SCORE_CAL_ROUNDS: int = 200
    TAU_CAL_ROUNDS: int = 200

    OUTPUT_DIR: str = "ssdf_final_results"
    TM_CACHE_FILE: str = "tm_cache.pkl"


def cfg_from_env() -> Config:
    c = Config()
    updates = {}
    for env_name, field in [
        ("SSDF_DEV_SEEDS_N", "DEV_SEEDS_N"),
        ("SSDF_SCORE_CAL_SEEDS_N", "SCORE_CAL_SEEDS_N"),
        ("SSDF_TAU_CAL_SEEDS_N", "TAU_CAL_SEEDS_N"),
        ("SSDF_EVAL_SEEDS_N", "EVAL_SEEDS_N"),
        ("SSDF_REGIME_SEEDS_N", "REGIME_SEEDS_N"),
        ("SSDF_LEB_SEEDS_N", "LEB_SEEDS_N"),
    ]:
        if env_name in os.environ:
            updates[field] = int(os.environ[env_name])
    if "SSDF_TM_N_SIM" in os.environ:
        updates["TM_N_SIM"] = int(os.environ["SSDF_TM_N_SIM"])
    for env_name, field in [
        ("SSDF_DEV_ROUNDS", "DEV_ROUNDS"),
        ("SSDF_SCORE_CAL_ROUNDS", "SCORE_CAL_ROUNDS"),
        ("SSDF_TAU_CAL_ROUNDS", "TAU_CAL_ROUNDS"),
        ("SSDF_EVAL_ROUNDS", "EVAL_ROUNDS"),
        ("SSDF_REGIME_ROUNDS", "REGIME_ROUNDS"),
    ]:
        if env_name in os.environ:
            updates[field] = int(os.environ[env_name])
    if "SSDF_OUTPUT_DIR" in os.environ:
        updates["OUTPUT_DIR"] = os.environ["SSDF_OUTPUT_DIR"]
    if "SSDF_SNR_GRID" in os.environ:
        updates["SNR_GRID"] = tuple(float(x) for x in os.environ["SSDF_SNR_GRID"].split(",") if x.strip())
    if "SSDF_MAIN_RHOS" in os.environ:
        updates["MAIN_RHOS"] = tuple(float(x) for x in os.environ["SSDF_MAIN_RHOS"].split(",") if x.strip())
    if "SSDF_STRESS_RHO" in os.environ:
        updates["STRESS_RHO"] = float(os.environ["SSDF_STRESS_RHO"])
    if "SSDF_ATTACKS" in os.environ:
        updates["ATTACKS"] = tuple(x.strip() for x in os.environ["SSDF_ATTACKS"].split(",") if x.strip())
    if updates:
        c = Config(**{**asdict(c), **updates})
    return c


def seed_pool(start: int, n: int) -> list[int]:
    return list(range(int(start), int(start) + int(n)))


def all_seed_pools(c: Config) -> dict[str, list[int]]:
    return {
        "development": seed_pool(c.DEV_SEED_START, c.DEV_SEEDS_N),
        "score_calibration": seed_pool(c.SCORE_CAL_SEED_START, c.SCORE_CAL_SEEDS_N),
        "tau_calibration": seed_pool(c.TAU_CAL_SEED_START, c.TAU_CAL_SEEDS_N),
        "evaluation": seed_pool(c.EVAL_SEED_START, c.EVAL_SEEDS_N),
        "regime": seed_pool(c.REGIME_SEED_START, c.REGIME_SEEDS_N),
        "leb": seed_pool(c.LEB_SEED_START, c.LEB_SEEDS_N),
    }
