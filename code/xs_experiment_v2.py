"""V2 experiment generation with explicit session/subgroup metadata.

For session_spec='fixed:20', the random streams and generation order match the V1
run_final_experiment.generate_paired_condition implementation. Other session schedules
are generated independently of sensing/reporting random streams.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable
import numpy as np
import run_final_experiment as R
import system_model_final as sm


@dataclass(frozen=True)
class SessionMeta:
    session_id: np.ndarray
    session_start: np.ndarray
    session_end: np.ndarray
    session_length: np.ndarray
    boundaries: list[int]
    lengths: list[int]
    mode: str


def _parse_session_spec(spec: str | int | None, default_len: int) -> tuple[str, list[int]]:
    if spec is None:
        return "fixed", [int(default_len)]
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError("session length must be positive")
        return "fixed", [int(spec)]
    s = str(spec).strip().lower()
    if s in {"default", "fixed"}:
        return "fixed", [int(default_len)]
    if ":" not in s:
        try:
            n = int(s)
        except ValueError as e:
            raise ValueError(f"invalid session spec {spec!r}; use fixed:N, cycle:A,B or random:A,B") from e
        if n <= 0:
            raise ValueError("session length must be positive")
        return "fixed", [n]
    mode, values = s.split(":", 1)
    lengths = [int(v.strip()) for v in values.split(",") if v.strip()]
    if not lengths or any(v <= 0 for v in lengths):
        raise ValueError(f"invalid session lengths in {spec!r}")
    if mode not in {"fixed", "cycle", "random", "random_balanced"}:
        raise ValueError("session mode must be fixed, cycle, random, or random_balanced")
    if mode == "fixed" and len(lengths) != 1:
        raise ValueError("fixed session spec requires exactly one length")
    return mode, lengths


@lru_cache(maxsize=128)
def _balanced_count_options(rounds: int, choices_tuple: tuple[int, ...]):
    import itertools
    vals = list(choices_tuple)
    max_count = rounds // min(vals) + 1
    feasible = []
    for counts in itertools.product(range(1, max_count + 1), repeat=len(vals)):
        used = sum(c * v for c, v in zip(counts, vals))
        remainder = rounds - used
        if remainder < 0:
            continue
        if remainder != 0 and not (min(vals) <= remainder <= max(vals)):
            continue
        spread = max(counts) - min(counts)
        variance = sum((c - sum(counts) / len(counts)) ** 2 for c in counts)
        rem_penalty = 0 if remainder == 0 else abs(remainder - (min(vals) + max(vals)) / 2)
        feasible.append(((spread, variance, rem_penalty), counts, remainder))
    if not feasible:
        return tuple()
    best_score = min(x[0] for x in feasible)
    return tuple((x[1], x[2]) for x in feasible if x[0] == best_score)


def _balanced_variable_lengths(rounds: int, choices: list[int], rng: np.random.Generator) -> list[int]:
    """Create a randomized, approximately balanced multiset of session lengths.

    Each requested nominal length is represented at least once. We choose counts
    with the smallest count spread whose total leaves either no remainder or a
    terminal remainder between min(choices) and max(choices). The resulting
    multiset is shuffled, so the order is unpredictable to the detector. This is
    a genuinely count-balanced variable schedule, unlike independent sampling.
    """
    vals = [int(v) for v in choices]
    if rounds <= 0 or not vals:
        raise ValueError("rounds and session choices must be positive")
    if rounds < sum(vals):
        # Small smoke tests cannot include every requested duration. Fall back
        # to random-variable sampling with a safe terminal segment.
        out = []
        remaining = rounds
        while remaining:
            if remaining <= max(vals):
                length = remaining
            else:
                feasible = [v for v in vals if remaining - v == 0 or remaining - v >= min(vals)]
                length = int(rng.choice(feasible or vals))
            out.append(min(length, remaining))
            remaining -= out[-1]
        rng.shuffle(out)
        return out

    options = _balanced_count_options(int(rounds), tuple(vals))
    if not options:
        raise RuntimeError(f"No balanced session schedule can cover {rounds} rounds with {vals}")
    counts, remainder = options[int(rng.integers(0, len(options)))]
    schedule = [v for count, v in zip(counts, vals) for _ in range(count)]
    if remainder:
        schedule.append(int(remainder))
    rng.shuffle(schedule)
    if sum(schedule) != rounds or any(v < min(vals) or v > max(vals) for v in schedule):
        raise RuntimeError("Internal error constructing balanced session schedule")
    return schedule


def make_session_meta(rounds: int, spec: str | int | None, default_len: int,
                      seed: int, condition_id: int) -> SessionMeta:
    mode, choices = _parse_session_spec(spec, default_len)
    schedule_rng = R.rng(seed, 140_000 + int(condition_id))
    session_id = np.empty(rounds, dtype=int)
    starts = np.zeros(rounds, dtype=bool)
    ends = np.zeros(rounds, dtype=bool)
    lengths_by_round = np.empty(rounds, dtype=int)
    boundaries: list[int] = []
    lengths: list[int] = []

    if mode == "random_balanced":
        planned_lengths = _balanced_variable_lengths(rounds, choices, schedule_rng)
    else:
        planned_lengths = None

    t = 0
    sid = 0
    while t < rounds:
        if mode == "fixed":
            length = choices[0]
        elif mode == "cycle":
            length = choices[sid % len(choices)]
        elif mode == "random":
            remaining = rounds - t
            if remaining <= max(choices):
                length = remaining
            else:
                feasible = [v for v in choices if remaining - v == 0 or remaining - v >= min(choices)]
                length = int(schedule_rng.choice(feasible or choices))
        elif mode == "random_balanced":
            length = int(planned_lengths[sid])
        actual = min(length, rounds - t)
        session_id[t:t + actual] = sid
        starts[t] = True
        ends[t + actual - 1] = True
        lengths_by_round[t:t + actual] = actual
        boundaries.append(t)
        lengths.append(actual)
        t += actual
        sid += 1
    return SessionMeta(session_id, starts, ends, lengths_by_round, boundaries, lengths, mode)



_PERSISTENCE_REFERENCE: dict[str, tuple[float, float]] = {}

def _persistence_reference(c) -> dict[str, tuple[float, float]]:
    """Reference mean/SD of log-gain marginals used by the V1 stress generator."""
    if not _PERSISTENCE_REFERENCE:
        r = np.random.default_rng(12345)
        h = np.log(sm.sample_f_composite_gain(200_000, c.F_M, c.F_MS, r))
        g = np.log(r.rayleigh(scale=c.RAYLEIGH_SCALE, size=200_000))
        _PERSISTENCE_REFERENCE["h"] = (float(h.mean()), float(h.std()))
        _PERSISTENCE_REFERENCE["g"] = (float(g.mean()), float(g.std()))
    return _PERSISTENCE_REFERENCE

def _mix_log_gain(persistent: np.ndarray, session_value: np.ndarray, w: float, ref: tuple[float, float]) -> np.ndarray:
    mu, sd = ref
    z = np.sqrt(w) * (np.log(persistent) - mu) / sd + np.sqrt(1.0 - w) * (np.log(session_value) - mu) / sd
    return np.exp(mu + sd * z)

def _mix_persistent_session(c, persistent: tuple[np.ndarray, np.ndarray, np.ndarray],
                            session_value: tuple[np.ndarray, np.ndarray, np.ndarray], w: float):
    ref = _persistence_reference(c)
    h = _mix_log_gain(persistent[0], session_value[0], w, ref["h"])
    g = _mix_log_gain(persistent[1], session_value[1], w, ref["g"])
    unc = np.clip(np.sqrt(w) * persistent[2] + np.sqrt(1.0 - w) * session_value[2],
                  -c.NOISE_UNC_DB, c.NOISE_UNC_DB)
    return h, g, unc

def _apply_attack_groups(T: np.ndarray, mal: np.ndarray, extreme: np.ndarray,
                         mild: np.ndarray, attack: str, shift_db: float,
                         extreme_db: float, mild_db: float) -> np.ndarray:
    out = np.asarray(T, dtype=float).copy()
    if attack == "none" or mal.size == 0:
        return out
    if attack == "masking":
        if extreme.size:
            out[extreme] *= 10.0 ** (float(extreme_db) / 10.0)
        if mild.size:
            out[mild] *= 10.0 ** (float(mild_db) / 10.0)
    elif attack in {"always_yes", "shift", "small_upward_shift"}:
        out[mal] *= 10.0 ** (abs(float(shift_db)) / 10.0)
    elif attack in {"always_no", "small_downward_shift"}:
        out[mal] /= 10.0 ** (abs(float(shift_db)) / 10.0)
    else:
        raise ValueError(f"unknown attack {attack!r}")
    return out


def generate_paired_condition_v2(
    c,
    seed: int,
    condition_id: int,
    snr_db: float,
    rho: float,
    attack: str,
    rounds: int,
    *,
    session_spec: str | int | None = None,
    extreme_fraction: float = 0.5,
    attack_shift_db: float | None = None,
    subgroup_assignment: str = "legacy_sorted",
    persistence_w: float = 0.0,
) -> dict:
    """Generate paired H0/H1 clean/attacked sequences plus complete ground-truth metadata.

    `extreme_fraction` is independent of rho. For masking, the number of extreme
    attackers is floor(L * extreme_fraction); the remainder are mild attackers.
    At fraction=0.5 this exactly retains V1's floor(L/2) split.
    """
    if not (0.0 <= float(extreme_fraction) <= 1.0):
        raise ValueError("extreme_fraction must be in [0,1]")
    if not (0.0 <= float(persistence_w) < 1.0):
        raise ValueError("persistence_w must lie in [0,1)")
    mal = sm.choose_malicious_idx(c.N, rho if attack != "none" else 0.0, seed, condition_id)
    n_extreme = int(np.floor(len(mal) * float(extreme_fraction)))
    if subgroup_assignment not in {"legacy_sorted", "random"}:
        raise ValueError("subgroup_assignment must be 'legacy_sorted' or 'random'")
    if subgroup_assignment == "random" and len(mal):
        subgroup_rng = R.rng(seed, 750_000 + int(condition_id))
        subgroup_order = subgroup_rng.permutation(mal)
    else:
        subgroup_order = mal.copy()
    extreme = subgroup_order[:n_extreme].copy() if attack == "masking" else np.empty(0, dtype=int)
    mild = subgroup_order[n_extreme:].copy() if attack == "masking" else np.empty(0, dtype=int)

    labels = np.full(c.N, "benign", dtype=object)
    labels[mal] = "malicious"
    subtype = np.full(c.N, "none", dtype=object)
    if attack == "masking":
        subtype[extreme] = "extreme"
        subtype[mild] = "mild"
    elif attack in {"always_yes", "shift", "small_upward_shift"}:
        subtype[mal] = "small_upward_shift"
    elif attack in {"always_no", "small_downward_shift"}:
        subtype[mal] = "small_downward_shift"

    session = make_session_meta(rounds, session_spec, c.SESSION_LENGTH, seed, condition_id)
    env = R.rng(seed, 100_000 + condition_id)
    h1_rng = R.rng(seed, 110_000 + condition_id)
    h0_rng = R.rng(seed, 120_000 + condition_id)
    atk_rng = R.rng(seed, 130_000 + condition_id)  # reserved to keep V1 stream contract

    H1_clean, H1_att, H0_clean, H0_att = [], [], [], []
    session_states = []
    h = g = unc = None
    persistent_state = R.make_session(c, R.rng(seed, 150_000 + condition_id)) if persistence_w > 0 else None
    shift_db = c.SHIFT_DB if attack_shift_db is None else float(attack_shift_db)

    for r in range(rounds):
        if session.session_start[r]:
            session_state = R.make_session(c, env)
            if persistence_w > 0:
                h, g, unc = _mix_persistent_session(c, persistent_state, session_state, float(persistence_w))
            else:
                h, g, unc = session_state
            session_states.append({"h": h.copy(), "g": g.copy(), "unc_db": unc.copy()})
        T1 = sm.local_energy_gamma(c.N, c.M_SAMPLES, snr_db, "H1", h, unc, h1_rng)
        T0 = sm.local_energy_gamma(c.N, c.M_SAMPLES, snr_db, "H0", h, unc, h0_rng)
        eta1 = env.normal(0.0, c.REPORT_NOISE_STD, size=c.N)
        eta0 = env.normal(0.0, c.REPORT_NOISE_STD, size=c.N)
        H1_clean.append(sm.reporting_channel(T1, g, eta1))
        H0_clean.append(sm.reporting_channel(T0, g, eta0))
        T1a = _apply_attack_groups(T1, mal, extreme, mild, attack, shift_db,
                                   c.MASK_EXTREME_DB, c.MASK_MILD_DB)
        T0a = _apply_attack_groups(T0, mal, extreme, mild, attack, shift_db,
                                   c.MASK_EXTREME_DB, c.MASK_MILD_DB)
        H1_att.append(sm.reporting_channel(T1a, g, eta1))
        H0_att.append(sm.reporting_channel(T0a, g, eta0))

    # Ground-truth sequence table is repeated only once per round; user-level rows
    # are materialized by the runner to avoid large duplicated metadata in memory.
    return {
        "malicious_idx": mal,
        "extreme_idx": extreme,
        "mild_idx": mild,
        "user_type": labels,
        "attacker_subtype": subtype,
        "H1_clean": H1_clean,
        "H1_att": H1_att,
        "H0_clean": H0_clean,
        "H0_att": H0_att,
        "session": session,
        "session_states": session_states,
        "attack": attack,
        "rho_requested": float(rho),
        "rho_actual": float(len(mal) / c.N),
        "extreme_fraction_requested": float(extreme_fraction),
        "extreme_count": int(len(extreme)),
        "mild_count": int(len(mild)),
        "session_spec": str(session_spec if session_spec is not None else f"fixed:{c.SESSION_LENGTH}"),
        "attack_shift_db": float(shift_db),
        "subgroup_assignment": subgroup_assignment,
        "persistence_w": float(persistence_w),
        "persistence_model": "log-domain sqrt(w) persistent + sqrt(1-w) session; unc linear mixture with clipping" if persistence_w > 0 else "independent-session (w=0)",
    }
