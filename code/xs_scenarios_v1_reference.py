"""Stress scenario: persistent user heterogeneity.

In the base simulator every session redraws (h_i, g_i, unc_i) independently. Here a fraction w of each
user's standardized log-gain variance is a PERSISTENT per-user component, fixed for the whole run:
    z_i(session) = sqrt(w) * z_persistent_i + sqrt(1-w) * z_session_i      (log-domain, standardized)
so the marginal mean/variance of log h and log g are preserved (shape becomes closer to Gaussian in log).
unc_i (uniform +-2 dB) is mixed the same way and clipped to +-2 dB.
w = 0 reproduces the base generator (regression-tested); w = 0.9 is strong persistence.
"""
from __future__ import annotations
import numpy as np
import run_final_experiment as R
import system_model_final as sm

_REF: dict = {}

def _ref(c):
    if not _REF:
        r = np.random.default_rng(12345)
        h = np.log(sm.sample_f_composite_gain(200_000, c.F_M, c.F_MS, r))
        g = np.log(r.rayleigh(scale=c.RAYLEIGH_SCALE, size=200_000))
        _REF["h"] = (float(h.mean()), float(h.std())); _REF["g"] = (float(g.mean()), float(g.std()))
    return _REF

def _mix_log(p, s, w, mu_sd):
    mu, sd = mu_sd
    z = np.sqrt(w) * (np.log(p) - mu) / sd + np.sqrt(1.0 - w) * (np.log(s) - mu) / sd
    return np.exp(mu + sd * z)

def mix_session(c, pers, sess, w):
    ref = _ref(c)
    h = _mix_log(pers[0], sess[0], w, ref["h"])
    g = _mix_log(pers[1], sess[1], w, ref["g"])
    unc = np.clip(np.sqrt(w) * pers[2] + np.sqrt(1.0 - w) * sess[2], -c.NOISE_UNC_DB, c.NOISE_UNC_DB)
    return h, g, unc

def generate_persistent_condition(c, seed, condition_id, snr_db, rho, attack, rounds, w, attack_shift_db=None):
    mal = sm.choose_malicious_idx(c.N, rho if attack != "none" else 0.0, seed, condition_id)
    env = R.rng(seed, 100_000 + condition_id)
    h1_rng = R.rng(seed, 110_000 + condition_id)
    h0_rng = R.rng(seed, 120_000 + condition_id)
    atk_rng = R.rng(seed, 130_000 + condition_id)
    pers = R.make_session(c, R.rng(seed, 150_000 + condition_id))
    H1_clean, H1_att, H0_clean, H0_att = [], [], [], []
    h = g = unc = None
    for r in range(rounds):
        if r % c.SESSION_LENGTH == 0:
            h, g, unc = mix_session(c, pers, R.make_session(c, env), w)
        T1 = sm.local_energy_gamma(c.N, c.M_SAMPLES, snr_db, "H1", h, unc, h1_rng)
        T0 = sm.local_energy_gamma(c.N, c.M_SAMPLES, snr_db, "H0", h, unc, h0_rng)
        eta1 = env.normal(0.0, c.REPORT_NOISE_STD, size=c.N)
        eta0 = env.normal(0.0, c.REPORT_NOISE_STD, size=c.N)
        H1_clean.append(sm.reporting_channel(T1, g, eta1)); H0_clean.append(sm.reporting_channel(T0, g, eta0))
        shift_db = c.SHIFT_DB if attack_shift_db is None else float(attack_shift_db)
        T1a = sm.apply_attack(T1, mal, attack, atk_rng, shift_db, c.MASK_EXTREME_DB, c.MASK_MILD_DB)
        T0a = sm.apply_attack(T0, mal, attack, atk_rng, shift_db, c.MASK_EXTREME_DB, c.MASK_MILD_DB)
        H1_att.append(sm.reporting_channel(T1a, g, eta1)); H0_att.append(sm.reporting_channel(T0a, g, eta0))
    return {"malicious_idx": mal, "H1_clean": H1_clean, "H1_att": H1_att, "H0_clean": H0_clean, "H0_att": H0_att}

def make_gen(w):
    """Returns a generator with the same signature as R.generate_paired_condition."""
    if w is None:
        return R.generate_paired_condition
    def gen(c, seed, cid, snr, rho, attack, rounds, attack_shift_db=None):
        return generate_persistent_condition(c, seed, cid, snr, rho, attack, rounds, w, attack_shift_db)
    return gen
