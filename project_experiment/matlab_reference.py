"""Python port of the supplied MATLAB script's active ``determ == 1`` branch.

This is kept separate from the A/B/C CTC extensions in ``ctc_validation.py``.
It matches the model equations, default time grid, stimulus epoch and primary
phase-coherence calculations. Random streams are Python/NumPy streams, not
MATLAB's stream, and the original helper ``mi.m`` was not supplied; therefore
trajectories and MI values are not expected to match MATLAB point-for-point.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, hilbert, lfilter

try:  # Import as project_experiment.matlab_reference or from notebook sys.path.
    from .ctc_validation import _mutual_information_bits
except ImportError:
    from ctc_validation import _mutual_information_bits


MATLAB_ACTIVE_DEFAULTS = {
    "dt_s": 50e-6,
    "runtime_steps": 100_000,
    "nits": 10,
    "tau_e_s": 0.003,
    "tau_i_s": 0.006,
    "s_ee": 1.4,
    "s_ei": 0.78,
    "s_ie": 2.1105,
    "s_ii": 0.1,
    "s_e_to_w": 0.5,
    "alpha_hz": 10.0,
    "alpha_amplitude": 2.0,
    "alpha_phase_jitter_sd_rad": 0.4,
    "alpha_to_inhibitory_gain": 3.0,
    "cortical_noise_sigma": 0.15 * 0.20,
    "v2e_delay_steps": 1,
    "signal_magnitude": 20.0,
    "signal_on_step": 50_000,
    "signal_off_step": 100_000,
    "downsample_step": 100,
}


def _phase_plv(phase_a, phase_b):
    return np.abs(np.mean(np.exp(1j * (phase_a - phase_b)), axis=-1))


def simulate_matlab_active_branch(*, seed=1, params=None, keep_traces=True):
    """Simulate MATLAB's default single-stream V1→V2 cortical model.

    The function follows the script's ``determ=1``, ``noV2=0``, ``ialpha=1``
    branch. Defaults match its 5 s / 20 kHz run and 2.5–5 s input pulse.
    ``seed`` makes the Python run repeatable, but cannot reproduce MATLAB's
    random-number stream (the source assigns ``rng='shuffle'`` rather than
    fixing a seed).
    """
    p = dict(MATLAB_ACTIVE_DEFAULTS)
    if params:
        p.update(params)
    n = int(p["runtime_steps"])
    nits = int(p["nits"])
    dt = float(p["dt_s"])
    if n < 2 or nits < 1 or dt <= 0:
        raise ValueError("runtime_steps and nits must be positive; dt_s must be > 0")
    delay = int(p["v2e_delay_steps"])
    if delay < 0:
        raise ValueError("v2e_delay_steps must be non-negative")

    rng = np.random.default_rng(seed)
    # MATLAB initializes the state at t=0 once before entering the nits loop.
    initial = rng.random(4)
    t = np.arange(n, dtype=np.float64) * dt
    alpha_jitter_1 = rng.standard_normal((nits, n))
    alpha_jitter_2 = rng.standard_normal((nits, n))
    noise = rng.standard_normal((nits, 4, n))
    alpha1 = p["alpha_amplitude"] * np.sin(
        2 * np.pi * p["alpha_hz"] * t[None, :]
        + p["alpha_phase_jitter_sd_rad"] * alpha_jitter_1
    )
    alpha2 = p["alpha_amplitude"] * np.sin(
        2 * np.pi * p["alpha_hz"] * t[None, :]
        + p["alpha_phase_jitter_sd_rad"] * alpha_jitter_2
    )

    # Store all trials, vectorizing the time recurrence over nits.
    v1e = np.empty((nits, n), dtype=np.float64)
    v1i = np.empty_like(v1e)
    v2e = np.empty_like(v1e)
    v2i = np.empty_like(v1e)
    v1e[:, 0] = initial[0]
    v1i[:, 0] = initial[1]
    v2e[:, 0] = initial[2]
    v2i[:, 0] = initial[3]
    signal = np.zeros(n, dtype=np.float64)
    stim_on = max(0, int(p["signal_on_step"]))
    stim_off = min(n, int(p["signal_off_step"]))
    signal[stim_on:stim_off] = float(p["signal_magnitude"])
    sigma = float(p["cortical_noise_sigma"])
    sqrt_dt = np.sqrt(dt)

    for k in range(1, n):
        source_k = max(0, k - delay)
        d1e = (
            dt * (-v1e[:, k-1] + p["s_ee"] * v1e[:, k-1]
                  - p["s_ei"] * v1i[:, k-1] + signal[k])
            + sigma * sqrt_dt * noise[:, 0, k]
        ) / p["tau_e_s"]
        d1i = (
            dt * (-v1i[:, k-1] - p["s_ii"] * v1i[:, k-1]
                  + p["s_ie"] * v1e[:, k-1]
                  + p["alpha_to_inhibitory_gain"] * alpha1[:, k-1])
            + sigma * sqrt_dt * noise[:, 1, k]
        ) / p["tau_i_s"]
        d2e = (
            dt * (-v2e[:, k-1] + p["s_ee"] * v2e[:, k-1]
                  - p["s_ei"] * v2i[:, k-1]
                  + p["s_e_to_w"] * v1e[:, source_k])
            + sigma * sqrt_dt * noise[:, 2, k]
        ) / p["tau_e_s"]
        d2i = (
            dt * (-v2i[:, k-1] - p["s_ii"] * v2i[:, k-1]
                  + p["s_ie"] * v2e[:, k-1]
                  + p["alpha_to_inhibitory_gain"] * alpha2[:, k-1])
            + sigma * sqrt_dt * noise[:, 3, k]
        ) / p["tau_i_s"]
        v1e[:, k] = v1e[:, k-1] + d1e
        v1i[:, k] = v1i[:, k-1] + d1i
        v2e[:, k] = v2e[:, k-1] + d2e
        v2i[:, k] = v2i[:, k-1] + d2i

    # Match the source's raw Hilbert phase-coherence calculations.
    phase1 = np.angle(hilbert(v1e, axis=1))
    phase2 = np.angle(hilbert(v2e, axis=1))
    phase_alpha1 = np.angle(hilbert(alpha1, axis=1))
    phase_alpha2 = np.angle(hilbert(alpha2, axis=1))
    coherence_raw = _phase_plv(phase2, phase1)
    coherence_alpha = _phase_plv(phase_alpha1, phase_alpha2)

    stride = int(p["downsample_step"])
    take = np.arange(0, n, stride)
    v1_down, v2_down = v1e[:, take], v2e[:, take]
    # MATLAB fir1(100,[.1,.2]) / fir1(100,[.01,.011]) after downsampling.
    filt_gamma = firwin(101, [0.1, 0.2], pass_zero=False)
    filt_low = firwin(101, [0.01, 0.011], pass_zero=False)
    filtered_coherence = np.empty(nits)
    lowband_coherence = np.empty(nits)
    for trial in range(nits):
        g1 = lfilter(filt_gamma, [1.0], v1_down[trial])
        g2 = lfilter(filt_gamma, [1.0], v2_down[trial])
        l1 = lfilter(filt_low, [1.0], v1_down[trial])
        l2 = lfilter(filt_low, [1.0], v2_down[trial])
        filtered_coherence[trial] = _phase_plv(
            np.angle(hilbert(g1)), np.angle(hilbert(g2))
        )
        lowband_coherence[trial] = _phase_plv(
            np.angle(hilbert(l1)), np.angle(hilbert(l2))
        )

    # Reproduce source's signal-window maxima/near-peak means (subplus ~= max(x,0)).
    lo = min(max(stim_on - 1, 0), n - 1)
    hi = min(max(int(p["signal_on_step"]) + 5_000, lo + 1), n)
    peak_v1 = np.empty(nits)
    peak_v2 = np.empty(nits)
    near_peak_v1 = np.empty(nits)
    near_peak_v2 = np.empty(nits)
    for trial in range(nits):
        segment1 = np.maximum(v1e[trial, lo:hi], 0.0)
        segment2 = np.maximum(v2e[trial, lo:hi], 0.0)
        peak_v1[trial] = segment1.max()
        peak_v2[trial] = segment2.max()
        # Source uses me+50000±1000 in MATLAB indexing, near the selected peak.
        center1 = lo + int(np.argmax(segment1)) + 1
        center2 = lo + int(np.argmax(segment2)) + 1
        for arr, center, out, trial_idx in (
            (v1e, center1, near_peak_v1, trial),
            (v2e, center2, near_peak_v2, trial),
        ):
            start = max(0, center - 1000)
            stop = min(n, center + 1001)
            out[trial_idx] = np.maximum(arr[trial_idx, start:stop], 0.0).mean()

    # The project did not include mi.m: provide a labeled estimator proxy only.
    pre_end = min(stim_on, n)
    post_start = min(stim_on, n)
    amplitude1, amplitude2 = np.abs(hilbert(v1e, axis=1)), np.abs(hilbert(v2e, axis=1))
    mi_proxy = {}
    for name, x, y in (
        ("activity", v1e, v2e),
        ("amplitude", amplitude1, amplitude2),
        ("phase", phase1, phase2),
    ):
        mi_proxy[f"{name}_pre_bits"] = np.array([
            _mutual_information_bits(x[j, :pre_end], y[j, :pre_end])
            for j in range(nits)
        ])
        mi_proxy[f"{name}_post_bits"] = np.array([
            _mutual_information_bits(x[j, post_start:], y[j, post_start:])
            for j in range(nits)
        ])

    result = {
        "params": p,
        "time_s": t,
        "stim_on_s": stim_on * dt,
        "coherence_raw_phase": coherence_raw,
        "coherence_alpha_input": coherence_alpha,
        "coherence_filtered_10_20_hz": filtered_coherence,
        "coherence_filtered_10_20_hz_source_scaled": filtered_coherence * (take.size / n),
        "coherence_filtered_lowband": lowband_coherence,
        "peak_v1e_positive": peak_v1,
        "peak_v2e_positive": peak_v2,
        "near_peak_mean_v1e_positive": near_peak_v1,
        "near_peak_mean_v2e_positive": near_peak_v2,
        "raw_signal_effect_peak_v2_minus_v1": peak_v2 - peak_v1,
        "raw_signal_effect_near_peak_v2_minus_v1": near_peak_v2 - near_peak_v1,
        "mi_proxy_not_matlab_mi": mi_proxy,
        "mi_comparability_note": "The MATLAB helper mi.m was not supplied; these quantile-binned values are proxies, not matched MI results.",
        "rng_note": "Seeded NumPy stream ensures repeatability here; MATLAB source does not set a comparable fixed stream.",
    }
    if keep_traces:
        result["traces"] = {
            "time_s": t,
            "v1e": v1e,
            "v1i": v1i,
            "v2e": v2e,
            "v2i": v2i,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "downsample_time_s": t[take],
            "v1e_downsampled": v1_down,
            "v2e_downsampled": v2_down,
        }
    return result
