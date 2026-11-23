"""Two-node CTC validation scaffold informed by the supplied Kang/Quax MATLAB.

The cortical E/I equations, time constants, noise scale, fixed V1E->V2E
projection, and 10 Hz alpha input to inhibitory populations follow the active
``determ == 1`` branch. The MATLAB alpha is a sine wave with independently
jittered phase at every sample; this module replaces that with explicit
phase dynamics so A/B/C communication controls can be compared. Those phase
dynamics and optional R-gated projection are experimental extensions, not
features claimed to be in the MATLAB source.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


DEFAULT_PARAMS = {
    "dt_s": 0.00005,             # MATLAB delt: 50 microseconds
    "duration_s": 1.0,           # shorter than the source script's 5 s
    "stim_on_s": 0.20,
    "stim_off_s": 0.60,
    "tau_e_s": 0.003,
    "tau_i_s": 0.006,
    "s_ee": 1.4,
    "s_ei": 0.78,
    "s_ie": 2.1105,
    "s_ii": 0.1,
    "s_e_to_w": 0.5,             # MATLAB S1E2E
    "alpha_magnitude": 2.0,      # MATLAB amag
    "alpha_to_inhibitory": 3.0,  # MATLAB alphafact1/2 and S*IP*I
    "alpha_frequency_hz": 10.0,
    "alpha_frequency_spread_hz": 0.5,
    "phase_diffusion_rad_sqrt_s": 0.35,
    "cortical_noise_sigma": 0.03,  # cornse * sig1E/sig1I = 0.20 * 0.15
    "delay_samples": 1,            # actual V2Eoff used in MATLAB execution
    "phase_offset_rad": 0.0,
    "r_window_ms": 50.0,
    "phase_gate_depth": 0.8,       # new proposed weight-gating extension
    "trace_stride": 20,            # retain traces at 1 ms for plotting/analysis
    "ignition_duration_ms": 50.0,
    "mi_bins": 6,
    "mi_surrogates": 12,
}


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _mutual_information_bits(x, y, bins=6):
    """Quantile-binned MI in bits; suitable as a transparent pilot estimator."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 20 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
    x_edges = np.unique(np.quantile(x, quantiles))
    y_edges = np.unique(np.quantile(y, quantiles))
    if len(x_edges) < 3 or len(y_edges) < 3:
        return 0.0
    counts, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    pxy = counts / counts.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    keep = pxy > 0.0
    return float(np.sum(pxy[keep] * np.log2(pxy[keep] / expected[keep])))


def _rolling_plv(phase_difference, window_samples):
    """Causal sliding-window PLV, one trace per trial."""
    z = np.exp(1j * np.asarray(phase_difference))
    cumulative = np.concatenate(
        [np.zeros((z.shape[0], 1), dtype=np.complex128), np.cumsum(z, axis=1)],
        axis=1,
    )
    indices = np.arange(z.shape[1])
    starts = np.maximum(0, indices - int(window_samples) + 1)
    sums = cumulative[:, indices + 1] - cumulative[:, starts]
    counts = (indices - starts + 1).astype(float)
    return np.abs(sums / counts[None, :])


def _ignition_summary(time_ms, target_trace, threshold, onset_ms, duration_ms):
    above = (target_trace >= threshold) & (time_ms >= onset_ms)
    samples = max(1, int(np.ceil(duration_ms / np.median(np.diff(time_ms)))))
    if samples > len(above):
        return False, float("nan")
    rolling = np.convolve(above.astype(np.int16), np.ones(samples, dtype=np.int16), mode="valid")
    candidates = np.flatnonzero(rolling == samples)
    if candidates.size == 0:
        return False, float("nan")
    onset = float(time_ms[candidates[0]])
    return True, onset - float(onset_ms)


def simulate_two_node_batch(
    configuration,
    coupling_rad_s,
    seeds,
    signal_magnitude,
    *,
    explicit_weight_gate=False,
    response_threshold=None,
    params=None,
):
    """Simulate a batch of V1/V2-like E/I pairs for one condition.

    Parameters
    ----------
    configuration : {"A", "B", "C"}
        A: mutual phase coupling and V1-to-V2 E projection.
        B: common external alpha phase drive, no inter-area E projection.
        C: V2 alpha phase is one-way dragged by V1, with the E projection.
    coupling_rad_s : float
        Phase interaction strength. In B this is common-drive locking; in C it
        is the restoring strength of the one-way phase-drag error.
    explicit_weight_gate : bool
        If true, multiply S1E2E by a new, explicit alpha-PLV/phase-difference
        gate. The supplied MATLAB does not implement this; false is the
        source-inspired comparison, and true is the proposed extension.
    response_threshold : float or None
        Threshold on V2E's abstract activity variable. This is not Hz. A
        threshold should be calibrated from no-signal controls before calling.
    """
    configuration = str(configuration).upper()
    if configuration not in {"A", "B", "C"}:
        raise ValueError("configuration must be A, B, or C")
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    if explicit_weight_gate and not 0.0 <= p["phase_gate_depth"] <= 1.0:
        raise ValueError("phase_gate_depth must be in [0, 1]")

    seeds = np.asarray(list(seeds), dtype=int)
    if seeds.ndim != 1 or seeds.size == 0:
        raise ValueError("seeds must be a non-empty one-dimensional sequence")
    dt = float(p["dt_s"])
    n_time = int(round(float(p["duration_s"]) / dt)) + 1
    time_s = np.arange(n_time, dtype=float) * dt
    time_ms = time_s * 1000.0
    n_trial = seeds.size

    # Independent reproducible noise streams for each paired trial.
    rngs = [np.random.default_rng(int(seed)) for seed in seeds]
    noise_cortical = [
        np.stack([rng.standard_normal(n_time) for _ in range(4)], axis=0)
        for rng in rngs
    ]
    noise_phase = [
        np.stack([rng.standard_normal(n_time) for _ in range(3)], axis=0)
        for rng in rngs
    ]
    cortical_noise = np.stack(noise_cortical, axis=0)
    phase_noise = np.stack(noise_phase, axis=0)

    omega_s = np.empty(n_trial)
    omega_w = np.empty(n_trial)
    theta_s = np.empty(n_trial)
    theta_w = np.empty(n_trial)
    for i, rng in enumerate(rngs):
        spread = p["alpha_frequency_spread_hz"]
        base = p["alpha_frequency_hz"]
        omega_s[i] = 2.0 * np.pi * rng.uniform(base - spread, base + spread)
        omega_w[i] = 2.0 * np.pi * rng.uniform(base - spread, base + spread)
        theta_s[i] = rng.uniform(-np.pi, np.pi)
        theta_w[i] = rng.uniform(-np.pi, np.pi)

    # Source code's V1E, V1I, V2E, and V2I variables.
    v1e = np.empty((n_trial, n_time), dtype=np.float64)
    v1i = np.empty_like(v1e)
    v2e = np.empty_like(v1e)
    v2i = np.empty_like(v1e)
    v1e[:, 0] = np.stack([r.uniform(0.0, 1.0) for r in rngs])
    v1i[:, 0] = np.stack([r.uniform(0.0, 1.0) for r in rngs])
    v2e[:, 0] = np.stack([r.uniform(0.0, 1.0) for r in rngs])
    v2i[:, 0] = np.stack([r.uniform(0.0, 1.0) for r in rngs])

    theta_s_trace = np.empty((n_trial, n_time), dtype=np.float64)
    theta_w_trace = np.empty_like(theta_s_trace)
    theta_s_trace[:, 0] = theta_s
    theta_w_trace[:, 0] = theta_w
    phase_error = _wrap_to_pi(theta_w - theta_s - p["phase_offset_rad"])

    window_samples = max(1, int(round(p["r_window_ms"] / (dt * 1000.0))))
    phase_ring = np.zeros((window_samples, n_trial), dtype=np.complex128)
    phase_ring_sum = np.zeros(n_trial, dtype=np.complex128)
    r_alpha = np.empty((n_trial, n_time), dtype=np.float32)
    effective_weight = np.zeros((n_trial, n_time), dtype=np.float32)
    gate_strength = p["phase_gate_depth"] if explicit_weight_gate else 0.0
    source_connected = configuration != "B"
    phase_diffusion = float(p["phase_diffusion_rad_sqrt_s"]) * np.sqrt(dt)
    delay = int(p["delay_samples"])
    if delay < 0:
        raise ValueError("delay_samples must be non-negative")

    stim_on = float(p["stim_on_s"])
    stim_off = float(p["stim_off_s"])
    omega_common = 2.0 * np.pi * float(p["alpha_frequency_hz"])

    for k in range(1, n_time):
        t_prev = time_s[k - 1]
        if configuration == "A":
            delta = _wrap_to_pi(theta_w - theta_s - p["phase_offset_rad"])
            mutual = float(coupling_rad_s) * np.sin(delta)
            theta_s = theta_s + (omega_s + mutual) * dt + phase_diffusion * phase_noise[:, 0, k]
            theta_w = theta_w + (omega_w - mutual) * dt + phase_diffusion * phase_noise[:, 1, k]
        elif configuration == "B":
            theta_common = omega_common * t_prev
            theta_s = theta_s + (
                omega_s + float(coupling_rad_s) * np.sin(theta_common - theta_s)
            ) * dt + phase_diffusion * phase_noise[:, 0, k]
            theta_w = theta_w + (
                omega_w + float(coupling_rad_s) * np.sin(theta_common - theta_w)
            ) * dt + phase_diffusion * phase_noise[:, 1, k]
        else:  # C: theta_W follows theta_S; W has no independent natural frequency.
            theta_s = theta_s + omega_s * dt + phase_diffusion * phase_noise[:, 0, k]
            phase_error += (
                -float(coupling_rad_s) * np.sin(phase_error) * dt
                + phase_diffusion * phase_noise[:, 2, k]
            )
            phase_error = _wrap_to_pi(phase_error)
            theta_w = theta_s + p["phase_offset_rad"] + phase_error

        theta_s = _wrap_to_pi(theta_s)
        theta_w = _wrap_to_pi(theta_w)
        theta_s_trace[:, k] = theta_s
        theta_w_trace[:, k] = theta_w
        delta = _wrap_to_pi(theta_w - theta_s)
        ring_idx = (k - 1) % window_samples
        phase_ring_sum -= phase_ring[ring_idx]
        current_z = np.exp(1j * delta)
        phase_ring[ring_idx] = current_z
        phase_ring_sum += current_z
        n_in_window = min(k, window_samples)
        r_alpha[:, k] = np.abs(phase_ring_sum / n_in_window)

        if source_connected:
            phase_factor = 1.0
            if explicit_weight_gate:
                phase_factor = np.maximum(
                    0.0,
                    1.0 + gate_strength * r_alpha[:, k]
                    * np.cos(delta - p["phase_offset_rad"]),
                )
            effective_weight[:, k] = p["s_e_to_w"] * phase_factor

        alpha_s = p["alpha_magnitude"] * np.sin(theta_s)
        alpha_w = p["alpha_magnitude"] * np.sin(theta_w)
        signal = float(signal_magnitude) if stim_on <= t_prev < stim_off else 0.0

        # Euler-Maruyama equations from the active determ == 1 MATLAB branch.
        d_v1e = (
            dt * (-v1e[:, k - 1] + p["s_ee"] * v1e[:, k - 1]
                  - p["s_ei"] * v1i[:, k - 1] + signal)
            + p["cortical_noise_sigma"] * np.sqrt(dt) * cortical_noise[:, 0, k]
        ) / p["tau_e_s"]
        d_v1i = (
            dt * (-v1i[:, k - 1] - p["s_ii"] * v1i[:, k - 1]
                  + p["s_ie"] * v1e[:, k - 1]
                  + p["alpha_to_inhibitory"] * alpha_s)
            + p["cortical_noise_sigma"] * np.sqrt(dt) * cortical_noise[:, 1, k]
        ) / p["tau_i_s"]

        source_idx = max(0, k - delay)
        interarea_input = effective_weight[:, k] * v1e[:, source_idx] if source_connected else 0.0
        d_v2e = (
            dt * (-v2e[:, k - 1] + p["s_ee"] * v2e[:, k - 1]
                  - p["s_ei"] * v2i[:, k - 1] + interarea_input)
            + p["cortical_noise_sigma"] * np.sqrt(dt) * cortical_noise[:, 2, k]
        ) / p["tau_e_s"]
        d_v2i = (
            dt * (-v2i[:, k - 1] - p["s_ii"] * v2i[:, k - 1]
                  + p["s_ie"] * v2e[:, k - 1]
                  + p["alpha_to_inhibitory"] * alpha_w)
            + p["cortical_noise_sigma"] * np.sqrt(dt) * cortical_noise[:, 3, k]
        ) / p["tau_i_s"]
        v1e[:, k] = v1e[:, k - 1] + d_v1e
        v1i[:, k] = v1i[:, k - 1] + d_v1i
        v2e[:, k] = v2e[:, k - 1] + d_v2e
        v2i[:, k] = v2i[:, k - 1] + d_v2i

    # Downsample saved traces to 1 ms; compute Hilbert phases on these traces.
    stride = max(1, int(p["trace_stride"]))
    selection = np.arange(0, n_time, stride)
    if selection[-1] != n_time - 1:
        selection = np.r_[selection, n_time - 1]
    traces = {
        "time_ms": time_ms[selection],
        "r_s": v1e[:, selection].astype(np.float32),
        "r_w": v2e[:, selection].astype(np.float32),
        "alpha_phase_s": theta_s_trace[:, selection].astype(np.float32),
        "alpha_phase_w": theta_w_trace[:, selection].astype(np.float32),
        "r_alpha": r_alpha[:, selection].astype(np.float32),
        "effective_weight": effective_weight[:, selection].astype(np.float32),
    }

    gamma_phase_s = np.angle(hilbert(traces["r_s"], axis=1))
    gamma_phase_w = np.angle(hilbert(traces["r_w"], axis=1))
    gamma_phase_diff = _wrap_to_pi(gamma_phase_s - gamma_phase_w)
    gamma_window_samples = max(
        1, int(round(p["r_window_ms"] / np.median(np.diff(traces["time_ms"]))))
    )
    traces["r_gamma"] = _rolling_plv(gamma_phase_diff, gamma_window_samples).astype(np.float32)

    during = (traces["time_ms"] >= stim_on * 1000.0) & (traces["time_ms"] < stim_off * 1000.0)
    post = traces["time_ms"] >= stim_off * 1000.0
    mi_rows = []
    for trial in range(n_trial):
        mi_during = _mutual_information_bits(
            traces["r_s"][trial, during], traces["r_w"][trial, during], p["mi_bins"]
        )
        mi_post = _mutual_information_bits(
            traces["r_s"][trial, post], traces["r_w"][trial, post], p["mi_bins"]
        )
        rng = np.random.default_rng(int(seeds[trial]) + 50_000)
        null_values = []
        x = traces["r_s"][trial, during]
        y = traces["r_w"][trial, during]
        if x.size > 40:
            low_shift = max(10, int(0.1 * x.size))
            high_shift = max(low_shift + 1, int(0.9 * x.size))
            for shift in rng.integers(low_shift, high_shift, size=int(p["mi_surrogates"])):
                null_values.append(_mutual_information_bits(x, np.roll(y, int(shift)), p["mi_bins"]))
        mi_bias = float(np.mean(null_values)) if null_values else 0.0
        mi_rows.append((mi_during, max(0.0, mi_during - mi_bias), mi_post, mi_bias))
    mi_rows = np.asarray(mi_rows, dtype=float)

    if response_threshold is None:
        response_threshold = 0.0
    success = []
    latency = []
    for trial in range(n_trial):
        ok, delay_ms = _ignition_summary(
            traces["time_ms"], traces["r_w"][trial], response_threshold,
            stim_on * 1000.0, p["ignition_duration_ms"],
        )
        success.append(ok)
        latency.append(delay_ms)

    summary = {
        "configuration": configuration,
        "coupling_rad_s": float(coupling_rad_s),
        "signal_magnitude": float(signal_magnitude),
        "seed": seeds.copy(),
        "response_success": np.asarray(success, dtype=bool),
        "response_latency_ms": np.asarray(latency, dtype=float),
        "mean_r_alpha_during": traces["r_alpha"][:, during].mean(axis=1),
        "mean_r_gamma_during": traces["r_gamma"][:, during].mean(axis=1),
        "mi_activity_during_bits": mi_rows[:, 0],
        "mi_activity_during_corrected_bits": mi_rows[:, 1],
        "mi_activity_post_bits": mi_rows[:, 2],
        "mi_shift_null_bits": mi_rows[:, 3],
        "mean_r_s_during": traces["r_s"][:, during].mean(axis=1),
        "mean_r_w_during": traces["r_w"][:, during].mean(axis=1),
        "max_r_w": traces["r_w"].max(axis=1),
        "traces": traces,
        "response_threshold": float(response_threshold),
        "explicit_weight_gate": bool(explicit_weight_gate),
    }
    return summary


def calibrate_response_threshold(
    seeds,
    *,
    coupling_rad_s=30.0,
    params=None,
    percentile=99.5,
    safety_margin=0.0,
):
    """Calibrate W's activity threshold from no-signal common-drive controls.

    The source MATLAB variables V1E/V2E are activity/voltage-like units, not
    firing rates in Hz. This threshold defines a model response criterion for
    Stage 1; the later GNW stage must use the original model's actual Hz rates.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    result = simulate_two_node_batch(
        "B", coupling_rad_s, seeds, 0.0,
        explicit_weight_gate=False,
        response_threshold=0.0,
        params=p,
    )
    traces = result["traces"]
    post = traces["time_ms"] >= p["stim_off_s"] * 1000.0
    values = traces["r_w"][:, post]
    dt_ms = float(np.median(np.diff(traces["time_ms"])))
    window = max(1, int(np.ceil(p["ignition_duration_ms"] / dt_ms)))
    if window > values.shape[1]:
        raise ValueError("post-stimulus period is shorter than the response window")
    # Calibrate against sustained activity, not isolated sample-level peaks.
    rolling_means = np.stack([
        np.convolve(row, np.ones(window) / window, mode="valid")
        for row in values
    ])
    threshold = float(np.percentile(rolling_means, percentile) + safety_margin)
    return threshold, result
