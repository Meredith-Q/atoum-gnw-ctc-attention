"""Two-node CTC Stage 0 using the shared GNW mean-field rate interface."""

from __future__ import annotations

import numpy as np

from gnw_mean_field import (
    GNWMeanFieldParameters,
    initialize_mean_field_state,
    mean_field_step,
)


DEFAULTS = {
    "duration_ms": 800.0,
    "dt_ms": 1.0,
    "stim_on_ms": 50.0,
    "stim_off_ms": 500.0,
    "stimulus_drive": 0.20,
    "g_sw": 0.95,  # same provisional global FLN gain in the GNW notebook
    "local_recurrent_gain": 0.05,
    "phase_frequency_s_hz": 10.0,
    "phase_frequency_w_hz": 10.3,
    "phase_coupling_rad_s": 1.5,
    "phase_diffusion_rad_sqrt_s": 0.15,
    "phase_gate_strength": 0.0,  # keep rate and phase readouts separable in Stage 0
    "phase_offset_rad": 0.0,
    "phase_window_ms": 50.0,
    "common_frequency_hz": 10.0,
    "ignition_threshold_hz": 20.0,
    "ignition_duration_ms": 50.0,
}


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _continuous_above(time_ms, rate_hz, threshold_hz, duration_ms, after_ms):
    dt_ms = float(np.median(np.diff(time_ms)))
    width = max(1, int(np.ceil(duration_ms / dt_ms)))
    valid = (rate_hz >= threshold_hz) & (time_ms >= after_ms)
    if width > valid.size:
        return False, float("nan")
    counts = np.convolve(valid.astype(np.int32), np.ones(width, dtype=np.int32), mode="valid")
    starts = np.flatnonzero(counts == width)
    if starts.size == 0:
        return False, float("nan")
    return True, float(time_ms[starts[0]] - after_ms)


def _gnw_params(overrides):
    names = set(GNWMeanFieldParameters.__dataclass_fields__)
    return GNWMeanFieldParameters(**{key: value for key, value in overrides.items() if key in names})


def simulate_stage0(configuration="A", *, seed=1, params=None):
    """Return two-node GNW rates (Hz) and phase traces (radians).

    Node S receives the sensory pulse. Node W receives a directed S→W rate
    projection plus optional local recurrence. Rate dynamics call the same
    ``mean_field_step`` function used by the 40-area GNW notebook. Phase
    dynamics are a separate Kuramoto-style addition, not part of that original
    mean-field model.

    A: bidirectional phase coupling and S→W rate projection.
    B: common phase drive, with the rate projection removed.
    C: one-way W phase drag from S, with S→W rate projection.
    """
    condition = str(configuration).upper()
    if condition not in {"A", "B", "C"}:
        raise ValueError("configuration must be A, B, or C")
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    dt = float(p["dt_ms"])
    duration = float(p["duration_ms"])
    if dt <= 0 or duration <= dt:
        raise ValueError("duration_ms and dt_ms must be positive and consistent")
    if not 0 <= float(p["phase_gate_strength"]) <= 1:
        raise ValueError("phase_gate_strength must be between 0 and 1")

    time_ms = np.arange(0.0, duration + 0.5 * dt, dt)
    n_time = time_ms.size
    rng = np.random.default_rng(seed)
    theta_s = np.empty(n_time)
    theta_w = np.empty(n_time)
    theta_s[0], theta_w[0] = rng.uniform(-np.pi, np.pi, size=2)

    mf = _gnw_params(p)
    rE = np.empty((n_time, 2), dtype=float)
    rI = np.empty_like(rE)
    sE = np.empty_like(rE)
    sI = np.empty_like(rE)
    rE[0], rI[0], sE[0], sI[0] = initialize_mean_field_state(2, mf)

    alignment = np.empty(n_time)
    phase_lock = np.empty(n_time)
    g_eff = np.zeros(n_time)
    phase_error = _wrap(theta_w[0] - theta_s[0] - p["phase_offset_rad"])
    omega_s = 2.0 * np.pi * float(p["phase_frequency_s_hz"])
    omega_w = 2.0 * np.pi * float(p["phase_frequency_w_hz"])
    omega_common = 2.0 * np.pi * float(p["common_frequency_hz"])
    coupling = float(p["phase_coupling_rad_s"])
    noise_scale = float(p["phase_diffusion_rad_sqrt_s"]) * np.sqrt(dt / 1000.0)
    win_n = max(1, int(round(float(p["phase_window_ms"]) / dt)))
    phase_ring = np.zeros(win_n, dtype=np.complex128)
    phase_sum = 0.0j

    for k in range(1, n_time):
        t0 = time_ms[k - 1]
        delta0 = _wrap(theta_w[k - 1] - theta_s[k - 1] - p["phase_offset_rad"])
        if condition == "A":
            theta_s[k] = theta_s[k-1] + (omega_s + coupling * np.sin(delta0)) * dt / 1000.0 + noise_scale * rng.standard_normal()
            theta_w[k] = theta_w[k-1] + (omega_w - coupling * np.sin(delta0)) * dt / 1000.0 + noise_scale * rng.standard_normal()
        elif condition == "B":
            common = omega_common * t0 / 1000.0
            theta_s[k] = theta_s[k-1] + (omega_s + coupling * np.sin(common-theta_s[k-1])) * dt / 1000.0 + noise_scale * rng.standard_normal()
            theta_w[k] = theta_w[k-1] + (omega_w + coupling * np.sin(common-theta_w[k-1])) * dt / 1000.0 + noise_scale * rng.standard_normal()
        else:
            theta_s[k] = theta_s[k-1] + omega_s * dt / 1000.0 + noise_scale * rng.standard_normal()
            phase_error = _wrap(phase_error - coupling * np.sin(phase_error) * dt / 1000.0 + noise_scale * rng.standard_normal())
            theta_w[k] = theta_s[k] + float(p["phase_offset_rad"]) + phase_error
        theta_s[k] = _wrap(theta_s[k])
        theta_w[k] = _wrap(theta_w[k])

        delta = _wrap(theta_w[k] - theta_s[k] - p["phase_offset_rad"])
        alignment[k] = 0.5 * (1.0 + np.cos(delta))
        phase_factor = 1.0 + float(p["phase_gate_strength"]) * alignment[k]
        g_eff[k] = 0.0 if condition == "B" else float(p["g_sw"]) * phase_factor
        ring_index = (k - 1) % win_n
        phase_sum -= phase_ring[ring_index]
        phase_ring[ring_index] = np.exp(1j * delta)
        phase_sum += phase_ring[ring_index]
        phase_lock[k] = abs(phase_sum / min(k, win_n))

        input_e = np.zeros(2, dtype=float)
        if float(p["stim_on_ms"]) <= t0 < float(p["stim_off_ms"]):
            input_e[0] += float(p["stimulus_drive"])
        input_e[1] += float(p["local_recurrent_gain"]) * sE[k - 1, 1]
        input_e[1] += g_eff[k] * sE[k - 1, 0]
        rE[k], rI[k], sE[k], sI[k] = mean_field_step(
            rE[k - 1], rI[k - 1], sE[k - 1], sI[k - 1], input_e, dt, mf
        )

    initial_delta = _wrap(theta_w[0] - theta_s[0] - p["phase_offset_rad"])
    alignment[0] = 0.5 * (1.0 + np.cos(initial_delta))
    phase_lock[0] = 1.0
    if condition != "B":
        g_eff[0] = float(p["g_sw"]) * (1.0 + float(p["phase_gate_strength"]) * alignment[0])
    ignited, latency = _continuous_above(
        time_ms, rE[:, 1], float(p["ignition_threshold_hz"]),
        float(p["ignition_duration_ms"]), float(p["stim_on_ms"]),
    )
    return {
        "configuration": condition,
        "seed": int(seed),
        "time_ms": time_ms,
        "r_s_hz": rE[:, 0],
        "r_w_hz": rE[:, 1],
        "r_i_s_hz": rI[:, 0],
        "r_i_w_hz": rI[:, 1],
        "s_e": sE,
        "s_i": sI,
        "theta_s_rad": theta_s,
        "theta_w_rad": theta_w,
        "phase_locking_value": phase_lock,
        "phase_alignment": alignment,
        "g_sw_effective": g_eff,
        "ignited": bool(ignited),
        "ignition_latency_ms": latency,
        "parameters": p,
    }
