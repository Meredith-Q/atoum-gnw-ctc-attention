"""Stage 0–2 runners backed by the original ``model_class.Model`` equations.

The model remains the original 40-area population-rate network. ``CTCModel``
adds phases and an optional gate to long-range E→E FLN input; this module only
selects readouts and experiment settings around that model.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from brian2 import Hz, ms, pA, second


ROOT = next(
    (path for path in (Path(__file__).resolve().parents[1], *Path(__file__).resolve().parents[2:])
     if (path / "model_class.py").is_file()),
    None,
)
if ROOT is None:
    raise RuntimeError("Could not find the repository root containing model_class.py")

import sys
import os
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _module_path in (ROOT / "simulations", ROOT / "toolboxes"):
    if str(_module_path) not in sys.path:
        sys.path.insert(0, str(_module_path))

from ctc_model import CTCModel
from model_class import Model
from template_kwargs import template_kwargs
from toolboxes.model_toolbox import get_full_kwargs


DEFAULT_WORKSPACE = (b"F1", b"F2", b"F3", b"F4", b"7a", b"7b")
_CALIBRATED_PARAMS = None


def _base_params():
    global _CALIBRATED_PARAMS
    if _CALIBRATED_PARAMS is None:
        os.chdir(ROOT)
        _CALIBRATED_PARAMS = get_full_kwargs(dict(template_kwargs))
    return deepcopy(_CALIBRATED_PARAMS)


def _rolling_plv(delta, width):
    z = np.exp(1j * np.asarray(delta, dtype=float))
    width = max(1, int(width))
    csum = np.concatenate(([0j], np.cumsum(z)))
    ix = np.arange(z.size)
    start = np.maximum(0, ix - width + 1)
    return np.abs((csum[ix + 1] - csum[start]) / (ix - start + 1))


def _phase_lock_time(time_ms, theta_s, theta_w, window_ms=100.0,
                     stable_ms=100.0, slope_limit_rad_s=0.75):
    dt = float(np.median(np.diff(time_ms)))
    win = max(3, int(round(window_ms / dt)))
    stable = max(1, int(round(stable_ms / dt)))
    delta = np.unwrap(np.asarray(theta_w) - np.asarray(theta_s))
    if delta.size < win + stable - 1:
        return float("nan")
    blocks = np.lib.stride_tricks.sliding_window_view(delta, win)
    x = (np.arange(win) - (win - 1) / 2) * dt / 1000.0
    slopes = (blocks @ x) / np.dot(x, x)
    valid = np.abs(slopes) <= slope_limit_rad_s
    counts = np.convolve(valid.astype(np.int32), np.ones(stable, dtype=np.int32), mode="valid")
    hits = np.flatnonzero(counts == stable)
    return float(time_ms[hits[0] + win - 1]) if hits.size else float("nan")


def run_original_model(
    *, seed=1, params=None, configuration="A", stimulus_schedule=None,
    phase_on=True,
):
    """Run one full-network trial and return V4/workspace rate and phase traces.

    ``stimulus_drive`` is expressed in nA-equivalent pA scale: 0.13 means
    130 pA. Explicit ``stimulus_current_pA`` overrides that conversion.
    Configurations A/B/C are respectively FLN phase coupling, common phase
    drive with E→E inter-area transmission disabled, and V4→workspace phase
    drag along a directed projection.
    """
    p = dict(params or {})
    condition = str(configuration).upper()
    if condition not in {"A", "B", "C"}:
        raise ValueError("configuration must be A, B, or C")

    network = _base_params()
    duration_ms = float(p.get("duration_ms", 800.0))
    dt_ms = float(p.get("dt_ms", 0.5))
    stim_on_ms = float(p.get("stim_on_ms", 50.0))
    stim_off_ms = float(p.get("stim_off_ms", 500.0))
    sensory_area = p.get("sensory_area", b"V4")
    if isinstance(sensory_area, str):
        sensory_area = sensory_area.encode()
    workspace = tuple(
        area.encode() if isinstance(area, str) else area
        for area in p.get("workspace_areas", DEFAULT_WORKSPACE)
    )
    current_pA = float(p.get("stimulus_current_pA", 1000.0 * float(p.get("stimulus_drive", 0.13))))
    if stimulus_schedule is None:
        run_t = duration_ms * ms
        stimulus_on = stim_on_ms * ms
        stimulus_off = stim_off_ms * ms
    else:
        schedule = np.asarray(stimulus_schedule, dtype=float)
        dt_ms = float(p.get("dt_ms", 0.5))
        duration_ms = (schedule.size - 1) * dt_ms
        run_t = duration_ms * ms
        stimulus_on = 0 * ms
        stimulus_off = run_t

    network.update({
        "run_t": run_t,
        "dt": dt_ms * ms,
        "stim_on": stimulus_on,
        "stim_off": stimulus_off,
        "i_ext": current_pA * pA,
        "active_area": [sensory_area],
        "workspace_areas": list(workspace),
        "sigma_e": float(p.get("sigma_e_pA", 0.0)) * pA,
        "sigma_i": float(p.get("sigma_i_pA", 0.0)) * pA,
        "attention_baseline_gain": float(p.get("attention_baseline_gain", 1.0)),
        "attention_task_gain": float(p.get("task_gain", 0.0)),
        "attention_task_on": stim_on_ms * ms,
        "attention_task_off": stim_off_ms * ms,
        "attention_task_tau": float(p.get("task_decay_tau_ms", 100.0)) * ms,
        "attention_ignition_threshold_hz": float(p.get("ignition_threshold_hz", 20.0)) * Hz,
        # Optional cue current to a selected frontal control area.
        "attention_control_area": p.get("attention_control_area"),
        "attention_control_current": float(p.get("attention_control_current_pA", 0.0)) * pA,
        "attention_control_on": float(p.get("attention_control_on_ms", stim_on_ms)) * ms,
        "attention_control_off": float(p.get("attention_control_off_ms", stim_off_ms)) * ms,
        # Optional workspace-triggered sensory re-entry. The native Model
        # already implements this pathway; expose its parameters to Stage 0–2.
        "attention_reentry_gain": float(p.get("attention_reentry_gain", 0.0)),
        "attention_reentry_current": float(p.get("attention_reentry_current_pA", 0.0)) * pA,
        "attention_sensory_floor_hz": float(p.get("attention_sensory_floor_hz", 5.0)) * Hz,
        "attention_ignition_slope_hz": float(p.get("attention_ignition_slope_hz", 2.0)) * Hz,
        "workspace_reentry_current": float(p.get("workspace_reentry_current_pA", 0.0)) * pA,
        "experiment_attention_mode": p.get("attention_mode", "both"),
        "experiment_decay_mode": p.get("decay_mode", "none"),
        "experiment_task_decay_tau_ms": float(p.get("task_decay_tau_ms", 100.0)),
        "experiment_competitor_on_ms": float(p.get("competitor_on_ms", np.inf)),
        "experiment_competitor_level": float(p.get("competitor_level", 1.0)),
        "experiment_competition_inhibition": float(p.get("competition_inhibition", 0.8)),
        "experiment_resource_depletion_per_s": float(p.get("resource_depletion_per_s", 2.0)),
        "experiment_resource_threshold_hz": float(p.get("resource_activity_threshold_hz", 15.0)),
        "experiment_trigger_duration_ms": float(p.get("ignition_duration_ms", 50.0)),
        "phase_frequency_overrides_hz": {
            sensory_area: float(p.get("phase_frequency_s_hz", 10.0)),
            **{
                area: float(p.get("phase_frequency_w_hz", 10.3))
                for area in workspace
            },
            **p.get("phase_frequency_overrides_hz", {}),
        },
        "phase_network_areas": list(p.get("phase_network_areas", [])),
        "phase_gate_areas": list(p.get("phase_gate_areas", [])),
    })
    np.random.seed(int(seed))
    phase_mode = (
        str(p.get("phase_mode_override"))
        if phase_on and p.get("phase_mode_override") is not None
        else {"A": "fln", "B": "common", "C": "sensory_to_workspace"}[condition]
    )
    disable_longrange_ee = bool(
        p.get("disable_longrange_ee_override", condition == "B")
    )
    model_type = CTCModel
    model = model_type(
        talky=False,
        network_params=network,
        phase_mean_hz=float(p.get("phase_frequency_hz", p.get("phase_frequency_s_hz", 10.0))),
        phase_frequency_spread_hz=float(p.get("phase_frequency_spread_hz", 0.3)),
        phase_coupling_rad_s=float(p.get("phase_coupling_rad_s", p.get("K_rad_s", 2.5)))
        if phase_on else 0.0,
        phase_diffusion_rad_sqrt_s=float(p.get("phase_diffusion_rad_sqrt_s", 0.0)),
        phase_gate_depth=float(p.get("phase_gate_strength", 0.5)) if phase_on else 0.0,
        phase_preferred_offset_rad=float(p.get("phase_offset_rad", 0.0)),
        phase_seed=10000 + int(seed),
        phase_mode=phase_mode if phase_on else "uncoupled",
        common_drive_frequency_hz=float(p.get("common_frequency_hz", 10.0)),
        disable_longrange_ee=disable_longrange_ee,
        phase_gate_scale=float(p.get("g_sw", 0.95)) / 0.95,
    )

    if stimulus_schedule is not None:
        source_index = model.network_params[b"area_list"].index(sensory_area)
        values = np.asarray(stimulus_schedule, dtype=float)
        expected = model.n_t
        if values.size != expected:
            values = np.interp(
                np.linspace(0.0, values.size - 1, expected),
                np.arange(values.size), values,
            )
        model.i_ext[:, source_index] = values * 1000.0 * pA

    model.run()
    names = model.network_params[b"area_list"]
    source_idx = names.index(sensory_area)
    workspace_idx = [names.index(area) for area in workspace]
    time_ms = np.asarray(model.t_plot / ms, dtype=float)
    rate_s = np.asarray(model.get_r(1)[:, source_idx] / Hz, dtype=float)
    workspace_rates = np.asarray(model.get_r(1)[:, workspace_idx] / Hz, dtype=float)
    rate_w = workspace_rates.mean(axis=1)
    phase = np.asarray(model.get_phase(), dtype=float)
    phase_s = phase[:, source_idx]
    phase_w = np.angle(np.exp(1j * phase[:, workspace_idx]).mean(axis=1))
    delta = (phase_w - phase_s + np.pi) % (2 * np.pi) - np.pi
    dt = float(np.median(np.diff(time_ms)))
    win = max(1, int(round(float(p.get("phase_window_ms", 50.0)) / dt)))
    phase_gate = np.asarray(model.get_phase_gate(), dtype=float)
    gain = np.mean(phase_gate[:, workspace_idx, source_idx], axis=1)

    ignition_threshold = float(p.get("ignition_threshold_hz", 20.0))
    ignition_duration = float(p.get("ignition_duration_ms", 50.0))
    onset_ms = float(p.get("ignition_after_ms", stim_on_ms))
    n_hold = max(1, int(np.ceil(ignition_duration / dt)))
    above = (rate_w >= ignition_threshold) & (time_ms >= onset_ms)
    counts = np.convolve(above.astype(np.int32), np.ones(n_hold, dtype=np.int32), mode="valid")
    hits = np.flatnonzero(counts == n_hold)
    ignited = bool(hits.size)
    latency = float(time_ms[hits[0]] - onset_ms) if ignited else float("nan")
    lock_time = _phase_lock_time(time_ms, phase_s, phase_w)
    task_trace = np.asarray(model.attention_gain - model.attention_baseline_gain, dtype=float)
    return {
        "configuration": condition,
        "seed": int(seed),
        "time_ms": time_ms,
        "stimulus": np.asarray(model.i_ext[:, source_idx] / pA, dtype=float),
        "r_s_hz": rate_s,
        "r_w_hz": rate_w,
        "workspace_area_rates_hz": workspace_rates,
        "theta_s_rad": phase_s,
        "theta_w_rad": phase_w,
        "theta_s": phase_s,
        "theta_w": phase_w,
        "delta_phase": delta,
        "phase_locking_value": _rolling_plv(delta, win),
        "phase_gate_strength_trace": gain,
        "g_sw_effective": gain,
        "ignited": ignited,
        "ignition_latency_ms": latency,
        "ignition_onset_ms": float(time_ms[hits[0]]) if ignited else float("nan"),
        "phase_lock_time_ms": lock_time,
        "phase_lock_latency_ms": max(0.0, lock_time - onset_ms) if np.isfinite(lock_time) else float("nan"),
        "g_task": task_trace,
        "parameters": p,
        "model": model,
    }


def simulate_stage0(configuration="A", *, seed=1, params=None):
    """Backward-compatible Stage 0 API, now running the original full model."""
    return run_original_model(seed=seed, params=params, configuration=configuration)


def simulate_joint(*, seed=1, params=None, stimulus_schedule=None,
                   attention_mode=None, decay_mode=None, phase_on=True):
    """Stage 3 compatibility API backed by the original ``Model`` dynamics."""
    p = dict(params or {})
    if attention_mode is not None:
        p["attention_mode"] = attention_mode
    if decay_mode is not None:
        p["decay_mode"] = decay_mode
    result = run_original_model(
        seed=seed, params=p, configuration="A",
        stimulus_schedule=stimulus_schedule, phase_on=phase_on,
    )
    # Experimental decay modes are intentionally read out as gain traces.
    # The base model's task gain follows its native exponential off-decay.
    result["g_task"] = result["g_task"]
    result["g_sw"] = result["g_sw_effective"]
    result["plv"] = result["phase_locking_value"]
    result["alignment"] = 0.5 * (1.0 + np.cos(result["delta_phase"]))
    return result
