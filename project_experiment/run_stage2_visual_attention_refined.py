"""Small, checkpointed Stage 2 visual-attention calibration and pilot scans.

The scan has three short phases: calibrate V4 stimulus strength, find a
non-ceiling phase-coupling setting, then separate frontal cue current from
task-dependent phase-coupling gain. It always preserves the original FLN rate
network; only selected phase dynamics and phase gates vary.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from brian2 import Hz, pA

ROOT = next(
    (p for p in (Path(__file__).resolve().parents[1], *Path(__file__).resolve().parents[2:])
     if (p / "model_class.py").is_file()),
    None,
)
if ROOT is None:
    raise RuntimeError("Could not find project root")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "project_experiment") not in sys.path:
    sys.path.insert(0, str(ROOT / "project_experiment"))

from original_model_runner import run_original_model  # noqa: E402
from ctc_validation import _mutual_information_bits  # noqa: E402

OUT = ROOT / "project_experiment"
FRONTAL = "8l"
VISUAL = ("V1", "V2", "V4")
STIMULUS_AREA = "V4"
PHASE_AREAS = (FRONTAL, *VISUAL)
SEEDS = (1, 6, 12, 23)
STIM_ON_MS = 50.0
STIM_OFF_MS = 1050.0
RUN_MS = 1600.0
PHASE_MEAN_HZ = 40.0
FREQUENCY_DETUNING_HZ = 1.0
COMMON_FREQUENCY_HZ = PHASE_MEAN_HZ + FREQUENCY_DETUNING_HZ / 2
FREQUENCIES = {
    FRONTAL: PHASE_MEAN_HZ,
    **{area: PHASE_MEAN_HZ + FREQUENCY_DETUNING_HZ for area in VISUAL},
}


def _mi_shift_corrected(x, y, seed, n_shifts=10, bins=6):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    raw = _mutual_information_bits(x, y, bins=bins)
    if x.size < 40:
        return float(raw)
    rng = np.random.default_rng(int(seed) + 71_000)
    lo, hi = max(10, x.size // 10), max(11, 9 * x.size // 10)
    null = [
        _mutual_information_bits(x, np.roll(y, int(k)), bins=bins)
        for k in rng.integers(lo, hi, size=n_shifts)
    ]
    return float(max(0.0, raw - np.mean(null)))


def _run_one(job):
    phase_mode = job["phase_condition"]
    if phase_mode == "anatomical_FLN":
        mode, gate, k = "fln", 1.0, float(job["K_rad_s"])
    elif phase_mode == "common_rhythm":
        mode, gate, k = "common", 1.0, float(job["K_rad_s"])
    elif phase_mode == "rate_only":
        mode, gate, k = "uncoupled", 0.0, 0.0
    else:
        raise ValueError(f"Unknown phase condition: {phase_mode}")

    cue_current = float(job["pfc_cue_current_pA"])
    task_gain = float(job["task_coupling_gain"])
    detuning_hz = float(job["frequency_detuning_hz"])
    frequency_overrides = {
        FRONTAL: PHASE_MEAN_HZ,
        **{area: PHASE_MEAN_HZ + detuning_hz for area in VISUAL},
    }
    params = {
        "duration_ms": RUN_MS,
        "dt_ms": 0.5,
        "stim_on_ms": STIM_ON_MS,
        "stim_off_ms": STIM_OFF_MS,
        "stimulus_current_pA": float(job["visual_stimulus_pA"]),
        "sensory_area": STIMULUS_AREA,
        "workspace_areas": list(VISUAL),
        "attention_control_area": FRONTAL,
        "attention_control_current_pA": cue_current,
        "attention_control_on_ms": STIM_ON_MS,
        "attention_control_off_ms": STIM_OFF_MS,
        "attention_mode": "synchrony",
        "task_gain": task_gain,
        "phase_frequency_overrides_hz": frequency_overrides,
        "phase_network_areas": list(PHASE_AREAS),
        "phase_gate_areas": list(PHASE_AREAS),
        "phase_coupling_rad_s": k,
        "phase_gate_strength": gate,
        "phase_mode_override": mode,
        "common_frequency_hz": PHASE_MEAN_HZ + detuning_hz / 2.0,
        "g_sw": 0.95,
        "workspace_reentry_current_pA": 0.0,
        "sigma_e_pA": 0.0,
        "sigma_i_pA": 0.0,
    }
    result = run_original_model(
        seed=int(job["seed"]), params=params, configuration="A"
    )
    model = result["model"]
    names = [x.decode() if isinstance(x, bytes) else str(x)
             for x in model.network_params[b"area_list"]]
    pfc_i = names.index(FRONTAL)
    v4_i = names.index(STIMULUS_AREA)
    visual_idx = [names.index(a) for a in VISUAL]
    t = np.asarray(result["time_ms"], dtype=float)
    during = (t >= STIM_ON_MS) & (t < STIM_OFF_MS)
    phase = np.asarray(model.get_phase(), dtype=float)
    rates = np.asarray(model.get_r(1) / Hz, dtype=float)
    delta = phase[during, pfc_i] - phase[during, v4_i]
    mpc = float(abs(np.exp(1j * delta).mean()))
    other_mpcs = [
        float(abs(np.exp(1j * (phase[during, pfc_i] - phase[during, j])).mean()))
        for j in visual_idx
    ]
    gate = np.asarray(model.get_phase_gate(), dtype=float)
    pathway_current = (
        model.s_1n[:, pfc_i] * model.w_e_n[v4_i, pfc_i]
        + model.s_1a[:, pfc_i] * model.w_e_a[v4_i, pfc_i]
    ) * gate[:, v4_i, pfc_i]
    pathway_current_pA = np.asarray(pathway_current / pA, dtype=float)

    row = dict(job)
    row.update({
        "MPC_8l_V4": mpc,
        "MPC_8l_visual_mean": float(np.mean(other_mpcs)),
        "MI_8l_V4_shift_corrected_bits": _mi_shift_corrected(
            rates[during, pfc_i], rates[during, v4_i], job["seed"]
        ),
        "mean_8l_rate_hz": float(rates[during, pfc_i].mean()),
        "mean_V1_rate_hz": float(rates[during, names.index("V1")].mean()),
        "mean_V2_rate_hz": float(rates[during, names.index("V2")].mean()),
        "mean_V4_rate_hz": float(rates[during, v4_i].mean()),
        "mean_gated_8l_to_V4_current_pA": float(pathway_current_pA[during].mean()),
        "max_single_visual_rate_hz": float(rates[:, visual_idx].max()),
    })
    return row


def _run_jobs(stage, jobs, output_name, key_columns):
    path = OUT / output_name
    rows = []
    completed = set()
    if path.exists():
        previous = pd.read_csv(path)
        rows.extend(previous.to_dict("records"))
        completed = {
            tuple(row[column] for column in key_columns)
            for row in rows
        }
    try:
        from tqdm.auto import tqdm
        progress = tqdm(total=len(jobs), desc=stage, unit="run")
    except ImportError:
        progress = None
    for job in jobs:
        key = tuple(job[column] for column in key_columns)
        if key not in completed:
            rows.append(_run_one(job))
            completed.add(key)
            pd.DataFrame(rows).drop_duplicates(key_columns, keep="last").to_csv(path, index=False)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    data = pd.DataFrame(rows).drop_duplicates(key_columns, keep="last")
    data.to_csv(path, index=False)
    return data


def run_refined_pilot():
    """Run the calibration, phase-range, and separated-cue pilot scans."""
    print("Stage 2 refined pilot: 4 paired seeds; all scans checkpoint after each run.")

    # 1) Select a visual-input level that leaves room for top-down modulation.
    calibration_jobs = [
        {
            "stage": "stimulus_calibration", "seed": seed,
            "phase_condition": "rate_only", "visual_stimulus_pA": current,
            "K_rad_s": 0.0, "frequency_detuning_hz": FREQUENCY_DETUNING_HZ,
            "pfc_cue_current_pA": 0.0, "task_coupling_gain": 0.0,
        }
        for current in (100.0, 200.0, 300.0, 400.0, 500.0)
        for seed in SEEDS
    ]
    calibration = _run_jobs(
        "Stage 2A V4 stimulus calibration", calibration_jobs,
        "stage2_visual_attention_refined_stimulus_trials.csv",
        ["visual_stimulus_pA", "seed"],
    )
    calibration_summary = calibration.groupby("visual_stimulus_pA").agg(
        n=("seed", "nunique"), mean_V4_rate_hz=("mean_V4_rate_hz", "mean"),
        sd_V4_rate_hz=("mean_V4_rate_hz", "std"),
        mean_8l_rate_hz=("mean_8l_rate_hz", "mean"),
        max_visual_rate_hz=("max_single_visual_rate_hz", "max"),
    ).reset_index()
    calibration_summary.to_csv(
        OUT / "stage2_visual_attention_refined_stimulus_summary.csv", index=False
    )
    target_rate_hz = 40.0
    selected_stimulus = float(calibration_summary.iloc[
        (calibration_summary["mean_V4_rate_hz"] - target_rate_hz).abs().argmin()
    ]["visual_stimulus_pA"])
    print("V4 stimulus calibration (target mean V4 response ~40 Hz):")
    print(calibration_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Selected visual stimulus for pilot: {selected_stimulus:g} pA")

    # 2) Find a non-ceiling K setting under a larger 1-Hz intrinsic detuning.
    coupling_jobs = []
    for k in (1.0, 3.0, 6.0, 10.8):
        for mode in ("anatomical_FLN", "common_rhythm"):
            for seed in SEEDS:
                coupling_jobs.append({
                    "stage": "coupling_scan", "seed": seed,
                    "phase_condition": mode, "visual_stimulus_pA": selected_stimulus,
                    "K_rad_s": k, "frequency_detuning_hz": FREQUENCY_DETUNING_HZ,
                    "pfc_cue_current_pA": 0.0, "task_coupling_gain": 0.0,
                })
    for seed in SEEDS:
        coupling_jobs.append({
            "stage": "coupling_scan", "seed": seed,
            "phase_condition": "rate_only", "visual_stimulus_pA": selected_stimulus,
            "K_rad_s": 0.0, "frequency_detuning_hz": FREQUENCY_DETUNING_HZ,
            "pfc_cue_current_pA": 0.0, "task_coupling_gain": 0.0,
        })
    coupling = _run_jobs(
        "Stage 2B phase-coupling range", coupling_jobs,
        "stage2_visual_attention_refined_coupling_trials.csv",
        ["phase_condition", "visual_stimulus_pA", "K_rad_s", "frequency_detuning_hz", "seed"],
    )
    coupling_summary = coupling.groupby(["phase_condition", "K_rad_s"]).agg(
        n=("seed", "nunique"), mean_MPC_8l_V4=("MPC_8l_V4", "mean"),
        sd_MPC_8l_V4=("MPC_8l_V4", "std"),
        mean_V4_rate_hz=("mean_V4_rate_hz", "mean"),
        mean_MI_bits=("MI_8l_V4_shift_corrected_bits", "mean"),
    ).reset_index()
    coupling_summary.to_csv(
        OUT / "stage2_visual_attention_refined_coupling_summary.csv", index=False
    )
    fln_scan = coupling_summary[coupling_summary["phase_condition"] == "anatomical_FLN"]
    selected_k = float(fln_scan.iloc[(fln_scan["mean_MPC_8l_V4"] - 0.70).abs().argmin()]["K_rad_s"])
    print("Phase-coupling scan:")
    print(coupling_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Selected K for the cue factorial (FLN MPC nearest 0.70): {selected_k:g} rad/s")

    # 3) Separate frontal cue current from task-dependent coupling gain.
    cue_jobs = []
    for mode in ("anatomical_FLN", "common_rhythm", "rate_only"):
        for cue_current in (0.0, 50.0):
            for task_gain in (0.0, 0.5):
                for seed in SEEDS:
                    cue_jobs.append({
                        "stage": "cue_factorial", "seed": seed,
                        "phase_condition": mode, "visual_stimulus_pA": selected_stimulus,
                        "K_rad_s": 0.0 if mode == "rate_only" else selected_k,
                        "frequency_detuning_hz": FREQUENCY_DETUNING_HZ,
                        "pfc_cue_current_pA": cue_current,
                        "task_coupling_gain": task_gain,
                    })
    cue_data = _run_jobs(
        "Stage 2C cue × coupling ablation", cue_jobs,
        "stage2_visual_attention_refined_cue_trials.csv",
        ["phase_condition", "visual_stimulus_pA", "K_rad_s", "frequency_detuning_hz", "pfc_cue_current_pA", "task_coupling_gain", "seed"],
    )
    cue_summary = cue_data.groupby([
        "phase_condition", "pfc_cue_current_pA", "task_coupling_gain"
    ]).agg(
        n=("seed", "nunique"), mean_MPC_8l_V4=("MPC_8l_V4", "mean"),
        mean_8l_rate_hz=("mean_8l_rate_hz", "mean"),
        mean_V4_rate_hz=("mean_V4_rate_hz", "mean"),
        mean_8l_to_V4_current_pA=("mean_gated_8l_to_V4_current_pA", "mean"),
        mean_MI_bits=("MI_8l_V4_shift_corrected_bits", "mean"),
        max_visual_rate_hz=("max_single_visual_rate_hz", "max"),
    ).reset_index()
    cue_summary.to_csv(
        OUT / "stage2_visual_attention_refined_cue_summary.csv", index=False
    )
    print("Cue-current × task-coupling factorial:")
    print(cue_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Refined Stage 2 pilot finished. Outputs are under stage2_visual_attention_refined_*.")


if __name__ == "__main__":
    run_refined_pilot()
