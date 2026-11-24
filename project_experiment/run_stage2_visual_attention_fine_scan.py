"""Fine paired scan of visual input, phase coupling, and frequency detuning.

Stage A refines V4 stimulus strength (100--200 pA) under anatomical and
common-rhythm phase conditions. Stage B scans K x intrinsic-frequency
detuning using the selected stimulus and reports paired anatomical-minus-
common-rhythm contrasts across the same seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from run_stage2_visual_attention_refined import (
    OUT,
    SEEDS,
    _run_jobs,
    _run_one,
)


def _summarize_stimulus(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby(["phase_condition", "visual_stimulus_pA"]).agg(
        n=("seed", "nunique"),
        mean_MPC_8l_V4=("MPC_8l_V4", "mean"),
        sd_MPC_8l_V4=("MPC_8l_V4", "std"),
        mean_V4_rate_hz=("mean_V4_rate_hz", "mean"),
        mean_8l_rate_hz=("mean_8l_rate_hz", "mean"),
        mean_MI_bits=("MI_8l_V4_shift_corrected_bits", "mean"),
        max_visual_rate_hz=("max_single_visual_rate_hz", "max"),
    ).reset_index()


def _paired_contrasts(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["K_rad_s", "frequency_detuning_hz", "seed"]
    metrics = [
        "MPC_8l_V4",
        "MI_8l_V4_shift_corrected_bits",
        "mean_V4_rate_hz",
        "mean_gated_8l_to_V4_current_pA",
    ]
    a = data[data.phase_condition == "anatomical_FLN"].set_index(keys)
    b = data[data.phase_condition == "common_rhythm"].set_index(keys)
    paired = pd.DataFrame(index=a.index.intersection(b.index)).reset_index()
    for metric in metrics:
        paired[f"delta_{metric}"] = (
            a.loc[paired.set_index(keys).index, metric].to_numpy()
            - b.loc[paired.set_index(keys).index, metric].to_numpy()
        )

    # n=4 gives a two-sided 95% t critical value of 3.182 (df=3).
    rows = []
    for (k, detuning), group in paired.groupby(["K_rad_s", "frequency_detuning_hz"]):
        row = {"K_rad_s": k, "frequency_detuning_hz": detuning, "n_pairs": len(group)}
        for metric in metrics:
            values = group[f"delta_{metric}"].to_numpy(dtype=float)
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
            half = 3.182 * sem if len(values) == 4 else np.nan
            stem = f"delta_{metric}"
            row[f"{stem}_mean"] = mean
            row[f"{stem}_ci95_low"] = mean - half
            row[f"{stem}_ci95_high"] = mean + half
            row[f"{stem}_positive_seeds"] = int(np.sum(values > 0))
        rows.append(row)
    return paired, pd.DataFrame(rows)


def run_fine_scan() -> None:
    print("Fine Stage 2 scan: 4 paired seeds; stimulus grid + K x detuning grid.")

    # Stage A: rate-only rows preserve the original rate response as a control;
    # active phase conditions reveal how phase gating changes the same input.
    stimulus_levels = (100.0, 125.0, 150.0, 175.0, 200.0)
    stimulus_jobs = []
    for current in stimulus_levels:
        for mode in ("anatomical_FLN", "common_rhythm"):
            for seed in SEEDS:
                stimulus_jobs.append({
                    "stage": "fine_stimulus_scan", "seed": seed,
                    "phase_condition": mode, "visual_stimulus_pA": current,
                    "K_rad_s": 3.0, "frequency_detuning_hz": 2.0,
                    "pfc_cue_current_pA": 0.0, "task_coupling_gain": 0.0,
                })
        for seed in SEEDS:
            stimulus_jobs.append({
                "stage": "fine_stimulus_scan", "seed": seed,
                "phase_condition": "rate_only", "visual_stimulus_pA": current,
                "K_rad_s": 0.0, "frequency_detuning_hz": 2.0,
                "pfc_cue_current_pA": 0.0, "task_coupling_gain": 0.0,
            })
    stimulus_data = _run_jobs(
        "Fine scan 1/2: V4 stimulus", stimulus_jobs,
        "stage2_visual_attention_fine_stimulus_trials.csv",
        ["phase_condition", "visual_stimulus_pA", "K_rad_s", "frequency_detuning_hz", "seed"],
    )
    stimulus_summary = _summarize_stimulus(stimulus_data)
    stimulus_summary.to_csv(OUT / "stage2_visual_attention_fine_stimulus_summary.csv", index=False)

    # Choose the stimulus whose anatomical-FLN V4 response is closest to 35 Hz,
    # while preferring a response below 60 Hz if one exists.
    anatomical = stimulus_summary[stimulus_summary.phase_condition == "anatomical_FLN"].copy()
    eligible = anatomical[anatomical.mean_V4_rate_hz <= 60.0]
    candidates = eligible if not eligible.empty else anatomical
    selected_stimulus = float(candidates.iloc[(candidates.mean_V4_rate_hz - 35.0).abs().argmin()]["visual_stimulus_pA"])
    print("Fine stimulus calibration:")
    print(stimulus_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Selected stimulus for K x detuning scan: {selected_stimulus:g} pA")

    # Stage B: two phase conditions share seeds and the anatomical rate matrix.
    k_levels = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    detuning_levels = (1.0, 2.0, 3.0, 4.0)
    # Run the full K curve at 1-Hz detuning, then spend the additional
    # detuning runs around the intermediate K values where synchrony is not
    # already at ceiling. This keeps the paired scan tractable.
    coupling_jobs = [
        {
            "stage": "fine_K_detuning_scan", "seed": seed,
            "phase_condition": mode, "visual_stimulus_pA": selected_stimulus,
            "K_rad_s": k, "frequency_detuning_hz": detuning,
            "pfc_cue_current_pA": 0.0, "task_coupling_gain": 0.0,
        }
        for detuning in detuning_levels
        for k in (k_levels if detuning == 1.0 else (3.0, 4.0))
        for mode in ("anatomical_FLN", "common_rhythm")
        for seed in SEEDS
    ]
    coupling_data = _run_jobs(
        "Fine scan 2/2: K x detuning", coupling_jobs,
        "stage2_visual_attention_fine_coupling_trials.csv",
        ["phase_condition", "visual_stimulus_pA", "K_rad_s", "frequency_detuning_hz", "seed"],
    )
    intended_grid = (
        ((coupling_data.frequency_detuning_hz == 1.0)
         & coupling_data.K_rad_s.isin(k_levels))
        | (coupling_data.frequency_detuning_hz.isin((2.0, 3.0, 4.0))
           & coupling_data.K_rad_s.isin((3.0, 4.0)))
    )
    coupling_data = coupling_data[
        intended_grid
        & coupling_data.phase_condition.isin(("anatomical_FLN", "common_rhythm"))
        & np.isclose(coupling_data.visual_stimulus_pA, selected_stimulus)
    ].copy()
    coupling_summary = coupling_data.groupby(
        ["phase_condition", "K_rad_s", "frequency_detuning_hz"]
    ).agg(
        n=("seed", "nunique"),
        mean_MPC_8l_V4=("MPC_8l_V4", "mean"),
        sd_MPC_8l_V4=("MPC_8l_V4", "std"),
        mean_V4_rate_hz=("mean_V4_rate_hz", "mean"),
        mean_MI_bits=("MI_8l_V4_shift_corrected_bits", "mean"),
        mean_gated_pathway_current_pA=("mean_gated_8l_to_V4_current_pA", "mean"),
        max_visual_rate_hz=("max_single_visual_rate_hz", "max"),
    ).reset_index()
    coupling_summary.to_csv(OUT / "stage2_visual_attention_fine_coupling_summary.csv", index=False)
    paired, contrasts = _paired_contrasts(coupling_data)
    paired.to_csv(OUT / "stage2_visual_attention_fine_paired_seed_contrasts.csv", index=False)

    # Flag candidate conditions without asserting a confirmatory conclusion:
    # the anatomical condition should beat common rhythm in MI and/or gated
    # pathway current, with synchrony below the near-ceiling band.
    anatomical_mpc = coupling_summary[
        coupling_summary.phase_condition == "anatomical_FLN"
    ][["K_rad_s", "frequency_detuning_hz", "mean_MPC_8l_V4"]].rename(
        columns={"mean_MPC_8l_V4": "anatomical_mean_MPC_8l_V4"}
    )
    contrasts = contrasts.merge(
        anatomical_mpc, on=["K_rad_s", "frequency_detuning_hz"], how="left"
    )
    contrasts.to_csv(OUT / "stage2_visual_attention_fine_paired_contrast_summary.csv", index=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    plot_metrics = [
        ("delta_MPC_8l_V4", "Paired ΔMPC (FLN − common rhythm)"),
        ("delta_MI_8l_V4_shift_corrected_bits", "Paired Δshift-corrected MI (bits)"),
        ("delta_mean_gated_8l_to_V4_current_pA", "Paired Δgated 8l→V4 current (pA)"),
    ]
    for ax, (metric, title) in zip(axes, plot_metrics):
        for detuning, subset in contrasts.groupby("frequency_detuning_hz"):
            subset = subset.sort_values("K_rad_s")
            x = subset["K_rad_s"].to_numpy(dtype=float)
            y = subset[f"{metric}_mean"].to_numpy(dtype=float)
            low = subset[f"{metric}_ci95_low"].to_numpy(dtype=float)
            high = subset[f"{metric}_ci95_high"].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker="o",
                        capsize=3, label=f"Δf={detuning:g} Hz")
        ax.axhline(0, color="black", linewidth=1, alpha=0.65)
        ax.set_title(title)
        ax.set_xlabel("Phase coupling K (rad/s)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Anatomical FLN minus common rhythm")
    axes[-1].legend(title="Intrinsic frequency mismatch", fontsize=8)
    fig.suptitle("Paired seed contrasts across K and frequency mismatch")
    fig.savefig(OUT / "stage2_visual_attention_fine_paired_contrasts.png", dpi=180)
    plt.close(fig)
    candidates = contrasts[
        contrasts.anatomical_mean_MPC_8l_V4.between(0.20, 0.90)
        & (
            (contrasts.delta_MI_8l_V4_shift_corrected_bits_ci95_low > 0)
            | (contrasts.delta_mean_gated_8l_to_V4_current_pA_ci95_low > 0)
        )
    ].copy()
    candidates.to_csv(OUT / "stage2_visual_attention_fine_candidate_conditions.csv", index=False)

    print("K x detuning condition summary:")
    print(coupling_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Paired anatomical_FLN minus common_rhythm contrasts (95% t intervals):")
    print(contrasts.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Candidate conditions are saved separately; they are exploratory, not a pass criterion.")
    print("Fine scan finished. Outputs: stage2_visual_attention_fine_*.csv")


if __name__ == "__main__":
    run_fine_scan()
