"""Analysis helpers for operational GNW ignition measurements."""

from __future__ import annotations

import numpy as np


def resolve_area_indices(area_names, requested_areas):
    """Map byte or string area labels to indices, with clear missing-area errors."""
    normalized = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in area_names
    ]
    requested = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in requested_areas
    ]
    missing = [name for name in requested if name not in normalized]
    if missing:
        raise ValueError(
            f"Unknown area(s): {missing}. Available areas: {normalized}"
        )
    return np.asarray([normalized.index(name) for name in requested], dtype=int)


def compute_ignition_metrics(
    firing_rates_hz,
    time_ms,
    workspace_indices,
    threshold_hz=20.0,
    min_duration_ms=50.0,
    onset_ms=0.0,
):
    """Measure sustained workspace activity for one trial.

    Parameters
    ----------
    firing_rates_hz : array, shape (time, areas)
        Activity of one excitatory population in Hz.
    time_ms : array, shape (time,)
        Time relative to stimulus onset, in milliseconds.
    workspace_indices : array-like
        Area indices included in the workspace mean.
    threshold_hz : float
        Provisional workspace mean-rate threshold.
    min_duration_ms : float
        Minimum continuous time above threshold required for ignition.
    onset_ms : float
        Earliest time at which an ignition event may begin.

    Returns
    -------
    dict
        Workspace rate trace, ignition flag, first ignition time, and latency.
        Time fields are NaN when no qualifying event is found.
    """
    rates = np.asarray(firing_rates_hz, dtype=float)
    times = np.asarray(time_ms, dtype=float)
    indices = np.asarray(workspace_indices, dtype=int)

    if rates.ndim != 2:
        raise ValueError("firing_rates_hz must have shape (time, areas)")
    if times.ndim != 1 or times.size != rates.shape[0]:
        raise ValueError("time_ms must have one value per firing-rate row")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("workspace_indices must contain at least one area")
    if indices.min() < 0 or indices.max() >= rates.shape[1]:
        raise ValueError("workspace_indices contains an out-of-range index")
    if times.size < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("time_ms must be strictly increasing with at least 2 samples")
    if threshold_hz < 0 or min_duration_ms < 0:
        raise ValueError("threshold_hz and min_duration_ms must be non-negative")

    workspace_rate_hz = np.mean(rates[:, indices], axis=1)
    dt_ms = float(np.median(np.diff(times)))
    samples_required = max(1, int(np.ceil(min_duration_ms / dt_ms)))
    above_threshold = (workspace_rate_hz >= threshold_hz) & (times >= onset_ms)

    ignition_index = None
    run_length = 0
    for index, is_active in enumerate(above_threshold):
        run_length = run_length + 1 if is_active else 0
        if run_length >= samples_required:
            ignition_index = index - samples_required + 1
            break

    ignited = ignition_index is not None
    ignition_time_ms = float(times[ignition_index]) if ignited else float("nan")
    latency_ms = ignition_time_ms - onset_ms if ignited else float("nan")

    return {
        "workspace_rate_hz": workspace_rate_hz,
        "ignited": ignited,
        "ignition_time_ms": ignition_time_ms,
        "latency_ms": latency_ms,
    }


def compute_cross_group_plv(phase_radians, source_indices, target_indices, window_samples):
    """Compute moving-window PLV between two groups of regional phases.

    The returned trace is NaN until a complete window is available. For every
    source-target pair, PLV is the magnitude of the mean phase-difference vector
    over the window; the output averages those pairwise PLVs.
    """
    phases = np.asarray(phase_radians, dtype=float)
    source_indices = np.asarray(source_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    window_samples = int(window_samples)

    if phases.ndim != 2:
        raise ValueError("phase_radians must have shape (time, areas)")
    if source_indices.size == 0 or target_indices.size == 0:
        raise ValueError("source_indices and target_indices cannot be empty")
    if window_samples < 1:
        raise ValueError("window_samples must be at least 1")
    if max(source_indices.max(), target_indices.max()) >= phases.shape[1]:
        raise ValueError("phase indices are out of range")
    if min(source_indices.min(), target_indices.min()) < 0:
        raise ValueError("phase indices must be non-negative")

    phase_difference = (
        phases[:, source_indices, None] - phases[:, None, target_indices]
    )
    unit_vectors = np.exp(1j * phase_difference)
    cumulative = np.concatenate(
        [np.zeros((1, *unit_vectors.shape[1:]), dtype=complex),
         np.cumsum(unit_vectors, axis=0)],
        axis=0,
    )
    result = np.full(phases.shape[0], np.nan, dtype=float)
    if phases.shape[0] >= window_samples:
        window_sums = cumulative[window_samples:] - cumulative[:-window_samples]
        pairwise_plv = np.abs(window_sums / window_samples)
        result[window_samples - 1:] = pairwise_plv.mean(axis=(1, 2))
    return result
