"""Compatibility import for the full-network original-Model Stage 0 runner.

The historical module name is retained so existing Stage 0/1 notebook cells
keep working. The implementation now runs ``model_class.Model`` through its
``CTCModel`` extension; it no longer defines a separate two-node rate model.
"""

from original_model_runner import run_original_model, simulate_stage0


DEFAULTS = {
    "duration_ms": 800.0,
    "dt_ms": 0.5,
    "stim_on_ms": 50.0,
    "stim_off_ms": 500.0,
    "stimulus_drive": 0.20,
    "g_sw": 0.95,
    "phase_frequency_s_hz": 10.0,
    "phase_frequency_w_hz": 10.3,
    "phase_coupling_rad_s": 2.5,
    "phase_diffusion_rad_sqrt_s": 0.0,
    "phase_gate_strength": 0.0,
    "phase_offset_rad": 0.0,
    "phase_window_ms": 50.0,
    "common_frequency_hz": 10.0,
    "ignition_threshold_hz": 20.0,
    "ignition_duration_ms": 50.0,
}


__all__ = ["DEFAULTS", "run_original_model", "simulate_stage0"]
