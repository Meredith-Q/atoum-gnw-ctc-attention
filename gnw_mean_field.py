"""Reusable rate/synapse update used by the GNW mean-field model.

The 40-area notebook supplies its own anatomical input and stimulus vector;
this module owns the shared local E/I transfer and state-update interface so
smaller models can use exactly the same population dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GNWMeanFieldParameters:
    """Provisional parameters currently used in the GNW integration notebook."""

    rE_max_hz: float = 80.0
    rI_max_hz: float = 60.0
    tau_E_ms: float = 10.0
    tau_I_ms: float = 8.0
    tau_sE_ms: float = 8.0
    tau_sI_ms: float = 8.0
    theta_E: float = 0.30
    slope_E: float = 0.12
    theta_I: float = 0.60
    slope_I: float = 0.12
    J_EI: float = 1.2
    J_IE: float = 0.8
    J_II: float = 0.3


def sigmoid_rate(input_value, max_rate, threshold, slope):
    """Bounded population input/output curve used in the GNW rate model."""
    z = np.clip((np.asarray(input_value) - threshold) / slope, -60.0, 60.0)
    return max_rate / (1.0 + np.exp(-z))


def initialize_mean_field_state(n_nodes, params=GNWMeanFieldParameters()):
    """Return the GNW notebook's low-activity initial E/I/synapse state."""
    rE0 = float(sigmoid_rate(0.0, params.rE_max_hz, params.theta_E, params.slope_E))
    rI0 = float(sigmoid_rate(0.0, params.rI_max_hz, params.theta_I, params.slope_I))
    rE = np.full(int(n_nodes), rE0, dtype=float)
    rI = np.full(int(n_nodes), rI0, dtype=float)
    sE = rE / params.rE_max_hz
    sI = rI / params.rI_max_hz
    return rE, rI, sE, sI


def mean_field_step(rE, rI, sE, sI, additive_e_input, dt_ms,
                    params=GNWMeanFieldParameters()):
    """Advance all nodes one Euler step using GNW's E/I rate equations.

    ``additive_e_input`` is the per-node excitatory input assembled by the
    caller (stimulus + anatomical/experimental projections + input noise).
    Local inhibitory feedback and the inhibitory population equation are
    handled here. Return order is ``rE_next, rI_next, sE_next, sI_next``.
    """
    rE = np.asarray(rE, dtype=float)
    rI = np.asarray(rI, dtype=float)
    sE = np.asarray(sE, dtype=float)
    sI = np.asarray(sI, dtype=float)
    additive_e_input = np.asarray(additive_e_input, dtype=float)
    if not (rE.shape == rI.shape == sE.shape == sI.shape == additive_e_input.shape):
        raise ValueError("E/I rates, synapses, and additive input must have the same shape")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")

    uE = additive_e_input - params.J_IE * sI
    uI = params.J_EI * sE - params.J_II * sI
    target_E = sigmoid_rate(uE, params.rE_max_hz, params.theta_E, params.slope_E)
    target_I = sigmoid_rate(uI, params.rI_max_hz, params.theta_I, params.slope_I)
    rE_next = rE + (dt_ms / params.tau_E_ms) * (target_E - rE)
    rI_next = rI + (dt_ms / params.tau_I_ms) * (target_I - rI)
    sE_next = sE + (dt_ms / params.tau_sE_ms) * (rE / params.rE_max_hz - sE)
    sI_next = sI + (dt_ms / params.tau_sI_ms) * (rI / params.rI_max_hz - sI)
    return rE_next, rI_next, sE_next, sI_next
