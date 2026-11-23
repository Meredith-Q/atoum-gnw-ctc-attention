"""CTC phase-gating extension for the original population-rate Model."""

from __future__ import annotations

import numpy as np
from brian2 import second

from model_class import Model


class CTCModel(Model):
    """Original AToum model with dynamic phase-gated E-to-E FLN inputs.

    All original firing-rate, local synaptic, anatomical FLN, and inhibitory
    long-range equations are inherited. A Kuramoto-style phase population is
    added, and only long-range excitatory inputs to the two excitatory
    populations are multiplied by a bounded phase gate.
    """

    def __init__(
        self,
        talky=False,
        network_params=None,
        timer=False,
        timer_keys=None,
        *,
        phase_mean_hz=40.0,
        phase_frequency_spread_hz=1.0,
        phase_coupling_rad_s=15.0,
        phase_diffusion_rad_sqrt_s=0.0,
        phase_gate_depth=0.5,
        phase_preferred_offset_rad=0.0,
        phase_seed=0,
    ):
        self.phase_mean_hz = float(phase_mean_hz)
        self.phase_frequency_spread_hz = float(phase_frequency_spread_hz)
        self.phase_coupling_rad_s = float(phase_coupling_rad_s)
        self.phase_diffusion_rad_sqrt_s = float(phase_diffusion_rad_sqrt_s)
        self.phase_gate_depth = float(phase_gate_depth)
        self.phase_preferred_offset_rad = float(phase_preferred_offset_rad)
        self.phase_seed = int(phase_seed)

        phase_parameters = np.asarray(
            [
                self.phase_mean_hz,
                self.phase_frequency_spread_hz,
                self.phase_coupling_rad_s,
                self.phase_diffusion_rad_sqrt_s,
                self.phase_gate_depth,
                self.phase_preferred_offset_rad,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(phase_parameters)):
            raise ValueError("All phase parameters must be finite")
        if self.phase_mean_hz <= 0:
            raise ValueError("phase_mean_hz must be positive")
        if self.phase_frequency_spread_hz < 0:
            raise ValueError("phase_frequency_spread_hz must be non-negative")
        if self.phase_frequency_spread_hz >= self.phase_mean_hz:
            raise ValueError("phase_frequency_spread_hz must be smaller than phase_mean_hz")
        if self.phase_coupling_rad_s < 0:
            raise ValueError("phase_coupling_rad_s must be non-negative")
        if self.phase_diffusion_rad_sqrt_s < 0:
            raise ValueError("phase_diffusion_rad_sqrt_s must be non-negative")
        if not 0 <= self.phase_gate_depth <= 1:
            raise ValueError("phase_gate_depth must be between 0 and 1")

        super().__init__(
            talky=talky,
            network_params=network_params,
            timer=timer,
            timer_keys=timer_keys,
        )
        self._initialize_phase_state()

    def _initialize_phase_state(self):
        """Create area phases and normalized incoming FLN phase weights."""
        rng = np.random.default_rng(self.phase_seed)
        self.phase_frequency_hz = rng.uniform(
            self.phase_mean_hz - self.phase_frequency_spread_hz,
            self.phase_mean_hz + self.phase_frequency_spread_hz,
            size=self.n_area,
        )
        self.phase_rad = np.empty((self.n_t, self.n_area), dtype=float)
        self.phase_rad[0] = rng.uniform(0.0, 2.0 * np.pi, size=self.n_area)
        self.phase_noise = rng.standard_normal((self.n_t, self.n_area))

        phase_weights = np.nan_to_num(
            np.asarray(self.fln_m, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        phase_weights = np.clip(phase_weights, 0.0, None)
        np.fill_diagonal(phase_weights, 0.0)
        row_sums = phase_weights.sum(axis=1, keepdims=True)
        self.phase_weights = np.divide(
            phase_weights,
            row_sums,
            out=np.zeros_like(phase_weights),
            where=row_sums > 0,
        )
        self.phase_gate_history = np.ones(
            (self.n_t, self.n_area, self.n_area), dtype=np.float32
        )

    def get_longrange_gate_matrix(self, i_t):
        """Advance area phases and return target-by-source E-to-E gates."""
        dt_seconds = float(self.network_params["dt"] / second)
        previous = self.phase_rad[i_t - 1]
        phase_difference = previous[None, :] - previous[:, None]
        network_phase_drive = np.sum(
            self.phase_weights * np.sin(phase_difference), axis=1
        )
        deterministic_increment = (
            2.0 * np.pi * self.phase_frequency_hz
            + self.phase_coupling_rad_s * network_phase_drive
        ) * dt_seconds
        stochastic_increment = (
            self.phase_diffusion_rad_sqrt_s
            * np.sqrt(dt_seconds)
            * self.phase_noise[i_t]
        )
        self.phase_rad[i_t] = np.mod(
            previous + deterministic_increment + stochastic_increment,
            2.0 * np.pi,
        )

        phase_difference = (
            self.phase_rad[i_t][None, :]
            - self.phase_rad[i_t][:, None]
            - self.phase_preferred_offset_rad
        )
        gate = 1.0 + self.phase_gate_depth * np.cos(phase_difference)
        self.phase_gate_history[i_t] = gate
        return gate

    def get_phase(self):
        """Return area phase traces in radians, shape (time, area)."""
        return self.phase_rad

    def get_phase_gate(self):
        """Return phase-gate traces, shape (time, target, source)."""
        return self.phase_gate_history
