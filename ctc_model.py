"""CTC phase-gating extension for the original population-rate Model."""

from __future__ import annotations

import numpy as np
from brian2 import Hz, ms, second

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
        phase_mode="fln",
        common_drive_frequency_hz=None,
        disable_longrange_ee=False,
        phase_gate_scale=1.0,
    ):
        self.phase_mean_hz = float(phase_mean_hz)
        self.phase_frequency_spread_hz = float(phase_frequency_spread_hz)
        self.phase_coupling_rad_s = float(phase_coupling_rad_s)
        self.phase_diffusion_rad_sqrt_s = float(phase_diffusion_rad_sqrt_s)
        self.phase_gate_depth = float(phase_gate_depth)
        self.phase_preferred_offset_rad = float(phase_preferred_offset_rad)
        self.phase_seed = int(phase_seed)
        self.phase_mode = str(phase_mode)
        self.common_drive_frequency_hz = float(
            phase_mean_hz if common_drive_frequency_hz is None
            else common_drive_frequency_hz
        )
        self.disable_longrange_ee = bool(disable_longrange_ee)
        self.phase_gate_scale = float(phase_gate_scale)
        self.experiment_attention_mode = "intensity"
        if self.phase_mode not in {"fln", "common", "sensory_to_workspace", "uncoupled"}:
            raise ValueError("phase_mode must be fln/common/sensory_to_workspace/uncoupled")
        if self.phase_gate_scale < 0:
            raise ValueError("phase_gate_scale must be non-negative")

        phase_parameters = np.asarray(
            [
                self.phase_mean_hz,
                self.phase_frequency_spread_hz,
                self.phase_coupling_rad_s,
                self.phase_diffusion_rad_sqrt_s,
                self.phase_gate_depth,
                self.phase_gate_scale,
                self.phase_preferred_offset_rad,
                self.common_drive_frequency_hz,
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
        self.experiment_attention_mode = str(
            self.network_params.get("experiment_attention_mode", "intensity")
        )
        self.experiment_decay_mode = str(
            self.network_params.get("experiment_decay_mode", "none")
        )
        if self.experiment_attention_mode not in {"none", "intensity", "synchrony", "both"}:
            raise ValueError("Unknown experiment_attention_mode")
        if self.experiment_decay_mode not in {"none", "target_decay", "competition", "resource"}:
            raise ValueError("Unknown experiment_decay_mode")
        self._task_gain_cache_index = -1
        self._task_gain_cache = 0.0
        self._task_trigger_index = None
        self._task_above_count = 0
        self._task_active_ms = 0.0
        self._initialize_phase_state()

    def _dynamic_task_gain(self, i_t):
        """Apply the optional Stage 2 gain-decay mechanism once per step."""
        if self._task_gain_cache_index == i_t:
            return self._task_gain_cache
        raw = float(self.attention_task_trace[i_t])
        p = self.network_params
        dt_ms = float(p["dt"] / ms)
        if self.workspace_indices:
            previous_rate = float(np.mean(self.r_1[i_t - 1, self.workspace_indices]) / Hz)
            threshold = float(p.get("attention_ignition_threshold_hz", 20 * Hz) / Hz)
            hold_ms = float(p.get("experiment_trigger_duration_ms", 50.0))
            if previous_rate >= threshold:
                self._task_above_count += 1
                if self._task_trigger_index is None and self._task_above_count * dt_ms >= hold_ms:
                    self._task_trigger_index = i_t - self._task_above_count + 1
            else:
                self._task_above_count = 0
            if previous_rate >= float(p.get("experiment_resource_threshold_hz", threshold)):
                self._task_active_ms += dt_ms
        time_ms = float(self.t_plot[i_t] / ms)
        if self.experiment_decay_mode == "target_decay":
            if self._task_trigger_index is None:
                raw = self.attention_task_amplitude if time_ms <= float(p.get("attention_task_off", p["stim_off"]) / ms) else 0.0
            else:
                trigger_ms = float(self.t_plot[self._task_trigger_index] / ms)
                tau_ms = float(p.get("experiment_task_decay_tau_ms", 100.0))
                raw = self.attention_task_amplitude * np.exp(-max(0.0, time_ms - trigger_ms) / tau_ms)
        elif self.experiment_decay_mode == "competition" and time_ms >= float(p.get("experiment_competitor_on_ms", np.inf)):
            steal = float(p.get("experiment_competitor_level", 1.0))
            inhibition = float(p.get("experiment_competition_inhibition", 0.8))
            raw /= 1.0 + steal * (1.0 - inhibition)
        elif self.experiment_decay_mode == "resource":
            depletion = float(p.get("experiment_resource_depletion_per_s", 2.0))
            raw *= np.exp(-depletion * self._task_active_ms / 1000.0)
        self._task_gain_cache_index = i_t
        self._task_gain_cache = float(raw)
        return self._task_gain_cache

    def get_attention_task_gain(self, i_t):
        if self.experiment_attention_mode not in {"intensity", "both"}:
            return 0.0
        return self._dynamic_task_gain(i_t)

    def _initialize_phase_state(self):
        """Create area phases and normalized incoming FLN phase weights."""
        rng = np.random.default_rng(self.phase_seed)
        self.phase_frequency_hz = rng.uniform(
            self.phase_mean_hz - self.phase_frequency_spread_hz,
            self.phase_mean_hz + self.phase_frequency_spread_hz,
            size=self.n_area,
        )
        frequency_overrides = self.network_params.get(
            "phase_frequency_overrides_hz", {}
        )
        area_names = self.network_params.get(b"area_list", [])
        for area, frequency in frequency_overrides.items():
            area = area.encode("utf-8") if isinstance(area, str) else area
            if area not in area_names:
                raise ValueError(f"Unknown phase-frequency area: {area!r}")
            frequency = float(frequency)
            if not np.isfinite(frequency) or frequency <= 0:
                raise ValueError("Per-area phase frequencies must be finite and positive")
            self.phase_frequency_hz[area_names.index(area)] = frequency
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
        if self.phase_mode == "sensory_to_workspace":
            names = self.network_params[b"area_list"]
            source_areas = self.network_params.get("active_area", [])
            workspace_areas = self.network_params.get("workspace_areas", [])
            if not source_areas or not workspace_areas:
                raise ValueError(
                    "sensory_to_workspace mode needs active_area and workspace_areas"
                )
            source = names.index(source_areas[0])
            targets = [names.index(area) for area in workspace_areas]
            self.phase_weights[:] = 0.0
            self.phase_weights[targets, source] = 1.0
        # Optional restriction for a targeted attention experiment. Phase
        # oscillators outside this set free-run, while the selected areas keep
        # the original directed FLN weights and are renormalized within-set.
        phase_network_areas = self.network_params.get("phase_network_areas")
        self.phase_network_indices = None
        if phase_network_areas:
            phase_indices = []
            for area in phase_network_areas:
                area = area.encode("utf-8") if isinstance(area, str) else area
                if area not in area_names:
                    raise ValueError(f"Unknown phase-network area: {area!r}")
                phase_indices.append(area_names.index(area))
            self.phase_network_indices = np.asarray(sorted(set(phase_indices)), dtype=int)
            mask = np.zeros_like(self.phase_weights, dtype=bool)
            mask[np.ix_(self.phase_network_indices, self.phase_network_indices)] = True
            self.phase_weights = np.where(mask, self.phase_weights, 0.0)
            row_sums = self.phase_weights.sum(axis=1, keepdims=True)
            self.phase_weights = np.divide(
                self.phase_weights, row_sums,
                out=np.zeros_like(self.phase_weights), where=row_sums > 0,
            )

        # When requested, only selected inter-area links receive a dynamic
        # phase gate; all other anatomical FLN links retain their native gain.
        phase_gate_areas = self.network_params.get("phase_gate_areas")
        self.phase_gate_indices = None
        if phase_gate_areas:
            gate_indices = []
            for area in phase_gate_areas:
                area = area.encode("utf-8") if isinstance(area, str) else area
                if area not in area_names:
                    raise ValueError(f"Unknown phase-gate area: {area!r}")
                gate_indices.append(area_names.index(area))
            self.phase_gate_indices = np.asarray(sorted(set(gate_indices)), dtype=int)
        self.phase_gate_history = np.ones(
            (self.n_t, self.n_area, self.n_area), dtype=np.float32
        )
        if self.disable_longrange_ee:
            self.phase_gate_history[:] = 0.0

    def get_longrange_gate_matrix(self, i_t):
        """Advance area phases and return target-by-source E-to-E gates."""
        dt_seconds = float(self.network_params["dt"] / second)
        previous = self.phase_rad[i_t - 1]
        phase_difference = previous[None, :] - previous[:, None]
        if self.phase_mode == "common":
            common_phase = (
                2.0 * np.pi * self.common_drive_frequency_hz
                * float(self.t_plot[i_t - 1] / second)
            )
            network_phase_drive = np.sin(common_phase - previous)
        elif self.phase_mode == "uncoupled":
            network_phase_drive = np.zeros(self.n_area, dtype=float)
        else:
            network_phase_drive = np.sum(
                self.phase_weights * np.sin(phase_difference), axis=1
            )
        if self.phase_network_indices is not None:
            selected = np.zeros(self.n_area, dtype=bool)
            selected[self.phase_network_indices] = True
            network_phase_drive = np.where(selected, network_phase_drive, 0.0)
        phase_task_gain = (
            self._dynamic_task_gain(i_t)
            if self.experiment_attention_mode in {"synchrony", "both"}
            else 0.0
        )
        coupling = self.phase_coupling_rad_s * (1.0 + phase_task_gain)
        deterministic_increment = (
            2.0 * np.pi * self.phase_frequency_hz
            + coupling * network_phase_drive
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
        gate = self.phase_gate_scale * (
            1.0 + self.phase_gate_depth * np.cos(phase_difference)
        )
        if self.phase_gate_indices is not None:
            scoped_gate = np.ones_like(gate)
            scoped_gate[np.ix_(self.phase_gate_indices, self.phase_gate_indices)] = (
                gate[np.ix_(self.phase_gate_indices, self.phase_gate_indices)]
            )
            gate = scoped_gate
        if self.disable_longrange_ee:
            gate = np.zeros_like(gate)
        self.phase_gate_history[i_t] = gate
        return gate

    def get_phase(self):
        """Return area phase traces in radians, shape (time, area)."""
        return self.phase_rad

    def get_phase_gate(self):
        """Return phase-gate traces, shape (time, target, source)."""
        return self.phase_gate_history
