# AToum: RTM Simulation Baseline

This project contains a macaque cortex population model and an introductory RTM simulation. The introductory notebook assigns one RTM excitatory cell and one WB inhibitory interneuron to each area, connects them locally, and plots responses to a V4 input pulse. The original large-scale model and paper figure workflow remain available separately.

## Environment

The notebook metadata and checked-in Python bytecode indicate that this project was developed with Python 3.8. The dependency list has been upgraded for a modern scientific Python stack. Python 3.12 is the recommended Conda environment; recent dependencies are also installed in the repository's `.venv` when following the setup below.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --prefix .venv --name atoum --display-name "Python (atoum)"
jupyter lab
```

Alternatively, create the recommended Conda environment with `conda env create -f environment.yml` and activate it using `conda activate atoum`.

Open `final_notebooks/RTM Baseline.ipynb` for the local RTM–WB circuit walkthrough or `final_notebooks/Figure 2 - XX.ipynb` for the original figure workflow. Select the `Python (atoum)` kernel and launch Jupyter from the repository root.

On Windows, Brian2 simulations should use its NumPy code-generation backend; C++ standalone mode may require a separate compiler toolchain.

## Code overview

- `model_class.py`: Brian2-based multi-area network model and state initialization.
- `template_kwargs.py`: default model, stimulus, and network parameters.
- `simulations/`: repeated runs, recording, density plots, and Plotly helpers.
- `toolboxes/`: network/model helper functions and notebook-facing utilities.
- `data/`: connectivity inputs, spine counts, and paper figure data.
- `final_notebooks/`: notebook used to generate paper figures.
- `final_notebooks/RTM Baseline.ipynb`: step-by-step simulation of one RTM excitatory cell and one WB inhibitory interneuron per area, with local reciprocal synapses and V4-only input.
- `final_notebooks/RTM Inter-area Connectivity.ipynb`: one RTM cell per area connected by FLN-weighted projections, comparing V4 stimulation with inter-area links disabled and enabled.
- `final_notebooks/RTM-WB Local Networks and FLN.ipynb`: one RTM cell and 40 locally connected WB cells per area, with FLN-weighted RTM projections between areas.
- `final_notebooks/RTM-WB GNW Integration.ipynb`: the main 40-area RTM–WB GNW mean-field workflow. Each area has an excitatory population-rate state (RTM-like) and inhibitory population-rate state (WB-like); it retains the FLN matrix, V4 stimulus, workspace ignition measure, local recurrence, and missing-reverse workspace feedback scans. The rate grid is 1 ms, so this avoids Brian2 integration of thousands of individual cells. Rates are continuous model outputs, not individual spikes. Transfer curves and gains are provisional.
- `gnw_mean_field.py`: shared GNW mean-field transfer, initialization, and E/I/synapse time-step interface used by the 40-area notebook and smaller experiments.
- `final_notebooks/Original Model with CTC Phase Gating.ipynb`: an incremental CTC experiment on the original `model_class.Model` population-rate dynamics. It adds phase oscillators and gates long-range E-to-E FLN inputs while preserving the original local rate and synaptic equations. The CTC parameters are provisional.
- `project_experiment/`: dedicated workspace for CTC-model experiments.
- `project_experiment/CTC Validation Before GNW.ipynb`: a two-node S→W control experiment inspired by the supplied Kang/Quax MATLAB. It reuses the active branch's cortical E/I coefficients and fixed one-way E projection, then adds explicit A/B/C phase controls as experimental extensions; it is not yet a numerical reproduction of the MATLAB outputs. Its response threshold is in activity units, not Hz; it separates the source-inspired mechanism from optional synchrony-gated weights.
- `project_experiment/CTC Stage 0 - GNW Mean Field.ipynb`: two-node S/W infrastructure using the same GNW mean-field update function as the 40-area notebook, with firing-rate and added phase readouts, a paired A/B/C coupling/gate scan, mean-strength-matched fixed-weight controls, and paired bootstrap summaries.
- `project_experiment/CTC Stage 1 - CTC Validation.ipynb`: confirmatory A/B/C scan with 50 paired seeds, whole-stimulus PLV, bias-corrected rate MI, ignition/latency outcomes, matched-mean-weight controls, paired bootstrap comparisons, and an optional K × stimulus map.
- `project_experiment/CTC Stage 2 - Joint Threshold Analysis.ipynb`: six gated experiments for strength × synchrony thresholds, attention-channel ablations, recurrence and hysteresis, ignition latency, attention-decay mechanisms, and ADHD-inspired parameter perturbations. Uses the Stage 1 candidate window and the shared GNW mean-field interface; long scans are opt-in.
- `project_experiment/stage0_two_node.py`: the Stage 0 S/W implementation; local E/I rates and synaptic states call the shared `gnw_mean_field.mean_field_step` interface.
- `project_experiment/ctc_validation.py`: vectorized two-node simulation and paired-seed metrics used by the CTC validation notebook.
- `project_experiment/matlab_reference.py`: separate port of the supplied MATLAB script's active `determ=1` V1→V2 branch, including source-default time grid, alpha jitter, stimulus epoch, and phase-coherence calculations. The MATLAB `mi.m` helper was not included, so its MI values are reported only as explicitly labeled proxies.
- `final_notebooks/RTM-WB GNW Spiking Reference.ipynb`: the previous Brian2 cell-by-cell RTM/WB implementation, preserved for comparison. It is computationally much heavier.
- `final_notebooks/RTM-WB GNW Mean-Field.ipynb`: a pointer to the main mean-field integration notebook.

The RTM equations and initial state follow the supplied `RTMmodel.m`; the local inhibitory cell uses Wang–Buzsaki dynamics. The MATLAB file does not specify current units; the notebooks provisionally interpret the pulse amplitude as `µA/cm²`. Each area currently contains just one RTM/WB pair in the local-circuit baseline, so it is not a population model. The separate inter-area notebook uses one RTM cell per area and the project's FLN matrix to demonstrate directional, weighted activity propagation. The V4-only input and inter-area conductance scale are provisional teaching parameters.

The main GNW integration notebook uses a phenomenological mean-field rate model; it is not a parameter-preserving reduction of the Brian2 RTM/WB cell equations. The previous spiking network remains available as a reference notebook. The original rate-based `Model` in `model_class.py` remains unchanged and is available separately.

## Known issues

- `timer.py` imports `time` from `timeit` but calls `time.time()`. That import is a function and has no `time` attribute; the timer path will fail until it imports the `time` module (or uses `timeit.default_timer`).
- Several simulation modules use sibling imports such as `from super_plot import *`. Run the notebook with the repository's `simulations` directory on `sys.path` as the notebook does, or package these modules before importing them from standalone scripts.
- The original dependency file was named `requieremnts.txt` and omitted runtime imports including `sympy` and `chart_studio`. `requirements.txt` corrects the name and includes those imports.

