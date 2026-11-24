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
- `final_notebooks/RTM-WB GNW Integration.ipynb`: deprecated pointer to the original-model CTC workflow. The separate GNW rate prototype has been retired.
- `final_notebooks/Original Model with CTC Phase Gating.ipynb`: an incremental CTC experiment on the original `model_class.Model` population-rate dynamics. It adds phase oscillators and gates long-range E-to-E FLN inputs while preserving the original local rate and synaptic equations. The CTC parameters are provisional.
- `project_experiment/`: dedicated workspace for CTC-model experiments.
- `project_experiment/CTC Validation Before GNW.ipynb`: a two-node S→W control experiment inspired by the supplied Kang/Quax MATLAB. It reuses the active branch's cortical E/I coefficients and fixed one-way E projection, then adds explicit A/B/C phase controls as experimental extensions; it is not yet a numerical reproduction of the MATLAB outputs. Its response threshold is in activity units, not Hz; it separates the source-inspired mechanism from optional synchrony-gated weights.
- `project_experiment/CTC Stage 0 - Original Model.ipynb`: Stage 0 infrastructure using the original 40-area `Model`; V4 is the sensory readout and the six-area workspace is the target readout.
- `project_experiment/CTC Stage 1 - CTC Validation.ipynb`: confirmatory A/B/C scan using the original model, with paired seeds, whole-stimulus PLV, rate MI, ignition/latency outcomes, matched-weight controls, paired bootstrap comparisons, and an optional K × stimulus map.
- `project_experiment/CTC Stage 2 - Visual Attention Synchrony.ipynb`: new visual-attention experiment using V4 and V1/V2 with provisional frontal control area 8l. Compares directed FLN phase coupling, a common-rhythm control, and ungated rate-only transmission; pilot scan is opt-in.
- `project_experiment/CTC Stage 3 - Joint Threshold Analysis.ipynb`: the former Stage 2 six-experiment notebook, renumbered and retained for strength × synchrony thresholds, attention-channel ablations, hysteresis, latency, attention decay, and ADHD-inspired parameter perturbations.
- `project_experiment/original_model_runner.py`: shared Stage 0–3 adapter that runs the original `Model`/`CTCModel`, preserving the full 40-area anatomy and population-rate equations.
- `project_experiment/stage0_two_node.py`: compatibility import for the Stage 0 runner; despite its historical filename, it no longer implements a separate two-node rate model.
- `project_experiment/legacy_two_node_prototype_results/`: archived pilot CSV/PNG outputs from the retired simplified rate prototype; rerun Stage 0–2 before interpreting results.
- `project_experiment/ctc_validation.py`: vectorized two-node simulation and paired-seed metrics used by the CTC validation notebook.
- `project_experiment/matlab_reference.py`: separate port of the supplied MATLAB script's active `determ=1` V1→V2 branch, including source-default time grid, alpha jitter, stimulus epoch, and phase-coherence calculations. The MATLAB `mi.m` helper was not included, so its MI values are reported only as explicitly labeled proxies.
- `final_notebooks/RTM-WB GNW Spiking Reference.ipynb`: the previous Brian2 cell-by-cell RTM/WB implementation, preserved for comparison. It is computationally much heavier.
- `final_notebooks/RTM-WB GNW Mean-Field.ipynb`: redirect to the original-model CTC notebook.

The RTM equations and initial state follow the supplied `RTMmodel.m`; the local inhibitory cell uses Wang–Buzsaki dynamics. The MATLAB file does not specify current units; the notebooks provisionally interpret the pulse amplitude as `µA/cm²`. Each area currently contains just one RTM/WB pair in the local-circuit baseline, so it is not a population model. The separate inter-area notebook uses one RTM cell per area and the project's FLN matrix to demonstrate directional, weighted activity propagation. The V4-only input and inter-area conductance scale are provisional teaching parameters.

The Stage 0–2 CTC experiments use the original rate-based `Model` in `model_class.py`; `CTCModel` adds phase dynamics and an optional gate over its long-range E→E FLN inputs. Results under `project_experiment/legacy_two_node_prototype_results/` come from the retired prototype and are preserved for provenance. Current Stage 0–2 outputs are generated by the original-model notebooks. The previous Brian2 cell-by-cell RTM/WB network remains available as a separate reference.

## Known issues

- `timer.py` imports `time` from `timeit` but calls `time.time()`. That import is a function and has no `time` attribute; the timer path will fail until it imports the `time` module (or uses `timeit.default_timer`).
- Several simulation modules use sibling imports such as `from super_plot import *`. Run the notebook with the repository's `simulations` directory on `sys.path` as the notebook does, or package these modules before importing them from standalone scripts.
- The original dependency file was named `requieremnts.txt` and omitted runtime imports including `sympy` and `chart_studio`. `requirements.txt` corrects the name and includes those imports.

