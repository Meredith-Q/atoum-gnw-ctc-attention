# AToum: RTM Simulation Baseline

This project contains a macaque cortex population model and an introductory RTM simulation. The current baseline notebook isolates RTM cells, assigns one cell to each area, and plots voltage and spike responses to a V4 input pulse. The original large-scale model and paper figure workflow remain available separately.

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

Open `final_notebooks/RTM Baseline.ipynb` for the RTM-only walkthrough or `final_notebooks/Figure 2 - XX.ipynb` for the original figure workflow. Select the `Python (atoum)` kernel and launch Jupyter from the repository root.

On Windows, Brian2 simulations should use its NumPy code-generation backend; C++ standalone mode may require a separate compiler toolchain.

## Code overview

- `model_class.py`: Brian2-based multi-area network model and state initialization.
- `template_kwargs.py`: default model, stimulus, and network parameters.
- `simulations/`: repeated runs, recording, density plots, and Plotly helpers.
- `toolboxes/`: network/model helper functions and notebook-facing utilities.
- `data/`: connectivity inputs, spine counts, and paper figure data.
- `final_notebooks/`: notebook used to generate paper figures.
- `final_notebooks/RTM Baseline.ipynb`: step-by-step simulation of one RTM cell per area, with V4-only input and voltage/spike plots.

The RTM equations and initial state follow the supplied `RTMmodel.m`. The MATLAB file does not specify current units; the notebook provisionally interprets the pulse amplitude as `µA/cm²`. The V4-only input is a regionalized teaching example; the MATLAB source itself describes a single cell, not an area-specific cortical network. Identical unstimulated RTM cells have identical inputs and initial states, so their traces overlap. Inter-area connections are not included in this baseline.

## Known issues

- `timer.py` imports `time` from `timeit` but calls `time.time()`. That import is a function and has no `time` attribute; the timer path will fail until it imports the `time` module (or uses `timeit.default_timer`).
- Several simulation modules use sibling imports such as `from super_plot import *`. Run the notebook with the repository's `simulations` directory on `sys.path` as the notebook does, or package these modules before importing them from standalone scripts.
- The original dependency file was named `requieremnts.txt` and omitted runtime imports including `sympy` and `chart_studio`. `requirements.txt` corrects the name and includes those imports.

