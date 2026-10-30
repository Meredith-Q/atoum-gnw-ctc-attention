# AToum: GNW–CTC Attention Model

AToum is a large-scale macaque cortex model being extended to study Global Neuronal Workspace (GNW) ignition, communication through coherence (CTC), and attention-related gain. The repository includes the model, simulation helpers, source data, paper figure notebook, and a provisional GNW/CTC baseline notebook. Attention-related effects and workspace labels remain configurable research hypotheses; see the notebooks before interpreting outputs.

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

Open `final_notebooks/GNW CTC Baseline.ipynb` for the provisional baseline or `final_notebooks/Figure 2 - XX.ipynb` for the original figure workflow. Select the `Python (atoum)` kernel and launch Jupyter from the repository root.

On Windows, Brian2 simulations should use its NumPy code-generation backend; C++ standalone mode may require a separate compiler toolchain.

## Code overview

- `model_class.py`: Brian2-based multi-area network model and state initialization.
- `template_kwargs.py`: default model, stimulus, and network parameters.
- `simulations/`: repeated runs, recording, density plots, and Plotly helpers.
- `toolboxes/`: network/model helper functions and notebook-facing utilities.
- `data/`: connectivity inputs, spine counts, and paper figure data.
- `final_notebooks/`: notebook used to generate paper figures.
- `gnw_metrics.py`: workspace ignition and phase-locking metrics used by the baseline notebook.

## Known issues

- `timer.py` imports `time` from `timeit` but calls `time.time()`. That import is a function and has no `time` attribute; the timer path will fail until it imports the `time` module (or uses `timeit.default_timer`).
- Several simulation modules use sibling imports such as `from super_plot import *`. Run the notebook with the repository's `simulations` directory on `sys.path` as the notebook does, or package these modules before importing them from standalone scripts.
- The original dependency file was named `requieremnts.txt` and omitted runtime imports including `sympy` and `chart_studio`. `requirements.txt` corrects the name and includes those imports.

