# Quantum Algorithms & Hardware Experiments

This repository bridges theoretical quantum computing algorithms with practical execution on noisy intermediate-scale quantum (NISQ) hardware. It contains modular implementations of advanced quantum algorithms, educational Jupyter notebooks, and automated scripts for running experiments on real IBM Quantum chips.

---

## 📂 Project Architecture

The project is structured using a domain-driven approach. Each algorithm (e.g., LCU, OAA) has its own dedicated ecosystem containing core logic, experiment scripts, and results.

* **`notebooks/`**
  Educational Jupyter notebooks detailing the mathematical theory, step-by-step circuit construction, and interactive visualizations. Start here if you want to understand the *why* and *how*.
* **`src/`**
  The production-ready codebase.
  * **`src/<problem>/core.py`**: Pure logic and circuit generation factories.
  * **`src/<problem>/data_handler.py`**: Automated tools for saving data and generating comparative plots.
  * **`src/<problem>/experiments/`**: Executable Python scripts configured to run specific test cases (e.g., ideal simulators vs. hardware stress tests).
* **`results/`**
  Aggregated outputs from the experiment scripts. Contains raw `.csv` data, human-readable `.txt` reports, and `.png` comparison charts. Detailed analysis of these results can be found in the specific sub-folders.

---

## 🚀 Getting Started

Follow these steps to set up the project locally and run your first quantum experiment.

### 1. Environment Setup
It is highly recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Clone the repository
git clone https://github.com/honzahavelka/ntc_quantum.git
cd ntc_quantum

# Create a virtual environment (Windows default)
python -m venv qiskit_env

# Activate the virtual environment (Windows default)
qiskit_env\Scripts\activate

# Note for macOS/Linux users:
# Run: `python3 -m venv qiskit_env` 
# Then run: `source qiskit_env/bin/activate`

# Install required packages
pip install -r requirements.txt
```

### 2. IBM Quantum Authentication
To execute scripts on real quantum hardware, you need an IBM Quantum API token. 

> **Security Note:** Never hardcode your API token into the experiment scripts. 

Save your token locally to your machine by running this command once in your Python console or Jupyter Notebook:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="YOUR_SECRET_TOKEN", overwrite=True)
```
The experiment scripts will automatically load this saved credential securely.

---

## 🧠 How to Run Experiments

Because the executable scripts are nested within the `src/` directory, they must be run as Python modules from the **root directory** of the project.

**Example: Running the LCU Baseline Experiment**
```bash
# Ensure you are in the root directory of the project
python -m src.lcu.experiments.run_exp_a_baseline

# Note for macOS/Linux users: 
# You may need to use `python3` instead of `python` depending on your system configuration.
```

Once the job is completed by the IBM queue, the script will automatically generate the corresponding data, text reports, and plots inside the `results/<problem>/` directory.

---

## 📊 Experimental Results & Analysis

Detailed analysis, physics interpretations, and conclusions drawn from the hardware executions are documented separately for clarity.

* Navigate to `results/<problem>/README.md` for the analysis of the problem.