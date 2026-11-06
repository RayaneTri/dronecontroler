# dronecontroler

**ST5 – Data Driven Control** : data-driven identification and control of a drone (quadcopter).

The assignment is described in EINT_student_version_EN.pdf.

## Objectives

1. **Validate the dataset** (vertical acceleration of the drone, input `u` → output `y`).
2. **Identify a model**: a linear model and a recurrent neural network (RNN) model.
3. **Develop a control law**: LQR on the linear model, MPC on the RNN, with state estimation using a Kalman filter (KF / EKF).

## Repository structure

| Path | Content |
|---|---|
| `bdd_in_mat_05*.csv`, `bdd_out_mat_05*.csv` | Input/output dataset (raw and `_postfiltre` = filtered), one row = one trial, `Ts = 0.05 s` |
| `dataset_all.mat` | 3D dataset (references `REF` and outputs `Y`) generated with Simulink |
| `data_analysis.py`, `plot.py` | Dataset analysis and visualization (static characteristic, filtering) |
| `Moindre_carres.py`, `linear_model_lq.py`, `linear_model_matlab.py`, `modele_lineaire/` | Least-squares identification of the linear model and its response |
| `modele_analysis.py` | RNN vs. linear model comparison (MAE, RMSE) |
| `rnn.ipynb`, `rnn_extended.ipynb`, `rnn_mpc.ipynb` | RNN training (PyTorch Lightning): 1D case, extended case (`dataset_all.mat`), and MPC loop |
| `model_rnn/`, `export_NN/` | RNN checkpoints and weights exported as `.txt` |
| `mpc.py` | `model_nn` model (RNN/LSTM/GRU) and RNN-based MPC controller |
| `KF.py`, `apply_EKF.py` | Kalman filter (linear model) and EKF on the augmented state (RNN hidden state, velocities, positions) |
| `commande/` | Piecewise polynomial trajectory generation (`COMMANDE`) and command space |
| `generate_consigne.py` | Generates a sinusoidal reference (`consigne.csv`) |
| `generate_training_Data/` | Simulink models/scripts to generate training data with APRBS excitation |
| `Matlab files/` | Simulink simulator (`UAV_model.slx`, `EINT_system_simulation.slx`, `system_simulation_FC.slx`), `MPC.m`, `LQR.m`, `LinearModel.m`, `main_simulation.m`, `generate_dataset.m` classes/scripts |
| `presentation/` | LaTeX (Beamer) slides and figures |

## Requirements

- **MATLAB / Simulink** R2025b.
- **Python 3** with: `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`, `lightning`, `jupyter`.

Example setup (the `.venv` folder at the repository root is the one expected by `main_simulation.m`):

Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib torch lightning jupyter
```

Windows : 
```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas scipy matplotlib torch lightning jupyter

```

## Usage

### Analysis and linear model identification

```bash
python data_analysis.py      # dataset validation
python Moindre_carres.py     # least-squares identification
python KF.py                 # Kalman filter on the linear model
python modele_analysis.py    # RNN vs. linear model comparison
```

### RNN training

Open and run the notebooks:

- `rnn.ipynb`: 1D input/output RNN on the CSV files;
- `rnn_extended.ipynb`: multi-output RNN trained on `dataset_all.mat`;
- `rnn_mpc.ipynb`: closed-loop MPC test (the "plant" is simulated by the RNN itself).

Trained models are saved in `model_rnn/lightning_logs/`.

### Drone control in Simulink

1. Generate data if needed: `generate_training_Data/generate_prbs_in_out.m`.
2. In MATLAB, from `Matlab files/`, run `main_simulation.m`. It initializes the simulation (`random_init.m`), configures the Python environment, and calls the MPC controller (`MPC.m` → `mpc_controller.py`) with the EKF.
3. Reference trajectories in space can be produced with the `COMMANDE` class (`commande/commande_space.py`).

## Methods

- **Linear model**: `x[k+1] = a·x[k] + (1−a)·N(u[k])`, `y[k] = x[k]`, where `N` is the identified static characteristic.
- **RNN**: sequence-to-sequence recurrent network mapping `u → y`, trained with PyTorch Lightning.
- **MPC**: optimization over a horizon `N` (default 20) using the RNN as the prediction model.
- **EKF**: augmented state (RNN hidden state, velocities, positions), with semi-implicit Euler integration consistent with the MPC.

## Presentation

Slides are in [presentation/](presentation/) (`slides.tex`, build with `latexmk -pdf slides.tex`); the compiled PDF is `presentation/slides.pdf`.

## Authors

Jules Marceau, Matthieu Moussaoui, Rayane Trigoust, Christian Brunet, Clément Tamboise
