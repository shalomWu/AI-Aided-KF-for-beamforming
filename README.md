# AI-Aided-KF-for-beamforming
KalmanNet-Based Channel Prediction for Beamforming
# AI-Aided Kalman Filter for Beamforming

A PyTorch implementation of a **KalmanNet**-based channel estimator for MIMO beamforming. The system trains a shared-weights neural Kalman filter across 32 scalar sub-channels extracted from a 4×4 complex MIMO channel matrix (QuaDRiGa simulation data), with optional Phase Noise (PN) modelling and a WMMSE-based Weighted Sum Rate (WSR) metric.

---

## Repository Structure

```
.
├── make_32_dataset_from_h.py              # Dataset creation from .mat channel file
├── main_linear_canonical_32_multilink_wsr_v_7.py  # Training script (shared-weight KalmanNet)
├── ChannelCheckKnetSNR.py                 # Testing / inference on a single link
├── Pipeline_EKF.py                        # Training and testing pipeline class
├── Simulations/
│   └── Linear_canonical/
│       ├── parameters.py                  # State-space model parameters (F, H, Q, R)
│       └── data/                          # Place .mat and generated .pt files here
├── KNet/
│   ├── KalmanNet_nn.py                    # KalmanNet neural network definition
│   └── ckpts/                             # Saved SNR-tagged model checkpoints
├── Rate_Calculation.py                    # WMMSE precoder and WSR (SINR-based)
└── Rate_Calculation_v2.py                 # Differentiable unrolled WSR (PyTorch)
```

---

## Overview

The pipeline has three stages:

1. **Dataset Generation** (`make_32_dataset_from_h.py`) — Loads a complex 4×4 MIMO channel `H` from a `.mat` file and produces 32 `.pt` dataset files (one per link component: real/imag × Tx1–4 × Rx1–4).

2. **Training** (`main_linear_canonical_32_multilink_wsr_v_7.py`) — Trains a single KalmanNet with shared weights across all 32 datasets simultaneously, minimising a per-link MSE loss. Optionally logs a CV-based WSR metric per epoch.

3. **Testing / Inference** (`ChannelCheckKnetSNR.py`) — Loads the nearest SNR-tagged checkpoint and runs inference on a selected (Rx, Tx) link from a `.mat` file, producing prediction and estimation plots and saving results.

---

## Dependencies

Install with pip:

```bash
pip install torch numpy scipy matplotlib
```

| Package | Purpose |
|---------|---------|
| `torch` | KalmanNet model, training, inference |
| `numpy` | Channel data processing |
| `scipy` | Loading `.mat` files |
| `matplotlib` | Plotting estimation results |

---

## Step 1 — Generate Datasets

**Script:** `make_32_dataset_from_h.py`

Edit the user parameters at the top of the file:

```python
MAT_PATH   = "Simulations/Linear_canonical/data/Channel_H_4x4_v4.mat"  # Path to your .mat file
MAT_VARNAME = "H"           # Variable name inside the .mat (expected shape: Ntx × Mrx × T)
OUT_DIR    = "Simulations/Linear_canonical/data/Siamese_14_11_2025"     # Output folder
N_E, N_CV, N_T = 1000, 100, 200  # Train / validation / test split sizes
T_SEQ      = 100            # Sequence length per sample
STRIDE     = 20             # Sliding window stride
SNR_dB     = 15.0           # Measurement noise SNR in dB
RNG_SEED   = 1234           # Reproducibility seed
```

**To add Phase Noise (PN) during dataset creation**, modify line 131:
```python
phi = 0.1 * rng.random(h_norm.shape) * 2.0 * np.pi  # PN amplitude (currently 0.1 × 2π)
```

Run:
```bash
python make_32_dataset_from_h.py
```

This produces 32 `.pt` files in `OUT_DIR`, named:
```
real_H_from_mat_Tx1Rx1_snr15dB_T100_stride20.pt
imag_H_from_mat_Tx1Rx1_snr15dB_T100_stride20.pt
...
imag_H_from_mat_Tx4Rx4_snr15dB_T100_stride20.pt
```

Each `.pt` file contains 9 tensors:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `train_input` | (1000, 2, 100) | Noisy measurements — training |
| `train_target` | (1000, 2, 100) | Clean channel — training |
| `cv_input` | (100, 2, 100) | Noisy measurements — validation |
| `cv_target` | (100, 2, 100) | Clean channel — validation |
| `test_input` | (200, 2, 100) | Noisy measurements — test |
| `test_target` | (200, 2, 100) | Clean channel — test |
| `train_lengthMask` | (1000, 2, 1) | Padding mask — training |
| `cv_lengthMask` | (100, 2, 1) | Padding mask — validation |
| `test_lengthMask` | (200, 2, 1) | Padding mask — test |

---

## Step 2 — Train KalmanNet

**Script:** `main_linear_canonical_32_multilink_wsr_v_7.py`

This trains a **single shared-weights KalmanNet** on all 32 per-link datasets simultaneously.

Key configurable parameters (via `args` / config):

| Parameter | Description |
|-----------|-------------|
| `args.T` | Training sequence length (must match `T_SEQ` from dataset generation) |
| `args.T_test` | Test sequence length |
| `args.n_steps` | Number of training epochs |
| `args.n_batch` | Batch size |
| `args.lr` | Learning rate (Adam optimiser) |
| `args.wd` | Weight decay (L2 regularisation) |
| `args.alpha` | Composition loss factor (if `CompositionLoss` is enabled) |
| `r2_db` | Measurement noise level (1/r² in dB; set to match dataset SNR) |
| `nu_db` | Process-to-measurement noise ratio ν = q²/r² in dB |

**To change the NMSE metric (prediction vs estimation)**, see line 226 of `ChannelCheckKnetSNRAllLinks_v7`.

Run:
```bash
python main_linear_canonical_32_multilink_wsr_v_7.py
```

The best model (lowest validation MSE) is saved to:
```
KNet/best-model.pt
```

SNR-tagged checkpoints are also saved to `KNet/ckpts/` for later inference at specific SNR values.

---

## Step 3 — Test / Inference

**Script:** `ChannelCheckKnetSNR.py`

Runs the trained KalmanNet on a single selected Tx–Rx link and plots the predicted vs true channel along with NMSE results.

Edit the user config at the top of the file:

```python
REPO_ROOT    = r"C:\path\to\KalmanNet_TSP"   # Root directory of the project
DATA_DIR     = os.path.join(REPO_ROOT, r"Simulations\Linear_canonical\data")
MAT_FILENAME = "Channel_H_4x4.mat"           # .mat file for testing
CKPT_DIR     = r"...\KNet\ckpts"             # Folder with SNR-tagged checkpoints

RX_IDX = 1      # 0-based Rx antenna index
TX_IDX = 1      # 0-based Tx antenna index
SNR_DB = 15.0   # Target SNR — script auto-selects nearest available checkpoint
PLOT_K = 2000   # Number of time samples to plot
```

**To add Phase Noise during testing**, modify line 106 of `ChannelCheckKnetSNR.py`.

Run:
```bash
python ChannelCheckKnetSNR.py
```

The script will:
- Auto-discover all SNR-tagged checkpoints in `CKPT_DIR`
- Select the nearest checkpoint to `SNR_DB`
- Run KalmanNet inference on the selected link
- Print NMSE (dB) for both the **prediction** and **estimation** stages
- Save plots to `KNet/test_plots/`

---

## Pipeline Class — `Pipeline_EKF`

`Pipeline_EKF.py` provides the `Pipeline_EKF` class used by the training and testing scripts.

Key methods:

| Method | Description |
|--------|-------------|
| `setssModel(ssModel)` | Attach the state-space model |
| `setModel(model)` | Attach the KalmanNet model |
| `setTrainingParams(args)` | Set learning rate, batch size, loss type, etc. |
| `NNTrain(...)` | Run full training loop; saves `best-model.pt` on validation improvement |
| `NNTest(...)` | Run inference; returns MSE array, average, and output trajectories |

---

## Key Notes

- The `.mat` channel file must contain a variable `H` with shape **(Ntx, Mrx, T)** and complex dtype.
- The number of sliding windows `K = floor((T - T_SEQ) / STRIDE) + 1` must be ≥ 1300 (`N_E + N_CV + N_T`). If not, increase `T` in the `.mat` or reduce `STRIDE`.
- Phase Noise can be independently enabled for **training** (dataset generation, line 131 of `make_32_dataset_from_h.py`) and **testing** (line 106 of `ChannelCheckKnetSNR.py`).
- NMSE metric (prediction vs estimation) is controlled at line 226 of `ChannelCheckKnetSNRAllLinks_v7`.
- The model uses an **Adam optimiser** with MSE loss. An optional composition loss (state + measurement reconstruction) is supported via `args.CompositionLoss`.

---

## Citation / Reference

This work builds on the **KalmanNet** framework. If you use this code, please also cite the original KalmanNet paper:

> Revach et al., "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics," *IEEE Transactions on Signal Processing*, 2022.
