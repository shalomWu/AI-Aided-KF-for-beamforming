"""main_linear_canonical_32_multilink.py

Train a single shared-weights KalmanNet on 32 per-link scalar datasets
(REAL/IMAG × Tx1..4 × Rx1..4), each with its own MSE loss, while sharing
all KalmanNet weights.

Additionally, compute a CV-based WSR metric at each epoch using
SINR-based rate with an unrolled WMMSE precoder (imported from
Rate_Calculation.py), and log epoch-wise metrics to a CSV file.

The resulting checkpoint is a standard KalmanNetNN module object,
compatible with existing code that expects best-model.pt style ckpts.
"""

import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from Simulations.Linear_sysmdl import SystemModel
import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure, m, m1_0
from KNet.KalmanNet_nn import KalmanNetNN

# WMMSE + WSR (SINR-based) reference implementation
from Rate_Calculation import get_precoder, WSR_Calculation
# Unrolled WSR (differentiable) from Rate_Calculation_v2
from Rate_Calculation_v2 import unrolled_wsr_sinr_torch


# ======================== Helpers ========================


def _set_device(args) -> torch.device:
    if getattr(args, "use_cuda", False):
        if not torch.cuda.is_available():
            raise RuntimeError("args.use_cuda=True but no CUDA device found")
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def _build_sys_model(args, r2_db: float, nu_db: float) -> SystemModel:
    """Build SystemModel with given noise settings.

    r2_db: 1/r^2 in dB (SNR target)
    nu_db: ν in dB, where ν = q2/r2
    """
    # r2_db is 1/r^2 in dB → r^2 = 1 / 10^(r2_db/10)
    r2 = torch.tensor([1.0 / (10.0 ** (r2_db / 10.0))])
    v = 10.0 ** (nu_db / 10.0)
    q2 = torch.mul(v, r2)

    print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[0]))
    print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[0]))

    Q = q2 * Q_structure
    R = r2 * R_structure

    sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
    # deterministic initial condition for now
    m2_0 = 0 * torch.eye(m)
    sys_model.InitSequence(m1_0, m2_0)

    print("State Evolution Matrix:", F)
    print("Observation Matrix:", H)
    return sys_model


def _load_per_link_datasets(device: torch.device,
                            data_root: str,
                            T_seq: int,
                            stride: int,
                            snr_tag: str = "0dB") -> List[Dict[str, Any]]:
    """Load 32 per-link datasets (REAL/IMAG × 4×4) from .pt files.

    Each .pt is expected to have been produced by make_32dataset_from_H.py
    and thus contain 9 tensors:
      [train_input, train_target,
       cv_input, cv_target,
       test_input, test_target,
       train_lengthMask, cv_lengthMask, test_lengthMask]

    Returns a list of dicts with tensors on the given device.
    """
    entries: List[Dict[str, Any]] = []
    idx = 1
    for tx in range(1, 5): #5
        for rx in range(1, 5): #5
            for comp in ("real", "imag"):
                fname = f"{comp}_H_from_mat_Tx{tx}Rx{rx}_snr{snr_tag}_T{T_seq}_stride{stride}.pt"
                fpath = os.path.join(data_root, fname)
                if not os.path.isfile(fpath):
                    raise FileNotFoundError(f"Per-link dataset not found: {fpath}")

                obj = torch.load(fpath, map_location=device)
                if len(obj) == 9:
                    (train_input, train_target,
                     cv_input, cv_target,
                     test_input, test_target,
                     train_len, cv_len, test_len) = obj
                else:
                    raise RuntimeError(f"Unexpected .pt structure in {fpath}: len={len(obj)}")

                print(f"[Load] #{idx:02d} {comp.upper()} Tx{tx}-Rx{rx}:")
                print("       train:", train_input.shape,
                      "cv:", cv_input.shape,
                      "test:", test_input.shape)

                entries.append({
                    "idx": idx,
                    "comp": comp,
                    "tx": tx,
                    "rx": rx,
                    "train_input": train_input.to(device),
                    "train_target": train_target.to(device),
                    "cv_input": cv_input.to(device),
                    "cv_target": cv_target.to(device),
                    "test_input": test_input.to(device),
                    "test_target": test_target.to(device),
                })
                idx += 1

    assert len(entries) == 32, f"Expected 32 per-link datasets, got {len(entries)}"
    return entries


def _run_knet_on_batch(model: KalmanNetNN,
                       sys_model: SystemModel,
                       y_batch: torch.Tensor,
                       init_mean: torch.Tensor) -> torch.Tensor:
    """Run KalmanNet sequentially over time for a batch.

    y_batch: [B, n, T]
    init_mean: [B, m, 1]

    Returns x_hat: [B, m, T]
    """
    device = model.device
    B, n, T = y_batch.shape
    assert T == sys_model.T, "Sequence length mismatch with sys_model.T"

    model.batch_size = B
    model.init_hidden_KNet()
    model.InitSequence(init_mean.to(device), sys_model.T)

    x_hat = torch.zeros(B, sys_model.m, T, device=device)
    for t in range(T):
        y_t = y_batch[:, :, t].unsqueeze(2)  # [B, n, 1]
        x_hat[:, :, t] = model(y_t).squeeze(2)
    return x_hat


def _mse_linear(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    """Return scalar MSE in linear scale."""
    with torch.no_grad():
        return torch.mean((x_hat - x_true) ** 2).item()


def _to_db(x: float) -> float:
    return 10.0 * torch.log10(torch.tensor([x])).item()


def _grad_norm(model: torch.nn.Module) -> float:
    """Compute total L2 norm of gradients of all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm().item()
            total += param_norm ** 2
    return total ** 0.5


# Flag: if True, use WSR-based loss (differentiable) instead of per-link MSE for training.
# Default is False to preserve existing behavior.
USE_WSR_LOSS: bool = False  # False


def wmmse_precoder_torch(H: torch.Tensor,
                         snr_db: float,
                         Pt: float = 1.0,
                         maxIter: int = 5) -> torch.Tensor:
    """Torch-based WMMSE-style precoder.

    This is a fully differentiable approximation of the classical WMMSE
    for a single-cell MU-MISO channel H (N_Rx, N_Tx) with K = N_Rx streams.

    It does *not* rely on NumPy or SciPy and therefore supports autograd
    all the way back to H (and hence to KNet parameters).

    H      : [N_Rx, N_Tx] complex
    snr_db : scalar SNR in dB (same convention as in Rate_Calculation)
    Pt     : total transmit power
    maxIter: number of WMMSE iterations (unrolling depth)
    """
    device = H.device
    dtype = H.dtype

    # Ensure complex dtype
    if not torch.is_complex(H):
        H = H.to(torch.complex64)

    N_Rx, N_Tx = H.shape
    K = N_Rx  # one stream per receive antenna (consistent with W shape [N_Tx, K])

    SNR_lin = 10.0 ** (snr_db / 10.0)
    sigma2 = 1.0 / SNR_lin  # same convention as Rate_Calculation (rho2)

    # --- Initialization via SVD (similar spirit to get_precoder) ---
    # H: [N_Rx, N_Tx]  → we want right singular vectors (V)
    U_svd, S_svd, Vh_svd = torch.linalg.svd(H)
    V = Vh_svd.conj().transpose(-2, -1)[:, :K]  # [N_Tx, K]

    # Normalize to total power Pt
    pow_V = (V.conj() * V).real.sum()
    V = V * torch.sqrt(Pt / (pow_V + 1e-12))

    I_Nt = torch.eye(N_Tx, dtype=H.dtype, device=device)

    for _ in range(maxIter):
        # Multi-user WMMSE updates in vectorized form
        # 1) Effective channels: G = H V  (N_Rx x K), G[k,j] = h_k v_j
        G = H @ V  # [N_Rx, K]
        power = (G.conj() * G).real  # [N_Rx, K]

        # denom_k = sum_j |h_k v_j|^2 + sigma2
        denom = power.sum(dim=1) + sigma2  # [N_Rx]

        # desired gain g_k = h_k v_k  (diagonal of G)
        g_diag = torch.diagonal(G, dim1=0, dim2=1)  # [K]

        # 2) Receive filters u_k (scalar per user)
        u = g_diag.conj() / (denom + 1e-12)  # [K] complex

        # 3) MSE & weights
        e = 1.0 - 2.0 * (u * g_diag).real + (u.abs() ** 2) * denom
        w = 1.0 / (e + 1e-12)  # [K] real positive

        # 4) Update precoder V using closed-form (up to power normalization)
        #    A = H^H diag(w |u|^2) H
        #    B = H^H diag(w u*)      (columns of B are beta_k * h_k^H)
        alpha = w * (u.abs() ** 2)  # [K] real
        beta = w * u.conj()         # [K] complex

        H_H = H.conj().transpose(0, 1)  # [N_Tx, N_Rx]

        # A = H^H diag(alpha) H = H^H (alpha[:,None] * H)
        A = H_H @ (alpha.unsqueeze(1) * H)  # [N_Tx, N_Tx]

        # B = H^H diag(beta)  → each column k of B is beta_k * h_k^H
        B = H_H * beta  # [N_Tx, K] (column-wise scaling)

        # Regularize A slightly for numerical stability
        A_reg = A + 1e-6 * I_Nt

        # Solve A_reg V0 = B  → V0 = A_reg^{-1} B
        V0 = torch.linalg.solve(A_reg, B)  # [N_Tx, K]

        # Enforce total power constraint Tr(V V^H) = Pt via scaling
        pow_V0 = (V0.conj() * V0).real.sum()
        V = V0 * torch.sqrt(Pt / (pow_V0 + 1e-12))

    return V


def wsr_sinr_torch(H_true: torch.Tensor,
                   W: torch.Tensor,
                   snr_db: float) -> torch.Tensor:
    """
    Torch version of WSR_Calculation's SINR-based sum-rate.
    H_true : [N_Rx, N_Tx] complex
    W      : [N_Tx, K]    complex
    """
    if not torch.is_complex(H_true):
        H_true = H_true.to(torch.complex64)
    if not torch.is_complex(W):
        W = W.to(torch.complex64)

    N_Rx, N_Tx = H_true.shape
    N_Tx2, K = W.shape
    assert N_Tx2 == N_Tx, "W dims must be (N_Tx, K)."

    SNR_lin = 10.0 ** (snr_db / 10.0)
    rho2 = 1.0 / SNR_lin

    # M = H W
    M = H_true @ W                      # [N_Rx, K]
    power = (M.conj() * M).real         # [N_Rx, K]

    # desired power on stream i (diag), padded if K < N_Rx
    desired = torch.zeros(N_Rx, dtype=power.dtype, device=H_true.device)
    diag = torch.diagonal(power, dim1=0, dim2=1)   # [min(N_Rx, K)]
    num_streams = min(N_Rx, K)
    desired[:num_streams] = diag[:num_streams]

    total = power.sum(dim=1)            # [N_Rx]
    interf = total - desired            # [N_Rx]

    sinr = desired / (interf + rho2)    # [N_Rx]
    rate_terms = torch.log2(1.0 + sinr / N_Rx)  # <- SAME as numpy version
    return rate_terms.sum().real


def train_wsr_unrolled(H_true_b: torch.Tensor,
                       H_est_b: torch.Tensor,
                       snr_db: float,
                       Pt: float,
                       maxIter: int) -> torch.Tensor:
    """Wrapper around Rate_Calculation_v2.unrolled_wsr_sinr_torch for a single snapshot.

    H_true_b, H_est_b : [N_Rx, N_Tx] complex, on the correct device.
    """
    return unrolled_wsr_sinr_torch(
        H_eval=H_true_b,
        H_pred=H_est_b,
        snr_db=snr_db,
        Pt=Pt,
        maxIter=maxIter,
    )


# ======================== Main training ========================


def main():
    print("Pipeline Start (shared-weights 32-link KNet + WSR)")
    today = datetime.today()
    now = datetime.now()
    strTime = today.strftime("%m.%d.%y") + "_" + now.strftime("%H:%M:%S")
    print("Current Time =", strTime)

    # ---- General settings ----
    args = config.general_settings()

    global USE_WSR_LOSS
    USE_WSR_LOSS = False

    # Sequence length (must match datasets)
    args.T = 100
    args.T_test = 100
    args.randomLength = False

    # Noise / SNR settings
    target_inv_r2_dB = 0  # 1/r^2 in dB (SNR target)
    nu_dB = 6 #6.0             # ν in dB, where ν = q2/r2

    # Training hyper-parameters
    args.use_cuda = False
    args.n_steps = 400
    args.n_batch = 32  # per-link batch size per epoch
    args.lr = 8e-4 #5e-4
    args.wd = 1e-3 #1e-3

    # WMMSE settings
    args.wmmse_iters = getattr(args, "wmmse_iters", 15)
    # Switch target for MSE→WSR (in dB). If None, no automatic switch.
    args.switch_target_mse_db = getattr(args, "switch_target_mse_db", -10)
    # Device
    device = _set_device(args)

    # System model
    sys_model = _build_sys_model(args, r2_db=target_inv_r2_dB, nu_db=nu_dB)

    # ---- Data: 32 per-link datasets ----
    data_root = os.path.join("Simulations", "Linear_canonical", "data", "Siamese_14_11_2025")
    #data_root = os.path.join("Simulations", "Linear_canonical", "data", "Siamese_LLM4CP_comparison_dataset")
    T_SEQ = 100
    STRIDE = 20  # must match make_32dataset_from_H.py
    SNR_TAG = "0dB"

    links = _load_per_link_datasets(device, data_root, T_SEQ, STRIDE, snr_tag=SNR_TAG)

    # Dict from (comp, tx, rx) to link index in `links`
    link_index: Dict[Tuple[str, int, int], int] = {}
    for i, link in enumerate(links):
        key = (link["comp"], link["tx"], link["rx"])
        link_index[key] = i

    # Sanity: align args.N_E/N_CV/N_T with dataset sizes (they should all be equal)
    args.N_E = links[0]["train_input"].shape[0]
    args.N_CV = links[0]["cv_input"].shape[0]
    args.N_T = links[0]["test_input"].shape[0]
    print(f"Using N_E={args.N_E}, N_CV={args.N_CV}, N_T={args.N_T} from per-link datasets")

    # ---- Build shared KalmanNet ----
    KalmanNet_model = KalmanNetNN()
    KalmanNet_model.NNBuild(sys_model, args)
    KalmanNet_model.to(device)
    print("Number of trainable parameters for KalmanNet:",
          sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(KalmanNet_model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss(reduction='mean')

    # ---- Output / ckpt paths ----
    path_results = "KNet"  # same root as existing scripts
    os.makedirs(path_results, exist_ok=True)
    ckpt_dir = os.path.join(path_results, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    inv_r2_dB = target_inv_r2_dB
    nu_dB_eff = nu_dB
    ckpt_name = f"knet_inv{inv_r2_dB:+.1f}dB_nu{nu_dB_eff:+.1f}dB_16x1_shared32_v403.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    ckpt_wsr_path = os.path.join(ckpt_dir, "best-wsr-model_v403.pt")
    last_ckpt_path = os.path.join(ckpt_dir, "last-model_v403.pt")

    # CSV for epoch-wise metrics
    metrics_csv = os.path.join(path_results, "knet_shared32_metrics_v403.csv")
    metrics_f = open(metrics_csv, "w", newline="")
    metrics_writer = csv.writer(metrics_f)
    #metrics_writer.writerow(["epoch", "train_mse_db", "cv_mse_db", "cv_wsr_sinr"])
    metrics_writer.writerow(["epoch", "train_mse_db", "cv_mse_db", "train_wsr_sinr", "cv_wsr_sinr", "grad_norm"])

    best_cv_mean = float("inf")
    best_cv_wsr = -float("inf")

    # Initial mean for InitSequence (same for all links & batches)
    with torch.no_grad():
        m1x0 = sys_model.m1x_0.reshape(1, sys_model.m, 1).to(device)

    N_Rx, N_Tx = 4, 4
    snr_db_for_wsr = target_inv_r2_dB
    Pt = 1.0

    # Fixed evaluation subset for train-WSR metric (and WSR-loss branch)
    N_E_full = links[0]["train_input"].shape[0]
    N_B_wsr = min(args.n_batch, N_E_full)
    wsr_eval_idx = torch.randperm(N_E_full, device=device)[:N_B_wsr]
    t_mid = sys_model.T // 2
    # ==================== Training loop ====================
    for epoch in range(args.n_steps):
        KalmanNet_model.train()
        optimizer.zero_grad()

        train_mse_lin_per_link: List[float] = []
        grad_norm_value = 0.0
        train_wsr = float("nan")


        if not USE_WSR_LOSS:
            # ---- Accumulate gradients over all 32 links (MSE loss) ----
            for link in links:
                train_input = link["train_input"]  # [N_E, n, T]
                train_target = link["train_target"]  # [N_E, m, T]
                N_E = train_input.shape[0]
                N_B = min(args.n_batch, N_E)

                # Random subset indices for this link
                idx = torch.randperm(N_E, device=device)[:N_B]
                y_batch = train_input[idx]  # [N_B, n, T]
                x_true = train_target[idx]

                B, n, T = y_batch.shape
                init_mean = m1x0.repeat(B, 1, 1)  # [B, m, 1]
                x_hat = _run_knet_on_batch(KalmanNet_model, sys_model, y_batch, init_mean)

                loss = loss_fn(x_hat, x_true)
                loss.backward()

                train_mse_lin = loss.item()
                train_mse_lin_per_link.append(train_mse_lin)

            # Single optimizer step after accumulating all link gradients
            grad_norm_value = _grad_norm(KalmanNet_model)
            optimizer.step()
            # --- Train WSR metric on the same idx_wsr (no grad) ---
            with torch.no_grad():
                # Collect mid-time true/est per link for this eval subset
                train_true_mid = [None] * len(links)
                train_hat_mid = [None] * len(links)

                for link in links:
                    train_input = link["train_input"]
                    train_target = link["train_target"]

                    y_batch = train_input[wsr_eval_idx]  # [N_B_wsr, n, T]
                    x_true = train_target[wsr_eval_idx]  # [N_B_wsr, m, T]

                    B, n, T = y_batch.shape
                    init_mean = m1x0.repeat(B, 1, 1)
                    x_hat = _run_knet_on_batch(KalmanNet_model, sys_model, y_batch, init_mean)

                    # Store mid-time scalar (first state component) for this link
                    h_true_mid = x_true[:, 0, t_mid]  # [N_B_wsr]
                    h_hat_mid = x_hat[:, 0, t_mid]  # [N_B_wsr]

                    link_idx = link["idx"] - 1
                    train_true_mid[link_idx] = h_true_mid
                    train_hat_mid[link_idx] = h_hat_mid

                # For each WSR sample and compute WSR
                rates = []
                for b in range(wsr_eval_idx.shape[0]):
                    H_true_b = torch.zeros((N_Rx, N_Tx), dtype=torch.complex64, device=device)
                    H_est_b = torch.zeros_like(H_true_b)

                    for rx in range(1, N_Rx + 1):
                        for tx in range(1, N_Tx + 1):
                            idx_real = link_index[("real", tx, rx)]
                            idx_imag = link_index[("imag", tx, rx)]

                            r_true = train_true_mid[idx_real][b]
                            i_true = train_true_mid[idx_imag][b]
                            r_est = train_hat_mid[idx_real][b]
                            i_est = train_hat_mid[idx_imag][b]

                            H_true_b[rx - 1, tx - 1] = torch.complex(r_true, i_true)
                            H_est_b[rx - 1, tx - 1] = torch.complex(r_est, i_est)

                    rate_b = train_wsr_unrolled(
                        H_true_b, H_est_b,
                        snr_db_for_wsr, Pt, args.wmmse_iters
                    )
                    rates.append(rate_b)

                if len(rates) > 0:
                    rates_t = torch.stack(rates)
                    train_wsr = float(rates_t.mean().item())
                else:
                    train_wsr = float("nan")
        else:
            # ---- WSR-based training over all 32 links (shared batch indices) ----
            N_E = links[0]["train_input"].shape[0]
            # N_B = min(args.n_batch, N_E)
            N_B = wsr_eval_idx.shape[0]  # same batch as for metric
            idx_batch = wsr_eval_idx
            # t_mid already defined before the loop

            # Per-link mid-time samples for WSR construction
            train_true_mid = [None] * len(links)
            train_hat_mid = [None] * len(links)

            for link in links:
                train_input = link["train_input"]  # [N_E, n, T]
                train_target = link["train_target"]  # [N_E, m, T]

                y_batch = train_input[idx_batch]  # [N_B, n, T]
                x_true = train_target[idx_batch]

                B, n, T = y_batch.shape
                init_mean = m1x0.repeat(B, 1, 1)
                x_hat = _run_knet_on_batch(KalmanNet_model, sys_model, y_batch, init_mean)

                # MSE for logging only (no grad from MSE in this branch)
                mse_link = loss_fn(x_hat, x_true)
                train_mse_lin_per_link.append(mse_link.item())

                # Store mid-time scalar (first state component) for this link
                h_true_mid = x_true[:, 0, t_mid]  # [N_B]
                h_hat_mid = x_hat[:, 0, t_mid]    # [N_B]

                link_idx = link["idx"] - 1
                train_true_mid[link_idx] = h_true_mid
                train_hat_mid[link_idx] = h_hat_mid

            # Now build WSR over the shared batch
            rates = []
            for b in range(N_B):
                # Reconstruct 4x4 complex H_true and H_est for sample b
                H_true_b = torch.zeros((N_Rx, N_Tx), dtype=torch.complex64, device=device)
                H_est_b = torch.zeros_like(H_true_b)

                for rx in range(1, N_Rx + 1):
                    for tx in range(1, N_Tx + 1):
                        idx_real = link_index[("real", tx, rx)]
                        idx_imag = link_index[("imag", tx, rx)]

                        r_true = train_true_mid[idx_real][b]
                        i_true = train_true_mid[idx_imag][b]
                        r_est = train_hat_mid[idx_real][b]
                        i_est = train_hat_mid[idx_imag][b]

                        H_true_b[rx - 1, tx - 1] = torch.complex(r_true, i_true)
                        H_est_b[rx - 1, tx - 1] = torch.complex(r_est, i_est)

                rate_b = train_wsr_unrolled(H_true_b, H_est_b,
                                            snr_db_for_wsr, Pt, args.wmmse_iters)
                rates.append(rate_b)

            if len(rates) == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                train_wsr = float("nan")
            else:
                rates_t = torch.stack(rates)  # [N_B]
                wsr_mean = rates_t.mean()
                train_wsr = float(wsr_mean.item())  # <-- this is what we'll log
                loss = -wsr_mean**2 #+ 0.1*mse_link # maximize WSR ⇒ minimize negative WSR
                #loss = wsr_mean  # maximize WSR ⇒ minimize negative WSR
                #loss = (wsr_mean - 1.2)**2

            loss.backward()
            grad_norm_value = _grad_norm(KalmanNet_model)
            optimizer.step()

        # ---- CV evaluation (no grad) ----
        KalmanNet_model.eval()
        cv_mse_lin_per_link: List[float] = []

        # For WSR: store mid-time true/est per link & CV sample
        t_mid = sys_model.T // 2
        cv_true_mid: List[np.ndarray] = [None] * len(links)
        cv_hat_mid: List[np.ndarray] = [None] * len(links)

        with torch.no_grad():
            for link in links:
                cv_input = link["cv_input"]
                cv_target = link["cv_target"]
                N_CV = cv_input.shape[0]

                y_batch = cv_input  # [N_CV, n, T]
                x_true = cv_target

                B, n, T = y_batch.shape
                init_mean = m1x0.repeat(B, 1, 1)
                x_hat = _run_knet_on_batch(KalmanNet_model, sys_model, y_batch, init_mean)

                mse_lin = _mse_linear(x_hat, x_true)
                cv_mse_lin_per_link.append(mse_lin)

                # Store mid-time true/est scalar (first state component) for WSR
                h_true_mid = x_true[:, 0, t_mid].detach().cpu().numpy().astype(np.float64)
                h_hat_mid = x_hat[:, 0, t_mid].detach().cpu().numpy().astype(np.float64)

                idx0 = link["idx"] - 1
                cv_true_mid[idx0] = h_true_mid
                cv_hat_mid[idx0] = h_hat_mid

        # ---- Aggregate MSE ----
        train_mean_lin = sum(train_mse_lin_per_link) / len(train_mse_lin_per_link)
        cv_mean_lin = sum(cv_mse_lin_per_link) / len(cv_mse_lin_per_link)

        train_mean_db = _to_db(train_mean_lin)
        cv_mean_db = _to_db(cv_mean_lin)

        # ---- Compute WSR on CV snapshots (SINR-based) ----
        # Use all CV samples; you can subsample here if you want.
        N_CV = links[0]["cv_input"].shape[0]
        wsr_sinr_list: List[float] = []

        for s in range(N_CV):
            # Reconstruct 4x4 complex H_true and H_est from 32 per-link scalars
            H_true = np.zeros((N_Rx, N_Tx), dtype=np.complex128)
            H_est = np.zeros_like(H_true)
            for rx in range(1, N_Rx + 1):
                for tx in range(1, N_Tx + 1):
                    idx_real = cv_true_mid[link_index[("real", tx, rx)]][s]
                    idx_imag = cv_true_mid[link_index[("imag", tx, rx)]][s]

                    r_true = cv_true_mid[link_index[("real", tx, rx)]][s]
                    i_true = cv_true_mid[link_index[("imag", tx, rx)]][s]
                    r_hat = cv_hat_mid[link_index[("real", tx, rx)]][s]
                    i_hat = cv_hat_mid[link_index[("imag", tx, rx)]][s]

                    H_true[rx - 1, tx - 1] = r_true + 1j * i_true
                    H_est[rx - 1, tx - 1] = r_hat + 1j * i_hat

            # W from unrolled WMMSE on estimated channel
            W = get_precoder(H_est, snr_db=snr_db_for_wsr, Pt=Pt,
                             maxIter=args.wmmse_iters, use_svd_init=True)
            _, rate_sinr = WSR_Calculation(H_true, W, snr_db=snr_db_for_wsr)
            wsr_sinr_list.append(rate_sinr)

        cv_wsr = float(np.mean(wsr_sinr_list)) if len(wsr_sinr_list) > 0 else float("nan")
        # Track & save best model (by CV WSR)
        if not np.isnan(cv_wsr) and cv_wsr > best_cv_wsr:
            best_cv_wsr = cv_wsr
            torch.save(KalmanNet_model, ckpt_wsr_path)
            print(f"[Best-WSR] New best avg CV WSR: {cv_wsr:.4f} bit/s/Hz → saved to {ckpt_wsr_path}")

        # ---- Logging ----
        print(
            f"Epoch {epoch:04d} | train MSE: {train_mean_db:+.3f} dB   "
            f"cv MSE: {cv_mean_db:+.3f} dB   "
            f"train WSR(SINR): {train_wsr:.4f}   cv WSR(SINR): {cv_wsr:.4f} bit/s/Hz"
        )
        # Per-link MSE print 1 / 32 links
        for i, link in enumerate(links[:0]):
            t_db = _to_db(train_mse_lin_per_link[i])
            c_db = _to_db(cv_mse_lin_per_link[i])
            print(
                f"   link#{link['idx']:02d} ({link['comp'].upper()} Tx{link['tx']}-Rx{link['rx']}): "
                f"train {t_db:+.3f} dB, cv {c_db:+.3f} dB"
            )

        # CSV line
        metrics_writer.writerow(
            [epoch, train_mean_db, cv_mean_db, train_wsr, cv_wsr, grad_norm_value]
        )
        metrics_f.flush()

        # ---- Track & save best model (by average CV MSE) ----
        if cv_mean_lin < best_cv_mean:
            best_cv_mean = cv_mean_lin
            # Save full module object (same style as Pipeline_EKF best-model.pt)
            torch.save(KalmanNet_model, ckpt_path)
            print(f"[Best] New best avg CV MSE: {cv_mean_db:+.3f} dB → saved to {ckpt_path}")
        # ---- Optional switch from MSE to WSR loss mode ----
        if (args.switch_target_mse_db is not None) and (not USE_WSR_LOSS):
            if cv_mean_db <= args.switch_target_mse_db:
                USE_WSR_LOSS = True
                args.lr = 2e-4
                print(
                    f"[Switch] Epoch {epoch:04d}: CV MSE {cv_mean_db:+.3f} dB ≤ "
                    f"target {args.switch_target_mse_db:+.3f} dB → WSR loss ON"
                )

    metrics_f.close()
    torch.save(KalmanNet_model, last_ckpt_path)
    print("Training finished.")
    print(f"Best model saved at: {ckpt_path}")
    print(f"Metrics CSV saved at: {metrics_csv}")
    print(f"Last-epoch model saved at: {last_ckpt_path}")


if __name__ == "__main__":
    main()
