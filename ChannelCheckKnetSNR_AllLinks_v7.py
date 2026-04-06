# ChannelCheckKnetSNR_AllLinks.py
# -----------------------------------------------------------------------------
# Runs the single-link flow from ChannelCheckKnetSNR.py over ALL (Rx,Tx) links,
# computes the average NMSE (linear; printed in dB), and saves the complex
# estimate tensor h_Knet with the same shape/order as Channel_H_4x4.mat.
# -----------------------------------------------------------------------------

import os
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat, savemat

# ===================== User config =====================

REPO_ROOT = r"C:\Users\shalomw\Documents\Education\Project ideas\Code\KalmanNet_TSP"
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import the single-link building blocks
import importlib
base = importlib.import_module("ChannelCheckKnetSNR")

# Data / outputs
DATA_DIR     = os.path.join(REPO_ROOT, r"Simulations\Linear_canonical\data")
#DATA_DIR     = os.path.join(REPO_ROOT, r"Simulations\Linear_canonical\data")
#MAT_FILENAME = "H_true_sub24_Hlong_v3.mat"       # expects key 'H' with shape (N_Rx, M_Tx, T)
MAT_FILENAME = "Channel_H_4x4.mat"
RESULTS_DIR  = os.path.join(REPO_ROOT, "KNet")  # fallback search root for 'best-model.pt'
CKPT_DIR     = os.path.join(REPO_ROOT, r"KNet\ckpts")
#CKPT_DIR     = os.path.join(REPO_ROOT, r"KNet")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Inference settings
SNR_DB = 0
SEED   = None

# State-space (must match training)
F_MATRIX = np.array([[1, 0.09],
                     [0.00, 1]], dtype=np.float32)

# Where to save the h_Knet matrix
SAVE_DIR  = os.path.join(REPO_ROOT, r"Simulations\Linear_canonical\data\Siamese_5_4_2026")
SAVE_NAME = f"h_Knet_Channel_H_4x4_SNR{int(round(SNR_DB)):+d}dB_v402_best_wsr.mat"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)

# ===================== Helpers =====================

def _load_H():
    mat_path = os.path.join(DATA_DIR, MAT_FILENAME)
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    M = loadmat(mat_path)
    if "H" not in M:
        raise KeyError(f"'H' not found in {mat_path}. Keys: {list(M.keys())}")
    H = M["H"]
    if H.ndim != 3:
        raise ValueError(f"Expected H to be 3D (N_Rx, M_Tx, T). Got shape {H.shape}")
    return H

def _select_checkpoint():
    target_inv = float(SNR_DB)
    cands = base.discover_snr_ckpts(CKPT_DIR)
    if cands:
        print(f"[CKPT] Discovered {len(cands)} SNR-tagged models in: {CKPT_DIR}")
        for d in sorted(cands, key=lambda x: x["inv"]):
            delta = abs(d["inv"] - target_inv)
            nu    = d.get("nu", None)
            nu_s  = f", nu={nu:+.1f} dB" if nu is not None else ""
            print(f"       inv={d['inv']:+.1f} dB (Δ={delta:.2f} dB){nu_s}  ← {Path(d['path']).name}")

    chosen = base.pick_ckpt_by_snr(CKPT_DIR, target_inv_dB=target_inv)
    if chosen is not None:
        print(f"[CKPT] Selected nearest SNR model: inv={chosen['inv']:+.1f} dB "
              f"(Δ={abs(chosen['inv']-target_inv):.2f} dB) → {chosen['path']}")
        return chosen["path"]

    fallback = base.find_fallback_best_model(RESULTS_DIR)
    if not fallback:
        raise FileNotFoundError(
            f"No SNR-tagged checkpoints in:\n  {CKPT_DIR}\n"
            f"and no 'best-model.pt' found under:\n  {RESULTS_DIR}"
        )
    print(f"[CKPT] No SNR-tagged ckpts found. Falling back to: {fallback}")
    return fallback

# Robustly infer a dtype from a model that may have no parameters
def _infer_model_dtype(model, default=torch.float32):
    try:
        it = model.parameters()
        p = next(it)
        return p.dtype
    except Exception:
        pass
    try:
        itb = model.buffers()
        b = next(itb)
        return b.dtype
    except Exception:
        pass
    return default

# ===================== Main =====================

def main():
    # ----- 0) Load checkpoint -----
    ckpt_path = _select_checkpoint()
    knet, device = base.load_knet_any(ckpt_path)
    print(f"[Device] Using: {device}")

    # If the state_dict loader produced a bare object without built layers,
    # prefer the full MODULE artifact if present, and allowlist safe globals.
    module_fallback = os.path.join(REPO_ROOT, "KNet", "ckpts", "best-core-module.pt")
    if (not hasattr(knet, "FC5")) and os.path.isfile(module_fallback):
        print(f"[CKPT] Loaded state_dict lacks built layers → switching to module: {module_fallback}")
        # Allow common nn classes in PyTorch 2.6 safe unpickler
        try:
            import torch.serialization as _ts
            if hasattr(_ts, "add_safe_globals"):
                _ts.add_safe_globals([
                    torch.nn.modules.container.Sequential,
                    torch.nn.Linear, torch.nn.ReLU, torch.nn.Sigmoid,
                    torch.nn.Tanh, torch.nn.GRU, torch.nn.ModuleList
                ])
        except Exception:
            pass
        # Force weights_only=False (trusted local artifact)
        try:
            knet = torch.load(module_fallback, map_location=device, weights_only=False)
        except TypeError:
            # Older torch without weights_only kw
            knet = torch.load(module_fallback, map_location=device)
        except Exception as e:
            print(f"[CKPT] Module fallback load failed: {e}")

    # === REQUIRED MINIMAL PATCH (no other changes) ===
    dev = device
    dtype = _infer_model_dtype(knet, torch.float32)

    # move module to device if possible
    try:
        knet.to(dev)
    except Exception:
        pass
    try:
        knet.eval()
    except Exception:
        pass

    # attach Python-only attributes the state_dict cannot carry
    if not hasattr(knet, "device"):
        knet.device = dev
    if not hasattr(knet, "h"):
        H = torch.tensor([[1.0, 0.0],
                          [0.0, 0.0]], dtype=dtype, device=dev)
        class _Obs(torch.nn.Module):
            def __init__(self, H):
                super().__init__()
                self.register_buffer("H", H)
            def forward(self, x):
                return self.H @ x
        knet.h = _Obs(H)
    if not hasattr(knet, "f"):
        F = torch.tensor(F_MATRIX, dtype=dtype, device=dev)
        class _Dyn(torch.nn.Module):
            def __init__(self, F):
                super().__init__()
                self.register_buffer("F", F)
            def forward(self, x):
                return self.F @ x
        knet.f = _Dyn(F)
    if not hasattr(knet, "seq_len_input"):
        knet.seq_len_input = 1
    if not hasattr(knet, "batch_size"):
        knet.batch_size = 1
    # === END PATCH ===

    # ----- 1) Load channel cube H -----
    H = _load_H()  # (N_Rx, M_Tx, T)
    N_Rx, M_Tx, T = H.shape
    print(f"[Data] Loaded H with shape (N_Rx, M_Tx, T) = {H.shape}")

    # Pre-allocate result tensor, same shape and dtype complex64
    h_Knet = np.zeros((N_Rx, M_Tx, T), dtype=np.complex64)
    h_Knet_pred = np.zeros((N_Rx, M_Tx, T), dtype=np.complex64)

    # ----- 2) Loop all links; compute NMSE per link on normalized scale -----
    nmse_linear_list = []
    nmse_linear_list_est=[]
    total_links = N_Rx * M_Tx
    link_count = 0

    for rx in range(N_Rx):
        for tx in range(M_Tx):
            link_count += 1
            c = H[rx, tx, :].reshape(T).astype(np.complex64)  # original scale

            # Normalize to unit average power
            c_norm, scale = base.power_normalize(c)

            # Add COMPLEX AWGN at requested SNR; then split AFTER noise
            y_complex, rho2 = base.add_complex_awgn(c_norm, snr_db=SNR_DB, seed=SEED)
            sigma2_comp = rho2 / 2.0  # variance of each real scalar component

            # Run KNet on real & imag (scalar duplicated → 2D) and combine
            y_real = y_complex.real.astype(np.float32)
            y_imag = y_complex.imag.astype(np.float32)

            s_hat_real, s_pred_real = base.run_knet_on_scalar_duplicated(
                y_scalar=y_real,
                knet=knet,
                sigma2_component=sigma2_comp,
                F_2x2=F_MATRIX,
                device=device,
                return_prediction=True,
            )

            # posterior + prediction for IMAG part
            s_hat_imag, s_pred_imag = base.run_knet_on_scalar_duplicated(
                y_scalar=y_imag,
                knet=knet,
                sigma2_component=sigma2_comp,
                F_2x2=F_MATRIX,
                device=device,
                return_prediction=True,
            )
            h_hat_norm = s_hat_real.astype(np.float32) + 1j*s_hat_imag.astype(np.float32)
            h_pred_norm = s_pred_real.astype(np.float32) + 1j * s_pred_imag.astype(np.float32)

            # NMSE on normalized complex input vs model output
            nmse_lin, nmse_db = base.nmse_complex(c_norm, h_pred_norm) # Change to h_hat_norm\h_pred_norm if NMSE to be measure on h_pred
            nmse_lin_est, nmse_db_est = base.nmse_complex(c_norm,h_hat_norm)  # Change to h_hat_norm\h_pred_norm if NMSE to be measure on h_pred
            nmse_linear_list.append(nmse_lin)
            nmse_linear_list_est.append(nmse_lin_est)
            # De-normalize back to original scale and store
            scale_safe = scale if scale != 0 else 1.0
            h_hat = (h_hat_norm / scale_safe).astype(np.complex64)
            h_pred = (h_pred_norm / scale_safe).astype(np.complex64)

            # estimation H_{t|t}
            h_Knet[rx, tx, :] = h_hat  # estimation H_{t|t}
            h_Knet_pred[rx, tx, :] = h_pred  # prediction H_{t|t-1} or H_{t+1|t}

            # Progress
            print(f"[{link_count:>2}/{total_links}] rx={rx}, tx={tx}  |  "
                  f"NMSE_link = {nmse_lin:.6e}  ({nmse_db:+.2f} dB)")

    # ----- 3) Average NMSE over all links (linear), print also in dB -----
    avg_nmse_linear = float(np.mean(nmse_linear_list)) if nmse_linear_list else float("nan")
    avg_nmse_linear_est = float(np.mean(nmse_linear_list_est)) if nmse_linear_list_est else float("nan")
    avg_nmse_db     = 10.0 * np.log10(max(avg_nmse_linear, 1e-30))
    avg_nmse_db_est = 10.0 * np.log10(max(avg_nmse_linear_est, 1e-30))
    print("\n================== SUMMARY ==================")
    print(f"SNR tested: {SNR_DB:.1f} dB")
    print(f"Average NMSE (linear) over all {total_links} links: {avg_nmse_linear:.6e}")
    print(f"Average NMSE (dB)     over all {total_links} links: {avg_nmse_db:+.2f} dB")
    print(f"Average NMSE Estimation (dB)     over all {total_links} links: {avg_nmse_db_est:+.2f} dB")
    print(f"Checkpoint used: {ckpt_path}")
    print("=============================================\n")

    # ----- 4) Save h_Knet (complex) to the requested folder -----
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    #savemat(SAVE_PATH, {"h_Knet": h_Knet})
    savemat(SAVE_PATH, {
        "h_Knet": h_Knet,  # existing estimation cube
        "h_Knet_pred": h_Knet_pred,  # new prediction cube
    })
    print(f"[SAVE] h_Knet shape {h_Knet.shape} saved to:\n  {SAVE_PATH}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
