import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import brentq

# New: PyTorch-based "Unrolled WSR" path (differentiable)
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # allow running the script without torch just for "My WSR" plots
    torch = None
    _TORCH_AVAILABLE = False


# -----------------------------
# WMMSE precoder (your version) → "My WSR" path
# -----------------------------
def wmmse_precoder_with_lambda_v1(H, Pt, sigma2, maxIter, W0=None):
    """Original NumPy WMMSE with λ root-finding (non‑diff, high‑quality).

    H : (K, M) complex (K users/Rx, M Tx)
    W : (M, K)
    """
    K, M = H.shape
    if W0 is not None:
        W = W0 * np.sqrt(Pt) / np.linalg.norm(W0, 'fro')
    else:
        W = (np.random.randn(M, K) + 1j * np.random.randn(M, K))
        W = W * np.sqrt(Pt) / np.linalg.norm(W, 'fro')

    eyeM = np.eye(M, dtype=complex)

    for _ in range(maxIter):
        # 1) RX combiners
        U = np.zeros(K, dtype=complex)
        for i in range(K):
            hi = H[i, :]
            C = np.sum(np.abs(hi @ W) ** 2) + sigma2
            U[i] = np.conj(hi @ W[:, i]) / C

        # 2) Weights
        V = np.zeros(K, dtype=float)
        for i in range(K):
            hi = H[i, :]
            e = (
                1
                - U[i] * (hi @ W[:, i])
                - np.conj(U[i] * (hi @ W[:, i]))
                + np.abs(U[i]) ** 2 * (np.sum(np.abs(hi @ W) ** 2) + sigma2)
            )
            V[i] = 1.0 / e.real

        # 3) Q, B
        Q = np.zeros((M, M), dtype=complex)
        for j in range(K):
            hj = H[j, :]
            Q += V[j] * np.abs(U[j]) ** 2 * np.outer(hj.conj(), hj)
        B = np.zeros((M, K), dtype=complex)
        for i in range(K):
            B[:, i] = np.conj(U[i]) * V[i] * H[i, :].conj()

        # 4) lambda via bracketing
        def power_err(lam):
            X = np.linalg.solve(Q + lam * eyeM, B)
            return np.linalg.norm(X, 'fro') ** 2 - Pt

        lam_low = 1e-6
        f_low = power_err(lam_low)
        if f_low <= 0:
            lam = lam_low
        else:
            lam_high, f_high = lam_low, f_low
            while f_high > 0:
                lam_high *= 10
                f_high = power_err(lam_high)
                if lam_high > 1e12:
                    raise RuntimeError("Unable to bracket lambda for WMMSE precoder")
            lam = brentq(power_err, lam_low, lam_high)

        # 5) Update
        W = np.linalg.solve(Q + lam * eyeM, B)

    return W  # (M, K)


# ========== Per-link normalization over time ==========
def normalize_links(H):
    """For each (n,m), scale H[n,m,:] by sqrt(T / sum_t |H[n,m,t]|^2), if sum>0.

    H: (N_Rx, N_Tx, T)
    """
    H = np.asarray(H)
    if H.ndim != 3:
        raise ValueError(f"Expected 3D channel (N_Rx, N_Tx, T). Got {H.shape}.")
    N_Rx, N_Tx, T = H.shape
    H = H.astype(np.complex128, copy=False)

    power = np.sum(np.abs(H) ** 2, axis=2)  # (N_Rx, N_Tx)
    scale = np.zeros_like(power, dtype=float)
    mask = power > 0
    scale[mask] = np.sqrt(T / power[mask])
    return H * scale[:, :, None]


# ========== From H_curr (estimate) → W (precoder) : "My WSR" path ==========
def get_precoder(H_curr, snr_db, Pt=1.0, maxIter=10, use_svd_init=True):
    """Build precoder using your (NumPy) WMMSE implementation.

    H_curr : (N_Rx, N_Tx) complex (normalized snapshot)
    """
    H_curr = np.asarray(H_curr, dtype=np.complex128)
    N_Rx, N_Tx = H_curr.shape

    SNR_lin = 10.0 ** (snr_db / 10.0)
    rho2 = 1.0 / SNR_lin  # with Pt=1.0
    sigma2 = rho2

    W0 = None
    if use_svd_init:
        U, S, Vh = np.linalg.svd(H_curr, full_matrices=False)
        K = min(N_Rx, N_Tx)
        V = Vh.conj().T[:, :K]  # (N_Tx, K)
        W0 = V

    W = wmmse_precoder_with_lambda_v1(H_curr, Pt=Pt, sigma2=sigma2, maxIter=maxIter, W0=W0)
    return W


# ========== From known H and W → two rates for that snapshot ("My WSR") ==========
def WSR_Calculation(H_known, W, snr_db):
    """Your existing WSR calculation (not suitable for gradients).

    H_known : (N_Rx, N_Tx) complex  (normalized snapshot for rate)
    W       : (N_Tx, K) complex

    Returns: (rate_det, rate_sinr)
    """
    H_known = np.asarray(H_known, dtype=np.complex128)
    W = np.asarray(W, dtype=np.complex128)
    N_Rx, N_Tx = H_known.shape
    assert W.shape[0] == N_Tx, "W dims must be (N_Tx, K)."

    SNR_lin = 10.0 ** (snr_db / 10.0)
    rho2 = 1.0 / SNR_lin

    # (1) det-based: log2 det( I + (SNR_lin/N_Rx) * H W W^H H^H )
    A = H_known @ W @ W.conj().T @ H_known.conj().T
    I = np.eye(N_Rx, dtype=complex)
    B = I + (SNR_lin / N_Rx) * A
    sign, logdet = np.linalg.slogdet(B)
    if sign <= 0:
        eigvals = np.linalg.eigvalsh(B)
        if np.any(eigvals <= 0):
            eps = 1e-12 - np.min(eigvals.real) + 1e-12
            B = B + eps * I
            sign, logdet = np.linalg.slogdet(B)
    rate_det = float(logdet / np.log(2.0))

    # (2) per-stream SINR aggregate: sum_i log2(1 + SINR_i / N_Rx)
    M = H_known @ W
    rate_terms = []
    for i in range(N_Rx):
        sig = np.abs(M[i, i]) ** 2 if i < M.shape[1] else 0.0
        tot = np.sum(np.abs(M[i, :]) ** 2)
        interf = tot - sig
        sinr_i = sig / (interf + rho2)
        rate_terms.append(np.log2(1.0 + sinr_i / N_Rx))
    rate_sinr = float(np.sum(rate_terms))

    return rate_det, rate_sinr


# ============================================================================
# New: PyTorch-based "Unrolled WSR" (SINR‑only) — using SAME unrolled WMMSE as Pipeline_EKF
# ============================================================================

if _TORCH_AVAILABLE:

    # ---------- helpers copied/adapted from Pipeline_EKF ----------
    def _svd_init_torch(H: torch.Tensor, Pt: float) -> torch.Tensor:
        """SVD-based precoder init with total power Pt (differentiable).

        H: (Nr,Nt) complex → W0: (Nt,d) complex with d = min(Nr,Nt)
        """
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        V = torch.conj(Vh).transpose(-2, -1)
        d = min(H.shape[0], H.shape[1])
        W0 = V[:, :d]
        p = torch.real(torch.trace(W0.conj().T @ W0))
        scale = torch.sqrt(
            torch.tensor(Pt, dtype=W0.real.dtype, device=W0.device) / (p + 1e-12)
        )
        return W0 * scale


    def _solve_precoder_bisection(A: torch.Tensor, B: torch.Tensor, Pt: float, max_iter: int = 30) -> torch.Tensor:
        """Solve (A + λI)W=B with λ s.t. tr(WW^H) ≤ Pt via bisection (differentiable).

        This is the exact same construction as used inside Pipeline_EKF._solve_precoder_bisection.
        """
        Nt = A.shape[0]
        I = torch.eye(Nt, dtype=A.dtype, device=A.device)

        def W_of(lmbd: torch.Tensor) -> torch.Tensor:
            M = A + lmbd * I
            return torch.linalg.solve(M, B)

        # grow λ_hi until power ≤ Pt
        lmbd_lo = torch.tensor(0.0, dtype=A.dtype, device=A.device)
        lmbd_hi = torch.tensor(1.0, dtype=A.dtype, device=A.device)
        for _ in range(30):
            W = W_of(lmbd_hi)
            p = torch.real(torch.trace(W.conj().T @ W))
            if p <= Pt:
                break
            lmbd_hi = lmbd_hi * 2.0

        for _ in range(max_iter):
            lmbd_mid = 0.5 * (lmbd_lo + lmbd_hi)
            W = W_of(lmbd_mid)
            p = torch.real(torch.trace(W.conj().T @ W))
            if p > Pt:
                lmbd_lo = lmbd_mid
            else:
                lmbd_hi = lmbd_mid

        return W_of(lmbd_hi)


    def _wmmse_unrolled_core(H: torch.Tensor, Pt: float, sigma2: float, iters: int) -> torch.Tensor:
        """Unrolled WMMSE as in Pipeline_EKF._wmmse_unrolled_torch.

        H : (Nr,Nt) complex; returns W : (Nt,d) complex.
        """
        # ensure complex double
        if not torch.is_complex(H):
            H = H.to(torch.cdouble)
        Nr, Nt = H.shape
        W = _svd_init_torch(H, Pt)  # (Nt,d)
        d = W.shape[1]

        I_r = torch.eye(Nr, dtype=H.dtype, device=H.device)
        I_d = torch.eye(d,  dtype=H.dtype, device=H.device)

        for _ in range(iters):
            # U step
            S = H @ W @ W.conj().T @ H.conj().T + sigma2 * I_r
            U = torch.linalg.solve(S, H @ W)  # (Nr,d)

            # Error & MMSE weights
            E = I_d - (U.conj().T @ H @ W)
            Wmmse = torch.linalg.inv(E.conj().T @ E + 1e-9 * I_d)

            # Precoder step with λ bisection
            A = H.conj().T @ U @ Wmmse @ U.conj().T @ H
            B = H.conj().T @ U @ Wmmse
            W = _solve_precoder_bisection(A, B, Pt)

        return W


    def _wsr_sinr_torch(H: torch.Tensor, W: torch.Tensor, sigma2: float) -> torch.Tensor:
        """Differentiable SINR-based sum-rate, identical to Pipeline_EKF._wsr_sinr_torch.

        H: (Nr,Nt) complex, W: (Nt,d) complex.
        We interpret d data streams; effective users = min(Nr,d).
        """
        if not torch.is_complex(H):
            H = H.to(torch.cdouble)
        if not torch.is_complex(W):
            W = W.to(torch.cdouble)

        Nr, Nt = H.shape
        d = W.shape[1]

        G = H @ W  # (Nr,d)
        d_eff = min(Nr, d)

        diag_pow = torch.abs(torch.diagonal(G, offset=0, dim1=-2, dim2=-1)) ** 2  # (d_eff,)
        tot_pow = torch.sum(torch.abs(G) ** 2, dim=1)  # (Nr,)

        sig = torch.zeros(Nr, dtype=G.dtype, device=G.device)
        sig[:d_eff] = diag_pow[:d_eff]
        inter = tot_pow - sig
        sinr = sig / (inter + sigma2)

        # Sum_i log2(1 + SINR_i)
        rate = torch.sum(torch.log2(1.0 + sinr.real))
        return rate


    # ---------- public API for this script ----------
    def wmmse_precoder_unrolled_torch(H, snr_db, Pt=1.0, maxIter=10):
        """Wrapper: from H and snr_db to unrolled WMMSE precoder.

        H : (Nr,Nt) complex tensor
        Returns W : (Nt,d) complex tensor.
        """
        if not torch.is_complex(H):
            H = H.to(torch.cdouble)
        SNR_lin = 10.0 ** (snr_db / 10.0)
        sigma2 = 1.0 / SNR_lin
        return _wmmse_unrolled_core(H, Pt, sigma2, maxIter)


    def unrolled_wsr_sinr_torch(H_eval, H_pred, snr_db, Pt=1.0, maxIter=10):
        """Unrolled WSR based on SINR only, using SAME logic as Pipeline_EKF.

        H_eval : (Nr,Nt) complex tensor — channel for rate evaluation (e.g., H_true)
        H_pred : (Nr,Nt) complex tensor — channel for precoder design (e.g., H_pred)
        snr_db : float
        Pt     : float
        maxIter: int (# of WMMSE iterations)
        """
        if not torch.is_complex(H_eval):
            H_eval = H_eval.to(torch.cdouble)
        if not torch.is_complex(H_pred):
            H_pred = H_pred.to(torch.cdouble)

        SNR_lin = 10.0 ** (snr_db / 10.0)
        sigma2 = 1.0 / SNR_lin

        # Build precoder from H_pred with unrolled WMMSE
        W_pred = _wmmse_unrolled_core(H_pred, Pt, sigma2, maxIter)

        # Rate on H_eval via SINR only (no log-det)
        rate_t = _wsr_sinr_torch(H_eval, W_pred, sigma2)
        return rate_t.real


    def unrolled_wsr_sinr_numpy(H_eval_np, H_pred_np, snr_db, Pt=1.0, maxIter=10, device="cpu"):
        """Helper wrapper for this script: NumPy → Torch → scalar.

        This is what the plotting code calls to generate the "Unrolled WSR" curve.
        For training, call `unrolled_wsr_sinr_torch` directly to keep gradients.
        """
        H_eval_t = torch.as_tensor(H_eval_np)
        H_pred_t = torch.as_tensor(H_pred_np)
        if not torch.is_complex(H_eval_t):
            H_eval_t = H_eval_t.to(torch.cdouble)
        if not torch.is_complex(H_pred_t):
            H_pred_t = H_pred_t.to(torch.cdouble)
        H_eval_t = H_eval_t.to(device)
        H_pred_t = H_pred_t.to(device)

        wsr_sinr_t = unrolled_wsr_sinr_torch(
            H_eval_t,
            H_pred_t,
            snr_db=snr_db,
            Pt=Pt,
            maxIter=maxIter,
        )
        return float(wsr_sinr_t.detach().cpu().numpy())


# ====================================================================================
# Main driver: load BOTH cubes, normalize, loop, plot side-by-side
#   - "My WSR" (det + SINR)           → NumPy implementation (as before)
#   - "Unrolled WSR" (SINR only)     → Torch implementation using Pipeline_EKF unrolled WMMSE
# ============================================================================

def process_channel_and_plot(
    mat_path_known,  # .mat file for the known/original H used for rate
    mat_path_curr,   # .mat file for H_curr (your KNet estimate)
    var_name_known="H",  # variable name for known H (auto-detect if missing)
    var_name_curr=None,   # variable name for H_curr (auto-detect if None)
    snr_db=15.0,
    Pt=1.0,
    maxIter=10,
    show=True,
    save_path=None,
    torch_device="cpu",  # where to run the unrolled path
):
    """For each time index ii:
      - H_known_ii = H_known_norm[:,:,ii]
      - H_curr_ii  = H_curr_norm[:,:,ii]  (loaded from your specified file)
      - W          = get_precoder(H_curr_ii, ...)                [My WSR]
      - rates from WSR_Calculation(H_known_ii, W, ...)           [My WSR]
      - unrolled WSR via Torch, SINR-only, same unrolled WMMSE as Pipeline_EKF.
    """

    # ----- Load known/original channel -----
    md_known = loadmat(mat_path_known)
    if var_name_known in md_known:
        H_known = md_known[var_name_known]
    else:
        H_known = None
        for k, v in md_known.items():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                H_known = v
                var_name_known = k
                break
        if H_known is None:
            raise ValueError(f"No 3D array found for known H in {mat_path_known}")

    # ----- Load H_curr (precoder input) -----
    md_curr = loadmat(mat_path_curr)
    if var_name_curr and var_name_curr in md_curr:
        H_curr_cube = md_curr[var_name_curr]
    else:
        H_curr_cube = None
        for k, v in md_curr.items():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                H_curr_cube = v
                var_name_curr = k
                break
        if H_curr_cube is None:
            raise ValueError(f"No 3D array found for H_curr in {mat_path_curr}")

    # Sanity: shapes must match (N_Rx, N_Tx, T)
    H_known = np.asarray(H_known, dtype=np.complex128)
    H_curr_cube = np.asarray(H_curr_cube, dtype=np.complex128)
    if H_known.shape != H_curr_cube.shape:
        raise ValueError(f"Shape mismatch: known {H_known.shape} vs H_curr {H_curr_cube.shape}")

    # ---- Normalize BOTH cubes per-link over time BEFORE any usage ----
    H_known_norm = normalize_links(H_known)
    H_curr_norm  = normalize_links(H_curr_cube)

    N_Rx, N_Tx, T = H_known_norm.shape
    rates_det = np.zeros(T, dtype=float)
    rates_sinr = np.zeros(T, dtype=float)
    rates_unrolled_sinr = np.zeros(T, dtype=float) if _TORCH_AVAILABLE else None

    for ii in range(T):
        H_known_ii = H_known_norm[:, :, ii]
        H_curr_ii  = H_curr_norm[:, :, ii]  # used to build W (estimate)

        # My WSR: build W from H_curr via NumPy WMMSE
        W = get_precoder(H_curr_ii, snr_db=snr_db, Pt=Pt, maxIter=maxIter, use_svd_init=True)

        # Rates on the known/original channel (My WSR)
        r1, r2 = WSR_Calculation(H_known_ii, W, snr_db=snr_db)
        rates_det[ii]  = r1
        rates_sinr[ii] = r2

        # Unrolled WSR — SINR only, using Pipeline_EKF unrolled WMMSE
        if _TORCH_AVAILABLE:
            rates_unrolled_sinr[ii] = unrolled_wsr_sinr_numpy(
                H_eval_np=H_known_ii,
                H_pred_np=H_curr_ii,
                snr_db=snr_db,
                Pt=Pt,
                maxIter=maxIter,
                device=torch_device,
            )

    mean_det = float(np.mean(rates_det))
    mean_sinr = float(np.mean(rates_sinr))
    mean_unrolled_sinr = float(np.mean(rates_unrolled_sinr)) if _TORCH_AVAILABLE else None

    # ---- Plot: side-by-side My WSR vs Unrolled WSR ----
    t = np.arange(T)

    if _TORCH_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

        # Left: My WSR (as-is: det + SINR)
        ax0 = axes[0]
        ax0.plot(t, rates_det, label="My WSR (det)")
        ax0.plot(t, rates_sinr, label="My WSR (SINR)", linestyle="--")
        ax0.set_xlabel("Time sample")
        ax0.set_ylabel("Rate [bits/use]")
        ax0.set_title(
            "My WSR (NumPy / non-diff)\n"
            f"mean det={mean_det:.3f}, mean SINR={mean_sinr:.3f}"
        )
        ax0.grid(True, alpha=0.3)
        ax0.legend()

        # Right: Unrolled WSR (SINR only)
        ax1 = axes[1]
        ax1.plot(t, rates_unrolled_sinr, label="Unrolled WSR (SINR only)")
        ax1.set_xlabel("Time sample")
        ax1.set_title(
            "Unrolled WSR (PyTorch / diff, SINR only)\n"
            f"mean={mean_unrolled_sinr:.3f}"
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        fig.suptitle(
            "Rate Comparison (per-link normalized)\n"
            f"H_known='{var_name_known}' from {mat_path_known}\n"
            f"H_curr='{var_name_curr}' from {mat_path_curr}\n"
            f"SNR={snr_db:.1f} dB, Pt={Pt}"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    else:
        # Fallback: only My WSR curves if torch is not available
        plt.figure(figsize=(9, 4.5))
        plt.plot(t, rates_det, label="My WSR (det)")
        plt.plot(t, rates_sinr, label="My WSR (SINR)", linestyle="--")
        plt.xlabel("Time sample")
        plt.ylabel("Rate [bits/use]")
        plt.title(
            "My WSR only (PyTorch not available)\n"
            f"H_known='{var_name_known}' from {mat_path_known}\n"
            f"H_curr='{var_name_curr}' from {mat_path_curr}\n"
            f"SNR={snr_db:.1f} dB, Pt={Pt} | means: det={mean_det:.3f}, SINR={mean_sinr:.3f}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    print(f"[INFO] Known H   : {mat_path_known} (var '{var_name_known}') shape={H_known.shape}")
    print(f"[INFO] H_curr (W): {mat_path_curr}   (var '{var_name_curr}')  shape={H_curr_cube.shape}")
    print(f"[RESULT] My WSR   (det) : {mean_det:.4f} bits/use")
    print(f"[RESULT] My WSR   (SINR): {mean_sinr:.4f} bits/use")
    if _TORCH_AVAILABLE:
        print(f"[RESULT] Unrolled WSR (SINR only, Pipeline-style): {mean_unrolled_sinr:.4f} bits/use")
    else:
        print("[WARN] PyTorch not available → Unrolled WSR not computed.")

    return rates_det, rates_sinr, rates_unrolled_sinr, mean_det, mean_sinr, mean_unrolled_sinr


# ----------------
# Example usage
# ----------------
if __name__ == "__main__":
    # Known/original channel (for rate):
    MAT_KNOWN = r"C:\\Users\\shalomw\\Documents\\Education\\Project ideas\\Code\\KalmanNet_TSP\\Simulations\\Linear_canonical\\data\\Channel_H_2x2_v1.mat"  # <-- adjust
    VAR_KNOWN = "H"  # change if your variable has a different name

    # H_curr (precoder input) from your path:
    MAT_CURR = r"C:\\Users\\shalomw\\Documents\\Education\\Project ideas\\Code\\KalmanNet_TSP\\Simulations\\Linear_canonical\\data\\Channel_H_2x2_v1.mat"  # or e.g. your KNet .mat
    VAR_CURR = None  # let the loader auto-detect the 3D array (or set to exact var name)

    SNR_DB = 20.0  # as in your filename
    PT = 1.0
    MAXITER = 10   # match Pipeline_EKF args.wmmse_iters when comparing

    process_channel_and_plot(
        mat_path_known=MAT_KNOWN,
        mat_path_curr=MAT_CURR,
        var_name_known=VAR_KNOWN,
        var_name_curr=VAR_CURR,
        snr_db=SNR_DB,
        Pt=PT,
        maxIter=MAXITER,
        show=True,
        save_path=None,  # e.g., r"C:\\...\\rates_vs_time_knet.png"
        torch_device="cpu",  # set to "cuda" if you want and have a GPU
    )
