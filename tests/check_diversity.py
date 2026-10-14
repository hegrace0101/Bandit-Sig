# src/sigbandit/tests/check_diversity.py
import numpy as np
import matplotlib.pyplot as plt
import os

from sigbandit.algos.ucb import sigUCB
from sigbandit.envs.context import bm, ou_stationary, gbm
from sigbandit.utils.signature import sig_length
from tqdm import tqdm

import iisignature as iis

def sig_list_levels(t_X: np.ndarray, N: int) -> list[np.ndarray]:
    """
    Returns [level1, level2, ..., levelN] where levelk has shape (d^k,).
    Uses iisignature's format=1 API.
    """
    return iis.sig(t_X, N, 1)

def prune_sig_levels_drop_last_channel(levels: list[np.ndarray], d: int, time_index: int = 0) -> np.ndarray:
    """
    Prune all signature coordinates whose word ends in the time channel.

    Convention here:
      - time_augment inserts time as channel 0 (first column),
      - so time_index = 0 in 0-based indexing.
    """
    kept = []
    for k, lev in enumerate(levels, start=1):
        T = lev.reshape((d,) * k)                 # tensor indexed by i1,...,ik
        idx = [i for i in range(d) if i != time_index]
        T_keep = np.take(T, idx, axis=-1)         # drop last-letter == time
        kept.append(T_keep.reshape(-1))
    return np.concatenate(kept, axis=0)

# ------------------------- config -------------------------
T = 100
raw_d = 1
N_total = T * 1000
window = N_total // T

depth_list = [1, 2, 3, 4]
sig_type = "sig"  # 'sig' or 'logsig'

lam = 1.0
gamma = 1.0
fit_intercept = False

n_runs = 100
seeds = np.arange(n_runs) + 123

channels_after_time = raw_d + 1  # time + spatial

# bm params
bm_rho = 0.7
idx = np.arange(raw_d)
SIGMA = bm_rho ** np.abs(idx[:, None] - idx[None, :])

def run_diversity_for_depth(
    sig_depth: int,
    process: str,
    gbm_mu: float = 0.1,
    gbm_sigma_scale: float = 1.0,
    gbm_x0_scale: float = 0.0,   # if > 0, x0 ~ Uniform[1, 1 + gbm_x0_scale] per seed
) -> tuple[np.ndarray, int]:
    # pick process function
    if process == "bm":
        proc_fn = bm
        log_norm = False
    elif process == "ou_s":
        proc_fn = ou_stationary
        log_norm = False
    elif process == "gbm":
        proc_fn = gbm
        log_norm = True
    else:
        raise ValueError(f"Unknown process '{process}'")

    d = channels_after_time
    sig_dim_pruned = sum((d ** (k - 1)) * (d - 1) for k in range(1, sig_depth + 1))
    feat_dim = sig_dim_pruned + raw_d

    print(f"[depth N={sig_depth}] | pruned sig dim: {sig_dim_pruned} | feat_dim: {feat_dim} | "
          f"gbm(mu={gbm_mu}, sigma_scale={gbm_sigma_scale}, x0_scale={gbm_x0_scale})")

    lm_min_depth = np.zeros((n_runs, T))

    for j, s in enumerate(tqdm(seeds, desc=f"N={sig_depth}")):
        seed = int(s)
        np.random.seed(seed)

        if process == "gbm":
            Sigma_local = (gbm_sigma_scale ** 2) * SIGMA  # variance scales by sigma^2
            if gbm_x0_scale > 0:
                x0_val = 1.0 + gbm_x0_scale * np.random.rand()
            else:
                x0_val = 1.0
            X = proc_fn(T, N_total, x0=x0_val, mu=gbm_mu, Sigma=Sigma_local)
        else:
            X = proc_fn(T, N_total, Sigma=SIGMA)

        Sigma = np.zeros((feat_dim, feat_dim))

        for t in range(T):
            ctx = X[t * window : (t + 1) * window, :]   # (window, raw_d)

            # Preserve RAW initial and terminal values before any log-norm
            # (sigUCB concatenates the raw terminal xT with the signature).
            x0_raw = ctx[0].ravel()
            xT_raw = ctx[-1].ravel()

            # Match sigUCB's log_norm=True: for GBM the algorithm computes
            # signatures on log(context / context[0]), which is Brownian-motion-like
            # and much better conditioned than signatures of the raw exponential path.
            if log_norm:
                ctx = np.log(ctx / ctx[0])

            # build time-augmented path
            tau = np.linspace(0, 1, num=ctx.shape[0], dtype=float)
            t_X = np.insert(ctx, 0, tau, axis=1)
            d = t_X.shape[1]

            # signature levels and pruning
            levels = sig_list_levels(t_X, sig_depth)     # list of level arrays
            sig_pruned = prune_sig_levels_drop_last_channel(levels, d=d, time_index=0)

            # Extra state feature. sigUCB uses xT (raw terminal); use x0 for
            # non-log-normed processes to preserve prior behavior.
            extra = xT_raw if log_norm else x0_raw

            feat = np.concatenate([sig_pruned, extra])
            assert feat.shape[0] == feat_dim, (feat.shape[0], feat_dim)

            Sigma += np.outer(feat, feat)
            Sigma_hat = Sigma / float(t + 1)

            eigs = np.linalg.eigvalsh(Sigma_hat)
            lm_min_depth[j, t] = eigs[0]

    return lm_min_depth, feat_dim


def main(process, gbm_mu=0.1, gbm_sigma_scale=1.0, gbm_x0_scale=0.0, tag=None):
    lm_min_all_depths = []
    feat_dims = []

    for sig_depth in depth_list:
        lm_min_depth, feat_dim = run_diversity_for_depth(
            sig_depth, process,
            gbm_mu=gbm_mu, gbm_sigma_scale=gbm_sigma_scale, gbm_x0_scale=gbm_x0_scale,
        )
        lm_min_all_depths.append(lm_min_depth)
        feat_dims.append(feat_dim)

    lam_min_all = np.stack(lm_min_all_depths, axis=0)

    lam_min_med = np.median(lam_min_all, axis=1)
    lam_min_p25 = np.percentile(lam_min_all, 25, axis=1)
    lam_min_p75 = np.percentile(lam_min_all, 75, axis=1)

    x = np.arange(1, T + 1)
    plt.figure(figsize=(8, 5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    linestyles = ["-", "--", "-.", ":"]

    for idx, sig_depth in enumerate(depth_list):
        med = lam_min_med[idx]
        p25 = lam_min_p25[idx]
        p75 = lam_min_p75[idx]

        ls = linestyles[idx % len(linestyles)]
        plt.plot(x, med, color=colors[idx], linestyle=ls, label=f"N={sig_depth}")
        plt.fill_between(x, p25, p75, color=colors[idx], alpha=0.2)

    plt.xlabel("Round t")
    plt.ylabel(r"$\lambda_{\min}(\widehat{\Sigma}_t)$")
    plt.title(f"Smallest Eigenvalue of Pruned Signature Feature Cov Matrix ({process})")

    ax = plt.gca()
    ax.set_yscale("log")

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_dir = "experiments/figs/diversity_assumption"
    os.makedirs(out_dir, exist_ok=True)
    tag_str = f"_{tag}" if tag else ""
    fname = f"sig_diversity_pruned_end_time_{process}{tag_str}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=180)
    print(f"Saved figure to {out_path}")
    plt.close()

    print("Global min λ_min(Σ̂_t):", lam_min_all.min())
    print("Global max λ_min(Σ̂_t):", lam_min_all.max())
    print("Feature dims per depth:", dict(zip(depth_list, feat_dims)))

if __name__ == "__main__":
    main(process='gbm')
