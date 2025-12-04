# src/sigbandit/tests/check_diversity.py
import numpy as np
import matplotlib.pyplot as plt
import os

from sigbandit.algos.ucb import sigUCB
from sigbandit.envs.context import bm, ou, ou_stationary, gbm
from sigbandit.utils.signature import sig_length
from tqdm import tqdm

# ------------------------- config -------------------------
T = 100                          # rounds per run
raw_d = 1                        # raw channels (before time-augment)
N_total = 100 * 1000             # total points in full series
window = N_total // T            # points per round

depth_list = [1, 2, 3, 4]        # signature depths to compare
sig_type = "logsig"              # 'sig' or 'logsig'

lam = 1.0                        # not used in Σ̂_t, but kept for reference
gamma = 10.0                     # irrelevant here, but needed to build sigUCB
fit_intercept = False

n_runs = 100
seeds = np.arange(n_runs) + 123

channels_after_time = raw_d + 1

def run_diversity_for_depth(
    sig_depth: int,
    process: str,
) -> tuple[np.ndarray, int]:
    """
    For a given signature depth, run n_runs trials and compute
    λ_min(Σ̂_t) for t=1..T.

    Returns
    -------
    lm_min_depth : np.ndarray
        Array of shape (n_runs, T) with λ_min(Σ̂_t) per run and round.
    feat_dim : int
        Dimension of the augmented feature vector bar{x}_t.
    """
    # pick process function
    if process == "bm":
        proc_fn = bm
        log_norm = False
    elif process == "ou":
        proc_fn = ou
        log_norm = False
    elif process == "ou_s":
        proc_fn = ou_stationary
        log_norm = False
    elif process == "gbm":
        proc_fn = gbm
        log_norm = True
    elif process == "unit_ball":
        proc_fn = unit_ball
        log_norm = False
    else:
        raise ValueError(f"Unknown process '{process}'")

    feat_dim = raw_d + sig_length(channels_after_time, sig_depth, sig_type)
    print(f"[depth N={sig_depth}] Feature dimension (bar x_t): {feat_dim}")

    lm_min_depth = np.zeros((n_runs, T))

    for j, s in enumerate(tqdm(seeds, desc=f"N={sig_depth}")):
        seed = int(s)
        np.random.seed(seed)

        # sample path X
        X = proc_fn(T, N_total)

        # use sigUCB object for sig_feat
        sig_ucb = sigUCB(
            n_actions=1,
            lm=lam,
            gamma=gamma,
            N=sig_depth,
            d=channels_after_time,
            sigType=sig_type,
            log_norm=log_norm,
            fit_intercept=fit_intercept,
        )

        # initialize Sigma
        Sigma = np.zeros((feat_dim, feat_dim))

        for t in range(T):
            # get signature feature
            ctx = X[t * window : (t + 1) * window, :]   # (window, raw_d)
            feat = sig_ucb.sig_feat(ctx)                # bar{x}_t ∈ R^{feat_dim}

            # update Sigma
            Sigma += np.outer(feat, feat)
            Sigma_hat = Sigma / float(t + 1)

            # compute eigs
            eigs = np.linalg.eigvalsh(Sigma_hat)
            lm_min_depth[j, t] = eigs[0]

    return lm_min_depth, feat_dim


def main(process):
    n_depths = len(depth_list)
    lm_min_all_depths = []
    feat_dims = []

    # run for each depth
    for sig_depth in depth_list:
        lm_min_depth, feat_dim = run_diversity_for_depth(sig_depth, process)
        lm_min_all_depths.append(lm_min_depth)
        feat_dims.append(feat_dim)

    # stack into array (n_depths, n_runs, T)
    lam_min_all = np.stack(lm_min_all_depths, axis=0)

    # summary stats across runs, per depth and t
    lam_min_med = np.median(lam_min_all, axis=1)
    lam_min_p25 = np.percentile(lam_min_all, 25, axis=1)
    lam_min_p75 = np.percentile(lam_min_all, 75, axis=1)

    # ---------- plot ----------
    x = np.arange(1, T + 1)
    plt.figure(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, n_depths))

    for idx, sig_depth in enumerate(depth_list):
        med = lam_min_med[idx]
        p25 = lam_min_p25[idx]
        p75 = lam_min_p75[idx]

        plt.plot(
            x, med,
            color=colors[idx],
            label=f"N={sig_depth}",
        )
        plt.fill_between(
            x, p25, p75,
            color=colors[idx],
            alpha=0.2,
        )

    plt.xlabel("Round t")
    plt.ylabel(r"$\lambda_{\min}(\widehat{\Sigma}_t)$")
    plt.title("Smallest Eigenvalue of Signature Feature Cov Matrix")

    ax = plt.gca()
    ax.set_yscale("log")

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_dir = "experiments/figs/diversity_assumption"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"sig_diversity_{process}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=180)
    print(f"Saved figure to {out_path}")

    plt.show()

    print("Global min λ_min(Σ̂_t):", lam_min_all.min())
    print("Global max λ_min(Σ̂_t):", lam_min_all.max())
    print("Feature dims per depth:", dict(zip(depth_list, feat_dims)))


if __name__ == "__main__":
    main(process='ou_s')