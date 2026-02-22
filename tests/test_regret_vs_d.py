# src/sigbandit/tests/check_sig_regret_vs_d.py
import numpy as np
import matplotlib.pyplot as plt
import os

from sigbandit.algos.ucb import sigUCB
from sigbandit.envs.bandit import Bandit
from sigbandit.utils.signature import sig_length
from sigbandit.envs.context import ContextualProcess, bm, ou, ou_stationary, gbm
from sigbandit.envs.reward import (
    min_max_reward,
    sum_a_reward,
    linear_reward,
    min_max_mix_reward,
)
from tqdm import tqdm

# ------------------------- config -------------------------
T = 100                          # rounds per run
K = 2                            # actions
N_total = 100 * 1000             # total points in full series
window = N_total // T            # points per round

lam = 1.0                        # ridge λ
gamma = 2.0                      # exploration γ
fit_intercept = False            # append constant 1.0 to features
sigma = 0.1                      # standard deviation of reward noise

n_runs = 100                     # number of independent runs
seeds = np.arange(n_runs) + 123  # reproducible seeds

# K values to sweep over
d_list = [1, 2, 3, 4]

# signature depths to sweep over
depth_list = [1, 2, 3]

# available processes
PROCESS_MAP = {
    "bm": bm,
    "ou": ou,
    "ou_s": ou_stationary,
    "gbm": gbm
}

# available reward functions
REWARD_MAP = {
    "min_max": min_max_reward,
    "sum_a":   sum_a_reward,
    "linear":  linear_reward,
}


def run_one(
    seed: int,
    K: int,
    d: int,
    process_fn,
    reward_fn,
    sig_type: str,
    sig_depth: int,
) -> np.ndarray:
    """
    Run one bandit experiment for a given seed, K, raw dimension, process,
    reward, and signature config.
    Returns sig_regrets of length T+1.
    """
    np.random.seed(seed)

    # simulate a d-dimensional process
    # (adjust call signature if your bm/ou/gbm already take dim as an arg)
    X = process_fn(T, N_total, d=d)

    log_norm = (process_fn == gbm)

    # context process: windows of length `window`
    context = ContextualProcess(X, T, window)

    # bandit with reward function
    bandit = Bandit(context, reward_fn, sigma, K, T)

    # channels seen by the signature after time-augmentation
    channels_after_time = d + 1

    sig_ucb = sigUCB(
        n_actions=K,
        lm=lam,
        gamma=gamma,
        N=sig_depth,
        d=channels_after_time,   # includes time channel
        sigType=sig_type,
        log_norm=log_norm,
        fit_intercept=fit_intercept,
    )

    sig_regrets = np.zeros(T + 1, dtype=float)
    t = 0

    while bandit.next_round():
        ctx = bandit.observe_context()

        oracle_r, _ = bandit.oracle()

        sig_a = sig_ucb.select(ctx)
        sig_r = bandit.play(sig_a)
        sig_ucb.update(ctx, sig_a, sig_r)

        t += 1
        sig_regrets[t] = sig_regrets[t - 1] + (oracle_r - sig_r)

    return sig_regrets


def main(
    process: str = "ou_s",
    reward_name: str = "min_max",
    sig_type: str = "logsig",
) -> None:
    if process not in PROCESS_MAP:
        raise ValueError(
            f"Unknown process '{process}'. Choose from {list(PROCESS_MAP.keys())}."
        )
    if reward_name not in REWARD_MAP:
        raise ValueError(
            f"Unknown reward '{reward_name}'. Choose from {list(REWARD_MAP.keys())}."
        )

    process_fn = PROCESS_MAP[process]

    final_med = {}   # depth -> array over d_list
    final_p25 = {}
    final_p75 = {}

    for sig_depth in depth_list:
        print(f"\n===== Depth N = {sig_depth} =====")
        med_list = []
        p25_list = []
        p75_list = []

        for d in d_list:
            print(f"\n--- Running sigUCB for d = {d} ---")

            if reward_name == "linear":
                # context is window x d; flatten for linear reward
                ctx_dim = window * d
                rng_beta = np.random.RandomState(0)  # fixed across dims/depths
                beta_star = rng_beta.uniform(low=-1.0, high=1.0, size=(K, ctx_dim))

                def reward_fn(action: int, ctx: np.ndarray, beta_star=beta_star) -> float:
                    ctx_flat = ctx.reshape(-1)
                    return linear_reward(action, ctx_flat, beta_star)
            else:
                reward_fn = REWARD_MAP[reward_name]

            runs = [
                run_one(
                    int(s),
                    K,
                    d,
                    process_fn,
                    reward_fn,
                    sig_type,
                    sig_depth,
                )
                for s in tqdm(
                    seeds,
                    desc=f"N={sig_depth}, d={d} ({process}, {reward_name}, {sig_type})",
                )
            ]

            all_sig = np.stack(runs, axis=0)  # (n_runs, T+1)
            final_vals = all_sig[:, -1]

            med = np.median(final_vals)
            p25 = np.percentile(final_vals, 25)
            p75 = np.percentile(final_vals, 75)

            med_list.append(med)
            p25_list.append(p25)
            p75_list.append(p75)

            print(f"N={sig_depth}, d={d} | Final median sigUCB regret: {med:.3f}")

        final_med[sig_depth] = np.array(med_list)
        final_p25[sig_depth] = np.array(p25_list)
        final_p75[sig_depth] = np.array(p75_list)

	# -------- plot: regret at time T vs (d + m), multiple depths ----------
    plt.figure(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(depth_list)))

    for i, sig_depth in enumerate(depth_list):
        med = final_med[sig_depth]

        # x-axis = d + m(d, N) for this depth
        x_vals = []
        for d in d_list:
            D_time_aug = d + 1  # channels_after_time: raw dims + time channel
            m = sig_length(D_time_aug, sig_depth, 'logsig')  # uses your helper
            x_vals.append(np.sqrt(d + m))

        plt.plot(
            x_vals,
            med,
            marker="o",
            linestyle="-",
            color=colors[i],
            label=f"N={sig_depth}",
        )

    plt.xlabel("sqrt(d + m) (dim + signature feature length)")
    plt.ylabel(f"Median Cumulative Regret at T={T} (sigUCB)")
    plt.title(
        f"sigUCB Regret vs d + m (K={K})\n"
        f"({process}, {reward_name}, γ={gamma}, λ={lam}, sig_type={sig_type})"
    )
    plt.legend()
    plt.tight_layout()

    out_dir = "experiments/figs/regret_vs_dim"
    os.makedirs(out_dir, exist_ok=True)
    fname = (
        f"sig_final_regret_vs_dplusm_multiN_"
        f"K{K}_{process}_{reward_name}_{sig_type}.png"
    )
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=180)
    print(f"Saved sigUCB final-regret vs d+m (multi-N) figure to {out_path}")

    plt.close()


if __name__ == "__main__":
    main(process="ou_s", reward_name="min_max", sig_type="sig")