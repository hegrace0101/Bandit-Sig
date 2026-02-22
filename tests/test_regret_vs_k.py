# src/sigbandit/tests/check_sig_regret_vs_K.py
import numpy as np
import matplotlib.pyplot as plt
import os

from sigbandit.algos.ucb import sigUCB
from sigbandit.envs.bandit import Bandit
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
raw_d = 1                        # raw channels (before time-augment)
N_total = 100 * 1000             # total points in full series
window = N_total // T            # points per round

lam = 1.0                        # ridge λ
gamma = 1.0                      # exploration γ
fit_intercept = False            # append constant 1.0 to features
sigma = 0.1                      # standard deviation of reward noise

n_runs = 100                     # number of independent runs
seeds = np.arange(n_runs) + 123  # reproducible seeds

# channels seen by the signature after time-augmentation
channels_after_time = raw_d + 1

# K values to sweep over
K_list = [2, 4, 8, 12, 16]

# signature depths to sweep over
depth_list = [1, 2, 3, 4]

# available processes
PROCESS_MAP = {
    "bm": bm,
    "ou": ou,
    "ou_s": ou_stationary,
    "gbm": gbm,
}

# available reward functions
REWARD_MAP = {
    "min_max":      min_max_reward,
    "sum_a":        sum_a_reward,
    "linear":       linear_reward,
    "min_max_mix":  min_max_mix_reward,
}

# ------------------------- single run -------------------------
def run_one(
    seed: int,
    K: int,
    process_fn,
    reward_fn,
    sig_type: str,
    sig_depth: int,
) -> np.ndarray:
    """
    Run one bandit experiment for a given seed, K, process, reward, and signature config.
    Returns sig_regrets of length T+1.
    """
    # set seed
    np.random.seed(seed)

    # initialize path
    X = process_fn(T, N_total)  # match your test_ucb_sig usage

    if process_fn == gbm:
        log_norm = True
    else:
        log_norm = False

    # context process: windows of length `window`
    context = ContextualProcess(X, T, window)

    # bandit with reward function
    bandit = Bandit(context, reward_fn, sigma, K, T, seed)

    # UCB over signature features
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

        # oracle (best possible reward)
        oracle_r, _ = bandit.oracle()

        # sigUCB choice + reward
        sig_a = sig_ucb.select(ctx)
        sig_r = bandit.play(sig_a)
        sig_ucb.update(ctx, sig_a, sig_r)

        t += 1
        sig_regrets[t] = sig_regrets[t - 1] + (oracle_r - sig_r)

    return sig_regrets


# ------------------------- main -------------------------
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

    # store stats per depth and K
    final_med = {}    # depth -> array over K_list
    final_p25 = {}
    final_p75 = {}

    for sig_depth in depth_list:
        print(f"\n===== Depth N = {sig_depth} =====")
        med_list = []
        p25_list = []
        p75_list = []

        for K in K_list:
            print(f"\n--- Running sigUCB experiments for K = {K} ---")

            if reward_name == "linear":
                rng_beta = np.random.RandomState(0)  # fixed seed
                beta_star = rng_beta.uniform(low=-1, high=1.0, size=(K, raw_d))

                def reward_fn(action: int, ctx: np.ndarray) -> float:
                    return linear_reward(action, ctx, beta_star)
            else:
                reward_fn = REWARD_MAP[reward_name]

            runs = [
                run_one(
                    int(s),
                    K,
                    process_fn,
                    reward_fn,
                    sig_type,
                    sig_depth,
                )
                for s in tqdm(
                    seeds,
                    desc=f"N={sig_depth}, K={K} ({process}, {reward_name}, {sig_type})",
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

            print(f"N={sig_depth}, K={K} | Final median sigUCB regret: {med:.3f}")

        final_med[sig_depth] = np.array(med_list)
        final_p25[sig_depth] = np.array(p25_list)
        final_p75[sig_depth] = np.array(p75_list)

    # ------------------------- plot: regret at time T vs K, multiple depths -------------------------
    plt.figure(figsize=(7, 5))

    # default Matplotlib color cycle (more readable than viridis here)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # different marker per depth
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    K_arr = np.array(K_list, dtype=float)
    sqrtK = np.sqrt(K_arr)

    for i, sig_depth in enumerate(depth_list):
        med = final_med[sig_depth]
        plt.plot(
            sqrtK,
            med,
            marker=markers[i % len(markers)],
            linestyle="-",
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=7,
            label=f"N={sig_depth}",
        )

    plt.xlabel(r"$\sqrt{K}$")
    plt.ylabel(f"Median Cumulative Regret at T={T} (sigUCB)")
    plt.title(
        f"Regret vs K\n"
        f"({process}, {reward_name}, λ={lam}, γ={gamma})"
    )
    plt.legend()
    plt.tight_layout()

    out_dir = "experiments/figs/regret_vs_K"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"sig_final_regret_vs_K_multiN_{process}_{reward_name}_{sig_type}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=180)
    print(f"Saved sigUCB final-regret vs K (multi-N) figure to {out_path}")

    plt.close()


if __name__ == "__main__":
    main(process="ou_s", reward_name="min_max_mix", sig_type="sig")
