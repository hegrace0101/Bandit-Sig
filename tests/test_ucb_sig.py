# src/sigbandit/tests/test_ucb_sig.py
import numpy as np
import matplotlib.pyplot as plt
import os

from sigbandit.algos.ucb import sigUCB, linUCB
from sigbandit.envs.bandit import Bandit
from sigbandit.envs.context import ContextualProcess, bm, ou, ou_stationary, gbm, frac_bm
from sigbandit.envs.reward import min_max_reward, sum_a_reward, linear_reward
from tqdm import tqdm

# ------------------------- config -------------------------
T = 100                          # rounds per run
K = 2                            # actions
raw_d = 1                        # raw channels (before time-augment)
N_total = 100*1000               # total points in full series
window = N_total // T            # points per round

lam = 1.0                        # ridge λ
gamma = 1.0                    	 # exploration γ
fit_intercept = False            # append constant 1.0 to features
sigma = 0.1                      # standard deviation of reward noise

n_runs = 100                     # number of independent runs
seeds = np.arange(n_runs) + 123  # reproducible seeds

# channels seen by the signature after time-augmentation
channels_after_time = raw_d + 1

# available processes
PROCESS_MAP = {
    "bm": bm,
    "ou": ou,
    "ou_s": ou_stationary,
    "gbm": gbm,
    "fbm": frac_bm
}

# available reward functions
REWARD_MAP = {
    "min_max": min_max_reward,
    "sum_a":   sum_a_reward,
    "linear":  linear_reward,
}

# ------------------------- runner -------------------------
def run_one(
    seed: int,
    process_fn,
    reward_fn,
    sig_type: str,
    sig_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one bandit experiment for a given seed, process, reward, and signature config.
    Returns (sig_regrets, lin_regrets), each of length T+1.
    """
    # set seed
    np.random.seed(seed)

    # initialize path
    X = process_fn(T, N_total)  # shape (N_total, raw_d)
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
    
    # linUCB (context dimension is window length here)
    lin_ucb = linUCB(
        n_actions=K,
        lm=lam,
        gamma=gamma,
        n=int(N_total / T),
        log_norm=log_norm,
        fit_intercept=fit_intercept,
    )

    sig_regrets = np.zeros(T + 1, dtype=float)  # start at 0 for nicer plotting
    lin_regrets = np.zeros(T + 1, dtype=float)  # start at 0 for nicer plotting
    t = 0

    while bandit.next_round():
        ctx = bandit.observe_context()

        # oracle (best possible reward)
        oracle_r, _ = bandit.oracle()

        # sigUCB choice + reward
        sig_a = sig_ucb.select(ctx)
        sig_r = bandit.play(sig_a)
        sig_ucb.update(ctx, sig_a, sig_r)
        
        # linUCB choice + reward
        lin_a = lin_ucb.select(ctx)
        lin_r = bandit.play(lin_a)
        lin_ucb.update(ctx, lin_a, lin_r)

        t += 1
        sig_regrets[t] = sig_regrets[t - 1] + (oracle_r - sig_r)
        lin_regrets[t] = lin_regrets[t - 1] + (oracle_r - lin_r)

    return sig_regrets, lin_regrets

# ------------------------- main -------------------------
def main(process: str = "bm", reward_name: str = "min_max", sig_type: str = "logsig", sig_depth: int = 4) -> None:
    if process not in PROCESS_MAP:
        raise ValueError(f"Unknown process '{process}'. "
                         f"Choose from {list(PROCESS_MAP.keys())}.")
    if reward_name not in REWARD_MAP:
        raise ValueError(f"Unknown reward '{reward_name}'. "
                         f"Choose from {list(REWARD_MAP.keys())}.")

    process_fn = PROCESS_MAP[process]

    if reward_name == "linear":
        rng_beta = np.random.RandomState(0)  # fixed seed
        beta_star = rng_beta.uniform(low=-1, high=1.0, size=(K, raw_d))

        def reward_fn(action: int, ctx: np.ndarray) -> float:
            return linear_reward(action, ctx, beta_star)
    else:
        reward_fn = REWARD_MAP[reward_name]

    # run many
    runs = [
        run_one(int(s), process_fn, reward_fn, sig_type, sig_depth)
        for s in tqdm(seeds, desc=f"Running sigUCB experiments ({process}, {reward_name}, {sig_type}, N={sig_depth})")
    ]

    all_sig = np.stack([r[0] for r in runs], axis=0)  # (n_runs, T+1)
    all_lin = np.stack([r[1] for r in runs], axis=0)  # (n_runs, T+1)

    sig_med = np.median(all_sig, axis=0)
    sig_p25 = np.percentile(all_sig, 25, axis=0)
    sig_p75 = np.percentile(all_sig, 75, axis=0)

    lin_med = np.median(all_lin, axis=0)
    lin_p25 = np.percentile(all_lin, 25, axis=0)
    lin_p75 = np.percentile(all_lin, 75, axis=0)

    print(
        f"Process={process}, reward={reward_name}, sig_type={sig_type}, depth={sig_depth} | "
        f"Runs: {n_runs} | Final med regret — sigUCB: {sig_med[-1]:.3f} | linUCB: {lin_med[-1]:.3f}"
    )

    # ------------------------- plot -------------------------
    x = np.arange(T + 1)

    plt.figure(figsize=(8, 4.8))

    # sigUCB
    plt.plot(x, sig_med, label="sigUCB (median)")
    plt.fill_between(x, sig_p25, sig_p75, alpha=0.20, label="sigUCB 25–75%")

    # linUCB
    plt.plot(x, lin_med, label="linUCB (median)", linestyle="--")
    plt.fill_between(x, lin_p25, lin_p75, alpha=0.15, label="linUCB 25–75%")

    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.title(
        f"Cumulative Regret ({process}, {reward_name}, "
        f"K={K}, depth={sig_depth}, λ={lam}, γ={gamma})"
    )
    plt.legend()
    plt.tight_layout()

    # save figure
    out_dir = "experiments/figs/sigUCB"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{process}_{reward_name}_{sig_type}_{sig_depth}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=180)

    plt.close()

if __name__ == "__main__":
    main(process="fbm", reward_name='min_max', sig_type="sig", sig_depth=1)