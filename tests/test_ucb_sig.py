import numpy as np
import matplotlib.pyplot as plt
import os

from sigbandit.algos.ucb import sigUCB, linUCB
from sigbandit.algos.kernel_ucb import KernelUCB
from sigbandit.envs.bandit import Bandit
from sigbandit.envs.context import (
    ContextualProcess,
    ou_stationary,
    ar1,
    arma11,
    regime_switching_ar1,
    trend_seasonal,
    bm,
    gbm,
)
from sigbandit.envs.reward import (
    min_max_reward,
    min_max_mix_reward,
    linear_reward,
    nested_extrema_reward,
    path_mix_reward,
    roughness_reward,
)
from tqdm import tqdm

# ------------------------- config -------------------------
T_tune = 100
T_eval = 100
K = 2
raw_d = 3
window = 1000

fit_intercept = True
sigma = 0.1

channels_after_time = 1 + raw_d # + raw_d # (t, X_t, X_0 t/T)

# bm params
bm_rho = 0.7
idx = np.arange(raw_d)
SIGMA = bm_rho ** np.abs(idx[:, None] - idx[None, :])

# separate seed pools
n_tune = 20
n_test = 100
tune_seeds = np.arange(n_tune) + 1000
test_seeds = np.arange(n_test) + 2000

PROCESS_MAP = {
    "ou_s": lambda T, N, Sigma: ou_stationary(T, N, Sigma=Sigma),
    # "ar": lambda T, N, Sigma: ar1(T, N, phi=np.array([0.2, 0.6, 0.9]), Sigma=Sigma),
    # "arma": lambda T, N, Sigma: arma11(
    #     T,
    #     N,
    #     phi=np.array([0.2, 0.6, 0.9]),
    #     theta=np.array([0.1, 0.3, 0.5]),
    #     Sigma=Sigma,
    # ),
    "bm": lambda T, N, Sigma: bm(T, N, Sigma=Sigma),
    "gbm": lambda T, N, Sigma: gbm(T, N, Sigma=Sigma),
    # "regime_ar": lambda T, N, Sigma: regime_switching_ar1(
    #     T,
    #     N,
    #     phi_regimes=(
    #         np.array([0.15, 0.25, 0.35]),
    #         np.array([0.50, 0.65, 0.75]),
    #         np.array([0.85, 0.92, 0.97]),
    #     ),
    #     mu_regimes=(
    #         np.array([-0.5,  1.5,  0.5]),
    #         np.array([ 1.0,  2.0,  -1.5]),
    #         np.array([ 0.5, -0.25, 0.25]),
    #     ),
    #     Sigma=Sigma,
    # ),
    # "trend_seasonal": lambda T, N, Sigma: trend_seasonal(
    #     T,
    #     N,
    #     slopes=np.array([0.002, 0.005, 0.008]),
    #     amplitudes=np.array([0.5, 1.0, 1.5]),
    #     periods=np.array([40.0, 60.0, 90.0]),
    #     phases=np.array([0.0, 0.8, 1.6]),
    #     noise_phi=np.array([0.2, 0.5, 0.8]),
    #     Sigma=Sigma,
    # ),
}

REWARD_MAP = {
    "min_max": min_max_reward,
    "min_max_mix": min_max_mix_reward,
    "linear": linear_reward,
    "nested_extrema": nested_extrema_reward,
    "path_mix": path_mix_reward,
    "roughness": roughness_reward,
}


# ------------------------- helpers -------------------------
def _get_log_norm(process: str) -> bool:
    return process == "gbm"


def _make_reward_fn(reward_name: str):
    if reward_name not in REWARD_MAP:
        raise ValueError(f"Unknown reward '{reward_name}'. Choose from {list(REWARD_MAP.keys())}.")

    if reward_name == "linear":
        rng_beta = np.random.RandomState(0)
        beta_star = rng_beta.uniform(low=-1.0, high=1.0, size=(K, raw_d))

        def reward_fn(action: int, ctx: np.ndarray) -> float:
            return linear_reward(action, ctx, beta_star)

        return reward_fn

    return REWARD_MAP[reward_name]


def _make_bandit(seed: int, process_fn, reward_fn, T_rounds: int):
    np.random.seed(seed)
    N_total_local = T_rounds * window
    X = process_fn(T_rounds, N_total_local, Sigma=SIGMA)
    context = ContextualProcess(X, T_rounds, window)
    return Bandit(context, reward_fn, sigma, K, T_rounds, seed)


def _ctx_avg(ctx: np.ndarray) -> np.ndarray:
    ctx = np.asarray(ctx, dtype=float)
    return ctx.mean(axis=0)


def _build_sig(log_norm: bool, sig_type: str, sig_depth: int, lam: float, gamma: float):
    return sigUCB(
        n_actions=K,
        lm=float(lam),
        gamma=float(gamma),
        N=int(sig_depth),
        d=channels_after_time,
        sigType=sig_type,
        log_norm=log_norm,
        fit_intercept=fit_intercept,
    )


def _build_lin(log_norm: bool, lam: float, gamma: float):
    return linUCB(
        n_actions=K,
        lm=float(lam),
        gamma=float(gamma),
        n=raw_d,
        log_norm=log_norm,
        fit_intercept=fit_intercept,
    )


def _build_kernel(log_norm: bool, lam: float, gamma: float, kernel_sigma: float, seed: int):
    return KernelUCB(
        n_actions=K,
        lm=float(lam),
        gamma=float(gamma),
        kernel="rbf",
        sigma=float(kernel_sigma),
        max_points=500,
        log_norm=log_norm,
        seed=seed,
    )


# ------------------------- collect full information on one seed -------------------------
def _collect_fullinfo(seed: int, process_fn, reward_fn, T_rounds: int):
    """
    Generate one trajectory and cache:
      - contexts ctx_t
      - averaged contexts xavg_t
      - full-information true rewards Y_t[a] = reward_fn(a, ctx_t)
      - oracle rewards
    """
    bandit = _make_bandit(seed, process_fn, reward_fn, T_rounds=T_rounds)

    ctx_list = []
    xavg_list = []
    Y_list = []
    oracle_list = []

    while bandit.next_round():
        ctx = np.asarray(bandit.observe_context(), dtype=float)
        xavg = _ctx_avg(ctx)

        y = np.array([reward_fn(a, ctx) for a in range(K)], dtype=float)
        oracle_r = float(np.max(y))

        ctx_list.append(ctx)
        xavg_list.append(xavg)
        Y_list.append(y)
        oracle_list.append(oracle_r)

    return ctx_list, xavg_list, np.stack(Y_list, axis=0), np.asarray(oracle_list)


def _collect_seedwise_datasets(seeds, process_fn, reward_fn, T_rounds: int):
    """
    Return a list of per-seed cached datasets.
    Each entry is (ctx_list, xavg_list, Y).
    """
    datasets = []
    for seed in seeds:
        ctx_list, xavg_list, Y, _ = _collect_fullinfo(
            seed=int(seed),
            process_fn=process_fn,
            reward_fn=reward_fn,
            T_rounds=T_rounds,
        )
        datasets.append((ctx_list, xavg_list, Y))
    return datasets


# ------------------------- offline replay tuning -------------------------
def _replay_sig_ucb(ctx_list, Y, log_norm, sig_type, sig_depth, lam, gamma):
    algo = _build_sig(
        log_norm=log_norm,
        sig_type=sig_type,
        sig_depth=sig_depth,
        lam=lam,
        gamma=gamma,
    )
    cum = 0.0
    for t, ctx in enumerate(ctx_list):
        a = algo.select(ctx)
        r = float(Y[t, a])
        oracle = float(np.max(Y[t]))
        algo.update(ctx, a, r)
        cum += (oracle - r)
    return cum


def _replay_lin_ucb(xavg_list, Y, log_norm, lam, gamma):
    algo = _build_lin(log_norm=log_norm, lam=lam, gamma=gamma)
    cum = 0.0
    for t, x in enumerate(xavg_list):
        a = algo.select(x)
        r = float(Y[t, a])
        oracle = float(np.max(Y[t]))
        algo.update(x, a, r)
        cum += (oracle - r)
    return cum


def _replay_kernel_ucb(xavg_list, Y, log_norm, lam, gamma, kernel_sigma, seed):
    algo = _build_kernel(
        log_norm=log_norm,
        lam=lam,
        gamma=gamma,
        kernel_sigma=kernel_sigma,
        seed=seed,
    )
    cum = 0.0
    for t, x in enumerate(xavg_list):
        a = algo.select(x)
        r = float(Y[t, a])
        oracle = float(np.max(Y[t]))
        algo.update(x, a, r)
        cum += (oracle - r)
    return cum


def tune_hyperparams_across_seeds(
    tune_seeds,
    process_name: str,
    process_fn,
    reward_fn,
    sig_type: str,
):
    """
    Tune once on a separate bank of seeds.

    For each hyperparameter candidate:
      - replay separately on each tuning seed
      - compute one final cumulative regret per seed
      - choose the candidate minimizing the median final regret across seeds
    """
    datasets = _collect_seedwise_datasets(
        seeds=tune_seeds,
        process_fn=process_fn,
        reward_fn=reward_fn,
        T_rounds=T_tune,
    )

    log_norm = _get_log_norm(process_name)

    lam_grid = [1e-10, 1e-3, 1e-2, 1e-1, 1.0, 2.5, 5.0]
    gamma_grid = [1e-10, 1e-3, 1e-2, 1e-1, 1.0, 2.5, 5.0]
    sig_depth_grid = [1, 2, 3]
    ker_sigma_grid = [0.5, 1.0, 2.5, 5.0, 10.0, 20.0]

    cfg = {
        "sig_depth": 2,
        "sig_lam": 1.0,
        "sig_gamma": 1.0,
        "lin_lam": 1.0,
        "lin_gamma": 1.0,
        "ker_lam": 1.0,
        "ker_gamma": 1.0,
        "ker_sigma": 1.0,
    }

    print(f"Tuning on {len(tune_seeds)} seeds x {T_tune} rounds each")

    linUCB
    best = None
    print("LinUCB")
    for lam in lam_grid:
        for gamma in gamma_grid:
            regs = []
            for _, xavg_list, Y in datasets:
                reg = _replay_lin_ucb(
                    xavg_list,
                    Y,
                    log_norm=log_norm,
                    lam=lam,
                    gamma=gamma,
                )
                regs.append(reg)
            med_reg = float(np.median(regs))
            print(f"lam={lam}, gamma={gamma}, median_reg={med_reg:.6f}")
            if best is None or med_reg < best[0]:
                best = (med_reg, lam, gamma)
    _, cfg["lin_lam"], cfg["lin_gamma"] = best

    sigUCB
    best = None
    print("SigUCB")
    for depth in sig_depth_grid:
        for lam in lam_grid:
            for gamma in gamma_grid:
                regs = []
                for ctx_list, _, Y in datasets:
                    reg = _replay_sig_ucb(
                        ctx_list,
                        Y,
                        log_norm=log_norm,
                        sig_type=sig_type,
                        sig_depth=depth,
                        lam=lam,
                        gamma=gamma,
                    )
                    regs.append(reg)
                med_reg = float(np.median(regs))
                print(
                    f"depth={depth}, lam={lam}, gamma={gamma}, "
                    f"median_reg={med_reg:.6f}"
                )
                if best is None or med_reg < best[0]:
                    best = (med_reg, depth, lam, gamma)
    _, cfg["sig_depth"], cfg["sig_lam"], cfg["sig_gamma"] = best

    # kernelUCB
    # best = None
    # print("kernelUCB")
    # ker_lam_grid = [1e-3, 1e-2, 1e-1, 1.0, 2.5, 5.0]
    # ker_gamma_grid = [1e-10, 1e-2, 1e-1, 1.0, 2.5, 5.0]

    # for lam in ker_lam_grid:
    #     for gamma in ker_gamma_grid:
    #         for ker_sigma in ker_sigma_grid:
    #             regs = []
    #             for i, (_, xavg_list, Y) in enumerate(datasets):
    #                 reg = _replay_kernel_ucb(
    #                     xavg_list,
    #                     Y,
    #                     log_norm=log_norm,
    #                     lam=lam,
    #                     gamma=gamma,
    #                     kernel_sigma=ker_sigma,
    #                     seed=i,
    #                 )
    #                 regs.append(reg)

    #             med_reg = float(np.median(regs))
    #             print(
    #                 f"lam={lam}, gamma={gamma}, ker_sigma={ker_sigma}, "
    #                 f"median_reg={med_reg:.6f}"
    #             )

    #             if best is None or med_reg < best[0]:
    #                 best = (med_reg, lam, gamma, ker_sigma)

    _, cfg["ker_lam"], cfg["ker_gamma"], cfg["ker_sigma"] = best

    return cfg


# ------------------------- one test-seed run with fixed hyperparams -------------------------
def run_one_fixed_cfg(
    seed: int,
    process_name: str,
    process_fn,
    reward_fn,
    sig_type: str,
    cfg: dict,
):
    """
    Evaluate one test seed using a single fixed global hyperparameter config.
    """
    log_norm = _get_log_norm(process_name)
    bandit = _make_bandit(seed, process_fn, reward_fn, T_rounds=T_eval)

    sig_ucb = _build_sig(
        log_norm=log_norm,
        sig_type=sig_type,
        sig_depth=cfg["sig_depth"],
        lam=cfg["sig_lam"],
        gamma=cfg["sig_gamma"],
    )
    lin_ucb = _build_lin(
        log_norm=log_norm,
        lam=cfg["lin_lam"],
        gamma=cfg["lin_gamma"],
    )
    # ker_ucb = _build_kernel(
    #     log_norm=log_norm,
    #     lam=cfg["ker_lam"],
    #     gamma=cfg["ker_gamma"],
    #     kernel_sigma=cfg["ker_sigma"],
    #     seed=seed,
    # )

    sig_reg = np.zeros(T_eval + 1, dtype=float)
    lin_reg = np.zeros(T_eval + 1, dtype=float)
    # ker_reg = np.zeros(T_eval + 1, dtype=float)

    t = 0
    while bandit.next_round():
        ctx = np.asarray(bandit.observe_context(), dtype=float)
        x_avg = _ctx_avg(ctx)

        oracle_r, _ = bandit.oracle()

        # sigUCB
        a_sig = sig_ucb.select(ctx)
        r_sig_obs = bandit.play(a_sig)
        r_sig_true = reward_fn(a_sig, ctx)
        sig_ucb.update(ctx, a_sig, r_sig_obs)

        # linUCB
        a_lin = lin_ucb.select(x_avg)
        r_lin_obs = bandit.play(a_lin)
        r_lin_true = reward_fn(a_lin, ctx)
        lin_ucb.update(x_avg, a_lin, r_lin_obs)

        # kernelUCB
        # a_ker = ker_ucb.select(x_avg)
        # r_ker_obs = bandit.play(a_ker)
        # r_ker_true = reward_fn(a_ker, ctx)
        # ker_ucb.update(x_avg, a_ker, r_ker_obs)

        t += 1
        sig_reg[t] = sig_reg[t - 1] + (oracle_r - r_sig_true)
        lin_reg[t] = lin_reg[t - 1] + (oracle_r - r_lin_true)
        # sw_reg[t] = sw_reg[t - 1] + (oracle_r - r_sw_true)
        # ker_reg[t] = ker_reg[t - 1] + (oracle_r - r_ker_true)

    return sig_reg, lin_reg, # sw_reg, ker_reg


# ------------------------- main -------------------------
def main(
    process: str = "bm",
    reward_name: str = "min_max",
    sig_type: str = "logsig",
) -> None:
    if process not in PROCESS_MAP:
        raise ValueError(
            f"Unknown process '{process}'. Choose from {list(PROCESS_MAP.keys())}."
        )

    process_fn = PROCESS_MAP[process]
    reward_fn = _make_reward_fn(reward_name)

    print(
        f"Global tuning on {n_tune} seeds for {T_tune} rounds each; "
        f"testing on {n_test} disjoint seeds for {T_eval} rounds each."
    )

    best_cfg = tune_hyperparams_across_seeds(
        tune_seeds=tune_seeds,
        process_name=process,
        process_fn=process_fn,
        reward_fn=reward_fn,
        sig_type=sig_type,
    )

    print("\nChosen global hyperparameters:")
    for k, v in best_cfg.items():
        print(f"  {k}: {v}")

    runs = []
    for s in tqdm(test_seeds, desc=f"Testing ({process}, {reward_name}, {sig_type})"):
        regs = run_one_fixed_cfg(
            seed=int(s),
            process_name=process,
            process_fn=process_fn,
            reward_fn=reward_fn,
            sig_type=sig_type,
            cfg=best_cfg,
        )
        runs.append(regs)

    all_sig = np.stack([r[0] for r in runs], axis=0)
    all_lin = np.stack([r[1] for r in runs], axis=0)
    # all_ker = np.stack([r[3] for r in runs], axis=0)

    sig_med = np.median(all_sig, axis=0)
    sig_p25 = np.percentile(all_sig, 25, axis=0)
    sig_p75 = np.percentile(all_sig, 75, axis=0)

    lin_med = np.median(all_lin, axis=0)
    lin_p25 = np.percentile(all_lin, 25, axis=0)
    lin_p75 = np.percentile(all_lin, 75, axis=0)

    # ker_med = np.median(all_ker, axis=0)
    # ker_p25 = np.percentile(all_ker, 25, axis=0)
    # ker_p75 = np.percentile(all_ker, 75, axis=0)

    print(
        f"\nProcess={process}, reward={reward_name}, sig_type={sig_type} | "
        f"Tune seeds={n_tune} | Test seeds={n_test} | Eval rounds={T_eval}\n"
        f"Final median regret:\n"
        f"  sigUCB    : {sig_med[-1]:.3f}\n"
        f"  linUCB    : {lin_med[-1]:.3f}\n"
        # f"  kernelUCB : {ker_med[-1]:.3f}\n"
    )

    x = np.arange(T_eval + 1)
    plt.figure(figsize=(8, 4.8))

    plt.plot(x, sig_med, label="sigUCB (median)")
    plt.fill_between(x, sig_p25, sig_p75, alpha=0.20)

    plt.plot(x, lin_med, linestyle="--", label="linUCB (median)")
    plt.fill_between(x, lin_p25, lin_p75, alpha=0.15)

    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.title(f"Cumulative Regret ({process}, {reward_name})")
    plt.legend()
    plt.tight_layout()

    out_dir = "experiments/figs/sigUCB"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{process}_{reward_name}_{sig_type}_regret_baselines_seedsplit_x0.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main(process="gbm", reward_name="min_max", sig_type="sig")