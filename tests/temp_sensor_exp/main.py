# tests/temp_sensor_exp/main.py
import numpy as np
import os
import matplotlib.pyplot as plt

from sigbandit.algos.ucb import sigUCB, linUCB
from sigbandit.algos.kernel_ucb import KernelUCB
from tests.temp_sensor_exp.load_data import load_intel_temp_matrix
from sigbandit.envs.temp_sensor_bandit import TempSensorBandit
from tqdm import tqdm

# ------------------------- helpers: ridge & kernels -------------------------
def _ridge_fit_predict(Xtr, Ytr, Xva, lam: float):
    """
    Xtr: (n_tr, d), Ytr: (n_tr, K)
    Xva: (n_va, d)
    returns Yhat: (n_va, K)
    """
    n_tr, d = Xtr.shape
    XtX = Xtr.T @ Xtr
    A = XtX + lam * np.eye(d)
    Ainv = np.linalg.inv(A)
    W = Ainv @ (Xtr.T @ Ytr)     # (d, K)
    return Xva @ W              # (n_va, K)

def _rbf_kernel(X1, X2, sigma: float):
    """
    X1: (n1,d), X2: (n2,d)
    return: (n1,n2) with exp(-||x-z||^2/(2 sigma^2))
    """
    X1 = np.asarray(X1, float)
    X2 = np.asarray(X2, float)
    sq1 = np.sum(X1**2, axis=1, keepdims=True)          # (n1,1)
    sq2 = np.sum(X2**2, axis=1, keepdims=True).T        # (1,n2)
    d2 = sq1 + sq2 - 2.0 * (X1 @ X2.T)
    return np.exp(-0.5 * d2 / (sigma**2))

def _krr_fit_predict_rbf(Xtr, Ytr, Xva, sigma: float, lam: float):
    """
    Kernel ridge regression with shared (sigma, lam) across arms.
    Xtr: (n_tr, d), Ytr: (n_tr, K), Xva: (n_va, d)
    returns Yhat: (n_va, K)
    """
    Ktt = _rbf_kernel(Xtr, Xtr, sigma) + lam * np.eye(len(Xtr))
    alpha = np.linalg.solve(Ktt, Ytr)                   # (n_tr, K)
    Kvt = _rbf_kernel(Xva, Xtr, sigma)                  # (n_va, n_tr)
    return Kvt @ alpha                                  # (n_va, K)

# ------------------------- warmup data collection -------------------------

def _collect_warmup_fullinfo(env: TempSensorBandit, K: int, window_L: int, warmup_hours: int):
    """
    Collect warmup samples for h = 0..warmup_hours-1:
      - ctx at hour h (current env.t)
      - x2 from ctx at hour h
      - Y labels computed on hour h+1 (after one step_time)

    After return: env is positioned at hour = warmup_hours (i.e., advanced warmup_hours times).
    """
    ctx_wins, ctx_avgs, Y = [], [], []

    for _ in range(warmup_hours):
        if not env.next_round():
            break

        # hour h context
        ctx_win_h = np.ascontiguousarray(env.observe_context_window(), dtype=np.float64)
        if ctx_win_h.shape[0] != window_L:
            break

        ctx_avg_h = ctx_win_h.mean(axis=0)

        # move to hour h+1 for label evaluation
        env.step_time()
        if not env.next_round():
            break

        # full-info labels on hour h+1 window
        y_next = np.zeros(K, dtype=float)
        for a in range(K):
            y_next[a] = env.reward(a)

        ctx_wins.append(ctx_win_h)
        ctx_avgs.append(ctx_avg_h)
        Y.append(y_next)

    return ctx_wins, np.stack(ctx_avgs, axis=0), np.stack(Y, axis=0)


# ---------------------- stage 1: model selection (simple CV) ----------------------
def _select_lin_lam(ctx_avgs, Y, lam_grid, val_hours=6):
    n = len(ctx_avgs)
    n_va = min(val_hours, max(1, n // 4))
    n_tr = n - n_va
    Xtr, Ytr = ctx_avgs[:n_tr], Y[:n_tr]
    Xva, Yva = ctx_avgs[n_tr:], Y[n_tr:]

    best = None
    for lam in lam_grid:
        Yhat = _ridge_fit_predict(Xtr, Ytr, Xva, lam=lam)
        mse = float(np.mean((Yhat - Yva) ** 2))
        if best is None or mse < best[0]:
            best = (mse, lam)
    return best[1]

def _select_sig_N_lam(ctx_wins, Y, K, adds_time, sig_type, N_grid, lam_grid, val_hours=6):
    """
    Uses sig features + ridge to pick (N, lam).
    """
    n = len(ctx_wins)
    n_va = min(val_hours, max(1, n // 4))
    n_tr = n - n_va

    best = None
    for N in N_grid:
        # build a tiny sigUCB to reuse its internal feature map (so we don’t guess signature internals)
        # n_actions=1 is fine; we only use .sig_feat
        tmp = sigUCB(
            n_actions=1,
            lm=1.0,
            gamma=1.0,
            N=N,
            d=(ctx_wins[0].shape[1]),
            sigType=sig_type,
            log_norm=False,
            fit_intercept=True,
            prune_end_time=True,
            time_augment_bool=adds_time,
        )

        Phi = []
        for w in ctx_wins:
            f = tmp.sig_feat(np.ascontiguousarray(w, dtype=np.float64))
            Phi.append(np.asarray(f, dtype=float).ravel())
        Phi = np.stack(Phi, axis=0)   # (n, m)

        Xtr, Ytr = Phi[:n_tr], Y[:n_tr]
        Xva, Yva = Phi[n_tr:], Y[n_tr:]

        for lam in lam_grid:
            Yhat = _ridge_fit_predict(Xtr, Ytr, Xva, lam=lam)
            mse = float(np.mean((Yhat - Yva) ** 2))
            if best is None or mse < best[0]:
                best = (mse, N, lam)

    _, Nbest, lambest = best
    return Nbest, lambest

def _select_kernel_sigma_lam(ctx_avgs, Y, sigma_grid, lam_grid, val_hours=6):
    n = len(ctx_avgs)
    n_va = min(val_hours, max(1, n // 4))
    n_tr = n - n_va
    Xtr, Ytr = ctx_avgs[:n_tr], Y[:n_tr]
    Xva, Yva = ctx_avgs[n_tr:], Y[n_tr:]

    best = None
    for sigma in sigma_grid:
        for lam in lam_grid:
            Yhat = _krr_fit_predict_rbf(Xtr, Ytr, Xva, sigma=sigma, lam=lam)
            mse = float(np.mean((Yhat - Yva) ** 2))
            if best is None or mse < best[0]:
                best = (mse, sigma, lam)
    _, sbest, lbest = best
    return sbest, lbest


# ------------------------- Stage-2: tune gamma via offline replay regret -------------------------
def _replay_lin_ucb(ctx_avgs, Y, K, lam, gamma):
    p = ctx_avgs.shape[1]    
    lin = linUCB(
        n_actions=K,
        lm=float(lam),
        gamma=float(gamma),
        n=p,
        log_norm=False,
        fit_intercept=True,
    )
    cum = 0.0
    for t in range(len(ctx_avgs)):
        ctx_lin = np.ascontiguousarray(ctx_avgs[t], dtype=float)[:, None]  # (2,1)
        a = lin.select(ctx_lin)
        r = float(Y[t, a])
        oracle = float(np.max(Y[t]))
        lin.update(ctx_lin, a, r)
        cum += (oracle - r)
    return cum


def _replay_sig_ucb(ctx_wins, Y, K, adds_time, sig_type, N, lam, gamma):
    d_ctx = ctx_wins[0].shape[1]
    sig = sigUCB(
        n_actions=K,
        lm=float(lam),
        gamma=float(gamma),
        N=int(N),
        d=int(d_ctx),
        sigType=sig_type,
        log_norm=False,
        fit_intercept=True,
        prune_end_time=True,
        time_augment_bool=adds_time,
    )
    cum = 0.0
    for t in range(len(ctx_wins)):
        ctx = np.ascontiguousarray(ctx_wins[t], dtype=np.float64)
        a = sig.select(ctx)
        r = float(Y[t, a])
        oracle = float(np.max(Y[t]))
        sig.update(ctx, a, r)
        cum += (oracle - r)
    return cum


def _replay_kernel_ucb(ctx_avgs, Y, K, sigma, lam, gamma, seed=0):
    ker = KernelUCB(
        n_actions=K,
        lm=float(lam),
        gamma=float(gamma),
        kernel="rbf",
        sigma=float(sigma),
        max_points=500,
        seed=int(seed),
    )
    cum = 0.0
    for t in range(len(ctx_avgs)):
        x = np.ascontiguousarray(ctx_avgs[t], dtype=float)
        a = ker.select(x)
        r = float(Y[t, a])
        oracle = float(np.max(Y[t]))
        ker.update(x, a, r)
        cum += (oracle - r)
    return cum


def _tune_gamma_lin(ctx_avgs, Y, K, lam, gamma_grid):
    best = None
    for g in gamma_grid:
        reg = _replay_lin_ucb(ctx_avgs, Y, K=K, lam=lam, gamma=g)
        if best is None or reg < best[0]:
            best = (reg, g)
    return best[1]


def _tune_gamma_sig(ctx_wins, Y, K, adds_time, sig_type, N, lam, gamma_grid):
    best = None
    for g in gamma_grid:
        reg = _replay_sig_ucb(ctx_wins, Y, K=K, adds_time=adds_time, sig_type=sig_type, N=N, lam=lam, gamma=g)
        if best is None or reg < best[0]:
            best = (reg, g)
    return best[1]


def _tune_gamma_kernel(ctx_avgs, Y, K, sigma, lam, gamma_grid, seed=0):
    best = None
    for g in gamma_grid:
        reg = _replay_kernel_ucb(ctx_avgs, Y, K=K, sigma=sigma, lam=lam, gamma=g, seed=seed)
        if best is None or reg < best[0]:
            best = (reg, g)
    return best[1]


# ------------------------- main experiment: warmup + bandit -------------------------

def run_one_intel(
    seed: int,
    txt_path: str,
    sig_type: str = "sig",
    sig_depth: int = 4,             # used if you disable model selection
    d0: int = 0,
    window_L: int = 12,
    reward_mode: str = "max",
    lam: float = 1.0,               # default / fallback
    gamma: float = 1.0,
    freq: str = "5min",
    adds_time: bool = True,
    warmup_hours: int = 24,
    do_model_select: bool = True,
):
    temp, times, sensor_ids = load_intel_temp_matrix(txt_path, freq=freq)
    K = temp.shape[1] - 2

    rng = np.random.default_rng(seed)
    anchors = rng.choice(np.arange(K), size=d0, replace=False).tolist() if d0 > 0 else []

    env = TempSensorBandit(
        temp_matrix=temp,
        times=times,
        anchors=anchors,
        window_L=window_L,
        reward_mode=reward_mode,
        seed=seed,
    )

    def _window_stats(env_obj):
        sl = env_obj._window_slice()
        field = env_obj.X[sl, :env_obj.K]  # (L, K)

        if env_obj.reward_mode == "max":
            per_sensor_stat = np.max(field, axis=0)   # (K,)
            true_stat = float(np.max(field))
        elif env_obj.reward_mode == "avg":
            per_sensor_stat = np.mean(field, axis=0)  # (K,)
            true_stat = float(np.mean(field))
        else:
            raise ValueError(f"Unknown reward_mode={env_obj.reward_mode!r}")

        return true_stat, per_sensor_stat

    # ---------- warmup: collect full-info labels ----------
    ctx_wins, ctx_avgs, Y = _collect_warmup_fullinfo(env, K=K, window_L=window_L, warmup_hours=warmup_hours)

    # ---------- Stage 1: supervised model selection on warmup (pick lam/N/sigma) ----------
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Ys = Yc / (Yc.std(axis=0, keepdims=True) + 1e-8)

    if do_model_select and len(ctx_wins) >= 8:
        lam_grid   = [1e-10, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0, 100.0]
        N_grid     = [1, 2]
        sigma_grid = [0.1, 1.0, 2.0, 5.0, 10.0]

        lin_lam = _select_lin_lam(ctx_avgs, Ys, lam_grid=lam_grid, val_hours=6)

        sig_N, sig_lam = _select_sig_N_lam(
            ctx_wins, Ys, K=K, adds_time=adds_time, sig_type=sig_type,
            N_grid=N_grid, lam_grid=lam_grid, val_hours=6
        )

        ker_sigma, ker_lam = _select_kernel_sigma_lam(
            ctx_avgs, Ys, sigma_grid=sigma_grid, lam_grid=lam_grid, val_hours=6
        )

        # ---------- Stage 2: offline bandit replay tuning for gamma (with stage-1 hypers frozen) ----------
        gamma_grid = [1e-10, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0]

        lin_gamma = _tune_gamma_lin(ctx_avgs, Y, K=K, lam=lin_lam, gamma_grid=gamma_grid)

        sig_gamma = _tune_gamma_sig(
            ctx_wins, Y, K=K, adds_time=adds_time, sig_type=sig_type,
            N=sig_N, lam=sig_lam, gamma_grid=gamma_grid
        )

        ker_gamma = _tune_gamma_kernel(ctx_avgs, Y, K=K, sigma=ker_sigma, lam=ker_lam, gamma_grid=gamma_grid, seed=seed)

        # print('LinUCB:', lin_lam, lin_gamma)
        # print('SigUCB:', sig_N, sig_lam, sig_gamma)
        # print('KerUCB:', ker_sigma, ker_lam, ker_gamma)

    else:
        lin_lam, lin_gamma = lam, gamma
        sig_N, sig_lam, sig_gamma = sig_depth, lam, gamma
        ker_sigma, ker_lam, ker_gamma = 0.5, lam, gamma

    # print(
    #     f"[seed={seed}] chosen signature depth N={sig_N} | "
    #     f"sig_lam={sig_lam} | sig_gamma={sig_gamma}"
    # )


    # ---------- init learners with chosen hyperparams ----------
    # infer sig dimension d from one window
    if len(ctx_wins) > 0:
        d_ctx = ctx_wins[0].shape[1]
    else:
        # fallback if warmup empty
        d_ctx = np.ascontiguousarray(env.observe_context_window(), dtype=np.float64).shape[1]

    sig_ucb = sigUCB(
        n_actions=K, 
        lm=float(sig_lam), 
        gamma=float(sig_gamma),
        N=int(sig_N), 
        d=int(d_ctx),
        sigType=sig_type, 
        log_norm=False, 
        fit_intercept=True,
        seed=seed,
        prune_end_time=True,
        time_augment_bool=adds_time,
    )

    lin_ucb = linUCB(
        n_actions=K, 
        lm=float(lin_lam), 
        gamma=float(lin_gamma),
        n=d_ctx, 
        log_norm=False, 
        fit_intercept=True,
        seed=seed,
    )

    ker_ucb = KernelUCB(
        n_actions=K, 
        lm=float(ker_lam), 
        gamma=float(ker_gamma),
        kernel="rbf", 
        sigma=float(ker_sigma),
        max_points=500, 
        seed=seed,
    )

    # ---------- bandit phase (remaining data): pending / next-hour evaluation ----------
    sig_reg, lin_reg, ker_reg = [], [], []
    sig_err, lin_err, ker_err = [], [], []
    sig_cum = lin_cum = ker_cum = 0.0
    sig_chosen_stat, lin_chosen_stat, ker_chosen_stat = [], [], []
    true_stat_series = []

    pending = None 

    while env.next_round():
        ctx_win = np.ascontiguousarray(env.observe_context_window(), dtype=np.float64)   # hour t window
        ctx_avg = ctx_win.mean(axis=0)
        ctx_lin = ctx_avg[:, None]

        # choose actions for next hour
        a_sig = sig_ucb.select(ctx_win)
        a_lin = lin_ucb.select(ctx_lin)
        a_ker = ker_ucb.select(ctx_avg)

        # move to hour t+1 (evaluation window)
        env.step_time()
        if not env.next_round():
            break

        if pending is not None:
            ctxw_prev, ctxlin_prev, ctx_avg_prev, as_prev, al_prev, ak_prev = pending

            true_stat, per_sensor_stat = _window_stats(env)

            true_stat_series.append(true_stat)
            sig_chosen_stat.append(float(per_sensor_stat[as_prev]))
            lin_chosen_stat.append(float(per_sensor_stat[al_prev]))
            ker_chosen_stat.append(float(per_sensor_stat[ak_prev]))

            oracle_r, _ = env.oracle()

            r_sig = env.reward(as_prev)
            sig_ucb.update(ctxw_prev, as_prev, r_sig)
            sig_cum += (oracle_r - r_sig)
            sig_reg.append(sig_cum)
            sig_err.append(-r_sig)

            r_lin = env.reward(al_prev)
            lin_ucb.update(ctxlin_prev, al_prev, r_lin)
            lin_cum += (oracle_r - r_lin)
            lin_reg.append(lin_cum)
            lin_err.append(-r_lin)

            r_ker = env.reward(ak_prev)
            ker_ucb.update(ctx_avg_prev, ak_prev, r_ker)
            ker_cum += (oracle_r - r_ker)
            ker_reg.append(ker_cum)
            ker_err.append(-r_ker)

        pending = (ctx_win, ctx_lin, ctx_avg, a_sig, a_lin, a_ker)

    return (
        np.array(sig_reg), np.array(lin_reg), np.array(ker_reg),
        np.array(sig_err), np.array(lin_err), np.array(ker_err),
        np.array(sig_chosen_stat), np.array(lin_chosen_stat), np.array(ker_chosen_stat),
        np.array(true_stat_series),
    )

def main(
    txt_path: str = "tests/temp_sensor_exp/temp_sensor.txt",
    freq: str = "5min",
    reward_mode: str = "max",      # "max" or "avg"
    sig_type: str = "sig",
    sig_depth: int = 3,
    d0: int = 3,
    window_L: int = 12,
    lam: float = 1.0,
    gamma: float = 1.0,
    n_runs: int = 20,
    seed0: int = 123,
    out_dir: str = "experiments/figs/temp_sensor",
    tag: str | None = None,
) -> None:
    seeds = np.arange(n_runs) + seed0

    runs = [
        run_one_intel(
            seed=int(s),
            txt_path=txt_path,
            sig_type=sig_type,
            sig_depth=sig_depth,
            d0=d0,
            window_L=window_L,
            reward_mode=reward_mode,
            lam=lam,
            gamma=gamma,
            freq=freq,
        )
        for s in tqdm(
            seeds,
            desc=f"Intel temp ({reward_mode}, sig={sig_type}, N={sig_depth}, d0={d0}, L={window_L}, freq={freq})"
        )
    ]

    # unpack
    sig_list = [r[0] for r in runs]
    lin_list = [r[1] for r in runs]
    ker_list = [r[2] for r in runs]
    sig_err_list = [r[3] for r in runs]
    lin_err_list = [r[4] for r in runs]
    ker_err_list = [r[5] for r in runs]
    sig_stat_list = [r[6] for r in runs]
    lin_stat_list = [r[7] for r in runs]
    ker_stat_list = [r[8] for r in runs]
    true_stat_list = [r[9] for r in runs]

    min_len = min(
        min(len(r) for r in sig_list),
        min(len(r) for r in lin_list),
        min(len(r) for r in ker_list),
        min(len(r) for r in sig_err_list),
        min(len(r) for r in lin_err_list),
        min(len(r) for r in ker_err_list),
        min(len(r) for r in sig_stat_list),
        min(len(r) for r in lin_stat_list),
        min(len(r) for r in ker_stat_list),
        min(len(r) for r in true_stat_list),
    )

    all_sig = np.stack([r[:min_len] for r in sig_list], axis=0)
    all_lin = np.stack([r[:min_len] for r in lin_list], axis=0)
    all_ker = np.stack([r[:min_len] for r in ker_list], axis=0)

    all_sig_err = np.stack([r[:min_len] for r in sig_err_list], axis=0)
    all_lin_err = np.stack([r[:min_len] for r in lin_err_list], axis=0)
    all_ker_err = np.stack([r[:min_len] for r in ker_err_list], axis=0)

    all_sig_stat = np.stack([r[:min_len] for r in sig_stat_list], axis=0)
    all_lin_stat = np.stack([r[:min_len] for r in lin_stat_list], axis=0)
    all_ker_stat = np.stack([r[:min_len] for r in ker_stat_list], axis=0)
    all_true_stat = np.stack([r[:min_len] for r in true_stat_list], axis=0)

    sig_med, sig_p25, sig_p75 = np.median(all_sig, 0), np.percentile(all_sig, 25, 0), np.percentile(all_sig, 75, 0)
    lin_med, lin_p25, lin_p75 = np.median(all_lin, 0), np.percentile(all_lin, 25, 0), np.percentile(all_lin, 75, 0)
    ker_med, ker_p25, ker_p75 = np.median(all_ker, 0), np.percentile(all_ker, 25, 0), np.percentile(all_ker, 75, 0)

    sigE_med, sigE_p25, sigE_p75 = np.median(all_sig_err,0), np.percentile(all_sig_err,25,0), np.percentile(all_sig_err,75,0)
    linE_med, linE_p25, linE_p75 = np.median(all_lin_err,0), np.percentile(all_lin_err,25,0), np.percentile(all_lin_err,75,0)
    kerE_med, kerE_p25, kerE_p75 = np.median(all_ker_err,0), np.percentile(all_ker_err,25,0), np.percentile(all_ker_err,75,0)

    sigM_med, sigM_p25, sigM_p75 = np.median(all_sig_stat,0), np.percentile(all_sig_stat,25,0), np.percentile(all_sig_stat,75,0)
    linM_med, linM_p25, linM_p75 = np.median(all_lin_stat,0), np.percentile(all_lin_stat,25,0), np.percentile(all_lin_stat,75,0)
    kerM_med, kerM_p25, kerM_p75 = np.median(all_ker_stat,0), np.percentile(all_ker_stat,25,0), np.percentile(all_ker_stat,75,0)
    truM_med, truM_p25, truM_p75 = np.median(all_true_stat,0), np.percentile(all_true_stat,25,0), np.percentile(all_true_stat,75,0)

    print(
        f"IntelTemp | reward={reward_mode} | runs={n_runs} | "
        f"Final med regret — sigUCB: {sig_med[-1]:.3f} | linUCB: {lin_med[-1]:.3f} | kernelUCB: {ker_med[-1]:.3f}"
    )

    stat_word = "Max" if reward_mode == "max" else "Avg"
    stat_word_lower = stat_word.lower()

    # regret plot
    x = np.arange(min_len)
    plt.figure(figsize=(8, 4.8))

    plt.plot(x, sig_med, label="sigUCB")
    plt.fill_between(x, sig_p25, sig_p75, alpha=0.20)

    plt.plot(x, lin_med, linestyle="--", label="linUCB")
    plt.fill_between(x, lin_p25, lin_p75, alpha=0.15)

    plt.plot(x, ker_med, linestyle=":", label="kernelUCB")
    plt.fill_between(x, ker_p25, ker_p75, alpha=0.12)

    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.title(f"Intel Lab: {stat_word} Temperature Monitoring")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    tag_str = f"_{tag}" if tag else ""
    fname = f"intel_{reward_mode}_{sig_type}_N{sig_depth}_d0{d0}_L{window_L}{tag_str}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=180)
    plt.close()

    # error plot 
    plt.figure(figsize=(8, 4.8))

    plt.plot(x, sigE_med, label="sigUCB")
    plt.fill_between(x, sigE_p25, sigE_p75, alpha=0.20)

    plt.plot(x, linE_med, linestyle="--", label="linUCB")
    plt.fill_between(x, linE_p25, linE_p75, alpha=0.15)

    plt.plot(x, kerE_med, linestyle=":", label="kernelUCB")
    plt.fill_between(x, kerE_p25, kerE_p75, alpha=0.12)

    plt.xlabel("Round")
    plt.ylabel(f"{stat_word} Temperature Absolute Error")
    plt.title(f"Intel Lab: {stat_word} Temperature Error Over Time")
    plt.legend()
    plt.tight_layout()

    fname_err = f"intel_error_{reward_mode}_{sig_type}_N{sig_depth}_d0{d0}_L{window_L}{tag_str}.png"
    plt.savefig(os.path.join(out_dir, fname_err), dpi=180)
    plt.close()

    # chosen-sensor max temp vs true field max
    plt.figure(figsize=(8, 4.8))

    plt.plot(x, truM_med, color="k", linewidth=2.0, label=f"true {stat_word_lower}")
    plt.fill_between(x, truM_p25, truM_p75, color="k", alpha=0.08)

    plt.plot(x, sigM_med, label="sigUCB")
    plt.fill_between(x, sigM_p25, sigM_p75, alpha=0.20)

    plt.plot(x, linM_med, linestyle="--", label="linUCB")
    plt.fill_between(x, linM_p25, linM_p75, alpha=0.15)

    plt.plot(x, kerM_med, linestyle=":", label="kernelUCB")
    plt.fill_between(x, kerM_p25, kerM_p75, alpha=0.12)


    plt.xlabel("Round")
    plt.ylabel(f"{stat_word} Temperature")
    plt.title(f"Chosen Sensor {stat_word} vs True {stat_word}")
    plt.legend()
    plt.tight_layout()

    fname_max = f"intel_chosen_vs_true_{reward_mode}_{sig_type}_N{sig_depth}_d0{d0}_L{window_L}{tag_str}.png"
    plt.savefig(os.path.join(out_dir, fname_max), dpi=180)
    plt.close()


if __name__ == "__main__":
    main(
        txt_path="tests/temp_sensor_exp/temp_sensor.txt",
        reward_mode="max",
        sig_type="sig",
        sig_depth=3,
        d0=4,
        window_L=12,
        lam=1.0,
        gamma=1.0,
        n_runs=10,
    )