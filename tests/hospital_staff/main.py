# tests/hospital_staff_exp/main.py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

from sigbandit.algos.ucb import sigUCB, linUCB
from sigbandit.algos.kernel_ucb import KernelUCB

from sigbandit.envs.hospital_bandit import HospitalBandit
from tests.hospital_staff.load_data import load_nyc_vomit_counts


# ------------------------- helpers: ridge & kernels -------------------------

def _ridge_fit_predict(Xtr, Ytr, Xva, lam: float):
    """
    Xtr: (n_tr, d), Ytr: (n_tr, K)
    Xva: (n_va, d)
    returns Yhat: (n_va, K)
    """
    Xtr = np.asarray(Xtr, float)
    Ytr = np.asarray(Ytr, float)
    Xva = np.asarray(Xva, float)

    n_tr, d = Xtr.shape
    XtX = Xtr.T @ Xtr
    A = XtX + lam * np.eye(d)
    W = np.linalg.solve(A, Xtr.T @ Ytr)   # (d, K)
    return Xva @ W                         # (n_va, K)


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
    Xtr = np.asarray(Xtr, float)
    Ytr = np.asarray(Ytr, float)
    Xva = np.asarray(Xva, float)

    Ktt = _rbf_kernel(Xtr, Xtr, sigma) + lam * np.eye(len(Xtr))
    alpha = np.linalg.solve(Ktt, Ytr)                   # (n_tr, K)
    Kvt = _rbf_kernel(Xva, Xtr, sigma)                  # (n_va, n_tr)
    return Kvt @ alpha                                  # (n_va, K)


# ------------------------- warmup data collection -------------------------

def _collect_warmup_fullinfo(env: HospitalBandit, K: int, warmup_weeks: int):
    """
    Warmup samples for w = 0..warmup_weeks-1:
      - ctx at week w (current env.t)
      - x2 summary from ctx (mean of sin/cos over window)
      - Y labels computed using realized demand of week w+1 (env.reward(a))

    After return: env is positioned at week = warmup_weeks (advanced warmup_weeks times).
    """
    ctx_wins, X2, Y = [], [], []

    for _ in range(warmup_weeks):
        if not env.next_round():
            break

        # week w context window
        ctx_win_w = np.ascontiguousarray(env.observe_context_window(), dtype=np.float64)
        if ctx_win_w.shape[0] != env.L:
            break

        # ctx columns: [daily_demand]
        x_w = ctx_win_w.mean(axis=0)

        # full-info labels for NEXT week (D_{w+1})
        y_next = np.zeros(K, dtype=float)
        for a in range(K):
            y_next[a] = env.reward(a)

        ctx_wins.append(ctx_win_w)
        X2.append(x_w)
        Y.append(y_next)

        env.step_time()

    if len(X2) == 0:
        d_ctx = 3
        return [], np.zeros((0, d_ctx)), np.zeros((0, K))

    return ctx_wins, np.stack(X2, axis=0), np.stack(Y, axis=0)


# ---------------------- stage 1: model selection (simple CV) ----------------------

def _select_lin_lam(X2, Y, lam_grid, val_weeks=13):
    n = len(X2)
    n_va = min(val_weeks, max(1, n // 4))
    n_tr = n - n_va
    Xtr, Ytr = X2[:n_tr], Y[:n_tr]
    Xva, Yva = X2[n_tr:], Y[n_tr:]

    best = None
    for lam in lam_grid:
        Yhat = _ridge_fit_predict(Xtr, Ytr, Xva, lam=lam)
        mse = float(np.mean((Yhat - Yva) ** 2))
        if best is None or mse < best[0]:
            best = (mse, lam)
    return best[1]


def _select_sig_N_lam(ctx_wins, Y, K, adds_time, sig_type, N_grid, lam_grid, val_weeks=13):
    """
    Uses sig features + ridge to pick (N, lam).
    """
    n = len(ctx_wins)
    n_va = min(val_weeks, max(1, n // 4))
    n_tr = n - n_va

    best = None
    for N in N_grid:
        # use sigUCB's internal feature map
        tmp = sigUCB(
            n_actions=K,
            lm=1.0,
            gamma=1.0,
            N=int(N),
            d=int(ctx_wins[0].shape[1]),
            sigType=sig_type,
            log_norm=False,
            fit_intercept=True,
            prune_end_time=True,
            time_augment_bool=adds_time
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
    return int(Nbest), float(lambest)


def _select_kernel_sigma_lam(X2, Y, sigma_grid, lam_grid, val_weeks=13):
    n = len(X2)
    n_va = min(val_weeks, max(1, n // 4))
    n_tr = n - n_va
    Xtr, Ytr = X2[:n_tr], Y[:n_tr]
    Xva, Yva = X2[n_tr:], Y[n_tr:]

    best = None
    for sigma in sigma_grid:
        for lam in lam_grid:
            Yhat = _krr_fit_predict_rbf(Xtr, Ytr, Xva, sigma=sigma, lam=lam)
            mse = float(np.mean((Yhat - Yva) ** 2))
            if best is None or mse < best[0]:
                best = (mse, sigma, lam)
    _, sbest, lbest = best
    return float(sbest), float(lbest)


# ------------------------- Stage-2: tune gamma via offline replay regret -------------------------

def _replay_lin_ucb(X2, Y, K, lam, gamma):
    lin = linUCB(
        n_actions=K,
        lm=float(lam),
        gamma=float(gamma),
        n=int(X2.shape[1]),
        log_norm=False,
        fit_intercept=True,
    )
    cum = 0.0
    for t in range(len(X2)):
        ctx_lin = np.ascontiguousarray(X2[t], dtype=float)[:, None]  # (2,1)
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
        d=int(ctx_wins[0].shape[1]),
        sigType=sig_type,
        log_norm=False,
        fit_intercept=True,
        prune_end_time=True,
        time_augment_bool=adds_time
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


def _replay_kernel_ucb(X2, Y, K, sigma, lam, gamma, seed=0):
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
    for t in range(len(X2)):
        x2 = np.ascontiguousarray(X2[t], dtype=float)
        a = ker.select(x2)
        r = float(Y[t, a])
        oracle = float(np.max(Y[t]))
        ker.update(x2, a, r)
        cum += (oracle - r)
    return cum



def _tune_gamma_lin(X2, Y, K, lam, gamma_grid):
    best = None
    for g in gamma_grid:
        reg = _replay_lin_ucb(X2, Y, K=K, lam=lam, gamma=g)
        if best is None or reg < best[0]:
            best = (reg, g)
    return float(best[1])


def _tune_gamma_sig(ctx_wins, Y, K, adds_time, sig_type, N, lam, gamma_grid):
    best = None
    for g in gamma_grid:
        reg = _replay_sig_ucb(ctx_wins, Y, K=K, adds_time=adds_time, sig_type=sig_type, N=N, lam=lam, gamma=g)
        if best is None or reg < best[0]:
            best = (reg, g)
    return float(best[1])


def _tune_gamma_kernel(X2, Y, K, sigma, lam, gamma_grid, seed=0):
    best = None
    for g in gamma_grid:
        reg = _replay_kernel_ucb(X2, Y, K=K, sigma=sigma, lam=lam, gamma=g, seed=seed)
        if best is None or reg < best[0]:
            best = (reg, g)
    return float(best[1])


# ------------------------- main experiment: warmup + bandit -------------------------

def run_one_hospital(
    seed: int,
    csv_path: str,
    sig_type: str = "sig",
    sig_depth: int = 4,             # used if you disable model selection
    window_L_days: int = 7,
    lam: float = 1.0,
    gamma: float = 1.0,
    adds_time: bool = True,         # kept for parity; Hospital context already has time encoding
    warmup_weeks: int = 52,
    do_model_select: bool = True,
):
    # load daily counts (already trimmed to Monday + full weeks if you used our loader)
    df = load_nyc_vomit_counts(csv_path)
    times = pd.DatetimeIndex(df["Date"])
    daily_counts = df["Count"].to_numpy(dtype=float)

    # build action grid from data scale (safer than hardcoding)
    # nurse demand is weekly_counts / 3, so actions should be comparable
    tmp_weekly = daily_counts.reshape(-1, 7).sum(axis=1) / 3.0
    lo = float(np.floor(np.percentile(tmp_weekly, 1) / 10) * 10)
    hi = float(np.ceil(np.percentile(tmp_weekly, 99) / 10) * 10)
    action_grid = np.arange(max(10.0, lo - 50), hi + 60, 10.0)

    # print("Action space:", action_grid)
    # print("Number of actions K:", len(action_grid))
    # print("Min action:", action_grid.min())
    # print("Max action:", action_grid.max())

    env = HospitalBandit(
        daily_counts=daily_counts,
        times=times,
        window_L_days=window_L_days,
        actions=action_grid,
        underage_cost=0.7,
        overage_cost=0.3,
        seed=seed,
    )
    K = env.K

    # ---------- warmup: collect full-info labels ----------
    ctx_wins, X2, Y = _collect_warmup_fullinfo(env, K=K, warmup_weeks=warmup_weeks)

    # ---------- Stage 1: supervised model selection on warmup ----------
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Ys = Yc / (Yc.std(axis=0, keepdims=True) + 1e-8)

    if do_model_select and len(ctx_wins) >= 16:
        lam_grid = [1e-10, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 5, 10, 15]
        N_grid = [1, 2]
        sigma_grid = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]

        lin_lam = _select_lin_lam(X2, Ys, lam_grid=lam_grid, val_weeks=13)

        sig_N, sig_lam = _select_sig_N_lam(
            ctx_wins, Ys, K=K, adds_time=adds_time, sig_type=sig_type,
            N_grid=N_grid, lam_grid=lam_grid, val_weeks=13
        )

        ker_sigma, ker_lam = _select_kernel_sigma_lam(
            X2, Ys, sigma_grid=sigma_grid, lam_grid=lam_grid, val_weeks=13
        )

        # ---------- Stage 2: tune gamma via offline replay regret ----------
        gamma_grid = [1e-14, 1e-10, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 2.0, 5.0]

        lin_gamma = _tune_gamma_lin(X2, Y, K=K, lam=lin_lam, gamma_grid=gamma_grid)
        sig_gamma = _tune_gamma_sig(ctx_wins, Y, K=K, adds_time=adds_time, sig_type=sig_type, N=sig_N, lam=sig_lam, gamma_grid=gamma_grid)
        ker_gamma = _tune_gamma_kernel(X2, Y, K=K, sigma=ker_sigma, lam=ker_lam, gamma_grid=gamma_grid, seed=seed)
    else:
        lin_lam, lin_gamma = lam, gamma
        sig_N, sig_lam, sig_gamma = sig_depth, lam, gamma
        ker_sigma, ker_lam, ker_gamma = 0.5, lam, gamma

    # print(
    # f"[seed={seed}] chosen params | "
    # f"lin: lam={lin_lam:.3g}, gamma={lin_gamma:.3g} | "
    # f"sig: N={sig_N}, lam={sig_lam:.3g}, gamma={sig_gamma:.3g}, adds_time={adds_time} | "
    # f"ker: sigma={ker_sigma:.3g}, lam={ker_lam:.3g}, gamma={ker_gamma:.3g} | "
    # )

    # ---------- init learners with chosen hyperparams ----------
    d_ctx = ctx_wins[0].shape[1] if len(ctx_wins) > 0 else env.observe_context_window().shape[1]
    d_lin = X2.shape[1] if len(X2) > 0 else d_ctx

    sig_ucb = sigUCB(
        n_actions=K,
        lm=float(sig_lam),
        gamma=float(sig_gamma),
        N=int(sig_N),
        d=int(ctx_wins[0].shape[1]),
        sigType=sig_type,
        log_norm=False,
        fit_intercept=True,
        seed=seed,
        prune_end_time=True,
        time_augment_bool=adds_time
    )

    lin_ucb = linUCB(
        n_actions=K,
        lm=float(lin_lam),
        gamma=float(lin_gamma),
        n=int(d_lin),
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

    # ---------- bandit phase ----------
    sig_reg, lin_reg, ker_reg = [], [], []

    sig_actions, lin_actions, ker_actions = [], [], []
    weekly_counts = []

    sig_cum = lin_cum = ker_cum = 0.0

    # After warmup, env.t is already at week = warmup_weeks (or as far as data allowed)
    while env.next_round():
        ctx_win = np.ascontiguousarray(env.observe_context_window(), dtype=np.float64)   # week t window
        x = ctx_win.mean(axis=0)
        ctx_lin = x[:, None]

        # choose actions for next week (t+1)
        a_sig = sig_ucb.select(ctx_win)
        a_lin = lin_ucb.select(ctx_lin)
        a_ker = ker_ucb.select(x)

        # record realized next-week demand and chosen staffing levels
        weekly_counts.append(float(env.weekly_counts[env.t + 1]))
        sig_actions.append(float(env.actions[a_sig]))
        lin_actions.append(float(env.actions[a_lin]))
        ker_actions.append(float(env.actions[a_ker]))

        # evaluate immediately using env.reward(), which uses D_{t+1}
        oracle_r, _ = env.oracle()

        r_sig = env.reward(a_sig)
        sig_ucb.update(ctx_win, a_sig, r_sig)
        sig_cum += (oracle_r - r_sig)
        sig_reg.append(sig_cum)

        r_lin = env.reward(a_lin)
        lin_ucb.update(ctx_lin, a_lin, r_lin)
        lin_cum += (oracle_r - r_lin)
        lin_reg.append(lin_cum)

        r_ker = env.reward(a_ker)
        ker_ucb.update(x, a_ker, r_ker)
        ker_cum += (oracle_r - r_ker)
        ker_reg.append(ker_cum)

        env.step_time()

    return (
        np.array(sig_reg), np.array(lin_reg), np.array(ker_reg),
        np.array(weekly_counts), np.array(sig_actions), np.array(lin_actions), np.array(ker_actions)
    )


def main(
    csv_path: str = "tests/hospital_staff/nyc_vomit_count.csv",
    sig_type: str = "sig",
    sig_depth: int = 3,
    window_L_days: int = 7,
    lam: float = 1.0,
    gamma: float = 1.0,
    warmup_weeks: int = 52,
    n_runs: int = 50,
    seed0: int = 123,
    out_dir: str = "experiments/figs/hospital_staff",
    tag: str | None = None,
) -> None:
    seeds = np.arange(n_runs) + seed0

    runs = [
        run_one_hospital(
            seed=int(s),
            csv_path=csv_path,
            sig_type=sig_type,
            sig_depth=sig_depth,
            window_L_days=window_L_days,
            lam=lam,
            gamma=gamma,
            warmup_weeks=warmup_weeks,
        )
        for s in tqdm(
            seeds,
            desc=f"Hospital staffing (warmup={warmup_weeks}w, sig={sig_type}, N={sig_depth}, L={window_L_days})"
        )
    ]

    sig_list = [r[0] for r in runs]
    lin_list = [r[1] for r in runs]
    ker_list = [r[2] for r in runs]
    weekly_count_list = [r[3] for r in runs]
    sig_action_list   = [r[4] for r in runs]
    lin_action_list   = [r[5] for r in runs]
    ker_action_list   = [r[6] for r in runs]


    min_len = min(
        min(len(r) for r in sig_list),
        min(len(r) for r in lin_list),
        min(len(r) for r in ker_list),
        min(len(r) for r in weekly_count_list),
        min(len(r) for r in sig_action_list),
        min(len(r) for r in lin_action_list),
        min(len(r) for r in ker_action_list),
    )

    all_sig = np.stack([r[:min_len] for r in sig_list], axis=0)
    all_lin = np.stack([r[:min_len] for r in lin_list], axis=0)
    all_ker = np.stack([r[:min_len] for r in ker_list], axis=0)

    all_weekly = np.stack([r[:min_len] for r in weekly_count_list], axis=0)
    all_sig_a  = np.stack([r[:min_len] for r in sig_action_list], axis=0)
    all_lin_a  = np.stack([r[:min_len] for r in lin_action_list], axis=0)
    all_ker_a  = np.stack([r[:min_len] for r in ker_action_list], axis=0)

    sig_med, sig_p25, sig_p75 = np.median(all_sig, 0), np.percentile(all_sig, 25, 0), np.percentile(all_sig, 75, 0)
    lin_med, lin_p25, lin_p75 = np.median(all_lin, 0), np.percentile(all_lin, 25, 0), np.percentile(all_lin, 75, 0)
    ker_med, ker_p25, ker_p75 = np.median(all_ker, 0), np.percentile(all_ker, 25, 0), np.percentile(all_ker, 75, 0)

    weekly_med = np.median(all_weekly, axis=0)
    sig_a_med, sig_a_p25, sig_a_p75 = np.median(all_sig_a, 0), np.percentile(all_sig_a, 25, 0), np.percentile(all_sig_a, 75, 0)
    lin_a_med, lin_a_p25, lin_a_p75 = np.median(all_lin_a, 0), np.percentile(all_lin_a, 25, 0), np.percentile(all_lin_a, 75, 0)
    ker_a_med, ker_a_p25, ker_a_p75 = np.median(all_ker_a, 0), np.percentile(all_ker_a, 25, 0), np.percentile(all_ker_a, 75, 0)
    
    print(
        f"Hospital | warmup={warmup_weeks}w | runs={n_runs} | "
        f"Final med regret — sigUCB: {sig_med[-1]:.3f} | linUCB: {lin_med[-1]:.3f} | kernelUCB: {ker_med[-1]:.3f}"
    )

    # regret plot
    x = np.arange(min_len)
    plt.figure(figsize=(8, 4.8))

    plt.plot(x, sig_med, label="sigUCB (median)")
    plt.fill_between(x, sig_p25, sig_p75, alpha=0.20)

    plt.plot(x, lin_med, linestyle="--", label="linUCB (median)")
    plt.fill_between(x, lin_p25, lin_p75, alpha=0.15)

    plt.plot(x, ker_med, linestyle="-.", label="kernelUCB (median)")
    plt.fill_between(x, ker_p25, ker_p75, alpha=0.12)

    plt.xlim(left=100, right=min_len-1)
    plt.ylim(bottom=10000)

    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.title("Hospital Nurse Staffing")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    tag_str = f"_{tag}" if tag else ""
    fname = f"hospital_regret_{sig_type}_N{sig_depth}_L{window_L_days}_warm{warmup_weeks}{tag_str}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=180)
    plt.close()

    # staffing vs weekly counts plot
    x = np.arange(min_len)
    
    def moving_avg(y, w=5):
        y = np.asarray(y, dtype=float)
        if w <= 1:
            return y
        pad = w // 2
        ypad = np.pad(y, (pad, pad), mode="edge")
        return np.convolve(ypad, np.ones(w) / w, mode="valid")

    fig, ax1 = plt.subplots(figsize=(9, 4.8))

    # smooth medians and quantiles separately
    weekly_med_plot = moving_avg(weekly_med, w=1)

    sig_med_plot = moving_avg(sig_a_med, w=5)
    sig_p25_plot = moving_avg(sig_a_p25, w=5)
    sig_p75_plot = moving_avg(sig_a_p75, w=5)

    lin_med_plot = moving_avg(lin_a_med, w=5)
    lin_p25_plot = moving_avg(lin_a_p25, w=5)
    lin_p75_plot = moving_avg(lin_a_p75, w=5)

    ker_med_plot = moving_avg(ker_a_med, w=5)
    ker_p25_plot = moving_avg(ker_a_p25, w=5)
    ker_p75_plot = moving_avg(ker_a_p75, w=5)

    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # right axis: staffing
    ax2.plot(x, sig_med_plot, linestyle="-", linewidth=2.0, label="sigUCB (median)")
    ax2.fill_between(x, sig_p25_plot, sig_p75_plot, step="mid", alpha=0.12)
    ax2.plot(x, lin_med_plot, linestyle="--", linewidth=2.0, label="linUCB (median)")
    ax2.fill_between(x, lin_p25_plot, lin_p75_plot, step="mid", alpha=0.12)
    ax2.plot(x, ker_med_plot, linestyle="-.", linewidth=2.0, label="kernelUCB (median)")
    ax2.fill_between(x, ker_p25_plot, ker_p75_plot, step="mid", alpha=0.12)

    ax2.set_ylabel("Assigned Nurse Staffing")

    # left axis: weekly ED counts
    ax1.plot(x, weekly_med_plot, linewidth=2.0, label="Weekly ED counts", color='black')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Weekly ED counts")
    ax1.set_title("Weekly ED Counts and Assigned Nurse Staffing")
    ax1.grid(True, alpha=0.25)

     # Match right axis to left axis divided by 3
    ax1.set_ylim(bottom=1200)
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim(y1_min / 3, y1_max / 3)

    # combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.xlim(left=100, right=min_len-1)
    fig.tight_layout()

    fname2 = f"hospital_staffing_vs_counts_{sig_type}_N{sig_depth}_L{window_L_days}_warm{warmup_weeks}{tag_str}.png"
    fig.savefig(os.path.join(out_dir, fname2), dpi=180)
    plt.close(fig)
    


if __name__ == "__main__":
    main(
        csv_path="tests/hospital_staff/nyc_vomit_count.csv",
        sig_type="sig",
        window_L_days=7,
        lam=1.0,
        gamma=1.0,
        warmup_weeks=26,
        n_runs=10,
    )