# tests/sleep_edf/main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sigbandit.algos.ucb import sigUCB, ammSigUCB, linUCB
from sigbandit.algos.kernel_ucb import KernelUCB
from sigbandit.utils.features import (
    temporal_stats_features, temporal_stats_dim,
    polynomial_features, polynomial_dim,
    make_rff,
)

from sigbandit.envs.sleep_bandit import SleepBandit
from tests.sleep_edf.load_data import load_sleep_cassette


# extra linUCB baselines: same UCB math, different feature transforms of the (L, d) EEG window.
# All four reuse the tuned (lin_lam, lin_gamma); RFF reuses the tuned KernelUCB RBF bandwidth.
LIN_VARIANT_NAMES = ("linUCB_stats", "linUCB_poly", "linUCB_rff")


def _build_lin_variants(K, lam, gamma, L, d, rff_sigma, rff_D, seed):
    return {
        "linUCB_stats": linUCB(n_actions=K, lm=lam, gamma=gamma,
                               n=temporal_stats_dim(d), feat_fn=temporal_stats_features, seed=seed),
        "linUCB_poly":  linUCB(n_actions=K, lm=lam, gamma=gamma,
                               n=polynomial_dim(d), feat_fn=polynomial_features, seed=seed),
        "linUCB_rff":   linUCB(n_actions=K, lm=lam, gamma=gamma,
                               n=int(rff_D),
                               feat_fn=make_rff(d=int(d), D=int(rff_D), sigma=float(rff_sigma), seed=int(seed)),
                               seed=seed),
    }

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json

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
    A = Xtr.T @ Xtr + lam * np.eye(d)
    W = np.linalg.solve(A, Xtr.T @ Ytr)     # (d, K)
    return Xva @ W

def _rbf_kernel(X1, X2, sigma: float):
    X1 = np.asarray(X1, float)
    X2 = np.asarray(X2, float)
    sq1 = np.sum(X1**2, axis=1, keepdims=True)
    sq2 = np.sum(X2**2, axis=1, keepdims=True).T
    d2 = sq1 + sq2 - 2.0 * (X1 @ X2.T)
    return np.exp(-0.5 * d2 / (sigma**2))

def _krr_fit_predict_rbf(Xtr, Ytr, Xva, sigma: float, lam: float):
    Ktt = _rbf_kernel(Xtr, Xtr, sigma) + lam * np.eye(len(Xtr))
    alpha = np.linalg.solve(Ktt, Ytr)
    Kvt = _rbf_kernel(Xva, Xtr, sigma)
    return Kvt @ alpha


# ------------------------- features for lin/kernel baselines -------------------------

def _epoch_feat_mean(ctx_win: np.ndarray) -> np.ndarray:
    """
    ctx_win: (L, d_channels)
    returns: (d_channels,) mean over time
    """
    return np.asarray(ctx_win, float).mean(axis=0)


# ------------------------- warmup collection (full-info) -------------------------

def _collect_warmup_fullinfo(env: SleepBandit, warmup_epochs: int):
    """
    Collect warmup samples:
      - context window at epoch t
      - x = simple vector feature from context window
      - Y = full-info rewards for all actions at epoch t (here Y[t] = P[t])
    """
    ctx_wins, Xfeat, Y = [], [], []

    for _ in range(warmup_epochs):
        if not env.next_round():
            break
        ctx = np.ascontiguousarray(env.observe_context_window(), dtype=np.float64)
        x = _epoch_feat_mean(ctx)

        # full-info: rewards for all actions at this round are just p[a]
        p = np.asarray(env.P[env.round], float)  # (K,)

        ctx_wins.append(ctx)
        Xfeat.append(x)
        Y.append(p)

        env.step_time()

    if len(Y) == 0:
        return [], np.zeros((0, 0)), np.zeros((0, env.K))

    return ctx_wins, np.stack(Xfeat, axis=0), np.stack(Y, axis=0)


# ---------------------- stage 1: model selection (simple CV) ----------------------

def _select_lin_lam(X, Y, lam_grid, val_steps=200):
    n = len(X)
    n_va = min(val_steps, max(1, n // 4))
    n_tr = n - n_va
    Xtr, Ytr = X[:n_tr], Y[:n_tr]
    Xva, Yva = X[n_tr:], Y[n_tr:]

    best = None
    for lam in lam_grid:
        Yhat = _ridge_fit_predict(Xtr, Ytr, Xva, lam=lam)
        mse = float(np.mean((Yhat - Yva) ** 2))
        if best is None or mse < best[0]:
            best = (mse, lam)
    return float(best[1])

def _select_sig_N_lam(ctx_wins, Y, K, sig_type, N_grid, lam_grid, val_steps=200):
    n = len(ctx_wins)
    n_va = min(val_steps, max(1, n // 4))
    n_tr = n - n_va

    best = None
    d_ctx = ctx_wins[0].shape[1]

    for N in N_grid:
        tmp = sigUCB(
            n_actions=1,
            lm=1.0,
            gamma=1.0,
            N=int(N),
            d=int(d_ctx),     # time_augment adds 1 dimension
            sigType=sig_type,
            log_norm=False,
            fit_intercept=True,
            seed=0,
            prune_end_time=True,
            time_augment_bool=True,
        )

        Phi = []
        for w in ctx_wins:
            f = tmp.sig_feat(np.ascontiguousarray(w, dtype=np.float64))
            Phi.append(np.asarray(f, float).ravel())
        Phi = np.stack(Phi, axis=0)

        Xtr, Ytr = Phi[:n_tr], Y[:n_tr]
        Xva, Yva = Phi[n_tr:], Y[n_tr:]

        for lam in lam_grid:
            Yhat = _ridge_fit_predict(Xtr, Ytr, Xva, lam=lam)
            mse = float(np.mean((Yhat - Yva) ** 2))
            if best is None or mse < best[0]:
                best = (mse, int(N), float(lam))

    _, Nbest, lambest = best
    return int(Nbest), float(lambest)

def _select_kernel_sigma_lam(X, Y, sigma_grid, lam_grid, val_steps=200):
    n = len(X)
    n_va = min(val_steps, max(1, n // 4))
    n_tr = n - n_va
    Xtr, Ytr = X[:n_tr], Y[:n_tr]
    Xva, Yva = X[n_tr:], Y[n_tr:]

    best = None
    for sigma in sigma_grid:
        for lam in lam_grid:
            Yhat = _krr_fit_predict_rbf(Xtr, Ytr, Xva, sigma=float(sigma), lam=float(lam))
            mse = float(np.mean((Yhat - Yva) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(sigma), float(lam))
    _, sbest, lbest = best
    return float(sbest), float(lbest)


# ------------------------- Stage-2: tune gamma via offline replay regret -------------------------

def _replay_lin_ucb(X, Y, K, lam, gamma, seed=0):
    alg = linUCB(
        n_actions=K,
        lm=float(lam),
        gamma=float(gamma),
        n=int(X.shape[1]),
        log_norm=False,
        fit_intercept=True,
        seed=int(seed),
    )
    cum = 0.0
    for t in range(len(X)):
        ctx = np.ascontiguousarray(X[t], float)[:, None]   # (d,1)
        a = alg.select(ctx)
        r = float(Y[t, a])
        oracle = float(np.max(Y[t]))
        alg.update(ctx, a, r)
        cum += (oracle - r)
    return cum

def _replay_sig_ucb(ctx_wins, Y, K, sig_type, N, lam, gamma, seed=0):
    d_ctx = ctx_wins[0].shape[1]
    alg = sigUCB(
        n_actions=K,
        lm=float(lam),
        gamma=float(gamma),
        N=int(N),
        d=int(d_ctx),
        sigType=sig_type,
        log_norm=False,
        fit_intercept=True,
        seed=int(seed),
        prune_end_time=True,
        time_augment_bool=True,
    )
    cum = 0.0
    for t in range(len(ctx_wins)):
        ctx = np.ascontiguousarray(ctx_wins[t], np.float64)
        a = alg.select(ctx)
        r = float(Y[t, a])
        oracle = float(np.max(Y[t]))
        alg.update(ctx, a, r)
        cum += (oracle - r)
    return cum

def _replay_kernel_ucb(X, Y, K, sigma, lam, gamma, seed=0):
    alg = KernelUCB(
        n_actions=K,
        lm=float(lam),
        gamma=float(gamma),
        kernel="rbf",
        sigma=float(sigma),
        max_points=800,
        seed=int(seed),
    )
    cum = 0.0
    for t in range(len(X)):
        x = np.ascontiguousarray(X[t], float)
        a = alg.select(x)
        r = float(Y[t, a])
        oracle = float(np.max(Y[t]))
        alg.update(x, a, r)
        cum += (oracle - r)
    return cum

def _replay_amm_sig_ucb(ctx_wins, Y, K, sig_type, N, sigma, delta, B, seed=0):
    d_ctx = ctx_wins[0].shape[1]
    alg = ammSigUCB(
        n_actions=K,
        N=int(N),
        d=int(d_ctx),
        sigma=float(sigma),
        delta=float(delta),
        B=float(B),
        sigType=sig_type,
        log_norm=False,
        fit_intercept=True,
        seed=int(seed),
        prune_end_time=True,
        time_augment_bool=True,
    )
    cum = 0.0
    for t in range(len(ctx_wins)):
        ctx = np.ascontiguousarray(ctx_wins[t], np.float64)
        a = alg.select(ctx)
        r = float(Y[t, a])
        oracle = float(np.max(Y[t]))
        alg.update(ctx, a, r)
        cum += (oracle - r)
    return cum


def _tune_gamma(fn, gamma_grid):
    best = None
    for g in gamma_grid:
        reg = fn(float(g))
        if best is None or reg < best[0]:
            best = (reg, float(g))
    return float(best[1])


# ------------------------- one run on one record -------------------------

def _rolling_metric(y_true, y_pred, window=200, metric="weighted_f1"):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out = np.full(len(y_true), np.nan, dtype=float)

    for t in range(len(y_true)):
        s = max(0, t - window + 1)
        yt = y_true[s:t+1]
        yp = y_pred[s:t+1]

        if metric == "accuracy":
            out[t] = accuracy_score(yt, yp)
        elif metric == "macro_f1":
            out[t] = f1_score(yt, yp, average="macro", zero_division=0)
        elif metric == "weighted_f1":
            out[t] = f1_score(yt, yp, average="weighted", zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return out


def run_one_sleep_record(
    record,
    seed: int,
    sig_type: str = "sig",
    sig_depth: int = 3,
    warmup_epochs: int = 200,     # ~100 min w/ 30s epochs
    do_model_select: bool = True,
    T_cut: int = 1000,
    eval_last_n: int = 200,
):
    K = record.P.shape[1]

    # truncate record views
    X_cut = record.X[: (T_cut + warmup_epochs) * record.epoch_len, :]
    P_cut = record.P[: (T_cut + warmup_epochs), :]

    env = SleepBandit(
        sleep_matrix=X_cut,
        stage_matrix=P_cut,
        stage_order=record.stage_order,
        epoch_len_samples=record.epoch_len,
        seed=seed,
    )

    # ---------- warmup (full-info) ----------
    ctx_wins, Xfeat, Y = _collect_warmup_fullinfo(env, warmup_epochs=warmup_epochs)

    # If warmup too short, fall back
    if len(Y) < 50:
        lin_lam = 1.0
        sig_N, sig_lam = sig_depth, 1.0
        ker_sigma, ker_lam = 1.0, 1.0
        lin_gamma = sig_gamma = ker_gamma = 1.0
        amm_sigma = 1.0
    else:
        lam_grid   = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        N_grid     = [1, 2]
        sigma_grid = [0.1, 0.2, 0.5, 1.0, 2.0]

        # stage-1 selection (supervised)
        if do_model_select:
            lin_lam = _select_lin_lam(Xfeat, Y, lam_grid=lam_grid, val_steps=min(200, len(Y)//4))
            sig_N, sig_lam = _select_sig_N_lam(ctx_wins, Y, K=K, sig_type=sig_type,
                                               N_grid=N_grid, lam_grid=lam_grid,
                                               val_steps=min(200, len(Y)//4))
            ker_sigma, ker_lam = _select_kernel_sigma_lam(Xfeat, Y, sigma_grid=sigma_grid, lam_grid=lam_grid,
                                                          val_steps=min(200, len(Y)//4))
        else:
            lin_lam = 1.0
            sig_N, sig_lam = sig_depth, 1.0
            ker_sigma, ker_lam = 1.0, 1.0

        # stage-2 gamma tuning via offline replay regret (using same warmup data)
        gamma_grid = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]

        lin_gamma = _tune_gamma(lambda g: _replay_lin_ucb(Xfeat, Y, K, lin_lam, g, seed=seed), gamma_grid)
        sig_gamma = _tune_gamma(lambda g: _replay_sig_ucb(ctx_wins, Y, K, sig_type, sig_N, sig_lam, g, seed=seed), gamma_grid)
        ker_gamma = _tune_gamma(lambda g: _replay_kernel_ucb(Xfeat, Y, K, ker_sigma, ker_lam, g, seed=seed), gamma_grid)

        amm_sigma_grid = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
        amm_sigma = _tune_gamma(
            lambda s: _replay_amm_sig_ucb(ctx_wins, Y, K, sig_type, sig_N, s, 0.05, 1.0, seed=seed),
            amm_sigma_grid,
        )
        
    # print(
    #     f"[seed={seed}] chosen signature depth N={sig_N} | "
    #     f"sig_lam={sig_lam} | sig_gamma={sig_gamma}"
    # )
        
    # ---------- init learners ----------
    d_ctx = int(X_cut.shape[1])
    d_lin = int(Xfeat.shape[1]) if len(Y) > 0 else d_ctx

    sig_alg = sigUCB(
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
        time_augment_bool=True,
    )
    lin_alg = linUCB(
        n_actions=K,
        lm=float(lin_lam),
        gamma=float(lin_gamma),
        n=int(d_lin),
        log_norm=False,
        fit_intercept=True,
        seed=seed,
    )
    ker_alg = KernelUCB(
        n_actions=K,
        lm=float(ker_lam),
        gamma=float(ker_gamma),
        kernel="rbf",
        sigma=float(ker_sigma),
        max_points=800,
        seed=seed,
    )

    # AMM-sigUCB: same signature features + depth as sigUCB, but the confidence
    # radius comes from the AMM analytic bound (no gamma to tune). sigma is a
    # regularization / noise scale; delta is the confidence level; B bounds ||theta*||.
    # amm_alg = ammSigUCB(
    #     n_actions=K,
    #     N=int(sig_N),
    #     d=int(d_ctx),
    #     sigma=float(amm_sigma),
    #     delta=0.05,
    #     B=1.0,
    #     sigType=sig_type,
    #     log_norm=False,
    #     fit_intercept=True,
    #     seed=seed,
    #     prune_end_time=True,
    #     time_augment_bool=True,
    # )

    L_ctx = int(ctx_wins[0].shape[0]) if len(ctx_wins) > 0 else int(env.observe_context_window().shape[0])
    lin_variants = _build_lin_variants(
        K=K, lam=lin_lam, gamma=lin_gamma,
        L=L_ctx, d=d_ctx, rff_sigma=ker_sigma, rff_D=64, seed=seed,
    )

    # ---------- online bandit phase ----------
    sig_reg, lin_reg, ker_reg = [], [], []
    # amm_reg = []
    sig_cum = lin_cum = ker_cum = 0.0
    # amm_cum = 0.0

    variant_reg = {name: [] for name in lin_variants}
    variant_cum = {name: 0.0 for name in lin_variants}
    y_variant = {name: [] for name in lin_variants}

    y_true = []
    y_sig, y_lin, y_ker = [], [], []
    # y_amm = []

    while env.next_round():
        ctx = np.ascontiguousarray(env.observe_context_window(), dtype=np.float64)
        x = _epoch_feat_mean(ctx)                     # (d_lin,)
        ctx_lin = x[:, None]                          # (d_lin, 1)

        a_sig = sig_alg.select(ctx)
        a_lin = lin_alg.select(ctx_lin)
        a_ker = ker_alg.select(x)
        # a_amm = amm_alg.select(ctx)

        oracle_r, _ = env.oracle()

        # true label at this round
        y_t = int(np.argmax(env.P[env.round]))
        y_true.append(y_t)

        # predicted labels = selected actions
        y_sig.append(int(a_sig))
        y_lin.append(int(a_lin))
        y_ker.append(int(a_ker))
        # y_amm.append(int(a_amm))

        r_sig = env.reward(a_sig); sig_alg.update(ctx, a_sig, float(r_sig))
        r_lin = env.reward(a_lin); lin_alg.update(ctx_lin, a_lin, float(r_lin))
        r_ker = env.reward(a_ker); ker_alg.update(x, a_ker, float(r_ker))
        # r_amm = env.reward(a_amm); amm_alg.update(ctx, a_amm, float(r_amm))

        sig_cum += float(oracle_r - r_sig)
        lin_cum += float(oracle_r - r_lin)
        ker_cum += float(oracle_r - r_ker)
        # amm_cum += float(oracle_r - r_amm)

        sig_reg.append(sig_cum)
        lin_reg.append(lin_cum)
        ker_reg.append(ker_cum)
        # amm_reg.append(amm_cum)

        for name, algo in lin_variants.items():
            a_v = algo.select(ctx)
            r_v = env.reward(a_v)
            algo.update(ctx, a_v, float(r_v))
            variant_cum[name] += float(oracle_r - r_v)
            variant_reg[name].append(variant_cum[name])
            y_variant[name].append(int(a_v))

        env.step_time()

    n_eval = min(eval_last_n, len(y_true))
    yt = np.asarray(y_true[-n_eval:], dtype=int)

    def _metrics(yp_list):
        yp = np.asarray(yp_list[-n_eval:], dtype=int)
        return {
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(yt, yp, labels=np.arange(K)),
            "report": classification_report(yt, yp, labels=np.arange(K), zero_division=0),
        }

    clf_metrics = {
        "sigUCB": _metrics(y_sig),
        "linUCB": _metrics(y_lin),
        "kernelUCB": _metrics(y_ker),
        "n_eval": int(n_eval),
    }
    for name in y_variant:
        clf_metrics[name] = _metrics(y_variant[name])

    rolling_window = n_eval

    metric_traj = {
        "sigUCB_acc": _rolling_metric(y_true, y_sig, window=rolling_window, metric="accuracy"),
        "linUCB_acc": _rolling_metric(y_true, y_lin, window=rolling_window, metric="accuracy"),
        "kernelUCB_acc": _rolling_metric(y_true, y_ker, window=rolling_window, metric="accuracy"),

        "sigUCB_weighted_f1": _rolling_metric(y_true, y_sig, window=rolling_window, metric="weighted_f1"),
        "linUCB_weighted_f1": _rolling_metric(y_true, y_lin, window=rolling_window, metric="weighted_f1"),
        "kernelUCB_weighted_f1": _rolling_metric(y_true, y_ker, window=rolling_window, metric="weighted_f1"),
    }
    for name in y_variant:
        metric_traj[f"{name}_acc"] = _rolling_metric(y_true, y_variant[name], window=rolling_window, metric="accuracy")
        metric_traj[f"{name}_weighted_f1"] = _rolling_metric(y_true, y_variant[name], window=rolling_window, metric="weighted_f1")


    return (
        np.array(sig_reg), np.array(lin_reg), np.array(ker_reg),
        clf_metrics,
        metric_traj,
        {name: np.array(reg) for name, reg in variant_reg.items()},
        # np.array(amm_reg),
    )


# ------------------------- main: multiple runs, random record each run -------------------------

def main(
    cassette_dir: str = "tests/sleep_edf/sleep-cassette",
    pick_channels = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"),
    max_records: int | None = None,
    sig_type: str = "sig",
    sig_depth: int = 3,
    warmup_epochs: int = 200,
    seed0: int = 123,
    out_dir: str = "experiments/figs/sleep_edf",
    tag: str | None = None,
):
    T_CUT = 1000
    
    recs = load_sleep_cassette(
        cassette_dir,
        pick_channels=list(pick_channels),
        max_records=max_records,
        intersect_time=True,
        trim_leading_w=True,
        min_sleep_run_sec=60.0,
        window_sec=30.0,
        target_fs=25.0,
    )
    if len(recs) == 0:
        raise RuntimeError(f"No records found in {cassette_dir}")

    # ---- filter to >= T_CUT + warmup rounds ----
    recs = [r for r in recs if r.P.shape[0] >= T_CUT + warmup_epochs]
    if len(recs) == 0:
        raise RuntimeError(f"No cassettes with >= {T_CUT + warmup_epochs} epochs found in {cassette_dir}")

    # ---- each run is one cassette ----
    n_runs = len(recs)
    seeds = np.arange(n_runs) + seed0

    runs = []
    for rec, s in tqdm(zip(recs, seeds), total=n_runs, desc="Sleep-EDF bandit"):
        runs.append(
            run_one_sleep_record(
                rec,
                seed=int(s),
                sig_type=sig_type,
                sig_depth=sig_depth,
                warmup_epochs=warmup_epochs,
                do_model_select=True,
                T_cut=T_CUT,
            )
        )

    # align lengths (truncate to min)
    sig_list = [r[0] for r in runs]
    lin_list = [r[1] for r in runs]
    ker_list = [r[2] for r in runs]
    metric_list = [r[3] for r in runs]
    traj_list = [r[4] for r in runs]
    variant_lists = {name: [r[5][name] for r in runs] for name in LIN_VARIANT_NAMES}
    # amm_list = [r[6] for r in runs]

    min_len = T_CUT

    all_sig = np.stack([x[:min_len] for x in sig_list], axis=0)
    all_lin = np.stack([x[:min_len] for x in lin_list], axis=0)
    all_ker = np.stack([x[:min_len] for x in ker_list], axis=0)
    # all_amm = np.stack([x[:min_len] for x in amm_list], axis=0)
    variant_stacked = {name: np.stack([x[:min_len] for x in arrs], 0) for name, arrs in variant_lists.items()}
    variant_med = {name: np.median(v, 0) for name, v in variant_stacked.items()}
    variant_p25 = {name: np.percentile(v, 25, 0) for name, v in variant_stacked.items()}
    variant_p75 = {name: np.percentile(v, 75, 0) for name, v in variant_stacked.items()}
    # amm_med = np.median(all_amm, 0)
    # amm_p25 = np.percentile(all_amm, 25, 0)
    # amm_p75 = np.percentile(all_amm, 75, 0)

    def _qstats(A):
        return np.median(A, 0), np.percentile(A, 25, 0), np.percentile(A, 75, 0)
    
    def _meanstd(A):
        mu = np.mean(A, axis=0)
        sd = np.std(A, axis=0, ddof=1)
        return mu, sd

    sig_med, sig_p25, sig_p75 = _qstats(all_sig)
    lin_med, lin_p25, lin_p75 = _qstats(all_lin)
    ker_med, ker_p25, ker_p75 = _qstats(all_ker)

    all_sig_acc = np.stack([m["sigUCB_acc"][:min_len] for m in traj_list], axis=0)
    all_lin_acc = np.stack([m["linUCB_acc"][:min_len] for m in traj_list], axis=0)
    all_ker_acc = np.stack([m["kernelUCB_acc"][:min_len] for m in traj_list], axis=0)

    all_sig_wf1 = np.stack([m["sigUCB_weighted_f1"][:min_len] for m in traj_list], axis=0)
    all_lin_wf1 = np.stack([m["linUCB_weighted_f1"][:min_len] for m in traj_list], axis=0)
    all_ker_wf1 = np.stack([m["kernelUCB_weighted_f1"][:min_len] for m in traj_list], axis=0)
    variant_wf1_stacked = {
        name: np.stack([m[f"{name}_weighted_f1"][:min_len] for m in traj_list], axis=0)
        for name in LIN_VARIANT_NAMES
    }
    variant_wf1_mean = {name: np.mean(v, 0) for name, v in variant_wf1_stacked.items()}
    variant_wf1_sd = {name: np.std(v, 0, ddof=1) for name, v in variant_wf1_stacked.items()}
    variant_wf1_med = {name: np.median(v, 0) for name, v in variant_wf1_stacked.items()}
    variant_wf1_p25 = {name: np.percentile(v, 25, 0) for name, v in variant_wf1_stacked.items()}
    variant_wf1_p75 = {name: np.percentile(v, 75, 0) for name, v in variant_wf1_stacked.items()}

    sig_acc_mean, sig_acc_sd = _meanstd(all_sig_acc)
    lin_acc_mean, lin_acc_sd = _meanstd(all_lin_acc)
    ker_acc_mean, ker_acc_sd = _meanstd(all_ker_acc)

    sig_wf1_mean, sig_wf1_sd = _meanstd(all_sig_wf1)
    lin_wf1_mean, lin_wf1_sd = _meanstd(all_lin_wf1)
    ker_wf1_mean, ker_wf1_sd = _meanstd(all_ker_wf1)

    sig_acc_med, sig_acc_p25, sig_acc_p75 = _qstats(all_sig_acc)
    lin_acc_med, lin_acc_p25, lin_acc_p75 = _qstats(all_lin_acc)
    ker_acc_med, ker_acc_p25, ker_acc_p75 = _qstats(all_ker_acc)

    sig_wf1_med, sig_wf1_p25, sig_wf1_p75 = _qstats(all_sig_wf1)
    lin_wf1_med, lin_wf1_p25, lin_wf1_p75 = _qstats(all_lin_wf1)
    ker_wf1_med, ker_wf1_p25, ker_wf1_p75 = _qstats(all_ker_wf1)

    method_stats = [
        ("sigUCB",     sig_med[-1], sig_p25[-1], sig_p75[-1]),
        # ("ammSigUCB",  amm_med[-1], amm_p25[-1], amm_p75[-1]),
        ("linUCB",     lin_med[-1], lin_p25[-1], lin_p75[-1]),
        ("kernelUCB",  ker_med[-1], ker_p25[-1], ker_p75[-1]),
    ] + [
        (name, variant_med[name][-1], variant_p25[name][-1], variant_p75[name][-1])
        for name in LIN_VARIANT_NAMES
    ]
    print(f"\n=== Sleep-EDF | runs={n_runs} — final cumulative regret ===")
    for name, med, p25, p75 in method_stats:
        print(f"  {name:14s} | median={med:10.3f} | p25={p25:10.3f} | p75={p75:10.3f}")

    os.makedirs(out_dir, exist_ok=True)
    tag_str = f"_{tag}" if tag else ""
    x = np.arange(min_len)

    # regret plot
    plt.figure(figsize=(8, 4.8))
    plt.plot(x, sig_med, label="sigUCB (median)")
    plt.fill_between(x, sig_p25, sig_p75, alpha=0.20)
    # plt.plot(x, amm_med, linestyle="-", label="AMM-sigUCB (median)")
    # plt.fill_between(x, amm_p25, amm_p75, alpha=0.18)
    plt.plot(x, lin_med, linestyle="--", label="linUCB (median)")
    plt.fill_between(x, lin_p25, lin_p75, alpha=0.15)
    plt.plot(x, ker_med, linestyle=":", label="kernelUCB (median)")
    plt.fill_between(x, ker_p25, ker_p75, alpha=0.12)
    for name in LIN_VARIANT_NAMES:
        plt.plot(x, variant_med[name], linestyle="-.", label=name)
        plt.fill_between(x, variant_p25[name], variant_p75[name], alpha=0.10)
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.title("Sleep Stage Bandit: Cumulative Regret")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"sleep_edf_regret_{sig_type}{tag_str}.png"), dpi=180)
    plt.close()

    # weight f1 plot
    plt.figure(figsize=(8, 4.8))
    plt.plot(x, sig_wf1_mean, label="sigUCB (mean)")
    lo = np.clip(sig_wf1_mean - sig_wf1_sd, 0.0, 1.0)
    hi = np.clip(sig_wf1_mean + sig_wf1_sd, 0.0, 1.0)
    plt.fill_between(x, lo, hi, alpha=0.20)

    plt.plot(x, lin_wf1_mean, linestyle="--", label="linUCB (mean)")
    lo = np.clip(lin_wf1_mean - lin_wf1_sd, 0.0, 1.0)
    hi = np.clip(lin_wf1_mean + lin_wf1_sd, 0.0, 1.0)
    plt.fill_between(x, lo, hi, alpha=0.15)

    plt.plot(x, ker_wf1_mean, linestyle=":", label="kernelUCB (mean)")
    lo = np.clip(ker_wf1_mean - ker_wf1_sd, 0.0, 1.0)
    hi = np.clip(ker_wf1_mean + ker_wf1_sd, 0.0, 1.0)
    plt.fill_between(x, lo, hi, alpha=0.12)

    for name in LIN_VARIANT_NAMES:
        plt.plot(x, variant_wf1_mean[name], linestyle="-.", label=f"{name} (mean)")
        lo = np.clip(variant_wf1_mean[name] - variant_wf1_sd[name], 0.0, 1.0)
        hi = np.clip(variant_wf1_mean[name] + variant_wf1_sd[name], 0.0, 1.0)
        plt.fill_between(x, lo, hi, alpha=0.10)

    plt.xlabel("Rounds")
    plt.ylabel("Rolling Weighted F1")
    plt.title("Sleep-EDF: Rolling Weighted F1 (window = 200)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sleep_edf_weighted_f1_rolling_{sig_type}{tag_str}.png"), dpi=180)
    plt.close()

    # accuracy plot
    plt.figure(figsize=(8, 4.8))
    plt.plot(x, sig_acc_mean, label="sigUCB (mean)")
    lo = np.clip(sig_acc_mean - sig_acc_sd, 0.0, 1.0)
    hi = np.clip(sig_acc_mean + sig_acc_sd, 0.0, 1.0)
    plt.fill_between(x, lo, hi, alpha=0.20)

    plt.plot(x, lin_acc_mean, linestyle="--", label="linUCB (mean)")
    lo = np.clip(lin_acc_mean - lin_acc_sd, 0.0, 1.0)
    hi = np.clip(lin_acc_mean + lin_acc_sd, 0.0, 1.0)
    plt.fill_between(x, lo, hi, alpha=0.15)

    plt.plot(x, ker_acc_mean, linestyle=":", label="kernelUCB (mean)")
    lo = np.clip(ker_acc_mean - ker_acc_sd, 0.0, 1.0)
    hi = np.clip(ker_acc_mean + ker_acc_sd, 0.0, 1.0)
    plt.fill_between(x, lo, hi, alpha=0.12)

    plt.xlabel("Rounds")
    plt.ylabel("Rolling Accuracy")
    plt.title("Sleep-EDF: Rolling Accuracy (window = 200)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sleep_edf_accuracy_rolling_{sig_type}{tag_str}.png"), dpi=180)
    plt.close()

    # weighted F1 plot
    plt.figure(figsize=(8, 4.8))

    plt.plot(x, sig_wf1_med, label="sigUCB (median)")
    plt.fill_between(x, sig_wf1_p25, sig_wf1_p75, alpha=0.20)

    plt.plot(x, lin_wf1_med, linestyle="-.", label="linUCB (median)")
    plt.fill_between(x, lin_wf1_p25, lin_wf1_p75, alpha=0.15)

    plt.plot(x, ker_wf1_med, linestyle="-", label="kernelUCB (median)")
    plt.fill_between(x, ker_wf1_p25, ker_wf1_p75, alpha=0.12)

    for name in LIN_VARIANT_NAMES:
        plt.plot(x, variant_wf1_med[name], linestyle=":", label=f"{name} (median)")
        plt.fill_between(x, variant_wf1_p25[name], variant_wf1_p75[name], alpha=0.10)

    plt.xlabel("Rounds")
    plt.ylabel("Rolling Weighted F1")
    plt.title("Sleep-EDF: Rolling Weighted F1 (window = 200)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"sleep_edf_weighted_f1_rolling_{sig_type}{tag_str}_q.png"),
        dpi=180
    )
    plt.close()

    # accuracy plot
    plt.figure(figsize=(8, 4.8))

    plt.plot(x, sig_acc_med, label="sigUCB (median)")
    plt.fill_between(x, sig_acc_p25, sig_acc_p75, alpha=0.20)

    plt.plot(x, lin_acc_med, linestyle="-.", label="linUCB (median)")
    plt.fill_between(x, lin_acc_p25, lin_acc_p75, alpha=0.15)

    plt.plot(x, ker_acc_med, linestyle="-", label="kernelUCB (median)")
    plt.fill_between(x, ker_acc_p25, ker_acc_p75, alpha=0.12)

    plt.xlabel("Rounds")
    plt.ylabel("Rolling Accuracy")
    plt.title("Sleep-EDF: Rolling Accuracy (window = 200)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"sleep_edf_accuracy_rolling_{sig_type}{tag_str}_q.png"),
        dpi=180
    )
    plt.close()
    
    print(f"\nClassification metrics over last {metric_list[0]['n_eval']} rounds:")
    for name in ["sigUCB", "linUCB", "kernelUCB", *LIN_VARIANT_NAMES]:
        acc = np.array([m[name]["accuracy"] for m in metric_list], dtype=float)
        mf1 = np.array([m[name]["macro_f1"] for m in metric_list], dtype=float)
        wf1 = np.array([m[name]["weighted_f1"] for m in metric_list], dtype=float)

        print(
            f"{name:14s} | "
            f"acc median={np.median(acc):.3f} "
            f"(p25={np.percentile(acc,25):.3f}, p75={np.percentile(acc,75):.3f}) | "
            f"macro-F1 median={np.median(mf1):.3f} "
            f"(p25={np.percentile(mf1,25):.3f}, p75={np.percentile(mf1,75):.3f}) | "
            f"weighted-F1 median={np.median(wf1):.3f} "
            f"(p25={np.percentile(wf1,25):.3f}, p75={np.percentile(wf1,75):.3f})"
        )

    # ---------------- save classification summary ----------------
    summary = {
        "n_runs": int(n_runs),
        "n_eval_last_rounds": int(metric_list[0]["n_eval"]),
        "methods": {}
    }

    for name in ["sigUCB", "linUCB", "kernelUCB", *LIN_VARIANT_NAMES]:
        acc = np.array([m[name]["accuracy"] for m in metric_list], dtype=float)
        mf1 = np.array([m[name]["macro_f1"] for m in metric_list], dtype=float)
        wf1 = np.array([m[name]["weighted_f1"] for m in metric_list], dtype=float)
        cm  = np.sum([m[name]["confusion_matrix"] for m in metric_list], axis=0)

        summary["methods"][name] = {
            "accuracy_per_record": acc.tolist(),
            "macro_f1_per_record": mf1.tolist(),
            "weighted_f1_per_record": wf1.tolist(),
            "accuracy_median": float(np.median(acc)),
            "accuracy_p25": float(np.percentile(acc, 25)),
            "accuracy_p75": float(np.percentile(acc, 75)),
            "macro_f1_median": float(np.median(mf1)),
            "macro_f1_p25": float(np.percentile(mf1, 25)),
            "macro_f1_p75": float(np.percentile(mf1, 75)),
            "weighted_f1_median": float(np.median(wf1)),
            "weighted_f1_p25": float(np.percentile(wf1, 25)),
            "weighted_f1_p75": float(np.percentile(wf1, 75)),
            "confusion_matrix_sum": cm.tolist(),
        }

    json_path = os.path.join(out_dir, f"sleep_edf_classification_{sig_type}_N{sig_depth}{tag_str}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved classification summary to: {json_path}")

if __name__ == "__main__":
    main(
        cassette_dir="tests/sleep_edf/sleep-cassette",
        sig_type="sig",
        warmup_epochs=200,
        seed0=123,
    )