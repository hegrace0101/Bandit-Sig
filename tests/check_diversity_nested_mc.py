"""Nested Monte Carlo estimate of the action-conditional feature second moment
for Assumption 5.1(ii):

    E[ x̄_t x̄_t^T | F̃_{t-1}, A_t = a ]  >=  rho * I.

For each (process, sig depth):
  For each of H histories:
    - Sample a base trajectory
    - Run sigUCB with generic gamma on this trajectory up to round max(T)
    - At each round t in T:
        * Snapshot the frozen bandit state
        * Sample M independent continuations of the context process starting
          from the round-(t-1) endpoint (Markov continuation)
        * For each continuation: compute sig-feature x̄_t^(j) and frozen bandit
          action A_t^(j) via bandit.select() (no update)
        * Group continuations by action, compute empirical Σ̂_{a,t} and λ_min
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sigbandit.algos.ucb import sigUCB
from sigbandit.envs.context import bm, gbm, ou_stationary
from sigbandit.envs.reward import min_max_reward
from sigbandit.utils.signature import (
    time_augment, sig_list_levels, prune_sig_levels_drop_last_channel, compute_signature,
)


class sigUCBExtra(sigUCB):
    """sigUCB with configurable "extra" feature appended to the pruned signature.

    extra_mode:
      'raw_xT'  : append xT (raw terminal value)         [same as vanilla sigUCB]
      'log_xT'  : append log(xT / x0)                    [scale-invariant]
      'no_xT'   : append nothing (signature only)
    """
    def __init__(self, *, extra_mode: str = 'raw_xT', **kwargs):
        self.extra_mode = extra_mode
        super().__init__(**kwargs)
        if extra_mode == 'no_xT':
            # Rebuild state without the raw_d extra dims.
            new_dim = self._dim - self.raw_d
            self._V     = [self.lm * np.eye(new_dim) for _ in range(self.K)]
            self._Vinv  = [np.eye(new_dim) / self.lm for _ in range(self.K)]
            self._b     = [np.zeros(new_dim) for _ in range(self.K)]
            self._theta = [np.zeros(new_dim) for _ in range(self.K)]
            self._dim = new_dim

    def sig_feat(self, context):
        context = np.asarray(context, dtype=float)
        x0 = context[0].ravel()
        xT = context[-1].ravel()

        if self.log_norm:
            context = np.log(context / context[0])

        if self.time_augment_bool:
            path_for_sig = time_augment(context)
        else:
            path_for_sig = context

        if self.sigType == 'sig' and self.prune_end_time:
            levels = sig_list_levels(path_for_sig, self.N)
            sig = prune_sig_levels_drop_last_channel(
                levels, d=path_for_sig.shape[1], time_index=self.time_index,
            )
        else:
            sig = compute_signature(path_for_sig, self.N, self.sigType)

        if self.extra_mode == 'raw_xT':
            extra = xT
        elif self.extra_mode == 'log_xT':
            extra = np.log(xT / x0)
        elif self.extra_mode == 'no_xT':
            extra = np.zeros(0)
        else:
            raise ValueError(f"Unknown extra_mode {self.extra_mode!r}")

        feat = np.concatenate([sig, extra])
        if self.fit_intercept:
            feat = np.concatenate([feat, np.array([1.0])])
        return feat

# ------------------------- config -------------------------
PROCESSES = ["bm", "gbm", "ou_s"]
DEPTHS = [1, 2, 3, 4]
T_CHECK = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
H = 20               # histories
M = 200              # continuations per (history, round)
K = 2                # arms (min_max_reward: max vs min)
WINDOW = 1000        # per-round window length (samples)
RAW_D = 1
LAM = 1.0
GAMMA = 1.0          # generic exploration coefficient
DT = 1.0 / WINDOW    # per-step dt within a round (matches bm/gbm/ou default T=1 per round)


def _make_bandit_probe(depth, log_norm, seed, extra_mode="raw_xT"):
    return sigUCBExtra(
        n_actions=K, lm=LAM, gamma=GAMMA, N=depth, d=RAW_D,
        sigType="sig", log_norm=log_norm, fit_intercept=False,
        seed=seed, prune_end_time=True, time_augment_bool=True,
        extra_mode=extra_mode,
    )


def _generate_base(process, T_max, seed):
    """Sample the base trajectory for one history (fixes F̃)."""
    np.random.seed(seed)
    N_total = T_max * WINDOW
    if process == "bm":
        return bm(T_max, N_total)
    if process == "gbm":
        return gbm(T_max, N_total)
    if process == "ou_s":
        return ou_stationary(T_max, N_total)
    raise ValueError(f"Unknown process {process!r}")


def _sample_continuation(process, x_start, L, rng):
    """Sample one continuation window of length L starting from x_start.

    Markov continuation: uses same dt/mu/sigma as the base process.
    """
    if process == "bm":
        # dX = dW, sigma=1
        dW = rng.normal(0.0, np.sqrt(DT), size=L)
        return (x_start + np.cumsum(dW)).reshape(-1, 1)

    if process == "gbm":
        # dX = X*(mu - 0.5*sigma^2)*dt + sigma*X*dW, defaults mu=0.1, sigma=1
        mu, sigma = 0.1, 1.0
        dW = rng.normal(0.0, np.sqrt(DT), size=L)
        log_incr = (mu - 0.5 * sigma ** 2) * DT + sigma * dW
        log_x = np.log(x_start) + np.cumsum(log_incr)
        return np.exp(log_x).reshape(-1, 1)

    if process == "ou_s":
        # dX = theta*(mu - X)*dt + sigma*dW, defaults mu=0, theta=1, sigma=1
        mu, theta, sigma = 0.0, 1.0, 1.0
        x = np.zeros(L, dtype=float)
        prev = float(x_start)
        for i in range(L):
            dW = rng.normal(0.0, np.sqrt(DT))
            prev = prev + theta * (mu - prev) * DT + sigma * dW
            x[i] = prev
        return x.reshape(-1, 1)

    raise ValueError(f"Unknown process {process!r}")


def run_one_process_depth(process, depth, extra_mode="raw_xT", log_norm_override=None):
    """For one (process, depth), run H histories and record per-action λ_min at
    each check round.

    Returns:
      lambda_min[a][t] : list of length H (nan if action a never chosen)
      n_a[a][t]        : list of length H (count of continuations landing in a)
    """
    log_norm = (process == "gbm") if log_norm_override is None else log_norm_override
    T_max = max(T_CHECK)

    lambda_min = {a: {t: [] for t in T_CHECK} for a in range(K)}
    n_a = {a: {t: [] for t in T_CHECK} for a in range(K)}

    for h in tqdm(range(H), desc=f"{process} N={depth}"):
        seed = 1000 + h
        X = _generate_base(process, T_max, seed)

        bandit = _make_bandit_probe(depth, log_norm, seed=seed, extra_mode=extra_mode)

        for t in range(1, T_max + 1):
            ctx = X[(t - 1) * WINDOW : t * WINDOW]

            # snapshot check: state is F̃_{t-1} because we haven't updated round t yet
            if t in T_CHECK:
                if t == 1:
                    x_start = float(X[0].item())
                else:
                    x_start = float(X[(t - 1) * WINDOW - 1].item())

                cont_rng = np.random.default_rng(seed * 100_003 + t)

                feats = []
                acts = []
                for _ in range(M):
                    cont = _sample_continuation(process, x_start, WINDOW, cont_rng)
                    feat = bandit.sig_feat(cont)
                    a = bandit.select(cont)
                    feats.append(feat)
                    acts.append(a)

                feats = np.asarray(feats, dtype=float)   # (M, dim)
                acts = np.asarray(acts, dtype=int)       # (M,)

                for a in range(K):
                    mask = acts == a
                    n = int(mask.sum())
                    n_a[a][t].append(n)
                    if n == 0:
                        lambda_min[a][t].append(np.nan)
                        continue
                    Fa = feats[mask]                    # (n, dim)
                    Sigma_a = (Fa.T @ Fa) / float(n)
                    eigs = np.linalg.eigvalsh(Sigma_a)
                    lambda_min[a][t].append(float(eigs[0]))

            # process round t on the BASE trajectory: select+update
            a_real = bandit.select(ctx)
            r_real = min_max_reward(a_real, ctx)
            bandit.update(ctx, a_real, float(r_real))

    return lambda_min, n_a


def plot_process(process, lambda_min_per_depth, n_a_per_depth, out_dir, tag=""):
    """Save one figure per action so each can be included independently."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]
    action_label = {0: "max", 1: "min"}

    os.makedirs(out_dir, exist_ok=True)
    tag_str = f"_{tag}" if tag else ""

    for a in range(K):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, depth in enumerate(DEPTHS):
            lm_dict = lambda_min_per_depth[depth][a]     # {t: [λ_min per history]}
            t_vals = sorted(lm_dict.keys())
            arr = np.array([lm_dict[t] for t in t_vals], dtype=float)  # (n_t, H)
            medians = np.nanmedian(arr, axis=1)
            p25 = np.nanpercentile(arr, 25, axis=1)
            p75 = np.nanpercentile(arr, 75, axis=1)

            ls = linestyles[i % len(linestyles)]
            ax.plot(t_vals, medians, color=colors[i], linestyle=ls, marker="o",
                    label=f"N={depth}")
            ax.fill_between(t_vals, p25, p75, color=colors[i], alpha=0.15)

        ax.set_xlabel("Round t")
        ax.set_ylabel(r"$\lambda_{\min}(\widehat{\Sigma}_{a,t})$")
        lbl = action_label.get(a, str(a))
        proc_label = {"bm": "BM", "gbm": "GBM", "ou_s": "OU"}.get(process, process)
        ax.set_title(f"{proc_label}, action a = {lbl}")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()

        fname = f"sig_diversity_nested_mc_{process}{tag_str}_a{a}.png"
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved figure to {out_path}")

    # short numerical summary
    print(f"[{process}] median λ_min(Σ̂_{{a,t}}) across histories:")
    print(f"  {'N':>2s} {'a':>2s}  " + "  ".join(f"t={t:>3d}" for t in sorted(T_CHECK)))
    for depth in DEPTHS:
        for a in range(K):
            lm_dict = lambda_min_per_depth[depth][a]
            vals = [np.nanmedian(lm_dict[t]) for t in sorted(T_CHECK)]
            avg_n = np.mean([np.mean(n_a_per_depth[depth][a][t]) for t in sorted(T_CHECK)])
            row = f"  {depth:>2d} {a:>2d}  " + "  ".join(f"{v:>7.2e}" for v in vals)
            row += f"   (avg n_a={avg_n:.1f}/{M})"
            print(row)


def main():
    out_dir = "experiments/figs/diversity_assumption"
    # BM and OU: default sigUCB feature (log_norm=False, raw_xT appended).
    # GBM: log_norm=True (matches sigUCB) but with the extra raw-xT feature
    # dropped ('no_xT'). Under GBM the base trajectory drifts multiplicatively,
    # so the raw terminal xT of the round-t window has variance proportional
    # to x_start^2 conditioned on the history. That column of Σ̂_{a,t} loses
    # variance as the history-dependent x_start decays, driving λ_min → 0.
    # Dropping xT and keeping only the log-normalized signature avoids that
    # collapse and gives a stable λ_min across rounds.
    for process in PROCESSES:
        extra_mode = "no_xT" if process == "gbm" else "raw_xT"
        lambda_min_per_depth = {}
        n_a_per_depth = {}
        for depth in DEPTHS:
            lm, n_a = run_one_process_depth(process, depth, extra_mode=extra_mode)
            lambda_min_per_depth[depth] = lm
            n_a_per_depth[depth] = n_a
        plot_process(process, lambda_min_per_depth, n_a_per_depth, out_dir)


if __name__ == "__main__":
    main()
