# experiments/run_ucb_many.py
import numpy as np
import matplotlib.pyplot as plt

from sigbandit.algos.ucb import sigUCB, linUCB
from sigbandit.envs.bandit import Bandit
from sigbandit.envs.context import ContextualProcess, bm, ou, gbm, unit_ball
from sigbandit.envs.reward import min_max_reward, sum_a_reward, linear_reward

# ------------------------- config -------------------------
T = 100                          # rounds per run
K = 2                            # actions
raw_d = 1                        # raw channels (before time-augment)
N_total = 100 * 1000             # total points in full series
window = N_total // T            # points per round
sig_depth = 4                    # signature depth
sig_type = "sig"                 # 'sig' or 'logsig'
lam = 5.0                        # ridge λ
beta = 2.0                       # exploration β
fit_intercept = False            # append constant 1.0 to features

n_runs = 100                     # number of independent runs
seeds = np.arange(n_runs) + 123  # reproducible seeds

# channels seen by the signature after time-augmentation
channels_after_time = raw_d + 1

# ------------------------- runner -------------------------
def run_one(seed: int) -> np.ndarray:
	rng = np.random.default_rng(seed)

	# make a Brownian motion path (or your own process)
	X = bm(T, N_total)  # shape (N_total, raw_d)
	# X = ou(T, N_total)
	# X = gbm(T, N_total)
	# X = unit_ball(T, N_total)
	log_norm = False

	# context process: windows of length `window`
	context = ContextualProcess(X, T, window)

	# bandit with your reward function
	bandit = Bandit(context, min_max_reward, K, T)
	# bandit = Bandit(context, linear_reward, K, T)
	# bandit = Bandit(context, sum_a_reward, K, T)

	# UCB over signature features
	sig_ucb = sigUCB(
		n_actions=K,
		lm=lam,
		beta=beta,
		N=sig_depth,
		d=channels_after_time,   # includes time channel
		sigType=sig_type,
		log_norm=log_norm,
		fit_intercept=fit_intercept,
	)
	
	# linUCB
	lin_ucb = linUCB(
		n_actions=K,
		lm=lam,
		beta=beta,
		n=int(N_total/T), 
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

		# sig UCB choice + reward
		sig_a = sig_ucb.select(ctx)
		sig_r = bandit.play(sig_a)
		sig_ucb.update(ctx, sig_a, sig_r)
		
		# lin UCB choice + reward
		lin_a = lin_ucb.select(ctx)
		lin_r = bandit.play(lin_a)
		lin_ucb.update(ctx, lin_a, lin_r)

		t += 1
		sig_regrets[t] = sig_regrets[t - 1] + (oracle_r - sig_r)
		lin_regrets[t] = lin_regrets[t - 1] + (oracle_r - lin_r)

	return sig_regrets, lin_regrets  # length T+1

# ------------------------- run many -------------------------
runs = [run_one(int(s)) for s in seeds]
all_sig = np.stack([r[0] for r in runs], axis=0)  # (n_runs, T+1)
all_lin = np.stack([r[1] for r in runs], axis=0)  # (n_runs, T+1)

sig_med = np.median(all_sig, axis=0)
sig_p25 = np.percentile(all_sig, 25, axis=0)
sig_p75 = np.percentile(all_sig, 75, axis=0)

lin_med = np.median(all_lin, axis=0)
lin_p25 = np.percentile(all_lin, 25, axis=0)
lin_p75 = np.percentile(all_lin, 75, axis=0)

print(f"Runs: {n_runs} | Final med regret — sigUCB: {sig_med[-1]:.3f} | linUCB: {lin_med[-1]:.3f}")

# ------------------------- plot -------------------------
x = np.arange(T + 1)

plt.figure(figsize=(8, 4.8))

# sigUCB
plt.plot(x, sig_med, label=f"sigUCB (median)")
plt.fill_between(x, sig_p25, sig_p75, alpha=0.20, label="sigUCB 25–75%")

# linUCB
plt.plot(x, lin_med, label=f"linUCB (median)", linestyle="--")
plt.fill_between(x, lin_p25, lin_p75, alpha=0.15, label="linUCB 25–75%")

plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title(f"UCB comparison (K={K}, depth={sig_depth}, β={beta}, λ={lam})")
plt.legend()
plt.tight_layout()
plt.savefig("regret_sig_vs_lin.png", dpi=180)
plt.show()