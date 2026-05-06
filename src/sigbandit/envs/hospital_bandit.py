# src/sigbandit/envs/hospital_bandit.py
from __future__ import annotations

import numpy as np
import pandas as pd


class HospitalBandit:
	"""
	Weekly decision cadence.
	  - Context at week t: last L days of (tau_day, daily_demand)
	  - Action A_t: staffing level for week t+1 from a discrete grid
	  - Reward r_t: negative Newsvendor loss based on realized weekly demand D_{t+1}
				   revealed after week t+1 ends (but before choosing A_{t+1})

	Assumptions for the numpy-week aggregation path:
	  - daily_counts is a 1D array of daily counts, no missing days
	  - times is daily and aligned with daily_counts
	  - data is trimmed so it starts on a Monday
	  - data contains only full weeks (len(daily_counts) % 7 == 0)
	"""

	def __init__(
		self,
		daily_counts: np.ndarray,   # (N_days,)
		times,                      # DatetimeIndex-like, length N_days, daily
		*,
		window_L_days: int = 7,
		actions: np.ndarray | None = None,  # e.g. np.arange(50, 401, 10)
		underage_cost: float = 0.7,
		overage_cost: float = 0.3,
		seed: int = 0,
	):
		self.rng = np.random.default_rng(seed)

		# --- actions ---
		self.actions = np.asarray(actions, dtype=float).reshape(-1)
		self.K = int(self.actions.size)

		# --- params ---
		self.L = int(window_L_days)
		self.b = float(underage_cost)
		self.h = float(overage_cost)

		# --- inputs ---
		self.daily_counts = np.asarray(daily_counts, dtype=float).reshape(-1)
		self.times = pd.DatetimeIndex(times)

		# --- derived daily series ---
		self.daily_demand = self.daily_counts / 3.0

		# --- seasonal encoding ---
		dayofyear = self.times.dayofyear.astype(float)                 # 1..365/366
		days_in_year = np.where(self.times.is_leap_year, 366.0, 365.0) # per-day denominator
		phase = 2.0 * np.pi * (dayofyear - 1.0) / days_in_year         # in [0, 2pi)
		self.tod_sin = np.sin(phase)
		self.tod_cos = np.cos(phase)

		# --- weekly aggregation via reshape ---
		self.weekly_counts = self.daily_counts.reshape(-1, 7).sum(axis=1)      # (n_weeks,)
		self.weekly_demand = self.weekly_counts / 3.0                          # (n_weeks,)
		self.weekly_times = self.times[6::7]                                   # Sundays if start is Monday

		self.week_end_to_day_idx = np.arange(6, len(self.daily_counts), 7)

		# first round with enough history for an L-day window
		self.t0 = int(np.searchsorted(self.week_end_to_day_idx, self.L - 1))
		self.t = self.t0


	def next_round(self) -> bool:
		return self.t < (len(self.weekly_demand) - 1)

	def _window_slice_days(self) -> slice:
		day_end = int(self.week_end_to_day_idx[self.t])
		return slice(day_end - self.L + 1, day_end + 1)

	def observe_context_window(self) -> np.ndarray:
		sl = self._window_slice_days()
		return self.daily_demand[sl][:, None]
		#return np.stack([self.tod_sin[sl], self.tod_cos[sl], self.daily_demand[sl]], axis=1)

	def reward(self, action_idx: int) -> float:
		a = float(self.actions[int(action_idx)])
		d_next = float(self.weekly_demand[self.t + 1])

		under = max(d_next - a, 0.0)
		over = max(a - d_next, 0.0)
		return float(-(self.b * under + self.h * over))

	def oracle(self):
		d_next = float(self.weekly_demand[self.t + 1])
		a = self.actions
		under = np.maximum(d_next - a, 0.0)
		over = np.maximum(a - d_next, 0.0)
		r_all = -(self.b * under + self.h * over)
		j = int(np.argmax(r_all))
		return float(r_all[j]), j

	def step_time(self):
		self.t += 1


if __name__ == "__main__":
	import argparse
	from tests.hospital_staff.load_data import load_nyc_vomit_counts

	parser = argparse.ArgumentParser()
	parser.add_argument("--csv", type=str, default="tests/hospital_staff/nyc_vomit_count.csv")
	parser.add_argument("--weeks", type=int, default=12)
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()

	# IMPORTANT: loader should already trim to Monday + full weeks
	df = load_nyc_vomit_counts(args.csv)  # Date, Count
	times = pd.DatetimeIndex(df["Date"])
	daily_counts = df["Count"].to_numpy(dtype=float)

	env = HospitalBandit(
		daily_counts=daily_counts,
		times=times,
		window_L_days=7,
		actions=np.arange(100, 1300, 50),
		underage_cost=0.7,
		overage_cost=0.3,
		seed=args.seed,
	)

	rng = np.random.default_rng(args.seed)
	for i in range(args.weeks):
		if not env.next_round():
			print("No more rounds available.")
			break

		ctx = env.observe_context_window()
		oracle_r, oracle_a_idx = env.oracle()

		a_idx = int(rng.integers(0, env.K))
		r = env.reward(a_idx)

		print(
			f"round={i:02d} week_end={env.weekly_times[env.t].date()} "
			f"ctx_shape={ctx.shape} D_next={env.weekly_demand[env.t+1]:.2f} "
			f"oracle=(a={env.actions[oracle_a_idx]:.0f}, r={oracle_r:+.3f}) "
			f"sample=(a={env.actions[a_idx]:.0f}, r={r:+.3f})"
		)
		env.step_time()

	print("Basic functionality checks passed.")