# src/sigbandit/envs/temp_sensor_bandit.py
import numpy as np

class TempSensorBandit:
    def __init__(
        self,
        temp_matrix: np.ndarray,   # (T_total, K+2)
        times,                     # DatetimeIndex-like
        anchors: list[int],        # indices in [0, K-1]
        window_L: int = 12,        # 12 bins = last hour if bins are 5-min
        reward_mode: str = "max",  # "max" or "avg"
        seed: int = 0,
    ):
        self.X = temp_matrix
        self.times = times
        self.anchors = np.array(anchors, dtype=int)
        self.L = int(window_L)
        self.reward_mode = reward_mode
        self.rng = np.random.default_rng(seed)
        self.K = int(self.X.shape[1] - 2)
        self.T_total = self.X.shape[0]
        self.t = self._first_hour_index()
        
    def _first_hour_index(self) -> int:
        start = self.L - 1
        for t in range(start, self.T_total):
            ts = self.times[t]
            if ts.minute == 0:
                return t
        raise ValueError("No on-the-hour timestamp found in the data after the first full window.")

    def next_round(self) -> bool:
        return self.t < self.T_total

    def _window_slice(self):
        return slice(self.t - self.L + 1, self.t + 1)

    def observe_context_window(self) -> np.ndarray:
        """
        Returns a path/window of shape (L, d0) where d0=len(anchors).
        Your sigUCB can time-augment internally.
        """
        sl = self._window_slice()
        anchor_path = self.X[sl][:, self.anchors]
        tod_path = self.X[sl][:, self.K:self.K+2]
        return anchor_path #np.concatenate([anchor_path, tod_path], axis=1)


    def reward(self, action: int) -> float:
        sl = self._window_slice()

        chosen = self.X[sl, action]             # (L,)
        field  = self.X[sl, :self.K]            # (L, K)

        if self.reward_mode == "max":
            chosen_stat = np.max(chosen)
            field_stat  = np.max(field)  # max over time and sensors in window
        elif self.reward_mode == "avg":
            chosen_stat = np.mean(chosen)
            field_stat  = np.mean(field) # mean over time and sensors in window

        r = -abs(chosen_stat - field_stat)

        return float(r)

    def oracle(self):
        """
        Best action = sensor whose window-stat is closest to the true field-stat.
        Returns (oracle_reward_noiseless, oracle_action)
        """
        sl = self._window_slice()
        field = self.X[sl, :self.K]  # (L, K)

        if self.reward_mode == "max":
            field_stat = np.max(field)
            per_sensor = np.max(field, axis=0)      # (K,)
        else:
            field_stat = np.mean(field)
            per_sensor = np.mean(field, axis=0)     # (K,)

        errors = np.abs(per_sensor - field_stat)
        a_star = int(np.argmin(errors))
        r_star = -float(errors[a_star])
        return r_star, a_star

    def step_time(self):
        self.t += self.L


if __name__ == "__main__":
    import argparse
    import pandas as pd

    # import your loader (adjust if your module path differs)
    from tests.temp_sensor_exp.load_data import load_intel_temp_matrix
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, default="tests/temp_sensor_exp/temp_sensor.txt")
    parser.add_argument("--freq", type=str, default="5min")
    parser.add_argument("--L", type=int, default=12)
    parser.add_argument("--d0", type=int, default=3)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--reward_mode", type=str, default="max", choices=["max", "avg"])
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    X, times, col_ids = load_intel_temp_matrix(args.txt, freq=args.freq)

    K = int(X.shape[1] - 2)
    print("Data loaded.")
    print(f"X shape: {X.shape}   (T, K+2)")
    print(f"K sensors inferred: {K}")
    print(f"time range: {times[0]} -> {times[-1]}")
    print(f"dt: {times[1] - times[0]}")
    print(f"last columns: {col_ids[-4:]}")
    print()

    # basic checks on TOD columns
    tod = X[:, K:K+2]
    print(f"tod_sin range: [{tod[:,0].min():.3f}, {tod[:,0].max():.3f}]")
    print(f"tod_cos range: [{tod[:,1].min():.3f}, {tod[:,1].max():.3f}]")
    print()

    rng = np.random.default_rng(args.seed)
    if args.d0 > K:
        raise ValueError(f"d0={args.d0} cannot exceed K={K}")
    anchors = rng.choice(np.arange(K), size=args.d0, replace=False).tolist()
    print(f"Anchors (indices into sensor columns 0..{K-1}): {anchors}")

    env = TempSensorBandit(
        temp_matrix=X,
        times=times,
        anchors=anchors,
        window_L=args.L,
        reward_mode=args.reward_mode,
        seed=args.seed,
    )

    print("\nStepping through rounds...\n")
    for i in range(args.steps):
        if not env.next_round():
            print("No more rounds available.")
            break

        ctx = env.observe_context_window()
        expected_shape = (args.L, args.d0 + 2)
        assert ctx.shape == expected_shape, f"ctx shape {ctx.shape} != {expected_shape}"

        oracle_r, oracle_a = env.oracle()
        assert 0 <= oracle_a < K, f"oracle action {oracle_a} out of range"
        assert np.isfinite(oracle_r), "oracle reward should be finite"

        a = int(rng.integers(0, K))
        r = env.reward(a)
        assert np.isfinite(r), "reward should be finite"

        ts = env.times[env.t]
        tod_sin, tod_cos = ctx[-1, -2], ctx[-1, -1]
        print(
            f"round={i:02d}  t_idx={env.t:05d}  ts={ts}  "
            f"ctx_shape={ctx.shape}  "
            f"tod(last)=(sin={tod_sin:+.3f}, cos={tod_cos:+.3f})  "
            f"oracle=(a={oracle_a}, r={oracle_r:+.3f})  "
            f"sample=(a={a}, r={r:+.3f})"
        )
        print(ctx)

        env.step_time()

    print("Basic functionality checks passed.")