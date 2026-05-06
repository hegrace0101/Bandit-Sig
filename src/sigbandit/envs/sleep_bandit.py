# src/sigbandit/envs/sleep_bandit.py
import numpy as np
import os

class SleepBandit:
    def __init__(
        self,
        sleep_matrix: np.ndarray,   # (T_total, 3)
        stage_matrix: np.ndarray,	# (n_epoch, K)
        stage_order: list,			# order of stages in stage_matrix
        epoch_len_samples: int,          # samples per epoch/window
        seed: int = 0,
    ):
        self.X = np.asarray(sleep_matrix, dtype=float)
        self.P = np.asarray(stage_matrix, dtype=float)
        self.stage_order = list(stage_order)

        self.rng = np.random.default_rng(seed)

        self.L = int(epoch_len_samples)  # samples per epoch/window
        self.K = int(len(self.stage_order))
        self.T_total = int(self.X.shape[0])
        self.n_epoch = int(self.P.shape[0])
        
        self.t = self.L # start at the end of the first epoch-window
        self.round = 0  # epoch index (0-based)
        

    def next_round(self) -> bool:
        return (self.t <= self.T_total) and (self.round < self.n_epoch)

    def _window_slice(self):
        return slice(self.t - self.L, self.t)

    def observe_context_window(self) -> np.ndarray:
        sl = self._window_slice()
        return self.X[sl]

    def reward(self, action: int) -> float:
        # linear reward in probability
        a = int(action)
        p = self.P[self.round]
        return p[a]

    def oracle(self):
        """
        Best action = correct sleep stage
        Returns (oracle_reward, oracle_action)
        """
        p = self.P[self.round]
        a_star = int(np.argmax(p))
        r_star = p[a_star]
        return r_star, a_star

    def step_time(self):
        self.t += self.L
        self.round += 1


if __name__ == "__main__":
    from tests.sleep_edf.load_data import load_sleep_cassette

    cassette_dir = os.path.join("tests", "sleep_edf", "sleep-cassette")
    if not os.path.isdir(cassette_dir):
        raise FileNotFoundError(
            f"Could not find {cassette_dir}. "
            "Point cassette_dir to the folder containing *-PSG.edf and *-Hypnogram.edf."
        )

    recs = load_sleep_cassette(
        cassette_dir,
        pick_channels=["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
        max_records=3,
        intersect_time=True,
        trim_leading_w=True,
        min_sleep_run_sec=60.0,
        window_sec=30.0,
        target_fs=25.0,
        label_mode="start",
    )
    rec = recs[2]

    env = SleepBandit(
        sleep_matrix=rec.X,
        stage_matrix=rec.P,
        stage_order=rec.stage_order,
        epoch_len_samples=rec.epoch_len,
        seed=0,
    )

    # Basic checks
    assert rec.X.shape[0] == rec.P.shape[0] * rec.epoch_len
    rs = rec.P.sum(axis=1)
    assert np.allclose(rs, 1.0, atol=1e-3)

    # Step a few rounds
    rng = np.random.default_rng(0)
    n_steps = rec.P.shape[0]
    print(n_steps)

    for i in range(n_steps):
        assert env.next_round()

        ctx = env.observe_context_window()
        assert ctx.shape[0] == env.L
        assert np.all(np.isfinite(ctx))

        a = int(rng.integers(0, env.K))
        r = env.reward(a)
        assert 0.0 <= r <= 1.0

        r_star, a_star = env.oracle()
        assert 0.0 <= r_star <= 1.0
        assert a_star == int(np.argmax(rec.P[env.round]))
        assert np.isclose(r_star, float(rec.P[env.round].max()))

        env.step_time()

    print(f"[OK] SleepBandit test passed for record={rec.record_id} steps={n_steps}")