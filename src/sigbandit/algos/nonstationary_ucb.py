import numpy as np
from collections import deque

class SWLinUCB:
    """
    Sliding-Window LinUCB (per-arm ridge regression on last w samples for that arm).

    For each arm a, we keep a deque of (t, phi, r) and, at select-time, we
    compute (Va, ba) from only samples with t > current_t - w.

    This is a practical SW-LinUCB baseline inspired by SW-UCB / SW-LinUCB ideas
    (Cheung, Simchi-Levi, Zhu 2018/2021).
    """
    def __init__(
        self,
        n_actions: int,
        n: int,
        w: int = 50,
        lm: float = 1.0,
        gamma: float = 1.0,
        fit_intercept: bool = True,
        seed: int = 0,
    ):
        self.K = int(n_actions)
        self.n0 = int(n)
        self.fit_intercept = bool(fit_intercept)
        self.d = self.n0 + (1 if self.fit_intercept else 0)

        self.w = int(w)
        if self.w <= 0:
            raise ValueError("w must be positive.")

        self.lm = float(lm)
        self.gamma = float(gamma)

        self.rng = np.random.default_rng(seed)

        # per-arm sliding window data
        self.hist = [deque() for _ in range(self.K)]  # each entry: (t, phi(d,1), r)
        self.t = 0  # global interaction counter

    def _phi(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float).reshape(-1, 1)  # (n0,1)
        if x.shape[0] != self.n0:
            raise ValueError(f"Expected context dimension {self.n0}, got {x.shape[0]}.")
        if self.fit_intercept:
            x = np.vstack([x, [[1.0]]])          # (d,1)
        return x

    def _prune(self, a: int):
        # keep only entries with time > t - w
        cutoff = self.t - self.w
        dq = self.hist[a]
        while dq and dq[0][0] <= cutoff:
            dq.popleft()

    def _build_stats(self, a: int):
        # Va = lm I + sum phi phi^T over window
        # ba =       sum phi r     over window
        V = self.lm * np.eye(self.d)
        b = np.zeros((self.d, 1))
        for (_, phi, r) in self.hist[a]:
            V += phi @ phi.T
            b += phi * float(r)
        return V, b

    def select(self, x: np.ndarray) -> int:
        phi = self._phi(x)

        best_a, best_val = 0, -np.inf
        for a in range(self.K):
            self._prune(a)
            V, b = self._build_stats(a)
            Vinv = np.linalg.inv(V)
            theta = Vinv @ b

            mean = float((phi.T @ theta)[0, 0])
            bonus = float(np.sqrt((phi.T @ Vinv @ phi)[0, 0]))
            val = mean + self.gamma * bonus

            if val > best_val:
                best_val, best_a = val, a

        return best_a

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        phi = self._phi(x)
        a = int(a)

        # increment global time at each observed reward
        self.t += 1

        # append and prune that arm
        self.hist[a].append((self.t, phi, float(r)))
        self._prune(a)
