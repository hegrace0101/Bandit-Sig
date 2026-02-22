import numpy as np

class KernelUCB:
    """
    Per-arm KernelUCB using kernel ridge regression:
      mu_a(x) = k_a(x)^T (K_a + lam I)^{-1} y_a
      s2_a(x) = k(x,x) - k_a(x)^T (K_a + lam I)^{-1} k_a(x)
      UCB = mu + gamma * sqrt(max(s2, 0))

    Notes:
    - Maintains per-arm inverse Gram matrix incrementally via block inversion.
    - Stores all past contexts per arm unless max_points is set.
    """

    def __init__(
        self,
        n_actions: int,
        lm: float = 1.0,
        gamma: float = 1.0,
        kernel: str = "rbf",
        sigma: float = 1.0,   # RBF lengthscale (bigger = smoother)
        poly_deg: int = 2,
        poly_c: float = 1.0,
        max_points: int | None = None,
        seed: int = 0,
    ):
        self.K = int(n_actions)
        self.lm = float(lm)
        self.gamma = float(gamma)

        self.kernel = kernel
        self.sigma = float(sigma)
        self.poly_deg = int(poly_deg)
        self.poly_c = float(poly_c)

        self.max_points = max_points
        self.rng = np.random.default_rng(seed)

        # per-arm memory
        self.X = [[] for _ in range(self.K)]         # list of (d,) arrays
        self.y = [[] for _ in range(self.K)]         # list of scalars
        self.Kinv = [None for _ in range(self.K)]    # (n,n) inverse of (Gram + lam I)

    # ---------- kernels ----------
    def _k(self, x: np.ndarray, z: np.ndarray) -> float:
        if self.kernel == "rbf":
            # exp(-||x-z||^2 / (2*sigma^2))
            diff = x - z
            return float(np.exp(-0.5 * (diff @ diff) / (self.sigma ** 2)))
        elif self.kernel == "poly":
            return float((x @ z + self.poly_c) ** self.poly_deg)
        else:
            raise ValueError(f"Unknown kernel '{self.kernel}'")

    def _k_xX(self, x: np.ndarray, X_list: list[np.ndarray]) -> np.ndarray:
        # returns shape (n,)
        return np.array([self._k(x, xi) for xi in X_list], dtype=float)

    def _k_xx(self, x: np.ndarray) -> float:
        # for RBF this is 1
        return float(self._k(x, x))

    # ---------- inverse update ----------
    @staticmethod
    def _block_inverse_update(Ainv: np.ndarray, b: np.ndarray, c: float) -> np.ndarray:
        """
        Given old inverse Ainv for A (n,n),
        update inverse of [[A, b],[b^T, c]] using block matrix inverse.
        """
        # Schur complement: s = c - b^T Ainv b
        Ab = Ainv @ b
        s = float(c - b.T @ Ab)
        # numerical guard
        if s <= 1e-12:
            s = 1e-12
        s_inv = 1.0 / s

        top_left = Ainv + np.outer(Ab, Ab) * s_inv
        top_right = -Ab * s_inv
        bottom_left = (-Ab * s_inv).reshape(1, -1)
        bottom_right = np.array([[s_inv]], dtype=float)

        return np.block([[top_left, top_right.reshape(-1, 1)],
                         [bottom_left, bottom_right]])

    def _maybe_cap(self, a: int):
        """
        Optional: cap memory per arm by random eviction (simple).
        If you want a smarter eviction (FIFO), change here.
        """
        if self.max_points is None:
            return
        n = len(self.X[a])
        if n <= self.max_points:
            return

        # evict one random point
        j = int(self.rng.integers(low=0, high=n))
        self.X[a].pop(j)
        self.y[a].pop(j)

        # rebuild inverse from scratch (simpler & safe)
        X = self.X[a]
        n = len(X)
        G = np.zeros((n, n), dtype=float)
        for i in range(n):
            for k in range(i, n):
                val = self._k(X[i], X[k])
                G[i, k] = val
                G[k, i] = val
        G += self.lm * np.eye(n)
        self.Kinv[a] = np.linalg.inv(G)

    # ---------- bandit API ----------
    def select(self, context: np.ndarray) -> int:
        x = np.ascontiguousarray(np.asarray(context, dtype=float).ravel())

        ucbs = np.zeros(self.K, dtype=float)
        for a in range(self.K):
            n = len(self.X[a])
            if n == 0:
                ucbs[a] = np.inf
                continue

            kx = self._k_xX(x, self.X[a])               # (n,)
            Kinv = self.Kinv[a]                         # (n,n)
            y = np.asarray(self.y[a], dtype=float)      # (n,)

            mu = float(kx @ (Kinv @ y))
            s2 = float(self._k_xx(x) - kx @ (Kinv @ kx))
            if s2 < 0:
                s2 = 0.0

            ucbs[a] = mu + self.gamma * np.sqrt(s2)

        m = ucbs.max()
        ties = np.flatnonzero(np.isclose(ucbs, m))
        return int(self.rng.choice(ties))


    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        a = int(action)
        x = np.ascontiguousarray(np.asarray(context, dtype=float).ravel())
        r = float(reward)

        X_a = self.X[a]
        y_a = self.y[a]

        if len(X_a) == 0:
            # first point: K = [k(x,x)] + lam I
            kxx = self._k_xx(x) + self.lm
            self.Kinv[a] = np.array([[1.0 / kxx]], dtype=float)
            X_a.append(x)
            y_a.append(r)
            return

        # append with inverse update
        kx = self._k_xX(x, X_a)              # (n,)
        kxx = self._k_xx(x) + self.lm        # scalar (add ridge on diagonal)
        Ainv = self.Kinv[a]                  # (n,n)

        Kinv_new = self._block_inverse_update(Ainv, kx, kxx)

        X_a.append(x)
        y_a.append(r)
        self.Kinv[a] = Kinv_new

        # cap memory if needed
        self._maybe_cap(a)
