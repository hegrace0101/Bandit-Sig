import numpy as np

class ContextualProcess:
    """
    A causal, stateful context process that serves rolling windows over:
      - a preloaded series X (T, d), or
      - a user-provided sampler() -> (d,) that generates the next point.

    Round t contract:
      - current() returns a dict with a window ending at index t (inclusive).
      - advance() moves to t+1 according to end_mode.

    Args:
      X:            time series of shape (T*window, d)
      T: 			number of rounds 
      normalize:    None | "online" | "pre" (z-score) normalization of the d channels
      end_mode:     behavior after last index for preloaded mode:
                       "stop" | "loop" | "repeat_last" | "pad"
      name:         identifier for logging/debug
    """
    def __init__(
        self,
        X: np.ndarray,
        T: int, 
        window: int
    ):
        
        self.X = X.astype(float, copy=False)
        self.T = T
        self.w = window
        self._round = 0 # current round
    
    def next_round(self) -> None:
        if self._round + 1 > self.T:
            return False
        else:
          self._round += 1
          return True
    
    def observe_context(self) -> None:
        curr_context = self.X[self.w*(self._round - 1):self.w*(self._round)]
        return curr_context
    

# ---------------------- Stationary Processes ----------------------
def ou_stationary(T, N, d=1, mu=0, theta=1, Sigma=1.0):
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim == 0:
        Sigma = Sigma.reshape(1, 1)

    d = Sigma.shape[0]
    if Sigma.shape != (d, d):
        raise ValueError(f"Sigma must be square, got {Sigma.shape}")

    dt = float(T / N)

    x = np.zeros((N + 1, d), dtype=float)

    # stationary covariance for isotropic theta:
    # Cov = Sigma / (2 theta)
    cov_stationary = Sigma / (2.0 * theta)
    x[0] = np.random.multivariate_normal(mean=np.full(d, mu, dtype=float), cov=cov_stationary)

    mu_arr = np.full(d, mu, dtype=float)

    for t in range(1, N + 1):
        dW = np.random.multivariate_normal(mean=np.zeros(d), cov=dt * Sigma)
        x[t] = x[t - 1] + theta * (mu_arr - x[t - 1]) * dt + dW

    return x


def ar1(T, N, d=1, phi=0.7, Sigma=1.0, x0=None):
    """
    Multivariate AR(1):
        x_t = phi * x_{t-1} + eps_t,   eps_t ~ N(0, Sigma)

    Returns array of shape (N+1, d).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim == 0:
        Sigma = Sigma.reshape(1, 1)
    d = Sigma.shape[0]

    if np.isscalar(phi):
        phi_arr = np.full(d, float(phi))
    else:
        phi_arr = np.asarray(phi, dtype=float)

    x = np.zeros((N + 1, d), dtype=float)

    if x0 is None:
        # stationary covariance coordinatewise approximation
        # Var_j ~ Sigma_jj / (1 - phi_j^2)
        var0 = np.diag(Sigma) / np.maximum(1.0 - phi_arr**2, 1e-12)
        x[0] = np.random.multivariate_normal(
            mean=np.zeros(d),
            cov=np.diag(var0),
        )
    else:
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.ndim == 0:
            x0_arr = np.full(d, float(x0_arr))
        elif x0_arr.shape != (d,):
            raise ValueError(f"x0 must be scalar or shape ({d},), got {x0_arr.shape}")
        x[0] = x0_arr

    for t in range(1, N + 1):
        eps_t = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma)
        x[t] = phi_arr * x[t - 1] + eps_t

    return x


def arma11(T, N, d=1, phi=0.6, theta=0.3, Sigma=1.0, x0=None):
    """
    Multivariate ARMA(1,1):
        x_t = phi * x_{t-1} + eps_t + theta * eps_{t-1},
        eps_t ~ N(0, Sigma)

    Returns array of shape (N+1, d).

    Notes:
    - phi and theta may be scalars or length-d arrays.
    - This uses diagonal AR/MA coefficients with correlated innovations Sigma.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim == 0:
        Sigma = Sigma.reshape(1, 1)
    d = Sigma.shape[0]

    if np.isscalar(phi):
        phi_arr = np.full(d, float(phi))
    else:
        phi_arr = np.asarray(phi, dtype=float)

    if np.isscalar(theta):
        theta_arr = np.full(d, float(theta))
    else:
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.shape != (d,):
            raise ValueError(f"theta must be scalar or shape ({d},), got {theta_arr.shape}")

    x = np.zeros((N + 1, d), dtype=float)

    if x0 is None:
        x[0] = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma)
    else:
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.ndim == 0:
            x0_arr = np.full(d, float(x0_arr))
        elif x0_arr.shape != (d,):
            raise ValueError(f"x0 must be scalar or shape ({d},), got {x0_arr.shape}")
        x[0] = x0_arr

    eps_prev = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma)

    for t in range(1, N + 1):
        eps_t = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma)
        x[t] = phi_arr * x[t - 1] + eps_t + theta_arr * eps_prev
        eps_prev = eps_t

    return x


import numpy as np


def regime_switching_ar1(
    T,
    N,
    d=1,
    phi_regimes=(0.2, 0.9),
    mu_regimes=None,
    Sigma=1.0,
    x0=None,
):
    """
    Multivariate deterministic regime-switching AR(1):
        x_t = mu_{s_t} + phi_{s_t} * (x_{t-1} - mu_{s_t}) + eps_t,
        eps_t ~ N(0, Sigma)

    where s_t changes deterministically at evenly spaced times.

    If there are m regimes, the horizon is split into m blocks of length about N/m.

    Parameters
    ----------
    T : int or float
        Included for interface consistency; not used directly.
    N : int
        Number of time steps.
    d : int
        State dimension.
    phi_regimes : sequence
        Regime-specific AR(1) coefficients. Each entry may be
        - a scalar, or
        - an array of shape (d,).
    mu_regimes : sequence or None
        Regime-specific means. Each entry may be
        - a scalar, or
        - an array of shape (d,).
        If None, all means are zero.
    Sigma : scalar or np.ndarray
        Innovation covariance, shape (d, d) if array.
    x0 : scalar, array, or None
        Initial state. If None, initialized from the stationary-ish
        distribution of the first regime using a diagonal approximation.

    Returns
    -------
    x : np.ndarray
        Array of shape (N+1, d).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim == 0:
        Sigma = Sigma.reshape(1, 1)

    d = Sigma.shape[0]
    if Sigma.shape != (d, d):
        raise ValueError(f"Sigma must be square, got {Sigma.shape}")

    m = len(phi_regimes)
    if m < 1:
        raise ValueError("Need at least one regime.")

    if mu_regimes is None:
        mu_regimes = [np.zeros(d, dtype=float) for _ in range(m)]
    elif len(mu_regimes) != m:
        raise ValueError("mu_regimes must have same length as phi_regimes.")

    # normalize phi, mu into arrays of shape (d,)
    phi_list = []
    mu_list = []

    for j in range(m):
        phi_j = phi_regimes[j]
        if np.isscalar(phi_j):
            phi_j = np.full(d, float(phi_j))
        else:
            phi_j = np.asarray(phi_j, dtype=float)
            if phi_j.shape != (d,):
                raise ValueError(
                    f"phi_regimes[{j}] must be scalar or shape ({d},), got {phi_j.shape}"
                )
        if np.any(np.abs(phi_j) >= 1.0):
            raise ValueError(
                f"All |phi| must be < 1 for stationary-within-regime dynamics. Bad regime {j}."
            )
        phi_list.append(phi_j)

        mu_j = mu_regimes[j]
        if np.isscalar(mu_j):
            mu_j = np.full(d, float(mu_j))
        else:
            mu_j = np.asarray(mu_j, dtype=float)
            if mu_j.shape != (d,):
                raise ValueError(
                    f"mu_regimes[{j}] must be scalar or shape ({d},), got {mu_j.shape}"
                )
        mu_list.append(mu_j)

    x = np.zeros((N + 1, d), dtype=float)

    # deterministic regime schedule
    block_len = max(1, N // m)

    def _regime_at_time(t: int) -> int:
        return min(t // block_len, m - 1)

    # initialize from first regime
    s0 = 0
    if x0 is None:
        phi0 = phi_list[s0]
        mu0 = mu_list[s0]
        var0 = np.diag(Sigma) / np.maximum(1.0 - phi0**2, 1e-12)
        x[0] = np.random.multivariate_normal(mean=mu0, cov=np.diag(var0))
    else:
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.ndim == 0:
            x0_arr = np.full(d, float(x0_arr))
        elif x0_arr.shape != (d,):
            raise ValueError(f"x0 must be scalar or shape ({d},), got {x0_arr.shape}")
        x[0] = x0_arr

    for t in range(1, N + 1):
        s_t = _regime_at_time(t)
        phi_t = phi_list[s_t]
        mu_t = mu_list[s_t]

        eps_t = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma)
        x[t] = mu_t + phi_t * (x[t - 1] - mu_t) + eps_t

    return x


def trend_seasonal(
    T,
    N,
    d=1,
    slopes=None,
    amplitudes=None,
    periods=None,
    phases=None,
    Sigma=1.0,
    noise_phi=0.5,
    x0=None,
):
    """
    Multivariate trend + seasonal + AR(1)-noise process:
        x_t = trend_t + seasonal_t + eta_t
        eta_t = noise_phi * eta_{t-1} + eps_t,   eps_t ~ N(0, Sigma)

    Returns array of shape (N+1, d).

    Parameters
    ----------
    T : int or float
        Included for interface consistency; not used directly.
    N : int
        Number of time steps.
    d : int
        State dimension.
    slopes : scalar or array shape (d,)
        Linear trend slopes.
    amplitudes : scalar or array shape (d,)
        Seasonal amplitudes.
    periods : scalar or array shape (d,)
        Seasonal periods in time steps.
    phases : scalar or array shape (d,)
        Seasonal phases.
    Sigma : scalar or np.ndarray
        Innovation covariance for AR noise.
    noise_phi : scalar or array shape (d,)
        AR(1) persistence of the noise.
    x0 : scalar, array, or None
        Optional initial value for x[0].
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim == 0:
        Sigma = Sigma.reshape(1, 1)

    d = Sigma.shape[0]
    if Sigma.shape != (d, d):
        raise ValueError(f"Sigma must be square, got {Sigma.shape}")

    def _to_arr(v, name, default):
        if v is None:
            return np.full(d, default, dtype=float)
        if np.isscalar(v):
            return np.full(d, float(v), dtype=float)
        v = np.asarray(v, dtype=float)
        if v.shape != (d,):
            raise ValueError(f"{name} must be scalar or shape ({d},), got {v.shape}")
        return v

    slopes = _to_arr(slopes, "slopes", 0.01)
    amplitudes = _to_arr(amplitudes, "amplitudes", 1.0)
    periods = _to_arr(periods, "periods", 50.0)
    phases = _to_arr(phases, "phases", 0.0)
    noise_phi = _to_arr(noise_phi, "noise_phi", 0.5)

    if np.any(periods <= 0):
        raise ValueError("All periods must be positive.")
    if np.any(np.abs(noise_phi) >= 1.0):
        raise ValueError("For stationary AR noise, all |noise_phi_j| must be < 1.")

    x = np.zeros((N + 1, d), dtype=float)
    eta = np.zeros((N + 1, d), dtype=float)

    # initialize AR noise near stationarity (diagonal approximation)
    var0 = np.diag(Sigma) / np.maximum(1.0 - noise_phi**2, 1e-12)
    eta[0] = np.random.multivariate_normal(mean=np.zeros(d), cov=np.diag(var0))

    if x0 is None:
        t0 = 0.0
        trend0 = slopes * t0
        seas0 = amplitudes * np.sin(2.0 * np.pi * t0 / periods + phases)
        x[0] = trend0 + seas0 + eta[0]
    else:
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.ndim == 0:
            x0_arr = np.full(d, float(x0_arr))
        elif x0_arr.shape != (d,):
            raise ValueError(f"x0 must be scalar or shape ({d},), got {x0_arr.shape}")
        x[0] = x0_arr

    for t in range(1, N + 1):
        eps_t = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma)
        eta[t] = noise_phi * eta[t - 1] + eps_t

        tt = float(t)
        trend_t = slopes * tt
        seas_t = amplitudes * np.sin(2.0 * np.pi * tt / periods + phases)

        x[t] = trend_t + seas_t + eta[t]

    return x


def bm(T, N, x0=0.0, Sigma=1.0):
    Sigma = np.asarray(Sigma, dtype=float)

    if Sigma.ndim == 0:
        Sigma = Sigma.reshape(1, 1)

    d = Sigma.shape[0]
    if Sigma.shape != (d, d):
        raise ValueError(f"Sigma must be square, got {Sigma.shape}")

    dt = float(T / N)

    x = np.zeros((N + 1, d), dtype=float)

    x0_arr = np.asarray(x0, dtype=float)
    if x0_arr.ndim == 0:
        x0_arr = np.full(d, float(x0_arr))
    elif x0_arr.shape != (d,):
        raise ValueError(f"x0 must be scalar or shape ({d},), got {x0_arr.shape}")

    x[0] = x0_arr

    dW = np.random.multivariate_normal(
        mean=np.zeros(d),
        cov=dt * Sigma,
        size=N
    )

    x[1:] = x[0] + np.cumsum(dW, axis=0)
    return x





def gbm(T, N, x0=1, d=1, mu=0.1, Sigma=1.0):
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim == 0:
        Sigma = Sigma.reshape(1, 1)

    d = Sigma.shape[0]
    if Sigma.shape != (d, d):
        raise ValueError(f"Sigma must be square, got {Sigma.shape}")

    dt = float(T / N)

    x = np.zeros((N + 1, d), dtype=float)

    x0_arr = np.asarray(x0, dtype=float)
    if x0_arr.ndim == 0:
        x0_arr = np.full(d, float(x0_arr))
    elif x0_arr.shape != (d,):
        raise ValueError(f"x0 must be scalar or shape ({d},), got {x0_arr.shape}")

    mu_arr = np.full(d, mu, dtype=float)

    x[0] = x0_arr

    # diagonal entries are per-coordinate variances
    var_diag = np.diag(Sigma)

    for t in range(1, N + 1):
        dW = np.random.multivariate_normal(mean=np.zeros(d), cov=dt * Sigma)
        x[t] = x[t - 1] * np.exp((mu_arr - 0.5 * var_diag) * dt + dW)

    return x


if __name__ == "__main__":
    # test
    T, N, d = 10, 100, 1
    window = N//T
    BM = bm(T, N)
    OU = ou(T, N)
    GBM = gbm(T, N)

    context = ContextualProcess(BM, T, window)

    while context.next_round():
        print(context.observe_context())
       
