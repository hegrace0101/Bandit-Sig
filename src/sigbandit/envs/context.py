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
      normalize:     None | "online" | "pre" (z-score) normalization of the d channels
      end_mode:      behavior after last index for preloaded mode:
                       "stop" | "loop" | "repeat_last" | "pad"
      name:          identifier for logging/debug
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
    

# ---------------------- Continuous-Time Stochastic Processes ----------------------
def bm(T, N, x0=0.0, d=1):
    dt = float(T / N)
    # initialize
    x = np.zeros((N + 1, d), dtype=float)
    x[0] = x0

    # generate bm process
    dW = np.random.normal(0.0, np.sqrt(dt), size=(N, d)).astype(float)
    x[1:] = x[0] + np.cumsum(dW, axis=0)

    return x

def ou(T, N, x0=0, d=1, mu=0, theta=1, sigma=1):
    dt = float(T/N)

    # initialize
    x = np.zeros((N+1,d))
    x[0] = x0

    # Generate the OU process
    for t in range(1, N+1):
        dW =  np.random.normal(0, np.sqrt(dt), size=d)
        x[t] = x[t-1] + theta * (mu - x[t-1]) * dt + sigma * dW

    return x


def ou_stationary(T, N, d=1, mu=0, theta=1, sigma=1):
    dt = float(T/N)

    # initialize
    x = np.zeros((N+1,d))

    # draw from normal distibution 
    var_stationary = sigma**2 / (2.0 * theta)
    x[0] = mu + np.random.normal(scale=np.sqrt(var_stationary), size=d)

    mu_arr = np.full(d, mu, dtype=float)

    # Generate the OU process
    for t in range(1, N+1):
        dW =  np.random.normal(0, np.sqrt(dt), size=d)
        x[t] = x[t-1] + theta * (mu_arr - x[t-1]) * dt + sigma * dW

    return x


def gbm(T, N, x0=1, d=1, mu=0.1, sigma=0.43):
    dt = float(T/N)

    # initialize
    x = np.zeros((N+1,d), dtype=float)
    x[0] = x0

    # generate gbm process
    for t in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt), size=d)
        x[t] = x[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    return x

def frac_bm(T, N, x0=0, d=1, H=0.75):
    dt = float(T) / float(N)

    # time grid
    t = np.arange(N + 1, dtype=float) * dt  # shape (N+1,)

    # Covariance for fBm:
    # Cov(B_H(t_i), B_H(t_j)) = 0.5*(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})
    ti = t[:, None]                       # (N+1, 1)
    tj = t[None, :]                       # (1, N+1)
    C = 0.5 * (ti**(2*H) + tj**(2*H) - np.abs(ti - tj)**(2*H))

    # Cholesky factor
    L = np.linalg.cholesky(C)             # (N+1, N+1)

    # Sample d independent paths
    Z = np.standard_normal(size=(N + 1, d))
    X = L @ Z                              # (N+1, d)

    # Add initial value x0
    x0_arr = np.asarray(x0, dtype=float)
    if x0_arr.ndim == 0:
        x0_arr = np.full((d,), float(x0_arr))
    elif x0_arr.shape != (d,):
        raise ValueError(f"x0 must be scalar or shape ({d},), got {x0_arr.shape}")

    X = X + x0_arr[None, :]
    return X

def hawkes(T, N):
    return


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
       
