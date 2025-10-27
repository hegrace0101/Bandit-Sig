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
def bm(T, N, x0=0, d=1):
    dt = float(T/N)
    
    # get gaussian increments
    dW = np.random.normal(0, np.sqrt(dt), (N, d)).astype(float)

    # construct bm
    x = x0 + np.cumsum(dW, axis=0)
    x = np.insert(x, 0, x0, axis=0)

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

def gbm(T, N, x0=1, d=1, mu=0.1, sigma=0.43):
    dt = float(T/N)

    # initialize
    x = np.zeros((N+1,d))
    x[0] = x0

    # generate gbm process
    for t in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt), size=d)
        x[t] = x[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    return x


def gaussian(T, N, d, cov):
    dt = float(T/N)
    x = np.zeros((N, ))


# ---------------------- Stationary Distributions ----------------------
def unit_ball(T, N):
    w = int(N/T)
    d = 1

    # sample vector from w-dimensional sphere T times
    def sample_unit_ball(w):
        z = np.random.normal(size=w)
        nrm = np.linalg.norm(z)

        # resample if needed
        while nrm == 0.0:
            z = np.random.normal(size=w)
            nrm = np.linalg.norm(z)
        u = z / nrm

        # radius
        r = np.random.uniform() ** (1.0 / w)
        return  r * u

    x = np.zeros((N+1,1), dtype=float)
    for t in range(T):
        u = sample_unit_ball(w)
        x[t*w : (t+1)*w, 0] = u

    return x


if __name__ == "__main__":
    # test
    T, N, d = 10, 100, 1
    window = N//T
    BM = bm(T, N)
    OU = ou(T, N)
    GBM = gbm(T, N)
    UB = unit_ball(T, N)

    context = ContextualProcess(UB, T, window)

    while context.next_round():
        print(context.observe_context())
       
