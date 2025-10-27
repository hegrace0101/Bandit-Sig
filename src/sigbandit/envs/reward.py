import numpy as np

def min_max_reward(action: int, ctx: np.ndarray) -> float:
    if action % 2 == 0:
        return np.abs(np.max(ctx))
    else:
        return np.abs(np.min(ctx))

def sum_a_reward(action: int, ctx: np.ndarray) -> float:
    return np.sum(ctx) * action


def linear_reward(action: int, ctx: np.ndarray) -> float:
    w = len(ctx)
    theta_even = np.array([1 if i < w/2 else -1 for i in range(w)]) 
    theta_odd = np.array([1 if i >= w/2 else -1 for i in range(w)]) 
    if action % 2 == 0: 
        return theta_even @ ctx
    else:
        return theta_odd @ ctx
