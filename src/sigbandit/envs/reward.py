import numpy as np

def min_max_reward(action: int, ctx: np.ndarray) -> float:
    if action % 2 == 0:
        return np.abs(np.max(ctx))
    else:
        return np.abs(np.min(ctx))
    
def min_max_mix_reward(action: int, ctx: np.ndarray) -> float:
    mx = float(np.abs(np.max(ctx)))
    mn = float(np.abs(np.min(ctx)))

    alpha = (action + 1) / (action + 2)

    if action % 2 == 0:
        base = alpha * mx + (1 - alpha) * mn
    else:
        base = alpha * mn + (1 - alpha) * mx
    return base

def sum_a_reward(action: int, ctx: np.ndarray) -> float:
    return np.sum(ctx) * action


def linear_reward(action: int, ctx: np.ndarray, beta: np.ndarray) -> float:
    return float((ctx @ beta[action]).sum()/len(ctx))

