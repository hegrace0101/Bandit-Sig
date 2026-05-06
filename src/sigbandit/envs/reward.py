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
    return float(np.mean(ctx, axis=0) @ beta[action]) 

def nested_extrema_reward(action: int, ctx: np.ndarray) -> float:
    if action == 0:
        # min over time for each feature, then take max across features
        return float(np.abs(np.max(np.min(ctx, axis=0))))
    elif action == 1:
        # max over time for each feature, then take min across features
        return float(np.abs(np.min(np.max(ctx, axis=0))))
    elif action == 2:
        # min over features at each time, then take max across time
        return float(np.abs(np.max(np.min(ctx, axis=1))))
    elif action == 3:
        # max over features at each time, then take min across time
        return float(np.abs(np.min(np.max(ctx, axis=1))))
    else:
        raise ValueError(f"Action must be in {{0,1,2,3}}, got {action}")
    

def path_mix_reward(action: int, ctx: np.ndarray) -> float:
    inc = np.diff(ctx, axis=0)
    rough = float(np.sum(inc**2))

    x = ctx[:, 0]
    y = ctx[:, 1]
    area = float(np.sum(x[:-1] * np.diff(y) - y[:-1] * np.diff(x)))
    area = abs(area)

    alpha = (action + 1) / (action + 2)

    if action % 2 == 0:
        return alpha * rough + (1 - alpha) * area
    else:
        return alpha * area + (1 - alpha) * rough
    
def roughness_reward(action: int, ctx: np.ndarray) -> float:
    lag = action + 1
    d = ctx.shape[1]
    col = action % d
    x = ctx[:, col]
    inc = x[lag:] - x[:-lag]
    return float(np.sum(inc**lag)) / lag