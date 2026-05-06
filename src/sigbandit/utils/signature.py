from typing import Optional, Literal
import numpy as np
import iisignature as iis


SigType = Literal["sig", "logsig"]


def compute_signature(t_X: np.ndarray, N: int, type: SigType):
    '''
    N = signature depth
    '''
    
    w, d = t_X.shape # w = number of points in interval/window, d = dimension of path 
    
    # log sig
    if type == 'logsig':
        s = iis.prepare(d, N)
        logsignature = iis.logsig(t_X, s)
        return logsignature
	# sig
    elif type == 'sig':
        signature = iis.sig(t_X, N)
        return signature
    raise ValueError(f"Unknown kind={type!r}; use 'sig' or 'logsig'.")


def time_augment(X: np.ndarray) -> np.ndarray:
    w, d = X.shape
    t = np.linspace(0, 1, num=w, dtype=float)
    t_W = np.insert(X, 0, t, axis=1)
    return t_W


def sig_length(d, N, type: SigType):
    # log sig
    if type == 'logsig':
        length = iis.logsiglength(d, N)
	# sig
    elif type == 'sig':
        length = iis.siglength(d, N)
    return length


def sig_feature(
    X: np.ndarray,
    N: int,
    type: SigType = "sig",
    add_time: bool = True,
    time_pruned: bool = False,
    time_index: int = 0,
    normalize: bool = False,
) -> np.ndarray:
    """
    Return the truncated signature feature vector used by sig_kernel.

    If time_pruned=True, uses pruned ordinary signature coordinates
    (drop words ending in the time channel). This is only supported for type='sig'.
    """
    X = np.asarray(X, dtype=float)

    if add_time:
        Xf = time_augment(X)
    else:
        Xf = X

    if time_pruned:
        if type != "sig":
            raise ValueError("time_pruned=True is only supported for type='sig'.")
        d = Xf.shape[1]
        levels = sig_list_levels(Xf, N)
        fx = prune_sig_levels_drop_last_channel(levels, d=d, time_index=time_index)
    else:
        fx = np.asarray(compute_signature(Xf, N, type), dtype=float).ravel()

    if not normalize:
        return fx

    nrm = float(np.linalg.norm(fx))
    if nrm <= 1e-12:
        return np.zeros_like(fx)
    return fx / nrm

def sig_kernel(
    X: np.ndarray,
    Z: np.ndarray,
    N: int,
    type: SigType = "sig",
    add_time: bool = True,
    normalize: bool = False,
    time_pruned: bool = False,
    time_index: int = 0,
) -> float:
    fx = sig_feature(
        X,
        N=N,
        type=type,
        add_time=add_time,
        time_pruned=time_pruned,
        time_index=time_index,
        normalize=normalize,
    )
    fz = sig_feature(
        Z,
        N=N,
        type=type,
        add_time=add_time,
        time_pruned=time_pruned,
        time_index=time_index,
        normalize=normalize,
    )
    return float(fx @ fz)

def sig_list_levels(t_X: np.ndarray, N: int) -> list[np.ndarray]:
    """
    Returns [level1, level2, ..., levelN], where level k has shape (d^k,).
    Uses iisignature format=1 API.
    """
    return iis.sig(t_X, N, 1)


def prune_sig_levels_drop_last_channel(
    levels: list[np.ndarray],
    d: int,
    time_index: int = 0,
) -> np.ndarray:
    """
    Drop all signature coordinates whose word ends in the time channel.

    Assumes time is inserted as channel 0 by time_augment / np.insert(..., 0, ...).
    """
    kept = []
    for k, lev in enumerate(levels, start=1):
        T = lev.reshape((d,) * k)
        idx = [i for i in range(d) if i != time_index]
        T_keep = np.take(T, idx, axis=-1)
        kept.append(T_keep.reshape(-1))
    return np.concatenate(kept, axis=0)


if __name__ == '__main__':
    X = np.array([[0,1,2,3],
                  [4,5,6,7],
                  [8,9,10,11]], dtype=float)
    t_X = time_augment(X)
    print(t_X)
    sig = compute_signature(t_X, 2, 'sig')
    print(sig)
