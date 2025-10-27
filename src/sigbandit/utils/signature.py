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


def sig_kernel():
    # to do later
    return


if __name__ == '__main__':
    X = np.array([[0,1,2,3],
                  [4,5,6,7],
                  [8,9,10,11]], dtype=float)
    t_X = time_augment(X)
    print(t_X)
    sig = compute_signature(t_X, 2, 'sig')
    print(sig)
