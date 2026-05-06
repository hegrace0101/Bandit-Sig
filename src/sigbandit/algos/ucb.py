# src/sigbandit/algos/ucb.py
import numpy as np
from ..utils.signature import compute_signature, time_augment, sig_length, sig_kernel, sig_list_levels, prune_sig_levels_drop_last_channel
import iisignature as iis


class sigUCB:
    def __init__(
        self,
        n_actions,
        lm,
        gamma,
        N,
        d,
        sigType='sig',
        log_norm=False,
        fit_intercept=False,
        seed=0,
        prune_end_time=True,
        time_index=0,
        time_augment_bool=True,
    ):
        self.K = int(n_actions)                        # number of arms (0,...,K-1)
        self.lm = float(lm)                            # ridge regression penalty
        self.gamma = gamma                             # exploration parameter
        self.N = N                                     # signature depth    
        self.sigType = sigType                         # sig or log sig
        self.fit_intercept = bool(fit_intercept)       # fit intercept
        self.log_norm = log_norm                       # for gbm
        self.prune_end_time = bool(prune_end_time)     # sig pruning
        self.time_index = int(time_index)              # column index of time channel
        self.time_augment_bool = bool(time_augment_bool)
        self.raw_d = int(d)                            # raw path dimension
        self.d = self.raw_d + 1 if self.time_augment_bool else self.raw_d    

        if self.prune_end_time and not self.time_augment_bool:
            raise ValueError("prune_end_time=True requires time_augment_bool=True")

        # get dimension
        if self.sigType == "sig" and self.prune_end_time:
            # sum_{k=1}^N d^(k-1) * (d-1)
            sig_dim = sum((self.d ** (k - 1)) * (self.d - 1) for k in range(1, self.N + 1))
        else:
            # fall back to original full signature/logsignature dimension
            sig_dim = sig_length(self.d, self.N, self.sigType)

        # append terminal raw state
        self._dim = sig_dim + self.raw_d

        if self.fit_intercept:
            self._dim += 1

        # per-arm state
        self._V     = [self.lm * np.eye(self._dim) for _ in range(self.K)]
        self._Vinv  = [np.eye(self._dim) / self.lm for _ in range(self.K)]
        self._b     = [np.zeros(self._dim) for _ in range(self.K)]
        self._theta = [np.zeros(self._dim) for _ in range(self.K)]

        self.rng = np.random.default_rng(seed)

    def sig_feat(self, context):
        context = np.asarray(context, dtype=float)

        # save initial/terminal condition for extra feature
        x0 = context[0].ravel()
        xT = context[-1].ravel()

        # preprocessing
        if self.log_norm:
            context = np.log(context / context[0])

        # optionally time-augment
        if self.time_augment_bool:
            path_for_sig = time_augment(context)
        else:
            path_for_sig = context

        # get signature feature
        if self.sigType == "sig" and self.prune_end_time:
            levels = sig_list_levels(path_for_sig, self.N)
            sig = prune_sig_levels_drop_last_channel(
                levels,
                d=path_for_sig.shape[1],
                time_index=self.time_index,
            )
        else:
            sig = compute_signature(path_for_sig, self.N, self.sigType)

        # feat = np.concatenate([sig, x0])
        feat = np.concatenate([sig, xT])

        if self.fit_intercept:
            feat = np.concatenate([feat, np.array([1.0])])

        return feat

    def select(self, context):
        # get signature feature
        feat = self.sig_feat(context)

        # compute UCBs
        ucbs = np.zeros(self.K, dtype=float)
        for a in range(self.K):
            theta = self._theta[a]
            Vinv = self._Vinv[a]
            mu = float(feat @ theta)                      # expected
            sd2 = max(0.0, float(feat @ (Vinv @ feat)))   # variance
            ucbs[a] = mu + self.gamma * np.sqrt(sd2)

        # select arm that maximizes ucb, randomly break ties
        m = ucbs.max()
        ties = np.flatnonzero(np.isclose(ucbs, m))
        return int(self.rng.choice(ties))

    def update(self, context, action, reward):
        # get signature feature
        feat = self.sig_feat(context)

        a = int(action)
        V    = self._V[a]
        b    = self._b[a]
        Vinv = self._Vinv[a]

        # rank-1 updates for the chosen arm only
        V += np.outer(feat, feat)
        b += reward * feat

        # Sherman–Morrison for inverse
        Au = Vinv @ feat
        denom = 1.0 + float(feat @ Au)
        Vinv -= np.outer(Au, Au) / denom

        # update theta
        self._theta[a] = Vinv @ b


class linUCB:
    def __init__(self, n_actions, lm, gamma, n, log_norm=False, fit_intercept=False, seed=0):
        self.K = int(n_actions)				        # number of arms (0,...,K-1)
        self.lm = float(lm)							# ridge regression penalty
        self.gamma = gamma							# exploration parameter	
        self.n = n									# number of points in window
        self.log_norm = log_norm                    # for gbm
        self.fit_intercept = bool(fit_intercept)	# fit intercept, do later?
        
        # per-arm state
        self._V     = [self.lm * np.eye(self.n)  for _ in range(self.K)]
        self._Vinv  = [np.eye(self.n) / self.lm  for _ in range(self.K)]
        self._b     = [np.zeros(self.n)          for _ in range(self.K)]
        self._theta = [np.zeros(self.n)          for _ in range(self.K)]

        self.rng = np.random.default_rng(seed)
    
    def lin_feat(self, context):
        context = np.asarray(context, dtype=float)

        if context.ndim == 1:
            feat = context
        else:
            feat = context[:, 0]

        if self.log_norm:
            x0 = feat[0]
            feat = np.log(feat / feat[0])
            feat[0] = x0

        return feat


    def select(self, context):
        # get feature
        feat = self.lin_feat(context)

        # compute UCBs
        ucbs = np.zeros(self.K, dtype=float)
        for a in range(self.K):
            theta = self._theta[a]
            Vinv = self._Vinv[a]
            mu = float(feat @ theta)              # expected
            sd2 = np.max([0, float(feat @ (Vinv @ feat))])     # variance 
            ucbs[a] = mu + self.gamma * np.sqrt(sd2)

        # select arm that maximizes ucb, randomly break ties
        m = ucbs.max()
        ties = np.flatnonzero(np.isclose(ucbs, m))
        return int(self.rng.choice(ties))
    

    def update(self, context, action, reward):
        # get feature
        feat = self.lin_feat(context)

        a = int(action)
        V    = self._V[a]    
        b    = self._b[a]    
        Vinv = self._Vinv[a] 

        # rank-1 updates for the chosen arm only
        V += np.outer(feat, feat)
        b += reward * feat

        # Sherman–Morrison for inverse
        Au = Vinv @ feat
        denom = 1.0 + float(feat @ Au)
        Vinv -= np.outer(Au, Au) / denom

        # update theta
        self._theta[a] = Vinv @ b