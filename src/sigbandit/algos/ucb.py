# src/sigbandit/algos/ucb.py
import numpy as np
from ..utils.signature import compute_signature, time_augment, sig_length, sig_kernel

class sigUCB:
    def __init__(self, n_actions, lm, gamma, N, d, sigType='sig', log_norm=False, fit_intercept=False, seed=0):
        self.K = int(n_actions)				        # number of arms (0,...,K-1)
        self.lm = float(lm)							# ridge regression penalty
        self.gamma = gamma							# exploration parameter	
        self.N = N 									# signature depth
        self.d = d									# dimension of path (including time-augmentation)
        self.sigType = sigType						# sig or log sig	
        self.fit_intercept = bool(fit_intercept)	# fit intercept, do later?
        self.log_norm = log_norm                    # for gbm
        
        self._dim = sig_length(d, N, sigType)+(d-1) # length of sig vector + (d-1) (for x0 - time augment)
        # per-arm state
        self._V     = [self.lm * np.eye(self._dim) for _ in range(self.K)]
        self._Vinv  = [np.eye(self._dim) / self.lm  for _ in range(self.K)]
        self._b     = [np.zeros(self._dim)          for _ in range(self.K)]
        self._theta = [np.zeros(self._dim)          for _ in range(self.K)]

        self.rng = np.random.default_rng(seed)

    def sig_feat(self, context):
        # save initial condition for sig feature
        x0 = context[0].ravel()

        # preprocessing
        if self.log_norm:
            context = np.log(context / (context[0]))

        # get signature feature
        t_X = time_augment(context)
        sig = compute_signature(t_X, self.N, self.sigType)
        feat = np.concatenate([sig, x0])

        return feat


    def select(self, context):
        # get signature featre
        feat = self.sig_feat(context)

        # compute UCBs
        ucbs = np.zeros(self.K, dtype=float)
        for a in range(self.K):
            theta = self._theta[a]
            Vinv = self._Vinv[a]
            mu = float(feat @ theta)                          # expected
            sd2 = np.max([0, float(feat @ (Vinv @ feat))])    # variance 
            ucbs[a] = mu + self.gamma * np.sqrt(sd2)

        # select arm that maximizes ucb, randomly break ties
        m = ucbs.max()
        ties = np.flatnonzero(np.isclose(ucbs, m))
        return int(self.rng.choice(ties))
    

    def update(self, context, action, reward):
        # get signature featre
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
        feat = context[:,0]

        if self.log_norm:
            x0 = context[0].ravel()
            feat = np.log(feat / (feat[0]))
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