from typing import Callable
from .context import ContextualProcess, bm, ou_stationary, gbm
from .reward import min_max_reward
import numpy as np

Context_Process = ContextualProcess
Context_Round = np.ndarray
Action  = int
Reward  = float

class Bandit:
    """
      - next_round():   advance round and ContextualProcess
      - play(a):        apply action a, return reward for current context
      - oracle():       return oracle action and reward for current context
    """

    def __init__(
            self, 
            context_process: Context_Process,
            reward_fn: Callable[[Action, Context_Round], Reward], 
            sigma: float,
            K: int, 
            T: int,
            seed: int
    ):
        self._context_process = context_process
        self._reward_fn  = reward_fn
        self._t          = 0
        self._sigma      = sigma
        self._n_actions  = K
        self._T          = T
        self._eps_t      = 0.0
        self._rng        = np.random.default_rng(seed)


    def next_round(self) -> bool:
        if self._t + 1 > self._T:
            return False
        else:
            self._t += 1
            self._context_process.next_round()
            self._eps_t = self._rng.normal(0.0, self._sigma)
            return True

    def observe_context(self) -> Context_Round:
        return self._context_process.observe_context()

    def play(self, action) -> Reward:
        context = self.observe_context()
        r = self._reward_fn(action, context) + self._eps_t
        return r
    
    def oracle(self):
        oracle_r = float('-inf')
        arg_a = 1

        context = self.observe_context() # get context for round

        # iterate through awards
        for a in range(0, self._n_actions):
            a_reward = self._reward_fn(a, context) # get reward

            # update accordingly
            if a_reward > oracle_r:
                oracle_r = a_reward
                arg_a = a

        return oracle_r, arg_a

    @property
    def t(self) -> int:
        return self._t

    @property
    def n_actions(self) -> int:
        return self._n_actions



if __name__ == "__main__":
    # test

    # intialize contextual process
    T, N, d = 10, 100, 1
    window = N//T
    BM = bm(T, N)
    context = ContextualProcess(BM, T, window)

    # intialize bandit
    bandit = Bandit(context, min_max_reward, 2, T, seed=0)

    # run rounds
    round = 1
    while bandit.next_round():
        print(f'Round: t={round}')
        print('Context:', bandit.observe_context())

        # oracle
        oracle_r, oracle_a = bandit.oracle()
        print(f'Oracle Action: {oracle_a}')
        print(f'Oracle Reward: {oracle_r}')

        # random action
        rand_a = np.random.choice([0,1])
        rand_a_r = bandit.play(rand_a)
        print(f'Chosen Action: {rand_a}')
        print(f'Observed Reward: {rand_a_r}')

        print('\n')

        round += 1