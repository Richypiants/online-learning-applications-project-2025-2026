import numpy as np

from bidders.bidder import Bidder
from environment.environment import Environment

class BaseUCBLikeBidder(Bidder):
    def __init__(self, B, T, valuations, environment: Environment):
        super().__init__(B, T, valuations, environment)

        self.range = np.asarray(valuations, dtype=float)  # per-campaign range for the confidence bounds, shape (N_CAMPAIGNS,)

        self.avg_f = np.array([np.zeros(self.K) for _ in range(self.N_CAMPAIGNS)])
        self.avg_c = np.array([np.zeros(self.K) for _ in range(self.N_CAMPAIGNS)])

    def bid(self):
        if self.B < sum(self.valuations):
            self.a_t = np.zeros(self.N_CAMPAIGNS, dtype=int)
            return self.a_t

        if self.t < self.K:
            self.a_t = self._exploration_actions()
            return self.a_t

        f_ucbs, c_lcbs = self._confidence_bounds()
        self.a_t = self._choose_actions(f_ucbs, c_lcbs)
        return self.a_t

    def _exploration_actions(self):
        raise NotImplementedError

    def _confidence_bounds(self):
        radius = np.broadcast_to(self.range[:, None], self.N_pulls.shape).copy()  # shape (N_CAMPAIGNS, K)

        mask = self.N_pulls != 0

        radius[mask] = radius[mask] * np.sqrt(2 * np.log(self.t) / self.N_pulls[mask])
        radius[~mask] = 1

        f_ucbs = np.clip(self.avg_f + radius, None, 1)
        c_lcbs = np.clip(self.avg_c - radius, 0, None)

        return f_ucbs, c_lcbs

    def _choose_actions(self, f_ucbs, c_lcbs):
        raise NotImplementedError

    def learn(self, f_t, c_t, m_t=None):
        super().learn(f_t, c_t)
        self.N_pulls[np.arange(self.N_CAMPAIGNS), self.a_t] += 1
        self.avg_f[np.arange(self.N_CAMPAIGNS), self.a_t] += (f_t - self.avg_f[np.arange(self.N_CAMPAIGNS), self.a_t]) / self.N_pulls[np.arange(self.N_CAMPAIGNS), self.a_t]
        self.avg_c[np.arange(self.N_CAMPAIGNS), self.a_t] += (c_t - self.avg_c[np.arange(self.N_CAMPAIGNS), self.a_t]) / self.N_pulls[np.arange(self.N_CAMPAIGNS), self.a_t]