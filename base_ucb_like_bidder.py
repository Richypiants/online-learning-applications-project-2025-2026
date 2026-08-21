import numpy as np

from environment import Environment

class BaseUCBLikeBidder:
    def __init__(self, B, T, valuations, environment: Environment):
        self.B = B  # budget
        self.T = T  # number of rounds (= number of auctions = number of users)
        self.RHO = B / T  # budget per round

        self.valuations = valuations  # true values of its own ads (assumed to be known)
        self.N_CAMPAIGNS = len(self.valuations)  # number of campaigns
        self.bids = environment.BIDS_SPACE[environment.BIDS_SPACE <= valuations[0]]  # possible bids = action space of size K
        self.K = len(self.bids)  # cardinality of the action space (possible bids)

        self.t = 0  # current round
        self.a_t = np.zeros(self.N_CAMPAIGNS, dtype=int)  # current action (as the index of the arm played, not the bid itself)
        self.range = valuations[0]  # range for the confidence bounds

        self.avg_f = np.array([self.valuations[0] - self.bids for _ in range(self.N_CAMPAIGNS)]) / 2        # ORIGINAL: np.zeros(self.K)
        self.avg_c = np.array([np.zeros(self.K) for _ in range(self.N_CAMPAIGNS)]) / 2                      # ALTERNATIVE: self.bids
        self.N_pulls = np.array([np.zeros(self.K) for _ in range(self.N_CAMPAIGNS)])

        self.environment = environment

    def bid(self):
        if self.B < sum(self.valuations):
            return np.zeros(self.N_CAMPAIGNS, dtype=int)

        # Should no longer be needed since we use the mask in the confidence bounds to avoid division by zero, 
        # but yeah we'll see if maybe do one exploration per campaign-bid pair
        # if self.t < self.K:
        #     self.a_t = self._exploration_actions()
        #     return self.a_t

        #print(self.avg_f)
        #print(self.avg_c)
        #print(self.N_pulls)

        f_ucbs, c_lcbs = self._confidence_bounds()
        self.a_t = self._choose_actions(f_ucbs, c_lcbs)
        return self.a_t

    def _exploration_actions(self):
        action_idx = self.t % self.K
        return np.full(self.N_CAMPAIGNS, action_idx, dtype=int)

    def _confidence_bounds(self):
        radius = np.full_like(self.N_pulls, self.range)
        #radius_f = np.full_like(self.N_pulls, self.valuations[0] - self.bids) / 2 #* self.range      ## TODO: for now there is still only one valuation 
        #radius_c = np.full_like(self.N_pulls, self.bids) / 2 #* self.range

        #print(radius_f)
        #print(radius_c)
        mask = self.N_pulls != 0

        radius[mask] = radius[mask] * np.sqrt(2 * np.log(self.t) / self.N_pulls[mask])

        #radius_f[mask] = radius_f[mask] * np.sqrt(2 * np.log(self.t) / self.N_pulls[mask])
        f_ucbs = np.clip(self.avg_f + radius, None, 1)

        #radius_c[mask] = radius_c[mask] * np.sqrt(2 * np.log(self.t) / self.N_pulls[mask])
        c_lcbs = np.clip(self.avg_c - radius, 0, None)

        # EXERCISE SESSION:
        #f_ucbs = self.avg_f + self.range*np.sqrt(2*np.log(self.T)/self.N_pulls)
        #c_lcbs = self.avg_c - self.range*np.sqrt(2*np.log(self.T)/self.N_pulls)

        return f_ucbs, c_lcbs

    def _choose_actions(self, f_ucbs, c_lcbs):
        raise NotImplementedError

    def learn(self, f_t, c_t):
        for i in range(self.N_CAMPAIGNS):
            arm = self.a_t[i]
            self.N_pulls[i][arm] += 1
            self.avg_f[i][arm] += (f_t[i] - self.avg_f[i][arm]) / self.N_pulls[i][arm]
            self.avg_c[i][arm] += (c_t[i] - self.avg_c[i][arm]) / self.N_pulls[i][arm]
        self.B -= np.sum(c_t)
        self.t += 1