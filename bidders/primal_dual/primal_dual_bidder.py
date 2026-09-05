import numpy as np

from bidders.bidder import Bidder
from bidders.primal_dual.hedge_primal_regret_minimizer import HedgePrimalRegretMinimizer

class PrimalDualBidder(Bidder):
    '''Primal-dual bidding strategy for multiple campaigns with budget constraint.

    Primal: exponential-weights (Hedge) over joint actions (campaigns subset + bids),
    assuming full feedback on the highest competing bids.
    Dual: online gradient ascent on the Lagrange multiplier lambda for the budget constraint.
    '''

    def __init__(self, B, T, valuations, environment, learning_rate_primal, learning_rate_dual, regret_minimizer=None):
        super().__init__(B, T, valuations, environment)

        self.learning_rate_primal = learning_rate_primal
        self.learning_rate_dual = learning_rate_dual
    
        if regret_minimizer is None:
            # K-1: only over the non-zero bids, since the 0.0 bid is already implicit in not bidding in the campaign
            self.regret_minimizer = HedgePrimalRegretMinimizer(environment, self.K - 1, self.learning_rate_primal, feasible_nonzero=self.feasible_nonzero)
        else:
            self.regret_minimizer = regret_minimizer

        self.lambda_t = 1.0
        self.lambda_history = []  # trace of the dual variable for plotting

    def bid(self):
        if self.B < sum(self.valuations):
            self.a_t = np.zeros(self.N_CAMPAIGNS, dtype=int)
            return self.a_t

        self.a_t = self.regret_minimizer.bid()
        return self.a_t

    def learn(self, f_t, c_t, m_t=None):
        super().learn(f_t, c_t, m_t)
        self.N_pulls[np.arange(self.N_CAMPAIGNS), self.a_t] += 1
        
        # full feedback: estimate f(b) and c(b) for every bid of every campaign from the observed highest competing bids m_t, instead of using only the bandit's f_t and c_t
        
        # wins against the highest competing bids in the campaigns
        my_wins = self.bids >= m_t[:, None]      # shape: (K,) >= (N_CAMPAIGNS, 1) -> (N_CAMPAIGNS, K)           
        f_t_full = (self.valuations[:, None] - self.bids[None, :]) * my_wins       # shape: ((N_CAMPAIGNS, 1) - (1, K)) * (N_CAMPAIGNS, K) = > (N_CAMPAIGNS, K)
        c_t_full = self.bids[None, :] * my_wins      # shape: (K,) * (N_CAMPAIGNS, K) = > (N_CAMPAIGNS, K)

        # remove the first column corresponding to the zero bid (no action)
        campaign_lagrangian = (f_t_full - self.lambda_t * c_t_full)[:, 1:]       # shape: (N_CAMPAIGNS, K)    
        campaign_expected_lagrangian = np.sum(self.regret_minimizer.bid_probs * campaign_lagrangian, axis=1)        # no RHO term in here to avoid summing it through multiple campaigns in a superarm 
        campaign_lagrangian += self.lambda_t * self.RHO          # sum back RHO to the original lagrangian

        superarm_lagrangian = np.full(2**self.N_CAMPAIGNS, self.lambda_t * self.RHO)        # initialize to the SINGLE missing RHO

        for a in range(2**self.N_CAMPAIGNS):
            for i in range(self.N_CAMPAIGNS):
                if a & (1 << i):
                    superarm_lagrangian[a] += campaign_expected_lagrangian[i]

        # primal update: update the regret minimizer with the rescaled lagrangian
        lagrangian_upper_bound = 1 - (1 / self.RHO) * (-self.RHO)
        lagrangian_lower_bound = 0 - (1 / self.RHO) * (1 - self.RHO)
        rescaled_campaign_lagrangian = (campaign_lagrangian - lagrangian_lower_bound) / (lagrangian_upper_bound - lagrangian_lower_bound)
        rescaled_superarm_lagrangian = (superarm_lagrangian - lagrangian_lower_bound) / (lagrangian_upper_bound - lagrangian_lower_bound)
        
        self.regret_minimizer.learn(1 - rescaled_campaign_lagrangian, 1 - rescaled_superarm_lagrangian)  # we need to maximize L

        # dual update: gradient ascent on lambda, projected to [0, 1/RHO]
        budget_violation = self.RHO - np.sum(c_t) 
        self.lambda_t = np.clip(self.lambda_t - self.learning_rate_dual * budget_violation, 0.0, 1.0 / self.RHO)
        self.lambda_history.append(self.lambda_t)
