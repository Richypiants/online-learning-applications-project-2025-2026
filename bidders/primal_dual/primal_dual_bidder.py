import numpy as np

from bidders.bidder import Bidder
from bidders.primal_dual.hedge_primal_regret_minimizer import HedgePrimalRegretMinimizer

class PrimalDualBidder(Bidder):
    '''Primal-dual bidding strategy for multiple campaigns with budget constraint.

    Primal: exponential-weights (Hedge) over joint actions (campaigns subset + bids),
    assuming full feedback on the highest competing bids.
    Dual: online gradient ascent on the Lagrange multiplier lambda for the budget constraint.
    '''

    def __init__(self, B, T, valuations, environment, learning_rate_primal=0.1, learning_rate_dual=0.1, regret_minimizer=None):
        super().__init__(B, T, valuations, environment)

        self.learning_rate_primal = learning_rate_primal
        self.learning_rate_dual = learning_rate_dual
    
        if regret_minimizer is None:
            self.regret_minimizer = HedgePrimalRegretMinimizer(environment, self.K - 1, self.learning_rate_primal)     # only over the non-zero bids, since the 0.0 bid is already in the superarm
        else:
            self.regret_minimizer = regret_minimizer

        self.lambda_t = 1.0

        # TODO: N_pulls here? To keep track of played arms

    def bid(self):
        if self.B < sum(self.valuations):
            self.a_t = np.zeros(self.N_CAMPAIGNS, dtype=int)
            return self.a_t

        self.a_t = self.regret_minimizer.bid()
        return self.a_t

    def learn(self, f_t, c_t, m_t=None):
        self.N_pulls[np.arange(self.N_CAMPAIGNS), self.a_t] += 1

        # full feedback: estimate f(b) and c(b) for every bid of every campaign from the observed highest competing bid
        
        # wins against the highest competing bids in the campaigns
        my_wins = self.bids >= m_t[:, None]      # shape: (K,) >= (N_CAMPAIGNS, 1) -> (N_CAMPAIGNS, K)           
        f_t_full = (self.valuations[:, None] - self.bids[None, :]) * my_wins       # shape: ((N_CAMPAIGNS, 1) - (1, K)) - * (N_CAMPAIGNS, K) = > (N_CAMPAIGNS, K)
        c_t_full = self.bids[None, :] * my_wins      # shape: (K,) * (N_CAMPAIGNS, K) = > (N_CAMPAIGNS, K)
        lagrangian = f_t_full - self.lambda_t * (c_t_full - self.RHO)        # shape: (N_CAMPAIGNS, K)
        lagrangian = lagrangian[:, 1:]      # remove the first column corresponding to the zero bid (no action)

        # primal update: update the regret minimizer with the rescaled lagrangian (to [0, 1])
        lagrangian_upper_bound = 1 - (1 / self.RHO) * (-self.RHO)
        lagrangian_lower_bound = 0 - (1 / self.RHO) * (1 - self.RHO)
        rescaled_lagrangian = (lagrangian - lagrangian_lower_bound) / (lagrangian_upper_bound - lagrangian_lower_bound)
        self.regret_minimizer.learn(1 - rescaled_lagrangian)  # we need to maximize L

        # dual update: gradient ascent on lambda, projected to [0, 1/RHO]
        budget_violation = self.RHO - np.sum(self.c_hat) 
        self.lambda_t = np.clip(self.lambda_t - self.learning_rate_dual * budget_violation, 0.0, 1.0 / self.RHO)

        self.B -= np.sum(c_t)
        self.t += 1
