import numpy as np
import scipy

from bidders.UCB_based.base_ucb_like_bidder import BaseUCBLikeBidder

class UCBLikeBidder(BaseUCBLikeBidder):
    def _exploration_actions(self):
        # round-robin over feasible arms per campaign
        actions = np.zeros(self.N_CAMPAIGNS, dtype=int)
        for i in range(self.N_CAMPAIGNS):
            feasible_arms = np.flatnonzero(self.feasible[i])        # global arm indices that are feasible for campaign i
            actions[i] = feasible_arms[self.t % len(feasible_arms)]
        return actions

    def _choose_actions(self, f_ucbs, c_lcbs):
        actions = np.zeros(self.N_CAMPAIGNS, dtype=int)
        for i in range(self.N_CAMPAIGNS):
            gamma_t = self.compute_bid_distribution(f_ucbs[i], c_lcbs[i], self.feasible[i])
            actions[i] = np.random.choice(self.K, p=gamma_t)
        return actions

    def compute_bid_distribution(self, f_ucbs, c_lcbs, feasible_mask=None):
        c = -f_ucbs
        a = c_lcbs
        b = np.ones(self.K)
        # equality constraints to force gamma = 0 on infeasible arms (length-K mask)
        if feasible_mask is not None:
            A_eq = np.vstack([b, np.eye(self.K)])             # first row: distribution sums to 1; subsequent rows: gamma_b = 0 if infeasible
            b_eq = np.concatenate(([1.0], np.zeros(self.K)))
            # zero out rows where the arm is feasible (no constraint on gamma)
            A_eq[1:][feasible_mask] = 0.0
        else:
            A_eq = [b]
            b_eq = [1]
        res = scipy.optimize.linprog(c, 
                                    A_ub=None if self.RHO == np.inf else [a], 
                                    b_ub=None if self.RHO == np.inf else [self.RHO], 
                                    A_eq=A_eq, 
                                    b_eq=b_eq, 
                                    bounds=(0, 1))
        gamma = res.x
        return gamma
