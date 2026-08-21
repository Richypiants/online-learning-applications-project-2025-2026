import numpy as np
import scipy
from base_ucb_like_bidder import BaseUCBLikeBidder

# In an online learning setting, you don't know the actual distribution $\mathcal{D}$ of the highest competing bid. An UCB-like method needs to estimate
#  $\bar{f}(b)$ and $\bar{f}(b)$ in the optimization program, before solving it for $\gamma$. Using optimism means that, instead of using the Monte Carlo 
# average of the samples, we replace them with $f_{UCB}(b)$ and $c_{LCB}(b)$, estimatred through bandit feedback. (REWRITE BETTER, also they should have
# bars too in theory, no?)

class UCBLikeBidder(BaseUCBLikeBidder):
    def _choose_actions(self, f_ucbs, c_lcbs):
        actions = np.zeros(self.N_CAMPAIGNS, dtype=int)
        for i in range(self.N_CAMPAIGNS):
            gamma_t = self.compute_bid_distribution(f_ucbs[i], c_lcbs[i])
            actions[i] = np.random.choice(self.K, p=gamma_t)
        return actions

    def compute_bid_distribution(self, f_ucbs, c_lcbs):
        # TODO: remove, it should never happen
        if np.sum(c_lcbs < np.zeros(len(c_lcbs))):                     ## if any of the lower confidence bounds are negative, then the linear program is infeasible, so we just pick the arm with the highest upper confidence bound
            gamma = np.zeros(len(f_ucbs))               ## check if it is truly infeasible, or if it is just convenient since a negative cost means the arm is "free" to play
            gamma[np.argmax(f_ucbs)] = 1                ## also, it should never happen that the lower confidence bound is negative, since the cost is always positive, so maybe we should just clip it to 0 instead 
            return gamma
        c = -f_ucbs
        a = c_lcbs
        b = np.ones(self.K)
        res = scipy.optimize.linprog(c, 
                                    A_ub=None if self.RHO == np.inf else [a], 
                                    b_ub=None if self.RHO == np.inf else [self.RHO], 
                                    A_eq=[b], 
                                    b_eq=[1], 
                                    bounds=(0, 1))
        gamma = res.x
        return gamma
