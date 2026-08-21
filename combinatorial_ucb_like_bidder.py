import numpy as np
import scipy
import mip
from base_ucb_like_bidder import BaseUCBLikeBidder

class CombinatorialUCBLikeBidder(BaseUCBLikeBidder):
    def _choose_actions(self, f_ucbs, c_lcbs):                      # shape of f_ucbs and c_lcbs: (N_CAMPAIGNS, K), where K is the number of possible bids (arms)
        #print("f_ucbs:", f_ucbs)
        #print("c_lcbs:", c_lcbs)
        actions = np.zeros(self.N_CAMPAIGNS, dtype=int)

        gamma_ts = np.zeros((self.N_CAMPAIGNS, self.K - 1))             # shape = (N_CAMPAIGNS, K - 1), the 0.0 bid is removed since it is implicit in the superarm choice
        for i in range(self.N_CAMPAIGNS):
            gamma_ts[i, :] = self.compute_bid_distribution(f_ucbs[i, 1:], c_lcbs[i, 1:])

        campaign_f_ucbs = np.sum(f_ucbs[:, 1:] * gamma_ts, axis=1)
        campaign_c_lcbs = np.sum(c_lcbs[:, 1:] * gamma_ts, axis=1)
        #print("campaign_f_ucbs:", campaign_f_ucbs)
        #print("campaign_c_lcbs:", campaign_c_lcbs)

        superarm_f_ucbs = np.zeros(2**self.N_CAMPAIGNS)
        superarm_c_lcbs = np.zeros(2**self.N_CAMPAIGNS)
        for a in range(2**self.N_CAMPAIGNS):
            for i in range(self.N_CAMPAIGNS):
                if a & (1 << i):  # if campaign i is included in the campaigns subset a:
                    superarm_f_ucbs[a] += campaign_f_ucbs[i]
                    superarm_c_lcbs[a] += campaign_c_lcbs[i]
        #print("superarm_f_ucbs:", superarm_f_ucbs)
        #print("superarm_c_lcbs:", superarm_c_lcbs)

        gamma_superarm = self.compute_optimal_superarm(superarm_f_ucbs, superarm_c_lcbs)        # shape = (2**N_CAMPAIGNS,)
        #print("gamma_superarm:", gamma_superarm)

        superarm = np.random.choice(2**self.N_CAMPAIGNS, p=gamma_superarm)        # sample a subset of non-conflicting campaigns on which to bid
        #print(superarm)

        for i in range(self.N_CAMPAIGNS):
            if superarm & (1 << i):  # if campaign i is included in the sampled campaigns subset:
                actions[i] = 1 + np.random.choice(self.K - 1, p=gamma_ts[i, :])  # sample the bid for campaign i according to the conditional probabilities, remember to displace by the discarded 0.0 bid by adding 1
            else:
                actions[i] = 0  # if campaign i is not included in the sampled campaigns subset, we don't bid on it (arm 0 corresponds to bidding 0.0)
        #print(actions)
        return actions

    def compute_bid_distribution(self, f_ucbs, c_lcbs):
        # TODO: remove, it should never happen
        if np.sum(c_lcbs < np.zeros(len(c_lcbs))):                     ## if any of the lower confidence bounds are negative, then the linear program is infeasible, so we just pick the arm with the highest upper confidence bound
            gamma = np.zeros(len(f_ucbs))               ## check if it is truly infeasible, or if it is just convenient since a negative cost means the arm is "free" to play
            gamma[np.argmax(f_ucbs)] = 1                ## also, it should never happen that the lower confidence bound is negative, since the cost is always positive, so maybe we should just clip it to 0 instead 
            return gamma
        c = -f_ucbs
        a = c_lcbs
        b = np.ones(self.K - 1)                         # still removing the 0.0 bid, since it is implicit in the superarm choice
        res = scipy.optimize.linprog(c, 
                                    A_ub=None if self.RHO == np.inf else [a], 
                                    b_ub=None if self.RHO == np.inf else [self.RHO], 
                                    A_eq=[b], 
                                    b_eq=[1], 
                                    bounds=(0, 1))
        gamma = res.x
        return gamma

    def compute_optimal_superarm(self, superarm_f_ucbs, superarm_c_lcbs):
        model = mip.Model()

        # Optimization variables
        gamma_superarm = {(a): model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=1) for a in range(2**self.N_CAMPAIGNS)}     # probabilities

        # Budget constraint
        model.add_constr(mip.xsum(gamma_superarm[a] * superarm_c_lcbs[a] for a in range(2**self.N_CAMPAIGNS)) <= self.RHO)

        # Probability distribution constraint
        model.add_constr(mip.xsum(gamma_superarm[a] for a in range(2**self.N_CAMPAIGNS)) == 1)

        # Conflicts constraint
        for conflict in self.environment.conflicts_graph.graph:
            campaign_a, campaign_b = conflict
            for a in range(2**self.N_CAMPAIGNS):
                if a & (1 << campaign_a) and a & (1 << campaign_b):  # if both campaigns are included in the campaigns subset a:
                    model.add_constr(gamma_superarm[a] == 0)

        # Objective function: maximize expected fbar_t(gamma)
        model.objective = mip.maximize(
            mip.xsum(gamma_superarm[a] * superarm_f_ucbs[a] for a in range(2**self.N_CAMPAIGNS))
        )
        model.optimize()

        gamma_superarm_values = np.array([
            gamma_superarm[a].x if abs(gamma_superarm[a].x) > 1e-10 else 0 for a in range(2**self.N_CAMPAIGNS)
        ])

        return gamma_superarm_values
        