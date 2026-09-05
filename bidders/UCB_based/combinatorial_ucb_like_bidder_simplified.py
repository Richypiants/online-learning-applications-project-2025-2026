import numpy as np
import scipy
import mip

from bidders.UCB_based.base_ucb_like_bidder import BaseUCBLikeBidder

class CombinatorialUCBLikeBidder(BaseUCBLikeBidder):
    def _exploration_actions(self):
        actions = np.zeros(self.N_CAMPAIGNS, dtype=int)

        feasible_superarms = [i for i in range(2**self.N_CAMPAIGNS)]
        for conflict in self.environment.conflicts_graph.graph:
            campaign_a, campaign_b = conflict
            for a in feasible_superarms:
                if a & (1 << campaign_a) and a & (1 << campaign_b):  # if both campaigns are included in the campaigns subset a:
                    feasible_superarms.remove(a)

        superarm = np.random.choice(feasible_superarms)

        for campaign in range(self.N_CAMPAIGNS):
            if superarm & (1 << campaign):  # if campaign campaign is included in the sampled campaigns subset:
                # sample uniformly over the feasible non-zero arms for campaign `campaign` (offset by +1 since arm 0 is the 0.0 bid)
                feasible_nonzero = np.flatnonzero(self.feasible_nonzero[campaign])
                actions[campaign] = 1 + np.random.choice(feasible_nonzero)  # +1 keeps the original 0-bid offset; choice is over feasible non-zero arms only
            else:
                actions[campaign] = 0  # if campaign campaign is not included in the sampled campaigns subset, we don't bid on it (arm 0 corresponds to bidding 0.0)

        return actions

    def _choose_actions(self, f_ucbs, c_lcbs):                      # shape of f_ucbs and c_lcbs: (N_CAMPAIGNS, K), where K is the number of possible bids (arms)
        actions = np.zeros(self.N_CAMPAIGNS, dtype=int)

        gamma_ts = np.zeros((self.N_CAMPAIGNS, self.K - 1))             # shape = (N_CAMPAIGNS, K - 1), the 0.0 bid is removed since it is implicit in the superarm choice
        for i in range(self.N_CAMPAIGNS):
            gamma_ts[i, :] = self.compute_bid_distribution(f_ucbs[i, 1:], c_lcbs[i, 1:], self.feasible_nonzero[i])

        campaign_f_ucbs = np.sum(f_ucbs[:, 1:] * gamma_ts, axis=1)
        campaign_c_lcbs = np.sum(c_lcbs[:, 1:] * gamma_ts, axis=1)

        superarm_f_ucbs = np.zeros(2**self.N_CAMPAIGNS)
        superarm_c_lcbs = np.zeros(2**self.N_CAMPAIGNS)
        for a in range(2**self.N_CAMPAIGNS):
            for i in range(self.N_CAMPAIGNS):
                if a & (1 << i):  # if campaign i is included in the campaigns subset a:
                    superarm_f_ucbs[a] += campaign_f_ucbs[i]
                    superarm_c_lcbs[a] += campaign_c_lcbs[i]

        gamma_superarm = self.compute_optimal_superarm(superarm_f_ucbs, superarm_c_lcbs)        # shape = (2**N_CAMPAIGNS,)

        superarm = np.random.choice(2**self.N_CAMPAIGNS, p=gamma_superarm)        # sample a subset of non-conflicting campaigns on which to bid

        for i in range(self.N_CAMPAIGNS):
            if superarm & (1 << i):  # if campaign i is included in the sampled campaigns subset:
                actions[i] = 1 + np.random.choice(self.K - 1, p=gamma_ts[i, :])  # sample the bid for campaign i according to the conditional probabilities, remember to displace by the discarded 0.0 bid by adding 1
            else:
                actions[i] = 0  # if campaign i is not included in the sampled campaigns subset, we don't bid on it (arm 0 corresponds to bidding 0.0)
        return actions

    def compute_bid_distribution(self, f_ucbs, c_lcbs, feasible_mask=None):
        c = -f_ucbs
        a = c_lcbs
        b = np.ones(self.K - 1)                         # still removing the 0.0 bid, since it is implicit when not bidding in the campaign
        # equality constraints to force gamma = 0 on infeasible non-zero arms (length-(K-1) mask)
        if feasible_mask is not None:
            A_eq = np.vstack([b, np.eye(self.K - 1)])
            b_eq = np.concatenate(([1.0], np.zeros(self.K - 1)))
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
        