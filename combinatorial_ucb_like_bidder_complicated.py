import numpy as np
import mip
from base_ucb_like_bidder import BaseUCBLikeBidder

class CombinatorialUCBLikeBidder(BaseUCBLikeBidder):
    def _choose_actions(self, f_ucbs, c_lcbs):
        joint_gamma_t, marginals_t = self.compute_opt(f_ucbs, c_lcbs)
        campaigns_subset = np.random.choice(2**self.N_CAMPAIGNS, p=marginals_t)  # sample a campaigns subset according to the marginal probabilities
        conditional_gamma_t = joint_gamma_t[campaigns_subset, :, :] / marginals_t[campaigns_subset] # compute the conditional probabilities given the sampled campaigns subset
        return np.array([np.random.choice(self.K, p=conditional_gamma_t[i, :]) for i in range(self.N_CAMPAIGNS)])  # sample the bids for each campaign according to the conditional probabilities

    def compute_opt(self, f_ucbs, c_lcbs):
        # TODO: remove, it should never happen
        if np.sum(c_lcbs < np.zeros(c_lcbs.shape)):     # if any of the lower confidence bounds are negative, then the linear program is infeasible, so we just pick the arm with the highest upper confidence bound
            gamma = np.zeros((2**self.N_CAMPAIGNS, self.N_CAMPAIGNS, len(self.bids)))               ## check if it is truly infeasible, or if it is just convenient since a negative cost means the arm is "free" to play
            best_bids = np.argmax(f_ucbs, axis=1)                   ## also, it should never happen that the lower confidence bound is negative, since the cost is always positive, so maybe we should just clip it to 0 instead 
            for c in range(self.N_CAMPAIGNS):
                gamma[-1, c, best_bids[c]] = 1
            marginals = np.zeros(2**self.N_CAMPAIGNS)
            marginals[-1] = 1                           ## dunno if it's the correct way to do this, but we just pick the last campaigns subset (all campaigns) with probability 1
            return gamma, marginals

        model = mip.Model()

        # Optimization variables
        gamma = {(c, b, a): model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=1) 
                 for c in range(self.N_CAMPAIGNS) for b in range(len(self.bids)) for a in range(2**self.N_CAMPAIGNS)}     # probabilities
        marginals = {(a): model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=1) for a in range(2**self.N_CAMPAIGNS)}  # marginal probabilities

        # Budget constraint
        model.add_constr(mip.xsum(gamma[c, b, a] * c_lcbs[c, b]
                                  for c in range(self.N_CAMPAIGNS) for b in range(len(self.bids)) for a in range(2**self.N_CAMPAIGNS)) <= self.RHO)

        # Marginalization constraint
        for a in range(2**self.N_CAMPAIGNS):
            for c in range(self.N_CAMPAIGNS):
                model.add_constr(mip.xsum(gamma[c, b, a] for b in range(len(self.bids))) == marginals[a])
                if a & (1 << c):  # if campaign c is included in the campaigns subset a:
                    model.add_constr(gamma[c, 0, a] == 0)
                else:
                    model.add_constr(gamma[c, 0, a] == marginals[a])

        # Probability distribution constraint
        model.add_constr(mip.xsum(marginals[a] for a in range(2**self.N_CAMPAIGNS)) == 1)

        # Conflicts constraint
        for conflict in self.environment.conflicts_graph.graph:
            campaign_a, campaign_b = conflict
            for a in range(2**self.N_CAMPAIGNS):
                if a & (1 << campaign_a) and a & (1 << campaign_b):  # if both campaigns are included in the campaigns subset a:
                    model.add_constr(marginals[a] == 0)

        # Objective function: maximize expected fbar_t(gamma)
        model.objective = mip.maximize(
            mip.xsum(gamma[c, b, a] * f_ucbs[c, b]
                     for c in range(self.N_CAMPAIGNS) for b in range(len(self.bids)) for a in range(2**self.N_CAMPAIGNS))
        )
        model.optimize()

        gamma_values = np.array([       # shape: (2**N_CAMPAIGNS, N_CAMPAIGNS, len(BIDS_SPACE))
            [[gamma[c, b, a].x if abs(gamma[c, b, a].x) > 1e-10 else 0 for b in range(len(self.bids))] for c in range(self.N_CAMPAIGNS)] for a in range(2**self.N_CAMPAIGNS)
        ])

        marginal_values = np.array([
            marginals[a].x if abs(marginals[a].x) > 1e-10 else 0 for a in range(2**self.N_CAMPAIGNS)       # for cleaning up errors due to mip's feasibility tolerance
        ])

        return gamma_values, marginal_values

