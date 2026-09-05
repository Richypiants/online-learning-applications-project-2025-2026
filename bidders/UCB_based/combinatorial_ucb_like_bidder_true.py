import numpy as np
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
    
    def _choose_actions(self, f_ucbs, c_lcbs):
        joint_gamma_t, marginals_t = self.compute_opt(f_ucbs, c_lcbs)

        # sample a campaigns subset according to the marginal probabilities
        campaigns_subset = np.random.choice(2**self.N_CAMPAIGNS, p=marginals_t)

        # compute the conditional probabilities given the sampled campaigns subset
        conditional_gamma_t = joint_gamma_t[campaigns_subset, :, :] / marginals_t[campaigns_subset]

        # sample the bids for each campaign according to the conditional probabilities
        return np.array([np.random.choice(self.K, p=conditional_gamma_t[i, :]) for i in range(self.N_CAMPAIGNS)])
    
    def compute_opt(self, f_ucbs, c_lcbs):
        model = mip.Model()

        # Optimization variables
        gamma = {(c, b, a): model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=1) 
                 for c in range(self.N_CAMPAIGNS) for b in range(len(self.bids)) for a in range(2**self.N_CAMPAIGNS)}     # probabilities
        marginals = {(a): model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=1) for a in range(2**self.N_CAMPAIGNS)}  # marginal probabilities

        # Budget constraint
        model.add_constr(mip.xsum(gamma[c, b, a] * c_lcbs[c, b]
                                  for c in range(self.N_CAMPAIGNS) for b in range(len(self.bids)) for a in range(2**self.N_CAMPAIGNS)) <= self.RHO)

        # Marginalization + feasibility constraint
        for a in range(2**self.N_CAMPAIGNS):
            for c in range(self.N_CAMPAIGNS):
                model.add_constr(mip.xsum(gamma[c, b, a] for b in range(len(self.bids))) == marginals[a])
                if a & (1 << c):  # if campaign c is included in the campaigns subset a:
                    model.add_constr(gamma[c, 0, a] == 0)
                else:
                    model.add_constr(gamma[c, 0, a] == marginals[a])
                # Feasibility: force gamma[c, b, a] = 0 for infeasible bids (bids[b] > valuations[c])
                for b in range(len(self.bids)):
                    if not self.feasible[c, b]:
                        model.add_constr(gamma[c, b, a] == 0)

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

