import numpy as np

from bidders.UCB_based.combinatorial_ucb_like_bidder import CombinatorialUCBLikeBidder
from bidders.UCB_based.cusum_change_detector import CUSUMChangeDetector

class CUSUMCombinatorialUCBLikeBidder(CombinatorialUCBLikeBidder):
    '''Combinatorial-UCB with a CUSUM change detector, for slightly non-stationary environments.

    Each (campaign, arm) pair is monitored with a CUSUM test on the observed utility; when a
    change is detected, the statistics of that arm are reset and forced exploration resumes.
    '''

    def __init__(self, B, T, valuations, environment, M, eps, h, alpha):
        super().__init__(B, T, valuations, environment)

        self.M = M            # number of initial samples per arm for calibration
        self.h = h            # detection threshold
        self.alpha = alpha    # probability of extra exploration (matching the notebook's CUSUM-UCB)
        self.eps = eps        # tolerance in the CUSUM test (drift allowed before signaling)

        self.change_detector = CUSUMChangeDetector(self.N_CAMPAIGNS, self.K, self.M, self.eps, self.h, self.alpha)

        self.n_resets = np.zeros((self.N_CAMPAIGNS, self.K), dtype=int)            # log: number of detected changes per arm
        self.reset_history = [[[] for _ in range(self.K)] for _ in range(self.N_CAMPAIGNS)]   # log: timesteps at which changes were detected

        # N_pulls gets reset to 0 on a change so the inherited _confidence_bounds stays correct;
        # we keep a running total of pulls separately.
        self.N_pulls_total = np.zeros((self.N_CAMPAIGNS, self.K))

    # Overriding the method to return the correct statistic
    def get_total_pulls_per_arm(self):
        return self.N_pulls_total

    # Should not be needed, since now we use N_pulls, thus it should be the same as the superclass 
    # def _confidence_bounds(self):
    #     mask = self.N_pulls != 0
    #     radius = np.full_like(self.N_pulls, self.range)
    #     radius[mask] *= np.sqrt(2 * np.log(self.t) / self.N_pulls[mask])
    #     f_ucbs = np.clip(self.avg_f + radius, None, 1)
    #     c_lcbs = np.clip(self.avg_c - radius, 0, None)
    #     return f_ucbs, c_lcbs

    def _choose_actions(self, f_ucbs, c_lcbs):
        campaigns_to_explore, arms_to_explore = self.change_detector.require_estimation()
        if campaigns_to_explore.size > 0:
            self._estimation_action(campaigns_to_explore, arms_to_explore)

        # otherwise: standard action with prob 1-alpha, or extra exploration with prob alpha
        if np.random.random() <= 1 - self.alpha:
            return super()._choose_actions(f_ucbs, c_lcbs)
        # extra exploration: pick a random feasible superarm and bid
        return self._exploration_actions()

    def _estimation_action(self, campaigns_to_explore, arms_to_explore):
        actions = np.zeros(self.N_CAMPAIGNS, dtype=int)
                    
        feasible_superarms = [i for i in range(2**self.N_CAMPAIGNS)]
        for conflict in self.environment.conflicts_graph.graph:
            campaign_a, campaign_b = conflict
            for a in feasible_superarms:
                if a & (1 << campaign_a) and a & (1 << campaign_b):  # if both campaigns are included in the campaigns subset a:
                    feasible_superarms.remove(a)

        for campaign in campaigns_to_explore:
            new_feasible_superarms = [a for a in feasible_superarms if (a & (1 << campaign))]  # keep only superarms that include this campaign
            if len(new_feasible_superarms) > 0:
                feasible_superarms = new_feasible_superarms
            else: 
                break

        # superarm = np.random.choice(feasible_superarms)  # pick a random feasible superarm
        superarm = feasible_superarms[0]  # pick the last feasible superarm (arbitrary choice)
        # TODO: Element 0 means do not bid in any other campaign, because feasible superarms are in increasing order
        # At the same time, picking the last one does not guarantee that the most campaigns are selected, and in general there isn't 
        # a choice that guaranteed the max reward without using a LP on the remaining campaigns 
        # One could choose a random arm, but maybe it is better to play conservatively, because we would be wasting budget if the 
        # new arm that is being estimated has become exceptionally good: the clairvoyant would know this, and would wait until it can play this
        # At the same time, waiting to bid in other campaigns wastes a lot of rounds, potentially too many, leading to the budget being 
        # under-utilized if RHO isn't updated accordingly

        for campaign in range(self.N_CAMPAIGNS):
            if campaign in campaigns_to_explore:
                actions[campaign] = arms_to_explore[campaign]  # pick the arm that was signaled for exploration
            elif (superarm & (1 << campaign)):
                actions[campaign] = 1 + np.random.choice(self.K - 1, p=np.full(self.K - 1, 1.0 / (self.K - 1)))  # sample the bid for campaign uniformly excluding the 0.0 bid
        return actions        

    def learn(self, f_t, c_t, m_t=None):
        super().learn(f_t, c_t)
        self.N_pulls_total[np.arange(self.N_CAMPAIGNS), self.a_t] += 1
        self._handle_change(self.change_detector.detect())

    def _handle_change(self, changed_arms):
        self.N_pulls[np.arange(self.N_CAMPAIGNS), self.a_t] *= ~changed_arms
        self.avg_f[np.arange(self.N_CAMPAIGNS), self.a_t] *= ~changed_arms
        self.avg_c[np.arange(self.N_CAMPAIGNS), self.a_t] *= ~changed_arms
        self.n_resets[np.arange(self.N_CAMPAIGNS), self.a_t] += changed_arms
        for campaign, changed in enumerate(changed_arms):
            if changed:
                self.reset_history[campaign][self.a_t[campaign]].append(self.t)
