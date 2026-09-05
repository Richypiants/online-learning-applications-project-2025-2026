import numpy as np

class CUSUMChangeDetector:
    def __init__(self, n_campaigns, K, M, epsilon, h, alpha, feasible=None):
        self.N_CAMPAIGNS = n_campaigns
        self.K = K
        self.M = M
        self.epsilon = epsilon
        self.h = h
        self.alpha = alpha

        # feasible: optional (N_CAMPAIGNS, K) bool mask; infeasible arms are initialized with 0 forced-exploration counters
        self.feasible = feasible if feasible is not None else np.ones((self.N_CAMPAIGNS, self.K), dtype=bool)

        self.exploration_counters = np.full((self.N_CAMPAIGNS, self.K), self.M)     # forced-exploration rounds remaining per-arm per-campaign
        self.exploration_counters[~self.feasible] = 0                               # infeasible arms are never scheduled for estimation
        self.f_bar = np.zeros((self.N_CAMPAIGNS, self.K))                           # estimated utilities
        self.g_plus = np.zeros((self.N_CAMPAIGNS, self.K))                          # positive CUSUM statistics
        self.g_minus = np.zeros((self.N_CAMPAIGNS, self.K))                         # negative CUSUM statistics

    def require_estimation(self):
        arms_to_explore_per_campaign = np.argsort(self.exploration_counters, axis=1)[:, -1]
        maximum_required_exploration_per_arm = self.exploration_counters[np.arange(self.N_CAMPAIGNS), arms_to_explore_per_campaign]
        campaign_priorities = np.argsort(maximum_required_exploration_per_arm)[::-1]
        campaign_priorities = [campaign for campaign in campaign_priorities if maximum_required_exploration_per_arm[campaign] > 0]      # TODO: check if this is correct

        return campaign_priorities, arms_to_explore_per_campaign
    
    def detect(self, f_t, played_arms):
        noise = f_t - self.f_bar[np.arange(self.N_CAMPAIGNS), played_arms]

        s_plus = noise - self.epsilon 
        s_minus = -noise - self.epsilon

        # handle the case where the arm is still being estimated (counters > 0): we don't reset anything for these
        samples_remaining = self.exploration_counters[np.arange(self.N_CAMPAIGNS), played_arms]
        is_being_estimated = samples_remaining > 0
        current_estimates = self.f_bar[np.arange(self.N_CAMPAIGNS), played_arms]
        self.f_bar[np.arange(self.N_CAMPAIGNS), played_arms] = np.where(is_being_estimated,
                                                                        current_estimates + (f_t - current_estimates) / (self.M - samples_remaining + 1),
                                                                        current_estimates
                                                                        )

        # then, check for changes and eventually reset
        self.g_plus[np.arange(self.N_CAMPAIGNS), played_arms] = np.clip(self.g_plus[np.arange(self.N_CAMPAIGNS), played_arms] + s_plus, 0, None)
        self.g_minus[np.arange(self.N_CAMPAIGNS), played_arms] = np.clip(self.g_minus[np.arange(self.N_CAMPAIGNS), played_arms] + s_minus, 0, None)

        positive_changes = self.g_plus[np.arange(self.N_CAMPAIGNS), played_arms] > self.h
        negative_changes = self.g_minus[np.arange(self.N_CAMPAIGNS), played_arms] > self.h
        changed = ~is_being_estimated & (positive_changes | negative_changes)

        self.exploration_counters[np.arange(self.N_CAMPAIGNS), played_arms] = np.where(changed, 
                                                                                       self.M, 
                                                                                       np.clip(samples_remaining - 1, 0, self.M)
                                                                                       )
        self.f_bar[np.arange(self.N_CAMPAIGNS), played_arms] *= ~changed
        self.g_plus[np.arange(self.N_CAMPAIGNS), played_arms] *= ~changed
        self.g_minus[np.arange(self.N_CAMPAIGNS), played_arms] *= ~changed

        return changed