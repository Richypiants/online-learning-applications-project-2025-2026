import numpy as np
import scipy

from campaigns.campaign import Campaign

class SlightlyNonStationaryCampaign(Campaign):
    def __init__(self, ad_qualities):
        super().__init__(ad_qualities)
        self.TYPE = "Slightly Non-Stationary"
        self.DESCRIPTION = "Competing bids sampled from a slightly non-stationary distribution, which is partitioned in stationary phases."

        self.n_phases = np.random.choice(np.arange(2, 4), size=1)[0]      # number of stationary phases in the non-stationary distribution of competing bids
        self.phase_change_ratios = np.random.uniform(low=0.0, high=1.0, size=self.n_phases - 1)
        self.phase_change_ratios.sort()
        self.phase_upper_bounds = np.random.uniform(low=0.0, high=1.0, size=self.n_phases)

    def generate_random_competing_bids(self, n_users, seed):
        super().generate_random_competing_bids(n_users, seed)

        self.phase_change_times = (self.phase_change_ratios * n_users).astype(int)

        self.competing_bids = np.zeros((n_users, self.N_COMPETITORS))
        current_phase = 0
        for t in range(n_users):
            if current_phase < self.n_phases - 1 and t == self.phase_change_times[current_phase]:
                current_phase += 1
            self.competing_bids[t, :] = np.random.uniform(low=0.0, high=self.phase_upper_bounds[current_phase], size=self.N_COMPETITORS)
        return self.competing_bids

    def get_win_probabilities(self, bids_space):
        # true win probabilities per phase: max of N_COMPETITORS uniforms in [0, high] follows Beta(N_COMPETITORS, 1) scaled by high
        win_probs_per_phase = np.array([
            scipy.stats.beta.cdf(bids_space / upper_bound, a=self.N_COMPETITORS, b=1)       # it is a Beta in this case only
            for upper_bound in self.phase_upper_bounds
        ])      # shape: (n_phases, len(bids_space))
        return win_probs_per_phase
    

    def __str__(self):
        return "".join([super().__str__(), f"\nNumber of phases: {self.n_phases}"])

    __repr__ = __str__