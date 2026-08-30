import numpy as np
import scipy

from campaigns.campaign import Campaign

class SlightlyNonStationaryCampaign(Campaign):
    def __init__(self, ad_qualities):
        super().__init__(ad_qualities)      # consider removing ad_qualities from the constructor of FirstPriceAuction and just passing it to the methods that need it, since the auction itself isn't really defined by the qualities
        self.TYPE = "Slightly Non-Stationary"
        self.DESCRIPTION = "Competing bids sampled from a slightly non-stationary distribution, which is partitioned in stationary phases."

        self.n_phases = np.random.choice(np.arange(2, 6), size=1)[0]      # number of stationary phases in the non-stationary distribution of competing bids

    def generate_random_competing_bids(self, n_users, seed):
        super().generate_random_competing_bids(n_users, seed)

        self.phase_change_times = np.random.choice(n_users, self.n_phases - 1, replace=False)
        self.phase_change_times.sort()
        self.phase_upper_bounds = np.random.uniform(low=0.0, high=1.0, size=self.n_phases)

        # TODO: change this to a more realistic non-stationary distribution, e.g., a mixture of Gaussians with changing means and variances over time, or a sinusoidal function with noise, or a random walk with drift, etc.
        self.competing_bids = np.zeros((n_users, self.N_COMPETITORS))
        current_phase = 0
        for t in range(n_users):
            if current_phase < self.n_phases - 1 and t == self.phase_change_times[current_phase]:
                current_phase += 1
            self.competing_bids[t, :] = np.random.uniform(low=0.0, high=self.phase_upper_bounds[current_phase], size=self.N_COMPETITORS)
        return self.competing_bids

    # def generate_random_competing_bids(self, n_users):
    #     # rounds are partitioned in intervals; in each interval the distribution of the highest competing bid is fixed (uniform with phase-dependent upper bound)
    #     self.competing_bids = np.zeros((n_users, self.N_COMPETITORS))
    #     self.phase_bounds = np.linspace(0.5, 1.0, self.n_phases)
    #     for t in range(n_users):
    #         phase = min(t * self.n_phases // n_users, self.n_phases - 1)
    #         self.competing_bids[t, :] = np.random.uniform(low=0.0, high=self.phase_bounds[phase], size=self.N_COMPETITORS)
    #     return self.competing_bids

    # TODO: fix according to the new random_generation after having made a more realistic one
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