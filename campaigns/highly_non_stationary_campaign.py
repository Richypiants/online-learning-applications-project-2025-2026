import numpy as np
import scipy

from campaigns.campaign import Campaign

class HighlyNonStationaryCampaign(Campaign):
    def __init__(self, ad_qualities):
        super().__init__(ad_qualities)
        self.TYPE = "Highly Non-Stationary"
        self.DESCRIPTION = "Competing bids sampled from a highly non-stationary distribution which changes quickly over time."

        # non-trivial scenario: competitors sample bids from a uniform with range varying over time
        self.low = np.random.uniform(0, 0.5)
        self.high = np.random.uniform(0.5, 1.0)

        # market prices move as sinusoids, but all at the same frequencies (e.g. sale periods, black friday, offers, holidays = market condition/regimes)
        self.market_level = lambda t, n_users, low, high: low + (high - low) * (1 - np.abs(np.sin(5 * t / n_users)))

    def generate_random_competing_bids(self, n_users, seed):
        super().generate_random_competing_bids(n_users, seed)

        self.competing_bids = np.array([np.random.uniform(0, self.market_level(t, n_users, self.low, self.high), size = self.N_COMPETITORS) for t in range(n_users)])
        return self.competing_bids

    def get_win_probabilities(self, bids_space):
        # NOTE: from the non-truthful auctions exercise session notebook: adversarial clairvoyant estimates the overall win probabilities over all time steps
        # and then chooses one best single arm in hindsight 
        # This is because usually (not always) it is difficult to write the expectation in closed form to be used to choose the best arm in hindsight

        # NOTE: For this scenario/pattern: phase-wise true probabilities are actually still a Beta distribution, just with quickly-changing in time
        # We just don't use this fact

        m_ts = self.get_max_competing_bids()      # shape: (n_users,)
        n_users = len(m_ts)

        win_probs = np.array([sum(b > m_ts) / n_users for b in bids_space])[None, :]
        return win_probs      # shape: (1, len(bids_space))