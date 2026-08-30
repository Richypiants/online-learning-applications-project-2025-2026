import numpy as np
import scipy

from campaigns.campaign import Campaign

class UniformBidsStochasticCampaign(Campaign):
    def __init__(self, ad_qualities):
        super().__init__(ad_qualities)      # consider removing ad_qualities from the constructor of FirstPriceAuction and just passing it to the methods that need it, since the auction itself isn't really defined by the qualities
        self.TYPE = "Uniform Bids"
        self.DESCRIPTION = "Competing bids sampled from a stochastic uniform distribution, maximum bid follows a Beta distribution."

    def generate_random_competing_bids(self, n_users, seed):
        super().generate_random_competing_bids(n_users, seed)
        self.competing_bids = np.random.uniform(low=0.0, high=1.0, size=(n_users, self.N_COMPETITORS))
        return self.competing_bids      # am I sure that I need to return these?

    def get_win_probabilities(self, bids_space):
        win_probs = np.array(scipy.stats.beta.cdf(bids_space, a=self.N_COMPETITORS, b=1))[None, :]      # shape: (1, len(bids_space))
        return win_probs