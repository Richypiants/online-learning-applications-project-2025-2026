import numpy as np
import scipy

import matplotlib.pyplot as plt

from campaigns.first_price_auction import FirstPriceAuction

class Campaign(FirstPriceAuction):
    def __init__(self, ad_qualities):
        super().__init__(ad_qualities)      # consider removing ad_qualities from the constructor of FirstPriceAuction and just passing it to the methods that need it, since the auction itself isn't really defined by the qualities
        self.N_ADVERTISERS = len(ad_qualities)
        self.N_COMPETITORS = self.N_ADVERTISERS - 1
        self.TYPE = "UNDEFINED"
        self.DESCRIPTION = "UNDEFINED"
        self.competing_bids = None
        self.phase_change_times = np.array([], dtype=np.int64)

    # TODO: add seed to seed everything, including the random generation of competing bids, so that we can have reproducible experiments
    # maybe seed here so that it is not forgotten and repeated?
    def generate_random_competing_bids(self, n_users, seed):
        np.random.seed(seed)
        #raise NotImplementedError()

    def get_max_competing_bids(self):
        if self.competing_bids is None:
            raise ValueError("Competing bids have not been generated yet.")
        return np.max(self.competing_bids, axis=1)
    
    def plot_max_competing_bids(self):
        plt.plot(self.get_max_competing_bids())
        plt.title('Sampled maximum bids in time')
        plt.xlabel('$t$')
        plt.ylabel('$m_t$')
        plt.show()

    def get_win_probabilities(self, bids_space):
        "Must return a shape (n_phases, len(bids_space)) for both stationary and non-stationary scenarios for compatibility."
        "For stationary scenarios, n_phases = 1."
        raise NotImplementedError()
    
    def single_campaign_clairvoyant(self, bids_space, bidder_valuation, bidder_rho):
        win_probs_per_phase = self.get_win_probabilities(bids_space)

        gamma_values = []
        f_bars = []
        c_bars = []

        for win_probabilities in win_probs_per_phase:
            c = -(bidder_valuation - bids_space) * win_probabilities        # = f_bar(b)
            a = bids_space * win_probabilities                              # = c_bar(b)
            b = np.ones(len(win_probabilities))

            result = scipy.optimize.linprog(c, 
                                            A_ub=None if bidder_rho == np.inf else [a], 
                                            b_ub=None if bidder_rho == np.inf else [bidder_rho], 
                                            A_eq=[b], 
                                            b_eq=[1], 
                                            bounds=(0, 1))
            gamma_values.append(result.x)
            f_bars.append(-result.fun)                                              # = f_bar(gamma)
            c_bars.append(np.sum(result.x * bids_space * win_probabilities))        # = c_bar(gamma)

        return np.array(gamma_values), np.array(f_bars), np.array(c_bars)           # shapes: (n_phases, len(bids_space)), (n_phases,), (n_phases)

    def __str__(self):
        return (
            "-- CAMPAIGN: --\n"
            f"Type: {self.TYPE}\n"
            f"Description: {self.DESCRIPTION}\n"
            f"Ad qualities: {self.qs}\n"
            f"Number of advertisers: {self.N_ADVERTISERS}\n"
            f"Number of competitors: {self.N_COMPETITORS}"
        )

    __repr__ = __str__