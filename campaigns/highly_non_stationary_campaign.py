import numpy as np
import scipy

from campaigns.campaign import Campaign

class HighlyNonStationaryCampaign(Campaign):
    def __init__(self, ad_qualities):
        super().__init__(ad_qualities)      # consider removing ad_qualities from the constructor of FirstPriceAuction and just passing it to the methods that need it, since the auction itself isn't really defined by the qualities
        self.TYPE = "Highly Non-Stationary"
        self.DESCRIPTION = "Competing bids sampled from a highly non-stationary distribution which changes quickly over time."

    def generate_random_competing_bids(self, n_users, seed):
        np.random.seed(seed)
        
        # TODO: the following arbitrary pattern/distribution is from the notebook, probably change it
        # non-trivial scenario: competitors sample bids from a uniform with range varying over time
        pattern = lambda t: 1 - np.abs(np.sin(5 * t / n_users))
        self.competing_bids = np.array([np.random.uniform(0, pattern(t), size = self.N_COMPETITORS) for t in range(n_users)]).T
        return self.competing_bids

    def get_win_probabilities(self, bids_space):
        # NOTE: from the non-truthful auctions exercise session notebook: adversarial clairvoyant estimates the overall win probabilities over all time steps
        # ad then chooses one best single arm in hindsight 
        # This is because usually (not always) it is difficult to write the expectation in closed form to be used to choose the best arm in hindsight

        # True probabilities for this scenario/pattern (from the notebook): are actually a quickly-changing Beta distribution

        m_ts = self.get_max_competing_bids()      # shape: (n_users,)
        n_users = len(m_ts)

        win_probs = np.array([sum(b > m_ts) / n_users for b in bids_space])[None, :]
        return win_probs      # shape: (1, len(bids_space))

        # # One could use the true distribution (Beta) and compute the true win probability in this case, but the following code should be fixed and expanded to do so
        # pattern = lambda t: 1 - np.abs(np.sin(5 * t / n_users))
        # win_probs = np.zeros((n_users, bids_space.shape[0]))
        # for t in range(n_users):
        #     win_probs[t] = scipy.stats.beta.cdf(bids_space / pattern(t), a=self.N_COMPETITORS, b=1)
        # return win_probs