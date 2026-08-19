import numpy as np
import scipy

from campaign import Campaign

class Environment:
    def __init__(self, bids_space):
        self.BIDS_SPACE = bids_space

        self.campaigns = []
        self.N_CAMPAIGNS = len(self.campaigns)

    def __str__(self):
        campaigns = "\n".join(
            f"    {line}"
            for campaign in self.campaigns
            for line in repr(campaign).splitlines()
        )

        return (
            "-- ENVIRONMENT: --\n"
            f"Bids space: {self.BIDS_SPACE}\n"
            f"Number of campaigns: {self.N_CAMPAIGNS}\n"
            f"Campaigns:\n{campaigns}"
        )

    __repr__ = __str__

    def set_bids(self, bids):
        self.BIDS_SPACE = bids

    def generate_random_campaigns(self, n_campaigns, min_competitors=0, max_competitors=10, seed=17):
        np.random.seed(seed)
    
        for _ in range(n_campaigns):
            n_advertisers = np.random.randint(min_competitors + 1, max_competitors + 1)   # includes the bidder's own ad, so the number of competitors is n_advertisers - 1
            # ad_qualities = np.random.beta(2.0, 5.0, n_advertisers)      # chosen to model the fact that it is rare in general that an ad is clicked by a user
            # TODO: confirm that we instead assume that all ad_qualities are 1:
            ad_qualities = np.ones(n_advertisers)
            campaign = Campaign(ad_qualities=ad_qualities)
            self.campaigns.append(campaign)

        self.N_CAMPAIGNS = len(self.campaigns)

    def set_campaigns(self, campaigns):   
        self.campaigns = campaigns
        self.N_CAMPAIGNS = len(campaigns)  # number of campaigns

    # TODO: at the moment, single campaign only
    def compute_clairvoyant_strategy(self, bidder):
        # TODO: move this outside, especially if making the distribution generalized
        WIN_PROBABILITIES = scipy.stats.beta.cdf(self.BIDS_SPACE, a=self.campaigns[0].N_COMPETITORS, b=1)  # win probabilities for each bid -> note that this is the cdf, not the pdf!

        c = -(bidder.valuation - self.BIDS_SPACE) * WIN_PROBABILITIES       # TODO: should restrict bids space only to bidder.bids, i.e. only the available bids
        a = self.BIDS_SPACE * WIN_PROBABILITIES
        b = np.ones(len(WIN_PROBABILITIES))

        result = scipy.optimize.linprog(c, 
                                        A_ub=None if bidder.RHO == np.inf else [a], 
                                        b_ub=None if bidder.RHO == np.inf else [bidder.RHO], 
                                        A_eq=[b], 
                                        b_eq=[1], 
                                        bounds=(0, 1))

        return result.x, -result.fun, np.sum(self.BIDS_SPACE * result.x * WIN_PROBABILITIES)

    # TODO: currently works only for a single campaign, but we want to extend it to multiple campaigns later
    def simulate_environment(self, bidder, n_users, seed=17):
        np.random.seed(seed)  # set a random seed for reproducibility

        campaign = self.campaigns[0]  # currently only simulating the first campaign; later we can extend this to multiple campaigns

        other_bids = np.random.uniform(low=0.0, high=1.0, size=(n_users, campaign.N_COMPETITORS))  # generate random bids for the other advertisers
        m_t = np.max(other_bids, axis=1)  # max of the competitors' bids at each round (Beta-distributed)

        utilities = []
        my_bids = []
        my_payments = []
        total_wins = 0

        for round in range(n_users):
            my_bid = bidder.bids[bidder.bid()]  # get the bid from the UCB-like bidder
            all_bids = np.concatenate(([my_bid], other_bids[round]))  # combine the bidder's bid with the competitors' bids
            winners, payments = campaign.round(all_bids)  # run the auction and get the winners and payments
            my_win = int(winners == 0)  # check if the UCB-like bidder won the auction
            f_t, c_t = my_win * (bidder.valuation - payments), my_win * payments  # calculate the utility and cost for the UCB-like bidder
            bidder.learn(f_t, c_t)  # update the UCB-like bidder's knowledge based on the outcome of the auction

            utilities.append(f_t)
            my_bids.append(my_bid)
            my_payments.append(c_t)
            total_wins += my_win

        return np.array(utilities), np.array(my_bids), np.array(my_payments), total_wins