import numpy as np
import scipy
import mip

from campaign import Campaign
from conflicts_graph import ConflictsGraph

class Environment:
    def __init__(self, bids_space):
        self.BIDS_SPACE = bids_space

        self.campaigns = []
        self.N_CAMPAIGNS = len(self.campaigns)
        
        self.conflicts_graph = None

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
            f"Campaigns:\n{campaigns}\n"
            f"Conflicts graph (1 = conflicting):\n{self.conflicts_graph}"
        )

    __repr__ = __str__

    def set_bids(self, bids):
        self.BIDS_SPACE = bids

    def generate_random_campaigns(self, n_campaigns, min_competitors=0, max_competitors=10, conflicts_percentage=0.0, seed=17):
        np.random.seed(seed)

        self.campaigns = []
        for _ in range(n_campaigns):
            n_advertisers = np.random.randint(min_competitors + 1, max_competitors + 2)   # includes the bidder's own ad, so the number of competitors is n_advertisers - 1
            # ad_qualities = np.random.beta(2.0, 5.0, n_advertisers)      # chosen to model the fact that it is rare in general that an ad is clicked by a user
            # TODO: confirm that we instead assume that all ad_qualities are 1:
            ad_qualities = np.ones(n_advertisers)
            campaign = Campaign(ad_qualities=ad_qualities)
            self.campaigns.append(campaign)

        self.N_CAMPAIGNS = len(self.campaigns)

        self.conflicts_graph = ConflictsGraph(self.N_CAMPAIGNS)
        self.conflicts_graph.generate_random_graph(conflicts_percentage=conflicts_percentage, seed=seed)

    def set_campaigns(self, campaigns):   
        self.campaigns = campaigns
        self.N_CAMPAIGNS = len(campaigns)  # number of campaigns

    # TODO: at the moment, single campaign only by specifying which campaign to compute the clairvoyant strategy for (this is to compute separate clairvoyants), 
    # but we want to extend it to multiple campaigns later (to compute an overall clairvoyant strategy for all campaigns with a single budget)
    def compute_clairvoyant_strategy_old(self, bidder, campaign_indices=None):
        print("Old implementation!")
        if campaign_indices is None:
            campaign_indices = [0]

        # TODO: move this outside, especially if making the distribution generalized
        # TODO: currently using the same beta distribution for all campaigns to model the environment
        WIN_PROBABILITIES = scipy.stats.beta.cdf(bidder.bids, a=self.campaigns[campaign_indices[0]].N_COMPETITORS, b=1)  # win probabilities for each bid -> note that this is the cdf, not the pdf!

        # TODO: also, currently this is valuations[campaign_index] but the LP will need to be generalized to multiple campaigns
        c = -(bidder.valuations[campaign_indices[0]] - bidder.bids) * WIN_PROBABILITIES       # TODO: should restrict bids space only to bidder.bids, i.e. only the available bids
        a = bidder.bids * WIN_PROBABILITIES
        b = np.ones(len(WIN_PROBABILITIES))

        result = scipy.optimize.linprog(c, 
                                        A_ub=None if bidder.RHO == np.inf else [a], 
                                        b_ub=None if bidder.RHO == np.inf else [bidder.RHO], 
                                        A_eq=[b], 
                                        b_eq=[1], 
                                        bounds=(0, 1))

        return result.x, [], -result.fun, np.sum(bidder.bids * result.x * WIN_PROBABILITIES)

    def compute_clairvoyant_strategy(self, bidder, campaign_indices=None):
        print("New implementation!")
        if campaign_indices is None:
            campaign_indices = list(range(self.N_CAMPAIGNS))

        # alternative implementation using mip (more intuitive to model constraints in code):
        # TODO: move this outside, especially if making the distribution generalized
        WIN_PROBABILITIES = np.array([scipy.stats.beta.cdf(bidder.bids, a=self.campaigns[campaign_index].N_COMPETITORS, b=1) for campaign_index in campaign_indices])
        # TODO: currently using the same beta distribution structure for all campaigns to model the environment (win_probabilities)
        fbar_t = np.array([(bidder.valuations[i] - bidder.bids) * WIN_PROBABILITIES[i] for i in range(len(campaign_indices))])  # expected utility for each bid and campaign
        cbar_t = bidder.bids * WIN_PROBABILITIES

        model = mip.Model()

        # Optimization variables
        gamma = {(c, b, a): model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=1) 
                 for c in range(len(campaign_indices)) for b in range(len(bidder.bids)) for a in range(2**len(campaign_indices))}     # probabilities
        marginals = {(a): model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=1) for a in range(2**len(campaign_indices))}  # marginal probabilities

        # Budget constraint
        model.add_constr(mip.xsum(gamma[c, b, a] * cbar_t[c, b]
                                  for c in range(len(campaign_indices)) for b in range(len(bidder.bids)) for a in range(2**len(campaign_indices))) <= bidder.RHO)

        # Marginalization constraint
        for a in range(2**len(campaign_indices)):
            for c in range(len(campaign_indices)):
                model.add_constr(mip.xsum(gamma[c, b, a] for b in range(len(bidder.bids))) == marginals[a])
                if a & (1 << c):  # if campaign c is included in the campaigns subset a:
                    model.add_constr(gamma[c, 0, a] == 0)
                else:
                    model.add_constr(gamma[c, 0, a] == marginals[a])

        # Probability distribution constraint
        model.add_constr(mip.xsum(marginals[a] for a in range(2**len(campaign_indices))) == 1)

        # Conflicts constraint
        for conflict in self.conflicts_graph.graph:
            campaign_a, campaign_b = conflict
            for a in range(2**len(campaign_indices)):
                if a & (1 << campaign_a) and a & (1 << campaign_b):  # if both campaigns are included in the campaigns subset a:
                    model.add_constr(marginals[a] == 0)

        # Objective function: maximize expected fbar_t(gamma)
        model.objective = mip.maximize(
            mip.xsum(gamma[c, b, a] * fbar_t[c, b]
                     for c in range(len(campaign_indices)) for b in range(len(bidder.bids)) for a in range(2**len(campaign_indices)))
        )
        model.optimize()

        gamma_values = np.array([       # shape: (2**N_CAMPAIGNS, N_CAMPAIGNS, len(BIDS_SPACE))
            [[gamma[c, b, a].x if abs(gamma[c, b, a].x) > 1e-10 else 0 for b in range(len(bidder.bids))] for c in range(len(campaign_indices))] for a in range(2**len(campaign_indices))
        ])

        marginal_values = np.array([
            marginals[a].x if abs(marginals[a].x) > 1e-10 else 0 for a in range(2**len(campaign_indices))       # for cleaning up errors due to mip's feasibility tolerance
        ])

        expected_payment = np.sum(
            gamma_values
            * bidder.bids[None, None, :]    # shape (1, 1, len(BIDS_SPACE))
            * WIN_PROBABILITIES[None, :, :]    # shape (1, len(campaign_indices), len(BIDS_SPACE))
        )

        return gamma_values, marginal_values, model.objective_value, expected_payment


    # TODO: currently works only for a single campaign, but we want to extend it to multiple campaigns later
    # def simulate_environment(self, bidder, n_users, seed=17):
    #     np.random.seed(seed)  # set a random seed for reproducibility

    #     campaign = self.campaigns[0]  # currently only simulating the first campaign; later we can extend this to multiple campaigns

    #     other_bids = np.random.uniform(low=0.0, high=1.0, size=(n_users, campaign.N_COMPETITORS))  # generate random bids for the other advertisers
    #     m_t = np.max(other_bids, axis=1)  # max of the competitors' bids at each round (Beta-distributed)

    #     utilities = []
    #     my_bids = []
    #     my_payments = []
    #     total_wins = 0

    #     for round in range(n_users):
    #         my_bid = bidder.bids[bidder.bid()]  # get the bid from the UCB-like bidder
    #         all_bids = np.concatenate(([my_bid], other_bids[round]))  # combine the bidder's bid with the competitors' bids
    #         winners, payments = campaign.round(all_bids)  # run the auction and get the winners and payments
    #         my_win = int(winners == 0)  # check if the UCB-like bidder won the auction
    #         f_t, c_t = my_win * (bidder.valuation - payments), my_win * payments  # calculate the utility and cost for the UCB-like bidder
    #         bidder.learn(f_t, c_t)  # update the UCB-like bidder's knowledge based on the outcome of the auction

    #         utilities.append(f_t)
    #         my_bids.append(my_bid)
    #         my_payments.append(c_t)
    #         total_wins += my_win

    #     return np.array(utilities), np.array(my_bids), np.array(my_payments), total_wins

    def simulate_environment(self, bidder, n_users, seed=17):
        np.random.seed(seed)  # set a random seed for reproducibility

        # campaign = self.campaigns[0]  # currently only simulating the first campaign; later we can extend this to multiple campaigns

        other_bids = [np.random.uniform(low=0.0, high=1.0, size=(n_users, campaign.N_COMPETITORS)) for campaign in self.campaigns]  # generate random bids for the other advertisers
        m_t = [np.max(other_bids[i], axis=1) for i, _ in enumerate(self.campaigns)]  # max of the competitors' bids at each round (Beta-distributed)

        utilities = []
        my_bids = []
        my_payments = []
        total_wins = [0 for _ in range(self.N_CAMPAIGNS)]

        for round in range(n_users):
            my_bid_indices = bidder.bid()
            f_t = []
            c_t = []

            for i, campaign in enumerate(self.campaigns):
                my_bid = bidder.bids[my_bid_indices[i]]  # get the bid from the UCB-like bidder
                all_bids = np.concatenate(([my_bid], other_bids[i][round]))  # combine the bidder's bid with the competitors' bids
                winners, payments = campaign.round(all_bids)  # run the auction and get the winners and payments
                my_win = int(winners == 0)  # check if the UCB-like bidder won the auction
                f_t.append(my_win * (bidder.valuations[i] - payments))
                c_t.append(my_win * payments)  # calculate the utility and cost for the UCB-like bidder

                total_wins[i] += my_win

            bidder.learn(f_t, c_t)  # update the UCB-like bidder's knowledge based on the outcome of the auction

            utilities.append(f_t)
            my_bids.append(bidder.bids[my_bid_indices])
            my_payments.append(c_t)

        return np.array(utilities), np.array(my_bids), np.array(my_payments), np.array(total_wins)