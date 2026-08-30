import numpy as np
import scipy
import mip

from campaigns.campaign import Campaign
from environment.conflicts_graph import ConflictsGraph

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
            f"Conflicts graph matrix (1 = conflicting):\n{self.conflicts_graph}"
        )

    __repr__ = __str__

    def set_bids(self, bids):
        self.BIDS_SPACE = bids

    def generate_random_campaigns(self, n_campaigns, campaign_type, min_competitors, max_competitors, conflicts_percentage=0.0, seed=47):
        np.random.seed(seed)

        self.campaigns = []
        for _ in range(n_campaigns):
            n_advertisers = np.random.randint(min_competitors + 1, max_competitors + 2)   # includes the bidder's own ad, so the number of competitors is n_advertisers - 1
            # ad_qualities = np.random.beta(2.0, 5.0, n_advertisers)      # chosen to model the fact that it is rare in general that an ad is clicked by a user
            # TODO: confirm that we instead assume that all ad_qualities are 1:
            ad_qualities = np.ones(n_advertisers)
            campaign = campaign_type(ad_qualities=ad_qualities)
            self.campaigns.append(campaign)

        self.N_CAMPAIGNS = len(self.campaigns)

        self.conflicts_graph = ConflictsGraph(self.N_CAMPAIGNS)
        self.conflicts_graph.generate_random_graph(conflicts_percentage=conflicts_percentage, seed=seed)

    def set_campaigns(self, campaigns):   
        self.campaigns = campaigns
        self.N_CAMPAIGNS = len(campaigns)  # number of campaigns

    # # TODO: at the moment, single campaign only by specifying which campaign to compute the clairvoyant strategy for (this is to compute separate clairvoyants), 
    # # but we want to extend it to multiple campaigns later (to compute an overall clairvoyant strategy for all campaigns with a single budget)
    # def compute_clairvoyant_strategy_old(self, bidder, campaign_indices=None):
    #     print("Old implementation!")
    #     if campaign_indices is None:
    #         campaign_indices = [0]

    #     # TODO: move this outside, especially if making the distribution generalized
    #     # TODO: currently using the same beta distribution for all campaigns to model the environment
    #     WIN_PROBABILITIES = scipy.stats.beta.cdf(bidder.bids, a=self.campaigns[campaign_indices[0]].N_COMPETITORS, b=1)  # win probabilities for each bid -> note that this is the cdf, not the pdf!

    #     # TODO: replace the above with this
    #     win_probabilities = np.array(self.campaigns[campaign_indices[0]].get_win_probabilities(bidder.bids))
    #     #win_probabilities = np.array(self.campaigns[campaign_index].get_win_probabilities(bidder.bids) for campaign_index in campaign_indices)

    #     # TODO: also, currently this is valuations[campaign_index] but the LP will need to be generalized to multiple campaigns
    #     c = -(bidder.valuations[campaign_indices[0]] - bidder.bids) * WIN_PROBABILITIES       # TODO: should restrict bids space only to bidder.bids, i.e. only the available bids
    #     a = bidder.bids * WIN_PROBABILITIES
    #     b = np.ones(len(WIN_PROBABILITIES))

    #     result = scipy.optimize.linprog(c, 
    #                                     A_ub=None if bidder.RHO == np.inf else [a], 
    #                                     b_ub=None if bidder.RHO == np.inf else [bidder.RHO], 
    #                                     A_eq=[b], 
    #                                     b_eq=[1], 
    #                                     bounds=(0, 1))

    #     return result.x, [], -result.fun, np.sum(bidder.bids * result.x * WIN_PROBABILITIES)

    # TODO: still, using a single set of bidder.bids which is the same across all campaigns regardless of the valuation...
    # But I think this could be solved easily in this case, because the gamma corresponding to the larger bids would be zero
    # NOTE: a LP is still needed instead of just picking the best performing subset for two reasons:
    # 1) a distribution might be optimal
    # 2) we need to enforce the constraints (could be done through the feasible superarms/subsets too actually)
    def compute_clairvoyant_strategy_combinatorial_simple(self, bidder, campaign_indices=None):
        print("Clairvoyant - combinatorial over campaigns - simple version")
        if campaign_indices is None:
            campaign_indices = list(range(self.N_CAMPAIGNS))

        campaign_phase_change_times = []
        campaign_gamma_values = []
        campaign_f_bars = []
        campaign_c_bars = []
        
        for campaign_index in campaign_indices:
            # TODO: problems: 
            # 2) the gamma_values contain the bid 0.0, which still must be removed somehow!
            # 3) I REALLY think that the LP should use the updated RHO value at each phase, otherwise this wouldn't be realistic -> actually no, because the LP 
            # constraint is enforcing an expected expense per-round, which is the same regardless of the phase -> we should be spending B/T at each round regardless 
            # of the phase, in a sense!
            # -> This question becomes important if the clairvoyant could decide to bid only in one of the phases (like the suboptimality proof for "pacing" in 
            # adversarial auctions, the one that splits the rounds in half and proves that a bidder cannot decide to wait out for a better phase), since in that case 
            # we don't want to bid RHO at each round, because RHO is different among the phases! But assuming to know the different phases in advance is unreasonable.
            # The problem that is truly important is: if we bid on multiple campaigns together, do we still satisfy the budget constraint in expectation since in 
            # theory we are bidding N*RHO at each round? -> likely not! the bidder is forced to not bid on anything with some probability if we still use RHO!
            # -> what if we use RHO/N in the combinatorial LP instead?
            gamma_values, f_bars, c_bars = self.campaigns[campaign_index].single_campaign_clairvoyant(bidder.bids[1:], bidder.valuations[campaign_index], bidder.RHO)
            phase_change_times = np.concatenate([np.array([0]), self.campaigns[campaign_index].phase_change_times])

            campaign_phase_change_times.append(phase_change_times)
            campaign_gamma_values.append(gamma_values)
            campaign_f_bars.append(f_bars)
            campaign_c_bars.append(c_bars)

        joined_phase_change_times = np.unique(np.concatenate(campaign_phase_change_times))
        joined_indices = [np.searchsorted(phase_times, joined_phase_change_times, side="right") - 1 for phase_times in campaign_phase_change_times]
        print("campaign_phase_change_times:", campaign_phase_change_times)
        print("joined_phase_change_times:", joined_phase_change_times)
        print("joined_indices:", joined_indices)
        print("campaign_gamma_values:", campaign_gamma_values)
        
        phase_wise_campaign_gamma_values = np.array([gamma_values[idx] for gamma_values, idx in zip(campaign_gamma_values, joined_indices)])         # shape: (N_CAMPAIGNS, len(joined_phase_change_times), len(bidder.bids))
        phase_wise_campaign_f_bars = np.array([f_bars[idx] for f_bars, idx in zip(campaign_f_bars, joined_indices)])     # shape: (N_CAMPAIGNS, len(joined_phase_change_times))
        phase_wise_campaign_c_bars = np.array([c_bars[idx] for c_bars, idx in zip(campaign_c_bars, joined_indices)])     # shape: (N_CAMPAIGNS, len(joined_phase_change_times))

        phase_wise_superarm_f_bars = np.zeros((2**self.N_CAMPAIGNS, len(joined_phase_change_times)))
        phase_wise_superarm_c_bars = np.zeros((2**self.N_CAMPAIGNS, len(joined_phase_change_times)))
        for a in range(2**self.N_CAMPAIGNS):
            for i in range(self.N_CAMPAIGNS):
                if a & (1 << i):  # if campaign i is included in the campaigns subset a:
                    phase_wise_superarm_f_bars[a] += phase_wise_campaign_f_bars[i]          # NOTE: check: should be summing shape (len(joined_phase_change_times),) with shape (len(joined_phase_change_times),) -> should be fine
                    phase_wise_superarm_c_bars[a] += phase_wise_campaign_c_bars[i]

        phase_wise_superarm_gamma_values = []
        phase_wise_objective_values = []
        phase_wise_expected_payments = []

        # running one separate LP for each phase_change_time
        for phase_index, phase_change_time in enumerate(joined_phase_change_times):
            model = mip.Model()
            
            # Optimization variables
            gamma_superarm = {(a): model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=1) for a in range(2**len(campaign_indices))}     # probabilities

            # Budget constraint
            model.add_constr(mip.xsum(gamma_superarm[a] * phase_wise_superarm_c_bars[a, phase_index] for a in range(2**len(campaign_indices))) <= bidder.RHO)

            # Probability distribution constraint
            model.add_constr(mip.xsum(gamma_superarm[a] for a in range(2**len(campaign_indices))) == 1)

            # Conflicts constraint
            for conflict in self.conflicts_graph.graph:
                campaign_a, campaign_b = conflict
                for a in range(2**len(campaign_indices)):
                    if a & (1 << campaign_a) and a & (1 << campaign_b):  # if both campaigns are included in the campaigns subset a:
                        model.add_constr(gamma_superarm[a] == 0)

            # Objective function: maximize expected fbar_t(gamma)
            model.objective = mip.maximize(
                mip.xsum(gamma_superarm[a] * phase_wise_superarm_f_bars[a, phase_index] for a in range(2**len(campaign_indices)))
            )
            model.optimize()

            superarm_gamma_values = np.array([
                gamma_superarm[a].x if abs(gamma_superarm[a].x) > 1e-10 else 0 for a in range(2**len(campaign_indices))
            ])

            expected_payment = np.sum(
                superarm_gamma_values * phase_wise_superarm_c_bars[:, phase_index]
            )

            phase_wise_superarm_gamma_values.append(superarm_gamma_values)
            phase_wise_expected_payments.append(expected_payment)
            phase_wise_objective_values.append(model.objective_value)

        phase_wise_superarm_gamma_values = np.array(phase_wise_superarm_gamma_values)       # shape: (len(joined_phase_change_times), 2**len(campaign_indices))
        phase_wise_objective_values = np.array(phase_wise_objective_values)                 # shape: (len(joined_phase_change_times),)
        phase_wise_expected_payments = np.array(phase_wise_expected_payments)               # shape: (len(joined_phase_change_times),)

        repeat_counts = np.diff(np.concatenate((joined_phase_change_times, [bidder.T])))
        phase_wise_objective_values = np.repeat(phase_wise_objective_values, repeat_counts) 
        phase_wise_expected_payments = np.repeat(phase_wise_expected_payments, repeat_counts) 

        # TODO: currently repeating the values according to the joined_phase_change_times, an alternative is to return the change times so that they can also be visualized
        return phase_wise_campaign_gamma_values, phase_wise_superarm_gamma_values, phase_wise_objective_values, phase_wise_expected_payments

    def compute_clairvoyant_strategy_combinatorial_complex(self, bidder, campaign_indices=None):
        print("Clairvoyant - combinatorial over campaigns - complex version")
        if campaign_indices is None:
            campaign_indices = list(range(self.N_CAMPAIGNS))

        # TODO: move this outside, especially if making the distribution generalized
        #WIN_PROBABILITIES = np.array([scipy.stats.beta.cdf(bidder.bids, a=self.campaigns[campaign_index].N_COMPETITORS, b=1) for campaign_index in campaign_indices])
        # TODO: currently using the same beta distribution structure for all campaigns to model the environment (win_probabilities)
        
        win_probabilities = np.array([self.campaigns[campaign_index].get_win_probabilities(bidder.bids) for campaign_index in campaign_indices])    # shape: (N_CAMPAIGNS, len(bidder.bids))

        # alternative implementation using mip (more intuitive to model constraints in code):
        fbar_t = np.array([(bidder.valuations[i] - bidder.bids) * win_probabilities[i] for i in range(len(campaign_indices))])  # expected utility for each bid and campaign
        cbar_t = bidder.bids * win_probabilities

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
            * win_probabilities[None, :, :]    # shape (1, len(campaign_indices), len(BIDS_SPACE))
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

    # def simulate_environment(self, bidder, n_users, seed=17):
    #     print("Newish but old implementation")
    #     np.random.seed(seed)  # set a random seed for reproducibility

    #     # campaign = self.campaigns[0]  # currently only simulating the first campaign; later we can extend this to multiple campaigns

    #     #other_bids = [np.random.uniform(low=0.0, high=1.0, size=(n_users, campaign.N_COMPETITORS)) for campaign in self.campaigns]  # generate random bids for the other advertisers (max is Beta-distributed)
    #     other_bids = [campaign.generate_random_competing_bids(n_users) for campaign in self.campaigns]  # generate random bids for the other advertisers
    #     m_t = [np.max(other_bids[i], axis=1) for i, _ in enumerate(self.campaigns)]  # max of the competitors' bids at each round 

    #     utilities = []
    #     my_bids = []
    #     my_payments = []
    #     total_wins = [0 for _ in range(self.N_CAMPAIGNS)]

    #     for round in range(n_users):
    #         my_bid_indices = bidder.bid()
    #         f_t = []
    #         c_t = []

    #         for i, campaign in enumerate(self.campaigns):
    #             my_bid = bidder.bids[my_bid_indices[i]]  # get the bid from the UCB-like bidder
    #             all_bids = np.concatenate(([my_bid], other_bids[i][round]))  # combine the bidder's bid with the competitors' bids
    #             winners, payments = campaign.round(all_bids)  # run the auction and get the winners and payments
    #             my_win = int(winners == 0)  # check if the UCB-like bidder won the auction
    #             f_t.append(my_win * (bidder.valuations[i] - payments))
    #             c_t.append(my_win * payments)  # calculate the utility and cost for the UCB-like bidder

    #             total_wins[i] += my_win

    #         bidder.learn(f_t, c_t, m_t=[m_t[i][round] for i in range(self.N_CAMPAIGNS)])  # update the bidder's knowledge based on the outcome of the auction (full feedback: highest competing bids)

    #         utilities.append(f_t)
    #         my_bids.append(bidder.bids[my_bid_indices])
    #         my_payments.append(c_t)

    #     return np.array(utilities), np.array(my_bids), np.array(my_payments), np.array(total_wins)

    def simulate_environment(self, bidder, n_users, seed=17):
        np.random.seed(seed)  # set a random seed for reproducibility

        other_bids = [campaign.generate_random_competing_bids(n_users, seed=seed+i*217) for i, campaign in enumerate(self.campaigns)]  # generate random bids for the other advertisers  shape: (N_CAMPAIGNS, n_users, N_COMPETITORS)
        m_t = np.array([campaign.get_max_competing_bids() for campaign in self.campaigns])  # max of the competitors' bids at each round -> shape: (N_CAMPAIGNS, n_users)

        utilities = []
        my_bids = []
        my_payments = []
        total_wins = [0 for _ in range(self.N_CAMPAIGNS)]

        for round in range(n_users):
            my_bid_indices = bidder.bid()
            f_t = []
            c_t = []

            for campaign_index, campaign in enumerate(self.campaigns):
                my_bid = bidder.bids[my_bid_indices[campaign_index]]  # get the bid from the UCB-like bidder
                all_bids = np.concatenate(([my_bid], other_bids[campaign_index][round]))  # combine the bidder's bid with the competitors' bids
                winners, payments = campaign.round(all_bids)  # run the auction and get the winners and payments
                my_win = int(winners == 0)  # check if the UCB-like bidder won the auction
                f_t.append(my_win * (bidder.valuations[campaign_index] - payments))
                c_t.append(my_win * payments)  # calculate the utility and cost for the UCB-like bidder

                total_wins[campaign_index] += my_win

            bidder.learn(f_t, c_t, m_t=m_t[:, round])  # update the bidder's knowledge based on the outcome of the auction (full feedback: highest competing bids)

            utilities.append(f_t)
            my_bids.append(bidder.bids[my_bid_indices])
            my_payments.append(c_t)

        return np.array(utilities), np.array(my_bids), np.array(my_payments), np.array(total_wins)