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
            # we assume instead that all ad_qualities are equal to 1
            ad_qualities = np.ones(n_advertisers)
            campaign = campaign_type(ad_qualities=ad_qualities)
            self.campaigns.append(campaign)

        self.N_CAMPAIGNS = len(self.campaigns)

        self.conflicts_graph = ConflictsGraph(self.N_CAMPAIGNS)
        self.conflicts_graph.generate_random_graph(conflicts_percentage=conflicts_percentage, seed=seed)

    def set_campaigns(self, campaigns):   
        self.campaigns = campaigns
        self.N_CAMPAIGNS = len(campaigns)  # number of campaigns

    # NOTE: a LP that outputs a distribution is always needed instead of just picking the best performing subset for two reasons:
    # 1) a distribution might be optimal
    # 2) we need to enforce the conflict constraints
    def compute_simplified_combinatorial_clairvoyant(self, bidder, campaign_indices=None):
        if campaign_indices is None:
            campaign_indices = list(range(self.N_CAMPAIGNS))

        campaign_phase_change_times = []
        campaign_gamma_values = []
        campaign_f_bars = []
        campaign_c_bars = []
        
        for campaign_index in campaign_indices:
            # NOTE: we are forcing a per-round budget expense equal to RHO for EACH campaign
            # When we bid on multiple campaigns, this will inevitably be exceeded, so the superarm LP will be conservative and choose 
            # either to bid on one campaign at a time or not on anything at all sometimes!

            # the 0.0 bid is removed since it is implicit in the choice of not bidding on a campaign
            feasible_nonzero_mask = bidder.bids[1:] <= bidder.valuations[campaign_index]                   # length-K-1 boolean mask
            campaign_feasible_bids = bidder.bids[1:][feasible_nonzero_mask]                                # per-campaign non-zero bids that are feasible
            gamma_values, f_bars, c_bars = self.campaigns[campaign_index].single_campaign_clairvoyant(campaign_feasible_bids, bidder.valuations[campaign_index], bidder.RHO)
            phase_change_times = np.concatenate([np.array([0]), self.campaigns[campaign_index].phase_change_times])

            campaign_phase_change_times.append(phase_change_times)
            # pad gamma_values to the common length K-1 by zero-filling infeasible arms, so the joined array stays rectangular across campaigns
            padded_gamma_values = np.zeros((gamma_values.shape[0], bidder.bids.shape[0] - 1))
            padded_gamma_values[:, feasible_nonzero_mask] = gamma_values
            campaign_gamma_values.append(padded_gamma_values)
            campaign_f_bars.append(f_bars)
            campaign_c_bars.append(c_bars)

        joined_phase_change_times = np.unique(np.concatenate(campaign_phase_change_times))
        joined_indices = [np.searchsorted(phase_times, joined_phase_change_times, side="right") - 1 for phase_times in campaign_phase_change_times]
        
        phase_wise_campaign_gamma_values = np.array([gamma_values[idx] for gamma_values, idx in zip(campaign_gamma_values, joined_indices)])         # shape: (N_CAMPAIGNS, len(joined_phase_change_times), K-1)
        phase_wise_campaign_f_bars = np.array([f_bars[idx] for f_bars, idx in zip(campaign_f_bars, joined_indices)])     # shape: (N_CAMPAIGNS, len(joined_phase_change_times))
        phase_wise_campaign_c_bars = np.array([c_bars[idx] for c_bars, idx in zip(campaign_c_bars, joined_indices)])     # shape: (N_CAMPAIGNS, len(joined_phase_change_times))

        phase_wise_superarm_f_bars = np.zeros((2**len(campaign_indices), len(joined_phase_change_times)))
        phase_wise_superarm_c_bars = np.zeros((2**len(campaign_indices), len(joined_phase_change_times)))
        for a in range(2**len(campaign_indices)):
            for i in range(len(campaign_indices)):
                if a & (1 << i):  # if campaign i is included in the campaigns subset a:
                    phase_wise_superarm_f_bars[a] += phase_wise_campaign_f_bars[i]
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

        return phase_wise_campaign_gamma_values, phase_wise_superarm_gamma_values, phase_wise_objective_values, phase_wise_expected_payments

    def compute_true_combinatorial_clairvoyant(self, bidder, campaign_indices=None):
        if campaign_indices is None:
            campaign_indices = list(range(self.N_CAMPAIGNS))

        # Per-campaign win probabilities, shape (N_CAMPAIGNS, n_phases_c, len(bidder.bids))
        # Each campaign may have its own number of phases (e.g. stationary = 1, slightly non-stationary = up to 5)
        campaign_win_probabilities = [self.campaigns[campaign_index].get_win_probabilities(bidder.bids) for campaign_index in campaign_indices]

        # Per-campaign phase change times (with 0 prepended), shape (N_CAMPAIGNS, n_phases_c + 1)
        campaign_phase_change_times = [np.concatenate([np.array([0]), self.campaigns[campaign_index].phase_change_times]) for campaign_index in campaign_indices]

        # Joined (union) phase change times across all campaigns
        joined_phase_change_times = np.unique(np.concatenate(campaign_phase_change_times))
        # For each campaign, find the index of its phase at each joined phase change time
        joined_indices = [np.searchsorted(phase_times, joined_phase_change_times, side="right") - 1 for phase_times in campaign_phase_change_times]

        # Phase-wise per-campaign expected utility (fbar) and expected cost (cbar)
        # shapes: (N_CAMPAIGNS, len(joined_phase_change_times), len(bidder.bids))
        phase_wise_campaign_f_bars = np.array([(bidder.valuations[i] - bidder.bids) * campaign_win_probabilities[i][idx] for i, idx in enumerate(joined_indices)])
        phase_wise_campaign_c_bars = np.array([bidder.bids * campaign_win_probabilities[i][idx] for i, idx in enumerate(joined_indices)])

        phase_wise_gamma_values = []
        phase_wise_marginal_values = []
        phase_wise_objective_values = []
        phase_wise_expected_payments = []

        # Running one separate LP for each joined phase_change_time
        for phase_index, phase_change_time in enumerate(joined_phase_change_times):
            fbar_t = phase_wise_campaign_f_bars[:, phase_index, :]      # shape: (N_CAMPAIGNS, len(bidder.bids))
            cbar_t = phase_wise_campaign_c_bars[:, phase_index, :]      # shape: (N_CAMPAIGNS, len(bidder.bids))
            win_probabilities_t = np.array([
                campaign_win_probabilities[i][joined_indices[i][phase_index]] for i in range(len(campaign_indices))
            ])      # shape: (N_CAMPAIGNS, len(bidder.bids))

            model = mip.Model()

            # Optimization variables
            gamma = {(c, b, a): model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=1)
                     for c in range(len(campaign_indices)) for b in range(len(bidder.bids)) for a in range(2**len(campaign_indices))}     # probabilities
            marginals = {(a): model.add_var(var_type=mip.CONTINUOUS, lb=0, ub=1) for a in range(2**len(campaign_indices))}  # marginal probabilities

            # Budget constraint
            model.add_constr(mip.xsum(gamma[c, b, a] * cbar_t[c, b]
                                      for c in range(len(campaign_indices)) for b in range(len(bidder.bids)) for a in range(2**len(campaign_indices))) <= bidder.RHO)

            # Marginalization + feasibility constraint
            for a in range(2**len(campaign_indices)):
                for c in range(len(campaign_indices)):
                    model.add_constr(mip.xsum(gamma[c, b, a] for b in range(len(bidder.bids))) == marginals[a])
                    if a & (1 << c):  # if campaign c is included in the campaigns subset a:
                        model.add_constr(gamma[c, 0, a] == 0)
                    else:
                        model.add_constr(gamma[c, 0, a] == marginals[a])
                    # Feasibility: force gamma[c, b, a] = 0 for infeasible bids (bids[b] > valuations[c])
                    for b in range(len(bidder.bids)):
                        if bidder.bids[b] > bidder.valuations[campaign_indices[c]]:
                            model.add_constr(gamma[c, b, a] == 0)

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
                * win_probabilities_t[None, :, :]    # shape (1, N_CAMPAIGNS, len(BIDS_SPACE))
            )

            phase_wise_gamma_values.append(gamma_values)
            phase_wise_marginal_values.append(marginal_values)
            phase_wise_objective_values.append(model.objective_value)
            phase_wise_expected_payments.append(expected_payment)

        phase_wise_gamma_values = np.array(phase_wise_gamma_values)                         # shape: (len(joined_phase_change_times), 2**N_CAMPAIGNS, N_CAMPAIGNS, len(BIDS_SPACE))
        phase_wise_gamma_values = phase_wise_gamma_values.transpose((1, 2, 0, 3))           # shape: (2**N_CAMPAIGNS, N_CAMPAIGNS, len(joined_phase_change_times), len(BIDS_SPACE))
        phase_wise_marginal_values = np.array(phase_wise_marginal_values)                   # shape: (len(joined_phase_change_times), 2**N_CAMPAIGNS)
        phase_wise_objective_values = np.array(phase_wise_objective_values)                 # shape: (len(joined_phase_change_times),)
        phase_wise_expected_payments = np.array(phase_wise_expected_payments)               # shape: (len(joined_phase_change_times),)

        repeat_counts = np.diff(np.concatenate((joined_phase_change_times, [bidder.T])))
        phase_wise_objective_values = np.repeat(phase_wise_objective_values, repeat_counts)
        phase_wise_expected_payments = np.repeat(phase_wise_expected_payments, repeat_counts)

        return phase_wise_gamma_values, phase_wise_marginal_values, phase_wise_objective_values, phase_wise_expected_payments


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