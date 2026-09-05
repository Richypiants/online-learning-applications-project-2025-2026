import numpy as np

from environment.environment import Environment

class Experiment:
    def __init__(self, environment: Environment, bidder_class, bidder_parameters):
        self.environment = environment
        self.n_campaigns = environment.N_CAMPAIGNS
        self.bids_space = environment.BIDS_SPACE
        self.bidder_class = bidder_class
        self.bidder_parameters = bidder_parameters

        self.n_users = bidder_parameters['T']
        self.starting_budget = bidder_parameters['B']
        self.my_valuations = bidder_parameters['valuations']

        self.bidders = []  # Store the bidders for each trial for eventual later analysis
        self.results = {}  # Store the results of the experiment for later analysis


    def __str__(self):
        return (
            "-- EXPERIMENT: --\n"
            f"Environment: {self.environment}\n"
            f"Bidder class: {self.bidder_class.__name__}\n"
            f"Bidder parameters: {self.bidder_parameters}\n"
            f"Number of users: {self.n_users}\n"
            f"Starting budget: {self.starting_budget}\n"
            f"My valuations: {self.my_valuations}\n"
        )

    __repr__ = __str__


    def run_trials(self, n_trials, clairvoyant_strategy_type='simplified', seed=10):
        self.results = {}
        self.results["cumulative_regrets"] = []
        self.results["cumulative_payments"] = []
        self.results["all_pulls"] = []

        self.results["bidder_utilities"] = []
        self.results["my_bids"] = []
        self.results["my_payments"] = []
        self.results["total_bidder_wins"] = []

        self.results["clairvoyant_campaign_gammas"] = []
        self.results["clairvoyant_superarm_gammas"] = []
        self.results["clairvoyant_expected_utilities"] = []
        self.results["clairvoyant_expected_payments"] = []
        self.bidders = []

        for trial in range(n_trials):
            bidder = self.bidder_class(**self.bidder_parameters)
            self.bidders.append(bidder)

            bidder_utilities, my_bids, my_payments, total_bidder_wins = self.environment.simulate_environment(bidder, n_users=bidder.T, seed=seed+trial)

            self.results["bidder_utilities"].append(bidder_utilities)
            self.results["my_bids"].append(my_bids)
            self.results["my_payments"].append(my_payments)
            self.results["total_bidder_wins"].append(total_bidder_wins)

            if clairvoyant_strategy_type == 'simplified':        
                campaign_gammas, superarm_gammas, clairvoyant_expected_utility, clairvoyant_expected_payment = self.environment.compute_simplified_combinatorial_clairvoyant(bidder, campaign_indices=None)
            elif clairvoyant_strategy_type == 'true':
                campaign_gammas, superarm_gammas, clairvoyant_expected_utility, clairvoyant_expected_payment = self.environment.compute_true_combinatorial_clairvoyant(bidder, campaign_indices=None)
            else:
                raise ValueError(f"Invalid clairvoyant_strategy_type: {clairvoyant_strategy_type}. Must be 'simplified' or 'true'.")

            self.results["clairvoyant_campaign_gammas"].append(campaign_gammas)
            self.results["clairvoyant_superarm_gammas"].append(superarm_gammas)
            self.results["clairvoyant_expected_utilities"].append(clairvoyant_expected_utility)
            self.results["clairvoyant_expected_payments"].append(clairvoyant_expected_payment)

            cumulative_payments = np.cumsum(my_payments, axis=0)
            #cumulative_utilities = np.cumsum(bidder_utilities.sum(axis=1), axis=0)
            cumulative_regrets = np.cumsum(clairvoyant_expected_utility - bidder_utilities.sum(axis=1), axis=0)    # shape: (1,) - (n_users, 1) = (n_users, 1)  # cumulative regret for each campaign

            self.results["cumulative_regrets"].append(cumulative_regrets)
            self.results["cumulative_payments"].append(cumulative_payments)
            self.results["all_pulls"].append(bidder.get_total_pulls_per_arm())

        self.results["cumulative_regrets"] = np.array(self.results["cumulative_regrets"])               # shape: (n_trials, n_users)
        self.results["cumulative_payments"] = np.array(self.results["cumulative_payments"])             # shape: (n_trials, n_users, N_CAMPAIGNS)
        self.results["all_pulls"] = np.array(self.results["all_pulls"])                                 # shape: (n_trials, N_CAMPAIGNS, len(BIDS_SPACE))

        self.results["total_cumulative_payments_by_trial"] = self.results["cumulative_payments"].sum(axis=-1)                       # shape: (n_trials,)

        self.results["avg_total_cumulative_payments"] = self.results["total_cumulative_payments_by_trial"].mean(axis=0)                       # shape: (n_trials,)
        self.results["std_total_cumulative_payments"] = self.results["total_cumulative_payments_by_trial"].std(axis=0)                       # shape: (n_trials,)

        self.results["avg_regrets"] = self.results["cumulative_regrets"].mean(axis=0)
        self.results["std_regrets"] = self.results["cumulative_regrets"].std(axis=0)

        self.results["avg_payments"] = self.results["cumulative_payments"].mean(axis=0)
        self.results["std_payments"] = self.results["cumulative_payments"].std(axis=0)

        self.results["avg_pulls"] = self.results["all_pulls"].mean(axis=0)
        self.results["std_pulls"] = self.results["all_pulls"].std(axis=0)