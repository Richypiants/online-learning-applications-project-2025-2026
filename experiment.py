import matplotlib.pyplot as plt
import numpy as np

from environment.environment import Environment

class Experiment:
    def __init__(self, environment: Environment, bidder_class, bidder_parameters): # n_trials, n_users, starting_budget, my_valuation):
        self.environment = environment
        self.n_campaigns = environment.N_CAMPAIGNS
        self.bids_space = environment.BIDS_SPACE
        self.bidder_class = bidder_class
        self.bidder_parameters = bidder_parameters

        # TODO: keep these here in common, or just pass them to the trial function as needed?
        self.n_users = bidder_parameters['T']
        self.starting_budget = bidder_parameters['B']
        self.my_valuations = bidder_parameters['valuations']

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


    # TODO: the error was in simple_clairvoyant because it was before the environment was simulated and therefore the bids still hadn't been generated, but now it's after
    # I should maybe move the bid generation somewhere better, probably not in the initialization of an instance, but I am going to lose the 
    # bids and the data! Maybe I should regenerate the whole Environment each time and store the trials separately? Probably not...
    def run_trials(self, n_trials, clairvoyant_strategy_type='simple'):
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

        for trial in range(n_trials):
            bidder = self.bidder_class(**self.bidder_parameters)

            bidder_utilities, my_bids, my_payments, total_bidder_wins = self.environment.simulate_environment(bidder, n_users=bidder.T, seed=10+trial) # seed=47   
            # shapes: (n_users, N_CAMPAIGNS), (n_users, N_CAMPAIGNS), (n_users, N_CAMPAIGNS), (n_users, N_CAMPAIGNS)    TODO: check

            self.results["bidder_utilities"].append(bidder_utilities)
            self.results["my_bids"].append(my_bids)
            self.results["my_payments"].append(my_payments)
            self.results["total_bidder_wins"].append(total_bidder_wins)
        
            # TODO: choose the clairvoyant strategy type based on the input parameter
            campaign_gammas, superarm_gammas, clairvoyant_expected_utility, clairvoyant_expected_payment = self.environment.compute_clairvoyant_strategy_combinatorial_simple(bidder, campaign_indices=None)

            # print(campaign_gammas)
            # print(superarm_gammas)
            # print(clairvoyant_expected_utility)
            # print(clairvoyant_expected_payment)

            self.results["clairvoyant_campaign_gammas"].append(campaign_gammas)
            self.results["clairvoyant_superarm_gammas"].append(superarm_gammas)
            self.results["clairvoyant_expected_utilities"].append(clairvoyant_expected_utility)
            self.results["clairvoyant_expected_payments"].append(clairvoyant_expected_payment)

            #print("Bid weights:", bidder.regret_minimizer.bid_weights)
            #print("Superarm weights:", bidder.regret_minimizer.superarm_weights)

            cumulative_payments = np.cumsum(my_payments, axis=0)
            #cumulative_utilities = np.cumsum(bidder_utilities.sum(axis=1), axis=0)
            cumulative_regrets = np.cumsum(clairvoyant_expected_utility - bidder_utilities.sum(axis=1), axis=0)    # shape: (1,) - (n_users, 1) = (n_users, 1)  # cumulative regret for each campaign

            self.results["cumulative_regrets"].append(cumulative_regrets)
            self.results["cumulative_payments"].append(cumulative_payments)
            self.results["all_pulls"].append(bidder.get_total_pulls_per_arm())

        self.results["cumulative_regrets"] = np.array(self.results["cumulative_regrets"])               # shape: (n_trials, n_users)
        self.results["cumulative_payments"] = np.array(self.results["cumulative_payments"])             # shape: (n_trials, n_users, N_CAMPAIGNS)
        self.results["all_pulls"] = np.array(self.results["all_pulls"])                                 # shape: (n_trials, N_CAMPAIGNS, len(BIDS_SPACE))
        #print(self.results["cumulative_regrets"].shape, self.results["cumulative_payments"].shape, self.results["all_pulls"].shape)

        self.results["avg_regrets"] = self.results["cumulative_regrets"].mean(axis=0)
        self.results["std_regrets"] = self.results["cumulative_regrets"].std(axis=0)

        self.results["avg_payments"] = self.results["cumulative_payments"].mean(axis=0)
        self.results["std_payments"] = self.results["cumulative_payments"].std(axis=0)

        self.results["avg_pulls"] = self.results["all_pulls"].mean(axis=0)
        self.results["std_pulls"] = self.results["all_pulls"].std(axis=0)

        self.plot_experiment_result(bidder)  # TODO: maybe remove the bidder argument?


    def plot_experiment_result(self, bidder):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        # --------------------------------------------------
        # 0. Cumulative payments (total across all campaigns)
        # --------------------------------------------------
        avg_total_cumulative_payments = self.results["avg_payments"].sum(axis=1)
        std_total_cumulative_payments = self.results["std_payments"].sum(axis=1)
        axes[0].plot(np.arange(self.n_users), avg_total_cumulative_payments)
        axes[0].fill_between(
            np.arange(self.n_users),
            avg_total_cumulative_payments - std_total_cumulative_payments,
            avg_total_cumulative_payments + std_total_cumulative_payments,
            alpha=0.3
        )

        axes[0].set_xlabel('$t$')
        axes[0].set_ylabel('$\\sum c_t$')
        axes[0].axhline(self.starting_budget, color='red', label='Budget')
        axes[0].legend()
        axes[0].set_title(f'Cumulative Payments of {self.bidder_class.__name__} across All Campaigns')

        # --------------------------------------------------
        # 1. Cumulative regret (total across all campaigns)
        # --------------------------------------------------
        # for i in range(auxiliary_bidder.N_CAMPAIGNS):
        axes[1].plot(np.arange(self.n_users), self.results["avg_regrets"])
        axes[1].fill_between(
            np.arange(self.n_users),
            self.results["avg_regrets"] - self.results["std_regrets"],
            self.results["avg_regrets"] + self.results["std_regrets"],
            alpha=0.3
        )

        axes[1].set_xlabel('$t$')
        axes[1].set_ylabel('$\\sum R_t$')
        axes[1].set_title(f'Cumulative Regret of {self.bidder_class.__name__} across All Campaigns')

        # --------------------------------------------------
        # 2. Cumulative payments (by campaign)
        # --------------------------------------------------
        for i in range(self.n_campaigns):
            axes[2].plot(np.arange(self.n_users), self.results["avg_payments"][:, i])
            axes[2].fill_between(
                np.arange(self.n_users),
                self.results["avg_payments"][:, i] - self.results["std_payments"][:, i],
                self.results["avg_payments"][:, i] + self.results["std_payments"][:, i],
                alpha=0.3
            )

        axes[2].set_xlabel('$t$')
        axes[2].set_ylabel('$\\sum c_t$')
        axes[2].axhline(self.starting_budget, color='red', label='Budget')
        axes[2].legend()
        axes[2].set_title(f'Cumulative Payments of {self.bidder_class.__name__} by Campaign')

        # --------------------------------------------------
        # 3. Chosen bids (by campaign)
        # --------------------------------------------------
        # Choosing NaN for the bids that are above the valuation for each campaign, so that they are not plotted
        avg_pulls_full = np.full((self.n_campaigns, len(self.bids_space)), np.nan)
        std_pulls_full = np.full((self.n_campaigns, len(self.bids_space)), np.nan)

        mask = self.bids_space <= bidder.valuations[0]  # TODO: or wherever valuation is stored for each campaign, if it is different per campaign
        avg_pulls_full[:, mask] = self.results["avg_pulls"]
        std_pulls_full[:, mask] = self.results["std_pulls"]

        # for i in range(self.n_campaigns):
        #     axes[3].plot(self.bids_space, self.results["avg_pulls"][i, :])
        #     axes[3].fill_between(
        #         self.bids_space,
        #         self.results["avg_pulls"][i, :] - self.results["std_pulls"][i, :],
        #         self.results["avg_pulls"][i, :] + self.results["std_pulls"][i, :],
        #         alpha=0.3
        #     )

        for i in range(self.n_campaigns):
            axes[3].plot(self.bids_space, avg_pulls_full[i, :])
            axes[3].fill_between(
                self.bids_space,
                avg_pulls_full[i, :] - std_pulls_full[i, :],
                avg_pulls_full[i, :] + std_pulls_full[i, :],
                alpha=0.3
            )

        axes[3].set_xlabel('$b$')
        axes[3].set_ylabel('$n(b)$')
        axes[3].set_title(f'Chosen Bids of {self.bidder_class.__name__} by Campaign')

        plt.tight_layout()
        plt.show()