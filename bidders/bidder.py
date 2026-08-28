import numpy as np

from environment.environment import Environment

class Bidder:
    def __init__(self, B, T, valuations, environment: Environment):
        self.B = B  # budget
        self.T = T  # number of rounds (= number of auctions = number of users)
        self.RHO = B / T  # budget per round

        self.valuations = valuations
        self.N_CAMPAIGNS = len(self.valuations)

        self.environment = environment

        self.bids = environment.BIDS_SPACE[environment.BIDS_SPACE <= valuations[0]]  # possible bids = action space of size K
        self.K = len(self.bids)  # cardinality of the action space (possible bids)

        self.t = 0  # current round
        self.a_t = np.zeros(self.N_CAMPAIGNS, dtype=int)  # current action (as the index of the arm played, not the bid itself)

        self.N_pulls = np.array([np.zeros(self.K) for _ in range(self.N_CAMPAIGNS)])

    def get_total_pulls_per_arm(self):
        return self.N_pulls

    def bid(self):
        # Implement the bidding strategy here
        raise NotImplementedError

    def learn(self, f_t, c_t, m_t=None):
        # Implement the learning mechanism here
        raise NotImplementedError