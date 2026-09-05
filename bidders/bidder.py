from abc import ABC, abstractmethod

import numpy as np

from environment.environment import Environment

class Bidder(ABC):
    def __init__(self, B, T, valuations, environment: Environment):
        self.B = B  # budget
        self.T = T  # number of rounds (= number of auctions = number of users)
        self.RHO = B / T  # budget per round

        self.valuations = np.array(valuations)
        self.N_CAMPAIGNS = len(self.valuations)

        self.environment = environment

        self.bids = environment.BIDS_SPACE  # full shared bid grid (length = len(BIDS_SPACE))
        self.K = len(self.bids)  # cardinality of the action space = len(BIDS_SPACE)

        # per-campaign feasibility mask: arm b is feasible for campaign i iff BIDS_SPACE[b] <= valuations[i]
        self.feasible = (self.bids[None, :] <= np.asarray(valuations)[:, None])  # shape: (N_CAMPAIGNS, K)
        self.feasible_nonzero = self.feasible[:, 1:]  # feasibility mask over non-zero arms (shape: (N_CAMPAIGNS, K-1))

        self.t = 0  # current round
        self.a_t = np.zeros(self.N_CAMPAIGNS, dtype=int)  # current action (as the index of the arm played, not the bid itself)

        self.N_pulls = np.array([np.zeros(self.K) for _ in range(self.N_CAMPAIGNS)])

    def get_total_pulls_per_arm(self):
        return self.N_pulls

    @abstractmethod
    def bid(self):
        # Here the bidding strategy must be implemented for each subclass
        pass

    @abstractmethod
    def learn(self, f_t, c_t, m_t=None):
        self.B -= np.sum(c_t)
        self.t += 1

        # Here the learning mechanism must be implemented for each subclass
        pass