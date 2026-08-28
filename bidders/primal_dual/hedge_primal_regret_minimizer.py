import numpy as np

class HedgePrimalRegretMinimizer:
    def __init__(self, environment, K, learning_rate):
        self.environment = environment
        self.N_CAMPAIGNS = environment.N_CAMPAIGNS
        self.K = K
        self.learning_rate = learning_rate

        # per-campaign weights and distributions over bids
        self.bid_weights = np.ones((self.N_CAMPAIGNS, self.K - 1))
        self.bid_probs = self.bid_weights / self.bid_weights.sum(axis=1, keepdims=True)

        # weights and distributions over superarms (campaigns subsets)
        self.superarm_weights = np.ones(2**self.N_CAMPAIGNS)
        for conflict in self.environment.conflicts_graph.graph:
            campaign_a, campaign_b = conflict
            for a in range(2**self.N_CAMPAIGNS):
                if a & (1 << campaign_a) and a & (1 << campaign_b):  # if both campaigns are included in the campaigns subset a:
                    self.superarm_weights[a] = 0.0
        self.superarm_probs = self.superarm_weights / self.superarm_weights.sum()

        self.t = 0
        self.a_t = None

    def bid(self):
        # sample a non-conflicting campaigns subset according to the primal weights
        superarm = np.random.choice(2**self.N_CAMPAIGNS, p=self.superarm_probs)

        # sample the bid for each active campaign according to the primal weights
        self.a_t = np.zeros(self.N_CAMPAIGNS, dtype=int)
        for i in range(self.N_CAMPAIGNS):
            if superarm & (1 << i):
                self.a_t[i] = 1 + np.random.choice(self.K - 1, p=self.bid_weights[i])
            else:
                self.a_t[i] = 0
        return self.a_t
    
    def learn(self, l_t):
        self.bid_weights *= np.exp(self.learning_rate * l_t)
        self.bid_probs = self.bid_weights / self.bid_weights.sum(axis=1, keepdims=True)

        superarm_lagrangian = np.zeros(2**self.N_CAMPAIGNS)
        for a in range(2**self.N_CAMPAIGNS):
            for i in range(self.N_CAMPAIGNS):
                if a & (1 << i):
                    superarm_lagrangian[a] += np.max(l_t[i])
        self.superarm_weights *= np.exp(self.learning_rate * superarm_lagrangian)
        self.superarm_probs = self.superarm_weights / self.superarm_weights.sum()
        
        self.t += 1