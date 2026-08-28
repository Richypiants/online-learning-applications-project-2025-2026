import numpy as np

class ConflictsGraph:
    def __init__(self, n_campaigns):
        self.n_campaigns = n_campaigns
        self.graph = set()  # set representation of the conflict graph, conflicts as ordered tuples (i, j) with i < j
        self.graph_matrix = np.zeros((n_campaigns, n_campaigns), dtype=int)  # adjacency matrix representation of the conflict graph

    def generate_random_graph(self, conflicts_percentage=0.2, seed=17):
        np.random.seed(seed)

        self.graph = set()
        self.graph_matrix = np.zeros((self.n_campaigns, self.n_campaigns), dtype=int)

        n_conflicts = int(conflicts_percentage * self.n_campaigns * (self.n_campaigns - 1) / 2)
        conflicts = np.array([(i, j) for i in range(self.n_campaigns) for j in range(i + 1, self.n_campaigns)])
        conflicts_selected = np.random.choice(
            np.arange(len(conflicts)),
            size=n_conflicts,
            replace=False
        )
        for i in conflicts_selected:
            self.add_edge(conflicts[i][0], conflicts[i][1])

    def add_edge(self, campaign_a, campaign_b):
        self.graph.add((campaign_a, campaign_b))
        self.graph_matrix[campaign_a, campaign_b] = 1
        self.graph_matrix[campaign_b, campaign_a] = 1

    def __str__(self):
        return str(self.graph_matrix)