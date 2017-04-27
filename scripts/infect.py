import numpy as np
import pandas as pd

np.random.seed(21)

adj = pd.read_csv('../data/csv/adjacency.csv', index_col=0)
W = np.matrix(adj)

class generate_unusual:
    """
    Iterative selector for regions from an adjacency matrix with preferential attatchment
    """
    
    def __init__(self, adj, pref_weight=1):

        self.adj = adj
        self.num_regions = adj.shape[0]
        self.pref_weight = pref_weight

        self.nodes = np.array(range(self.num_regions))
        self.weights = np.ones(self.num_regions)
        self.unusual = []

        self.form_probabilities()

    def form_probabilities(self):
        """
        Calculate the vector of probabilities corresponding to each region being picked as the next regions to be selected
        """
        total_weight = self.weights.sum()
        self.prob = np.array([weight/total_weight for weight in self.weights])

    def select_node(self):
        new_node = np.random.choice(self.nodes, p = self.prob)
        return new_node

    def remove_node(self, new_node):
        index = np.where(self.nodes == new_node)[0][0]
        self.nodes = np.delete(self.nodes, index)
        self.weights = np.delete(self.weights, index)

    def update_weights(self, new_node):
        new_node_adj = np.array(self.adj[new_node])[0]
        neib = np.where(new_node_adj == 1)[0]
        #get the indicies of each neighbour in the remaining list of nodes
        indicies = np.concatenate([np.where(self.nodes == node)[0] for node in neib]) 
        self.weights[indicies] = self.weights[indicies] + self.pref_weight

    def sample(self, sample_size):
        
        for i in range(sample_size):
            new_node = self.select_node()
            self.remove_node(new_node)
            self.update_weights(new_node)
            self.form_probabilities()
            self.unusual.append(new_node)

gen = generate_unusual(W, pref_weight=20)
gen.sample(15)

unusual = gen.unusual
np.savetxt('../data/csv/prefUnusual.csv', unusual)
