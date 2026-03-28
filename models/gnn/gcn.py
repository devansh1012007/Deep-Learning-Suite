import torch
import torch.nn as nn

class GCNLayer(nn.Module): # This class defines a single layer of the Graph Convolutional Network (GCN) model. 
    # It takes in the number of input features and output features as arguments and initializes a linear layer that will be used to transform the aggregated node features.
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: (N, F)
        # adj: (N, N)

        out = torch.matmul(adj, x)   # message passing
        out = self.linear(out)

        return out
    
class GCN(nn.Module):# This class defines the overall GCN model, which consists of two GCNLayers. 
    # The first layer takes in the input features and produces hidden features, while the second layer takes in the hidden features and produces the final output features (number of classes). 
    # The forward method defines how the input data flows through the layers of the model, applying a ReLU activation function after the first layer.
    def __init__(self, in_features, hidden_dim, num_classes):
        super().__init__()

        self.layer1 = GCNLayer(in_features, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, num_classes)

        self.relu = nn.ReLU()
    # adjacency matrix (adj) is used to represent the connections between nodes in a graph. 
    # It is a square matrix where the entry at row i and column j indicates the presence or absence of an edge between node i and node j. 
    # In the context of GCNs, the adjacency matrix is used to perform message passing, where each node aggregates information from its neighbors based on the connections defined by the adjacency matrix. 
    # This allows the GCN to learn representations that capture both the features of individual nodes and the structure of the graph.
    
    def forward(self, x, adj):
        x = self.relu(self.layer1(x, adj)) # This line applies the first GCN layer to the input features (x) and adjacency matrix (adj),
        # and then applies a ReLU activation function to the output.
        x = self.layer2(x, adj)
        return x

def normalize_adj(adj): # This function normalizes the adjacency matrix using symmetric normalization. 
    # It adds self-loops to the adjacency matrix, computes the degree of each node,
    # and then applies the normalization formula D^(-1/2) * A * D^(-1/2), where D is the degree matrix and A is the adjacency matrix.
    adj = adj + torch.eye(adj.size(0)).to(adj.device)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    D = torch.diag(deg_inv_sqrt)
    return D @ adj @ D

# Usecase of GCN:
# - Node classification: GCNs can be used for node classification tasks, where the goal is to predict the class labels of nodes in a graph based on their features and the structure of the graph. 
# For example, in a social network, you can use GCNs to predict the interests or demographics of users based on their connections and interactions with other users.

# Difference between GCN and GraphSAGE:
# - GCNs perform convolution operations on the graph by aggregating information from neighboring nodes using the adjacency matrix, 
# while GraphSAGE uses a sampling-based approach to aggregate information from a fixed number of neighbors. 

# - GCNs typically use a fixed aggregation function (e.g., mean, sum, or max) to combine the features of neighboring nodes, 
# while GraphSAGE allows for more flexible aggregation functions (e.g., LSTM, pooling) that can capture more complex relationships between nodes.

# sampling based approch means that instead of aggregating information from all neighboring nodes, GraphSAGE samples a fixed number of neighbors for each node and 
# aggregates information from those sampled neighbors.
