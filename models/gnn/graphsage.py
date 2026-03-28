import torch
import torch.nn as nn
class GraphSAGELayer(nn.Module):# This class defines a single layer of the GraphSAGE model.
    # It takes in the number of input features and output features as arguments and initializes a linear layer that will be used to transform the
    # concatenated node features and neighbor features.
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features) # defines a single layer of the GraphSAGE model. 
        # It takes in the number of input features and output features as arguments and initializes a linear layer that will be used to transform the concatenated node features and neighbor features. 

    def forward(self, x, adj):
        neighbor_sum = torch.matmul(adj, x)
        concat = torch.cat([x, neighbor_sum], dim=1) # This line concatenates the original node features (x) with the aggregated neighbor features (neighbor_sum) along the feature dimension (dim=1).
        return self.linear(concat)
    
class GraphSAGE(nn.Module): # This class defines the overall GraphSAGE model, which consists of two GraphSAGELayers.
    # The first layer takes in the input features and produces hidden features, while the second layer takes in the hidden features and produces the final output features (number of classes).
    # The forward method defines how the input data flows through the layers of the model, applying a ReLU activation function after the first layer.
    def __init__(self, in_features, hidden_dim, num_classes):
        super().__init__()
        # u can make this model deeper by just adding more layers in the __init__ method and then defining the forward method accordingly. 
        # For example, you can add a third layer by initializing another GraphSAGELayer and then applying it in the forward method after the second layer. 
        # This way, you can easily experiment with different architectures and depths for your GraphSAGE model without having to modify the existing code structure significantly.
        self.layer1 = GraphSAGELayer(in_features, hidden_dim)
        self.layer2 = GraphSAGELayer(hidden_dim, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.layer1(x, adj))
        x = self.layer2(x, adj)
        return x
    
# usecase of this model and architure:
# - Node classification: GraphSAGE can be used for node classification tasks, where the goal is to predict the class labels of nodes in a graph based on their features and the structure of the graph. 
# For example, in a social network, you can use GraphSAGE to predict the interests or demographics of users based on their connections and interactions with other users.

# - Link prediction: GraphSAGE can also be applied to link prediction tasks, where the goal is to predict the existence of edges between nodes in a graph. 
# For instance, in a recommendation system, you can use GraphSAGE to predict whether two users are likely to become friends based on their existing connections and shared interests.

# - Graph classification: GraphSAGE can be extended to graph classification tasks, where the goal is to predict the class labels of entire graphs rather than individual nodes. 
# This can be useful in applications such as chemical compound classification, where you want to predict the properties of a molecule based on its structure and features.

# - GraphSAGE can be used in various domains such as social networks, recommendation systems, knowledge graphs, and biological networks, where the data can be represented as graphs with nodes and edges. The ability of GraphSAGE to aggregate information from neighboring nodes allows it to capture the local structure and relationships in the graph, making it a powerful tool for learning representations and making predictions on graph-structured data.
