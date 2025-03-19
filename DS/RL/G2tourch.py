import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np


class GraphTransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Initialize the Graph Transition Model using PyTorch Geometric.
        
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers in GNN.
            output_dim (int): Dimension of the final graph embedding.
            num_layers (int): Number of GCN layers.
        """
        super(GraphTransitionModel, self).__init__()
        
        # GCN layers
        self.layers = nn.ModuleList()
        # First layer: input_dim -> hidden_dim
        self.layers.append(GCNConv(input_dim, hidden_dim))
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        # Final layer: hidden_dim -> output_dim (node-level)
        self.layers.append(GCNConv(hidden_dim, output_dim))
        
        # Fully connected layer to refine graph embedding
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, data):
        """
        Forward pass to compute the graph embedding.
        
        Args:
            data: Can be either:
                - torch_geometric.data.Data object with attributes x and edge_index
                - dict with keys 'x' and 'edge_index'
        
        Returns:
            torch.Tensor: Graph-level embedding (fixed-size vector).
        """
        # Get node features and edge indices, handling both Data objects and dictionaries
        if isinstance(data, dict):
            x = data['x']
            edge_index = data['edge_index']
        else:
            x = data.x
            edge_index = data.edge_index
            
        # Convert numpy arrays to torch tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index).long()
        
        # Apply GCN layers with ReLU activation
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:  # Apply ReLU except for the last layer
                x = F.relu(x)
        
        # Pool node embeddings into a graph-level embedding (mean pooling)
        # Handle case where there might be no nodes (empty graph)
        if x.size(0) > 0:
            graph_embedding = torch.mean(x, dim=0)  # Shape: [output_dim]
        else:
            # Return zero embedding for empty graphs
            graph_embedding = torch.zeros(self.layers[-1].out_channels, device=x.device)
        
        # Refine with a fully connected layer
        graph_embedding = self.fc(graph_embedding)  # Shape: [output_dim]
        graph_embedding = F.relu(graph_embedding)
        
        return graph_embedding


# Test the model
if __name__ == "__main__":
    from DS.RL.Env import GraphTransformationEnv
    from DS.Trans_Interface.src_trg_interface import GraphTransformationInterface

    TI = GraphTransformationInterface(num_patterns=3, num_transformations=1)
    env = GraphTransformationEnv(TI)    

    def get_default_src():
        G = nx.DiGraph()
        nodes = [(4, {'type': 'T', 'label': 0}), (6, {'type': 'T', 'label': 0}), (10, {'type': 'T', 'label': 0})]
        edges = [(4, 6, {'type': '1', 'label': '1'}), (6, 10, {'type': '1', 'label': '1'})]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G

    def get_default_trg():
        G = nx.DiGraph()
        nodes = [(0, {'type': 'T', 'label': 1}), (1, {'type': 'T', 'label': 1}), (2, {'type': 'T', 'label': 0}), (3, {'type': 'T', 'label': 1}), 
                (4, {'type': 'T', 'label': 0}), (5, {'type': 'T', 'label': 0}), (6, {'type': 'T', 'label': 1}), (7, {'type': 'T', 'label': 0}), 
                (8, {'type': 'T', 'label': 0}), (9, {'type': 'T', 'label': 0})]
        edges = [(0, 3, {'type': 'replacement', 'label': '1'}), (0, 6, {'type': '1', 'label': '1'}), (2, 4, {'type': 'replacement', 'label': '1'}), 
                (5, 2, {'type': 'replacement', 'label': '1'}), (5, 9, {'type': '1', 'label': '1'}), (7, 6, {'type': 'replacement', 'label': '1'}), 
                (8, 0, {'type': 'replacement', 'label': '1'}), (8, 7, {'type': '1', 'label': '1'})]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G

    
    env.TI.source_graph = get_default_src()
    env.TI.target_graph = get_default_trg()
    env.TI.re_init()
    env.TI.print_graps()
    obs, info = env.reset()


    # Hyperparameters
    input_dim = 2    # Node feature dimension (changed from 8 to 2 to match actual input)
    hidden_dim = 16  # Hidden layer dimension
    output_dim = 32  # Graph embedding dimension
    num_layers = 3   # Number of GCN layers
    
    # Initialize the model
    model = GraphTransitionModel(input_dim, hidden_dim, output_dim, num_layers)
    
    # Create an example graph
    graph_data = obs 
    
    # Print the type and structure of the input data for debugging
    print(f"Input type: {type(graph_data)}")
    if isinstance(graph_data, dict):
        print(f"Keys: {graph_data.keys()}")
        if 'x' in graph_data:
            print(f"x shape: {graph_data['x'].shape}")
        if 'edge_index' in graph_data:
            print(f"edge_index shape: {graph_data['edge_index'].shape}")
    
    # Compute the embedding
    embedding = model(graph_data)
    print(f"Graph embedding shape: {embedding.shape}")  # Expected: [32]
    print(f"Graph embedding: {embedding[:5]}...")  # Print first 5 values