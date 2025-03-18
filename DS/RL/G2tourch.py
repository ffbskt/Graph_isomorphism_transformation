import numpy as np
import networkx as nx
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def preprocess_categorical_features(G, node_attr='type', edge_attr='relation'):
    """
    Converts categorical node and edge attributes into numerical float representations.
    
    Args:
        G (networkx.Graph): The input graph.
        node_attr (str): The node attribute to encode.
        edge_attr (str): The edge attribute to encode.

    Returns:
        dict: Processed node and edge features.
    """
    # Collect unique categories
    node_categories = list(set(nx.get_node_attributes(G, node_attr).values()))
    edge_categories = list(set(nx.get_edge_attributes(G, edge_attr).values()))

    # Create encoders
    node_encoder = OneHotEncoder(sparse=False)
    edge_encoder = OneHotEncoder(sparse=False)

    # Fit encoders
    node_encoder.fit(np.array(node_categories).reshape(-1, 1))
    edge_encoder.fit(np.array(edge_categories).reshape(-1, 1))

    # Encode nodes
    node_features = []
    for node in G.nodes():
        category = G.nodes[node].get(node_attr, None)
        if category is not None:
            encoded = node_encoder.transform([[category]])[0]  # One-hot encoded vector
        else:
            encoded = np.zeros(len(node_categories))  # Default if missing
        node_features.append(encoded)
    
    node_features = np.array(node_features, dtype=np.float32)

    # Encode edges
    edge_features = []
    for src, tgt in G.edges():
        category = G.edges[src, tgt].get(edge_attr, None)
        if category is not None:
            encoded = edge_encoder.transform([[category]])[0]
        else:
            encoded = np.zeros(len(edge_categories))
        edge_features.append(encoded)

    edge_features = np.array(edge_features, dtype=np.float32)

    return node_features, edge_features

def graph_to_observation_with_edges(G):
    """
    Converts a networkx Graph to an RL observation, handling both node and edge categorical attributes.
    
    Returns:
        dict: Observation dictionary with float-based node and edge features.
    """
    num_nodes = G.number_of_nodes()
    
    # Convert categorical features to numerical float format
    node_features, edge_features = preprocess_categorical_features(G)

    # Edge Index: Convert edges to (2, num_edges) format
    edge_list = np.array(list(G.edges), dtype=np.int64).T  # Shape (2, num_edges)

    return {
        'x': torch.tensor(node_features, dtype=torch.float32),  # Processed node features
        'edge_index': torch.tensor(edge_list, dtype=torch.long),  # Edge list
        'edge_attr': torch.tensor(edge_features, dtype=torch.float32)  # Processed edge features
    }


if __name__ == '__main__':
    

    # Example Usage
    G = nx.Graph()

    # Adding nodes with categorical attributes
    G.add_nodes_from([
        (0, {'type': 'A'}),
        (1, {'type': 'B'}),
        (2, {'type': 'C'}),
        (3, {'type': 'A'})
    ])

    # Adding edges with categorical attributes
    G.add_edges_from([
        (0, 1, {'relation': 'friendship'}),
        (1, 2, {'relation': 'follows'}),
        (2, 3, {'relation': 'message'}),
        (0, 3, {'relation': 'friendship'})
    ])

    # Convert to observation
    observation = graph_to_observation_with_edges(G)
    print("Node Features (x):", observation['x'])
    print("Edge Index:", observation['edge_index'])
    print("Edge Features (edge_attr):", observation['edge_attr'])
