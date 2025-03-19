import torch
import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
import copy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from DS.Trans_Interface.src_trg_interface import GraphTransformationInterface


class GraphTransformationEnv(gym.Env):
    """
    Custom RL environment for transforming a graph G to match a target graph G_t.
    """
    def __init__(self, TI: GraphTransformationInterface(), max_steps=100, node_categories=[1,0], edge_categories=['1', 'replacement']):
        super(GraphTransformationEnv, self).__init__()
        self.TI = TI
        self.max_steps = max_steps
        self.cur_steps = 0
        self.node_categories = node_categories
        self.edge_categories = edge_categories

        # Define observation space (example: adjacency matrices of G and G_t)
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=-np.inf, high=np.inf, shape=(50, 10), dtype=np.float32),  # Assume max 100 nodes
            "edge_index": spaces.Box(low=0, high=100, shape=(2, 200), dtype=np.int64)  # Assume max 500 edges
        })
        # Define action space (example: choosing an index of a transformation graph g_i)
        self.num_actions = self.TI.get_number_of_patterns()  # Example fixed number of transformations
        self.action_space = spaces.Discrete(self.num_actions)
        
        
        
    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state and returns the initial observation correctly formatted."""
        self.TI.re_init()
        self.cur_steps = 0
        obs = self._get_observation()
        return obs, {}  # ✅ Return a tuple: (obs, info)

    def step(self, action):
        """Applies transformation function based on the chosen action."""
        self.TI.apply_pattern(action)
        
        # Compute reward
        reward = self.TI.get_cur_score()
        
        # Check termination condition
        done = self.TI.get_cur_score() == 1 or self.cur_steps >= self.max_steps
        
        self.cur_steps += 1
        obs = self._get_observation()
        
        return obs, reward, done, {}, {}  # ✅ Return a tuple (obs, reward, done, truncated, info)


    def _get_observation(self):
        """Encodes G and G_t as an observation."""
        return graph_to_observation_with_edges(self.TI.get_current_G(), node_categories=self.node_categories, edge_categories=self.edge_categories)

    #def graph2observation(self, graph):
    #    return graph_to_observation_with_edges(graph, node_categories=None, edge_categories=None)
    
    def render(self, mode='human'):
        """Renders the current state of the graph (optional)."""
        nx.draw(self.TI.get_current_G(), with_labels=True)

    def close(self):
        pass



def preprocess_categorical_features(G, node_attr='label', edge_attr='type', 
                                    node_categories=None, edge_categories=None):
    """
    Converts categorical node and edge attributes into numerical float representations.
    
    Args:
        G (networkx.Graph): The input graph.
        node_attr (str): The node attribute to encode.
        edge_attr (str): The edge attribute to encode.
        node_categories (list, optional): Predefined node categories for one-hot encoding.
        edge_categories (list, optional): Predefined edge categories for one-hot encoding.

    Returns:
        tuple: Processed node and edge features, and edge index list.
    """
    # Use provided categories or extract unique ones from the graph
    if node_categories is None:
        node_categories = list(set(nx.get_node_attributes(G, node_attr).values()))
    if edge_categories is None:
        edge_categories = list(set(nx.get_edge_attributes(G, edge_attr).values()))

    # Create and fit encoders
    node_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    edge_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
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

    # Encode edges as additional nodes
    edge_features = []
    edge_index_list = []
    edge_to_node_map = {}  # Mapping of (source, target) -> edge node index

    edge_node_start_index = G.number_of_nodes()  # Start indexing edge nodes after original nodes

    for edge_id, (src, tgt) in enumerate(G.edges()):
        # Edge node index
        edge_node_idx = edge_node_start_index + edge_id
        edge_to_node_map[(src, tgt)] = edge_node_idx

        # Extract edge features
        category = G.edges[src, tgt].get(edge_attr, None)
        if category is not None:
            encoded = edge_encoder.transform([[category]])[0]
        else:
            encoded = np.zeros(len(edge_categories))  # Default if missing
        edge_features.append(encoded)

        # Connect edge node to its source and target nodes
        edge_index_list.append([src, edge_node_idx])
        edge_index_list.append([tgt, edge_node_idx])
    
    edge_features = np.array(edge_features, dtype=np.float32)

    # Combine all node features (real nodes + edge nodes)
    # SHeety features
    # print('node_features', node_features)
    # print('edge_features', edge_features) 
    #print('x', G.nodes(data=True), G.edges(data=True))
    x = np.vstack([node_features, edge_features]).astype(np.float32)

    # Convert edge list to numpy array (shape: [2, num_edges])
    edge_index = np.array(edge_index_list, dtype=np.int64).T

    return x, edge_index

def graph_to_observation_with_edges(G, node_categories, edge_categories):
    """Converts a networkx Graph to an RL observation, ensuring correct format."""
    x, edge_index = preprocess_categorical_features(G, node_categories=node_categories, edge_categories=edge_categories)

    return {
        'x': np.array(x, dtype=np.float32),  # ✅ Convert tensors to NumPy arrays
        'edge_index': np.array(edge_index, dtype=np.int64)
    }



if __name__ == "__main__":
    from DS.Logger.logger import JSONLogger
    log = JSONLogger()
    log.set_caller("Env")
    # Example usage:
    TI = GraphTransformationInterface(num_patterns=30, num_transformations=1)
    env = GraphTransformationEnv(TI)
    obs = env.reset()
    print(TI.GC.G.nodes(data=True))

    for i in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, done, _, _ = env.step(action)
        print(obs, reward, done)
        print(f"{i} Reward: {reward}, Done: {done}")
        if done:
            break
