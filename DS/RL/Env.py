import torch
import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
import copy
from sklearn.preprocessing import OneHotEncoder

from DS.Trans_Interface.src_trg_interface import GraphTransformationInterface


class GraphTransformationEnv(gym.Env):
    """Custom Environment for graph transformation tasks"""
    def __init__(self, TI: GraphTransformationInterface(), max_steps=100, node_categories=[1,0], edge_categories=['1', 'replacement']):
        super().__init__()
        self.TI = TI
        self.max_steps = max_steps
        self.cur_steps = 0
        self.node_categories = node_categories
        self.edge_categories = edge_categories

        # Calculate feature dimensions
        self.n_node_features = len(node_categories)  # One-hot encoding size for nodes
        self.n_edge_features = len(edge_categories)  # One-hot encoding size for edges
        
        # Define observation space with fixed dimensions
        max_nodes = 50  # Maximum number of nodes
        max_edges = 200  # Maximum number of edges
        
        self.observation_space = spaces.Dict({
            # Node features matrix: [num_nodes, num_features]
            "x": spaces.Box(
                low=0, 
                high=1, 
                shape=(max_nodes, self.n_node_features), 
                dtype=np.float32
            ),
            # Edge index matrix: [2, num_edges] (PyTorch Geometric format)
            "edge_index": spaces.Box(
                low=0, 
                high=max_nodes-1, 
                shape=(2, max_edges), 
                dtype=np.int64
            )
        })
        
        # Action space is discrete (selecting transformation patterns)
        self.action_space = spaces.Discrete(self.TI.get_number_of_patterns())
        
        # Initialize encoders
        self._init_encoders()
        
    def _init_encoders(self):
        """Initialize one-hot encoders for node and edge features"""
        self.node_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.edge_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit encoders with all possible categories
        self.node_encoder.fit(np.array(self.node_categories).reshape(-1, 1))
        self.edge_encoder.fit(np.array(self.edge_categories).reshape(-1, 1))
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.TI.re_init()
        self.cur_steps = 0
        
        obs = self._get_observation()
        info = {}
        return obs, info
        
    def step(self, action):
        """Execute action and return new state"""
        # Apply the transformation
        success = self.TI.apply_pattern(action)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self.TI.get_cur_score()
        
        # Check termination conditions
        terminated = self.TI.get_cur_score() == 1  # Task completed
        truncated = self.cur_steps >= self.max_steps  # Max steps reached
        
        # Update step counter
        self.cur_steps += 1
        
        # Create info dictionary
        info = {
            'success': success,
            'score': self.TI.get_cur_score(),
            'steps': self.cur_steps
        }
        
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self):
        """Convert current graph state to observation"""
        # Get current graph
        G = self.TI.get_current_G()
        
        # Process graph features
        x, edge_index = preprocess_graph_features(
            G,
            self.node_encoder,
            self.edge_encoder,
            max_nodes=self.observation_space.spaces['x'].shape[0],
            max_edges=self.observation_space.spaces['edge_index'].shape[1]
        )
        
        return {
            'x': x,
            'edge_index': edge_index
        }
        
    def render(self, mode='human'):
        """Render current graph state"""
        nx.draw(self.TI.get_current_G(), with_labels=True)
        
    def close(self):
        pass


def preprocess_graph_features(G, node_encoder, edge_encoder, max_nodes=50, max_edges=200):
    """
    Process graph features into format suitable for GNN.
    
    Args:
        G: NetworkX graph
        node_encoder: Fitted OneHotEncoder for node features
        edge_encoder: Fitted OneHotEncoder for edge features
        max_nodes: Maximum number of nodes to pad/truncate to
        max_edges: Maximum number of edges to pad/truncate to
        
    Returns:
        tuple: (node_features, edge_index)
    """
    # Extract and encode node features
    node_features = []
    for node in G.nodes():
        category = G.nodes[node].get('label', None)
        if category is not None:
            encoded = node_encoder.transform([[category]])[0]
        else:
            encoded = np.zeros(node_encoder.n_features_in_)
        node_features.append(encoded)
    
    # Convert to numpy array
    node_features = np.array(node_features, dtype=np.float32)
    
    # Pad or truncate node features
    if len(node_features) < max_nodes:
        padding = np.zeros((max_nodes - len(node_features), node_features.shape[1]))
        node_features = np.vstack([node_features, padding])
    else:
        node_features = node_features[:max_nodes]
    
    # Extract and process edges
    edge_list = []
    for src, tgt in G.edges():
        if src < max_nodes and tgt < max_nodes:  # Only include edges between valid nodes
            edge_list.append([src, tgt])
    
    # Convert to numpy array and ensure correct shape
    if edge_list:
        edge_index = np.array(edge_list, dtype=np.int64).T  # Shape: [2, num_edges]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    
    # Pad or truncate edge index
    if edge_index.shape[1] < max_edges:
        padding = np.zeros((2, max_edges - edge_index.shape[1]), dtype=np.int64)
        edge_index = np.hstack([edge_index, padding])
    else:
        edge_index = edge_index[:, :max_edges]
    
    return node_features, edge_index


if __name__ == "__main__":
    # Test environment
    try:
        from DS.Logger.logger import JSONLogger
        log = JSONLogger()
        log.set_caller("Env")
        
        # Create and initialize environment
        TI = GraphTransformationInterface(num_patterns=30, num_transformations=1)
        env = GraphTransformationEnv(TI)
        
        # Test reset
        obs, info = env.reset()
        print("\nInitial observation shapes:")
        print(f"Node features (x): {obs['x'].shape}")
        print(f"Edge index: {obs['edge_index'].shape}")
        
        # Run test episode
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"\nStep {i}:")
            print(f"Action: {action}")
            print(f"Reward: {reward:.4f}")
            print(f"Success: {info['success']}")
            
            if terminated or truncated:
                print("\nEpisode ended")
                break
                
        print(f"\nTotal reward: {total_reward:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\nEnvironment closed.")
