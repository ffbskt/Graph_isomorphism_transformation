import torch
import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
import copy
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from DS.Visualisation.visg import VisG

from DS.Trans_Interface.src_trg_interface import GraphTransformationInterface


class GraphTransformationEnv(gym.Env):
    """Custom Environment for graph transformation tasks"""
    def __init__(self, TI: GraphTransformationInterface(), max_steps=100, node_categories=[1,0], edge_categories=['1', 'replacement'], add_base_patterns=True):
        super().__init__()
        self.TI = TI
        if add_base_patterns:
            self.add_base_patterns()
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
        """Execute action and return new state with gradient rewards"""
        # Get the previous score
        prev_score = self.TI.get_cur_score()

        # Apply the transformation
        success = self.TI.apply_pattern(action)

        # Get new observation
        obs = self._get_observation()

        # Get updated score
        cur_score = self.TI.get_cur_score()

        # Reward system
        reward = -0.1  # Default penalty for each step to encourage efficiency

        if cur_score > prev_score:
            reward += (cur_score - prev_score) * 5  # Reward improvement

        if cur_score == 1:
            reward += 10  # Large bonus for reaching the target

        # Check termination conditions
        terminated = cur_score == 1  # Task completed
        truncated = self.cur_steps >= self.max_steps  # Max steps reached
        # if len(self.TI.get_current_G()) > len(self.TI.get_target())
        if len(self.TI.get_current_G()) > len(self.TI.get_target()):
            truncated = True
            reward -= 100  # Large penalty for going over the target

        # Update step counter
        self.cur_steps += 1

        # Create info dictionary
        info = {
            'success': success,
            'score': cur_score,
            'steps': self.cur_steps
        }

        return obs, reward, terminated, truncated, info

        
    def _get_observation(self):
        """Convert current graph state to observation"""
        # Get current graph
        G = self.TI.get_current_G()
        
        # Process graph features
        observation = graph_to_observation_with_edges(
            G,
            self.node_categories,
            self.edge_categories
        )
        
        return observation
        
    def render(self, mode='human'):
        """Render current graph state using VisG visualization"""
        # Get the current graph and target graph
        G = self.TI.get_current_G()
        target_G = self.TI.get_target()
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(20, 8))
        
        # First subplot for current graph
        ax1 = fig.add_subplot(1, 2, 1)
        visg_current = VisG()
        visg_current.set_graph(G)
        visg_current.draw(layout='spring', title="Current Graph State", ax=ax1)
        
        # Second subplot for target graph
        ax2 = fig.add_subplot(1, 2, 2)
        visg_target = VisG()
        visg_target.set_graph(target_G)
        visg_target.draw(layout='spring', title="Target Graph State", ax=ax2)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        
    def close(self):
        pass


    def add_base_patterns(self):
        patterns = [
            ([(0, {'type': 'T', 'label': 0}), (1, {'type': 'T', 'label': 1})], [(0, 1, {'type': '1', 'label': '1'})], 0, 1),
            ([(0, {'type': 'T', 'label': 0}), (1, {'type': 'T', 'label': 0})], [(0, 1, {'type': '1', 'label': '1'})], 0, 1),
            ([(0, {'type': 'T', 'label': 0}), (1, {'type': 'T', 'label': 1})], [(0, 1, {'type': 'replacement', 'label': 'Re'})], 0, 1),
            ([(0, {'type': 'T', 'label': 1}), (1, {'type': 'T', 'label': 0})], [(0, 1, {'type': 'replacement', 'label': '1'})], 0, 1)
        ]
        
        common_nodes = [('B', {'type': 'base', 'label': 'B'}), ('H', {'type': 'head', 'label': 'H'})]
        common_edges = [('B', 0, {'type': 'hierarchy', 'label': '1'}), ('H', 1, {'type': 'hierarchy', 'label': '1'}), ('B', 'H', {'type': 1, 'label': '1'})]
        
        for nodes, edges, pbase, phead in patterns:
            pattern = nx.DiGraph()
            pattern.add_nodes_from(nodes + common_nodes)
            pattern.add_edges_from(edges + common_edges)
            pattern.graph['pbase'] = nx.subgraph(pattern, [pbase])
            pattern.graph['phead'] = nx.subgraph(pattern, [phead])
            self.TI.patterns.append(pattern)



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
        'x': np.array(x, dtype=np.float32),  # Convert tensors to NumPy arrays
        'edge_index': np.array(edge_index, dtype=np.int64)
    }



if __name__ == "__main__":
    # Test environment
    from DS.Logger.logger import JSONLogger
    log = JSONLogger()
    log.set_caller("Env")
    
    # Create and initialize environment
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
    print('obs', obs, env.TI.get_cur_score())
    env.TI.print_graps()

    for act in [5, 5, 4, 3, 4, 3]:
        obs, reward, terminated, truncated, info = env.step(act)
        print('---------obs', reward, env.TI.get_cur_score(), terminated, truncated)
        env.TI.print_graps()
    
    print('obs', obs, reward, env.TI.get_cur_score())
    env.render()
 
    # print(env.TI.get_current_G().nodes(data=True), env.TI.get_current_G().edges(data=True))
    # print(env.TI.get_target().nodes(data=True), env.TI.get_target().edges(data=True))

    """
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
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"\nStep {i+1}:")
            print(f"Action: {action}")
            print(f"Reward: {reward:.4f}")
            print(f"Success: {info['success']}")
            print(f"Score: {info['score']:.4f}")
            
            if terminated or truncated:
                print("\nEpisode terminated early")
                break
                
        print(f"\nTotal reward: {total_reward:.4f}")

        env.TI.print_graps()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\nEnvironment closed.")
    """