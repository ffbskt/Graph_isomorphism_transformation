import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces

class GraphTransformationEnv(gym.Env):
    """
    Custom RL environment for transforming a graph G to match a target graph G_t.
    """
    def __init__(self):
        super(GraphTransformationEnv, self).__init__()
        
        # Define observation space (example: adjacency matrices of G and G_t)
        self.graph_size = 5  # Example fixed size
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.graph_size, self.graph_size, 2), dtype=np.float32)
        
        # Define action space (example: choosing an index of a transformation graph g_i)
        self.num_actions = 10  # Example fixed number of transformations
        self.action_space = spaces.Discrete(self.num_actions)
        
        # Initialize graphs
        self.G = nx.erdos_renyi_graph(self.graph_size, 0.5)  # Random starting graph
        self.G_t = nx.erdos_renyi_graph(self.graph_size, 0.5)  # Random target graph
        
    def reset(self):
        """Resets the environment to the initial state and returns the initial observation."""
        self.G = nx.erdos_renyi_graph(self.graph_size, 0.5)  # Reset G
        return self._get_observation()

    def step(self, action):
        """Applies transformation function based on the chosen action."""
        
        # Placeholder for actual transformation logic
        # Right now, it just adds a random edge as a dummy transformation
        edge = np.random.choice(self.graph_size, 2, replace=False)
        self.G.add_edge(edge[0], edge[1])
        
        # Compute reward (placeholder: negative difference in adjacency matrices)
        reward = -np.sum(np.abs(nx.to_numpy_array(self.G) - nx.to_numpy_array(self.G_t)))
        
        # Check if task is complete
        done = nx.to_numpy_array(self.G).tolist() == nx.to_numpy_array(self.G_t).tolist()
        
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """Encodes G and G_t as an observation."""
        obs_G = nx.to_numpy_array(self.G)
        obs_Gt = nx.to_numpy_array(self.G_t)
        return np.stack([obs_G, obs_Gt], axis=-1).astype(np.float32)
    
    def render(self, mode='human'):
        """Renders the current state of the graph (optional)."""
        nx.draw(self.G, with_labels=True)

    def close(self):
        pass

# Example usage:
env = GraphTransformationEnv()
obs = env.reset()
for _ in range(5):
    action = env.action_space.sample()  # Random action
    obs, reward, done, _ = env.step(action)
    print(f"Reward: {reward}, Done: {done}")
    if done:
        break
