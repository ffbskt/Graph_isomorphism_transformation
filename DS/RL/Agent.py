import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN
# from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy

from stable_baselines3.common.utils import get_device
import gymnasium as gym
from gymnasium import spaces
import numpy as np





class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim=64):
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim=embedding_dim)
        self.conv1 = GCNConv(observation_space['x'].shape[1], 128)
        self.conv2 = GCNConv(128, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, observations):
        x, edge_index = observations['x'], observations['edge_index']
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        return x.mean(dim=0)  # Global pooling



class CustomDQNPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomDQNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GNNFeatureExtractor,
            **kwargs
        )
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))




if __name__ == '__main__':
    num_nodes = 5
    num_node_features = 10
    num_edges = 10
    num_actions = 10

    class GraphEnv(gym.Env):
        def __init__(self):
            super(GraphEnv, self).__init__()
            self.observation_space = spaces.Dict({
                'x': spaces.Box(low=-1, high=1, shape=(num_nodes, num_node_features), dtype=np.float32),
                'edge_index': spaces.Box(low=0, high=num_nodes, shape=(2, num_edges), dtype=np.int64)
            })
            self.action_space = spaces.Discrete(num_actions)

        def reset(self):
            # Initialize graph data
            x = np.random.randn(num_nodes, num_node_features).astype(np.float32)
            edge_index = np.random.randint(0, num_nodes, (2, num_edges)).astype(np.int64)
            return {'x': x, 'edge_index': edge_index}

        def step(self, action):
            # Implement environment dynamics
            reward = 0
            done = False
            return self.reset(), reward, done, {}

    env = GraphEnv()
    model = DQN(CustomDQNPolicy, env, verbose=1)
    model.learn(total_timesteps=100)
