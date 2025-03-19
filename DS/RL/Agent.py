import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.utils import get_device
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from DS.RL.Env import GraphTransformationEnv
from DS.Trans_Interface.src_trg_interface import GraphTransformationInterface


class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim=32):  # Reduced embedding dim for smaller model
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim=embedding_dim)
        self.conv1 = GCNConv(observation_space.spaces['x'].shape[1], 64)
        self.conv2 = GCNConv(64, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, observations):
        x, edge_index = observations['x'], observations['edge_index']
        
        # Ensure edge_index has the correct shape by padding
        max_edges = 200  # Match the observation space shape
        padded_edge_index = torch.zeros((2, max_edges), dtype=torch.int64)
        num_edges = min(edge_index.shape[1], max_edges) if edge_index.numel() > 0 else 0
        if num_edges > 0:
            padded_edge_index[:, :num_edges] = edge_index[:, :num_edges]
        
        x = self.relu(self.conv1(x, padded_edge_index))
        x = self.relu(self.conv2(x, padded_edge_index))
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
    TI = GraphTransformationInterface(num_patterns=30, num_transformations=1)
    env = GraphTransformationEnv(TI)
    
    # Initialize the model with reduced buffer size, batch size, and learning steps
    model = DQN(CustomDQNPolicy, env, verbose=1, buffer_size=10000, batch_size=32, learning_starts=10, train_freq=1, target_update_interval=10)
    
    # Test with manual input
    test_observation = {
        "x": torch.rand((5, 10), dtype=torch.float32),  # Example 5 nodes, 10 features each
        "edge_index": torch.randint(0, 5, (2, 8), dtype=torch.int64)  # Example 8 edges
    }
    
    print("Test observation input:", test_observation)
    
    # Forward pass through the feature extractor
    extractor = GNNFeatureExtractor(env.observation_space)
    test_embedding = extractor.forward(test_observation)
    print("Output of GNNFeatureExtractor:", test_embedding)
    
    model.learn(total_timesteps=50)  # Reduce total learning steps for quick testing
