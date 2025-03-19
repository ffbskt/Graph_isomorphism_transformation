import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.utils import get_device
import gymnasium as gym
import numpy as np

from DS.RL.Env import GraphTransformationEnv
from DS.Trans_Interface.src_trg_interface import GraphTransformationInterface


class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim=64):
        super().__init__(observation_space, features_dim=embedding_dim)
        
        # Get input dimension from observation space
        n_node_features = observation_space.spaces['x'].shape[1]
        
        # GNN layers
        self.conv1 = GCNConv(n_node_features, 128)
        self.conv2 = GCNConv(128, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def _to_pyg_data(self, x, edge_index, batch_size=1):
        """Convert numpy arrays or tensors to PyTorch Geometric Data format"""
        # Convert to tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.LongTensor(edge_index)
        
        # Get device from the model's parameters
        device = next(self.parameters()).device
            
        # Move to device
        x = x.to(device)
        edge_index = edge_index.to(device)
        
        # Handle batched input
        if len(x.shape) == 3:  # Batched input
            batch_size = x.shape[0]
            x = x.view(-1, x.shape[-1])  # Flatten batch dimension
            
        # Create batch assignment
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        if batch_size > 1:
            nodes_per_graph = x.shape[0] // batch_size
            for i in range(batch_size):
                batch[i * nodes_per_graph:(i + 1) * nodes_per_graph] = i
                
        return Data(x=x, edge_index=edge_index, batch=batch)

    def forward(self, observations):
        # Extract features from observations
        x = observations['x']
        edge_index = observations['edge_index']
        
        # Convert to PyG format
        data = self._to_pyg_data(x, edge_index)
        
        # Apply GNN layers
        x = self.dropout(self.relu(self.conv1(data.x, data.edge_index)))
        x = self.dropout(self.relu(self.conv2(x, data.edge_index)))
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, data.batch)
        
        return x


class CustomDQNPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GNNFeatureExtractor,
            **kwargs
        )


if __name__ == '__main__':
    try:
        # Create environment
        TI = GraphTransformationInterface(num_patterns=30, num_transformations=1)
        env = GraphTransformationEnv(TI)
        
        # Create model with proper hyperparameters
        model = DQN(
            policy=CustomDQNPolicy,
            env=env,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.4,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            max_grad_norm=10,
            tensorboard_log="./dqn_graph_tensorboard/",
            verbose=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("Starting training...")
        # Train the agent
        model.learn(
            total_timesteps=50000,
            log_interval=10,
            progress_bar=True
        )
        
        print("\nTraining completed. Saving model...")
        model.save("dqn_graph_transformation")
        
        print("\nTesting trained agent...")
        # Test the trained agent
        obs, _ = env.reset()
        episode_reward = 0
        steps_without_success = 0
        max_steps_without_success = 20
        
        for i in range(100):  # Test episodes
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            success = info.get('success', False)
            
            print(f"Step {i}: Action={action}, Reward={reward:.2f}, Success={success}")
            
            if success:
                steps_without_success = 0
            else:
                steps_without_success += 1
            
            if steps_without_success >= max_steps_without_success:
                print(f"\nStopping early - agent appears to be stuck after {steps_without_success} unsuccessful steps")
                break
                
            if terminated or truncated:
                print(f"\nEpisode finished after {i+1} steps with total reward {episode_reward:.2f}")
                break
                
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\nEnvironment closed.")
