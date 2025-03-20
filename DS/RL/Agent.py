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
from DS.RL.G2tourch import GraphTransitionModel


class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim=64):
        super().__init__(observation_space, features_dim=embedding_dim)
        
        # Get input dimension from observation space
        n_node_features = observation_space.spaces['x'].shape[1]
        
        # Use GraphTransitionModel for feature extraction
        self.gnn_model = GraphTransitionModel(
            input_dim=n_node_features, 
            hidden_dim=128, 
            output_dim=embedding_dim,
            num_layers=2
        )
        
        # Additional processing if needed
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, observations):
        # The model takes a dictionary with 'x' and 'edge_index' keys directly
        # No need for separate _to_pyg_data conversion
        
        # Make sure we're working with dictionary input
        if isinstance(observations, dict) and 'x' in observations and 'edge_index' in observations:
            # Get features from the GraphTransitionModel
            x = observations['x']
            edge_index = observations['edge_index']
            
            # Convert to tensors with proper types if needed
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.LongTensor(edge_index)
            else:
                edge_index = edge_index.long()  # Ensure long type even if already tensor
            
            # Get device from the model's parameters and move tensors
            device = next(self.parameters()).device
            x = x.to(device)
            edge_index = edge_index.to(device)
            
            # Handle batched input by processing each graph separately
            if len(x.shape) == 3:  # Batched input [batch_size, num_nodes, features]
                batch_size = x.shape[0]
                embeddings = []
                
                for i in range(batch_size):
                    # Process each graph in the batch
                    single_graph = {
                        'x': x[i],
                        'edge_index': edge_index[i]
                    }
                    
                    # Get embedding for this graph
                    graph_embedding = self.gnn_model(single_graph)
                    embeddings.append(graph_embedding)
                
                # Stack embeddings
                embedding = torch.stack(embeddings)
            else:
                # Single graph processing
                single_graph = {
                    'x': x,
                    'edge_index': edge_index
                }
                embedding = self.gnn_model(single_graph)
                
            # Additional processing if needed
            embedding = self.relu(self.fc(embedding))
            
            return embedding
        else:
            raise ValueError("Observations must be a dictionary with 'x' and 'edge_index' keys")


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
    from DS.Logger.logger import JSONLogger
    JSONLogger().disable_logging()
    try:
        # Create environment
        TI = GraphTransformationInterface(num_patterns=3, num_transformations=1)
        env = GraphTransformationEnv(TI)
        
        # Wrap the environment to ensure consistent observation space
        original_reset = env.reset
        
        def wrapped_reset(*args, **kwargs):
            obs, info = original_reset(*args, **kwargs)
            
            # Pad node features to fixed size (50, 2) and truncate if necessary
            if 'x' in obs:
                num_nodes = min(obs['x'].shape[0], 50)  # Limit to 50 nodes
                # Create padded array filled with zeros
                padded_x = np.zeros((50, obs['x'].shape[1]), dtype=np.float32)
                # Copy actual node features (truncate if more than 50)
                padded_x[:num_nodes] = obs['x'][:num_nodes]
                obs['x'] = padded_x
            
            # Ensure edge_index has valid indices and right shape
            if 'edge_index' in obs:
                # Only keep edges where both nodes exist and are within the 50-node limit
                valid_edges = obs['edge_index'][:, obs['edge_index'][0] < num_nodes]
                valid_edges = valid_edges[:, valid_edges[1] < num_nodes]
                
                # Pad edge_index to fixed size (2, 200)
                padded_edge_index = np.zeros((2, 200), dtype=np.int64)
                num_edges = min(valid_edges.shape[1], 200)  # Take at most 200 edges
                padded_edge_index[:, :num_edges] = valid_edges[:, :num_edges]
                obs['edge_index'] = padded_edge_index
                
            return obs, info
        
        env.reset = wrapped_reset
        
        original_step = env.step
        
        def wrapped_step(action):
            obs, reward, terminated, truncated, info = original_step(action)
            
            # Pad node features to fixed size (50, 2) and truncate if necessary
            if 'x' in obs:
                num_nodes = min(obs['x'].shape[0], 50)  # Limit to 50 nodes
                # Create padded array filled with zeros
                padded_x = np.zeros((50, obs['x'].shape[1]), dtype=np.float32)
                # Copy actual node features (truncate if more than 50)
                padded_x[:num_nodes] = obs['x'][:num_nodes]
                obs['x'] = padded_x
            
            # Ensure edge_index has valid indices and right shape
            if 'edge_index' in obs:
                # Only keep edges where both nodes exist and are within the 50-node limit
                valid_edges = obs['edge_index'][:, obs['edge_index'][0] < num_nodes]
                valid_edges = valid_edges[:, valid_edges[1] < num_nodes]
                
                # Pad edge_index to fixed size (2, 200)
                padded_edge_index = np.zeros((2, 200), dtype=np.int64)
                num_edges = min(valid_edges.shape[1], 200)  # Take at most 200 edges
                padded_edge_index[:, :num_edges] = valid_edges[:, :num_edges]
                obs['edge_index'] = padded_edge_index
                
            return obs, reward, terminated, truncated, info
        
        env.step = wrapped_step
        
        # Create model with proper hyperparameters
        model = DQN(
            policy=CustomDQNPolicy,
            env=env,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=50,
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
            total_timesteps=15000,
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
