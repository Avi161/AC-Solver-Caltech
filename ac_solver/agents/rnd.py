"""
Random Network Distillation (RND) module for curiosity-driven exploration.

RND provides intrinsic rewards by measuring prediction error between a fixed
random target network and a trained predictor network. States that the predictor
hasn't seen before produce high prediction error, yielding a large intrinsic
reward that encourages the agent to visit novel states.

Reference: Burda et al., "Exploration by Random Network Distillation", ICLR 2019
"""

import torch
import torch.nn as nn
import numpy as np


class RNDNetwork(nn.Module):
    """A simple MLP used for both the target and predictor networks."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class RunningMeanStd:
    """Tracks running mean and variance for online normalization."""

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch):
        """Update running statistics with a new batch of values."""
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count


class RNDModule:
    """
    Random Network Distillation module.

    Manages a fixed random target network and a trainable predictor network.
    The intrinsic reward is the MSE between their outputs, normalized by
    running statistics to keep the reward scale stable.

    Args:
        input_dim: Dimension of the observation vector.
        hidden_dim: Hidden layer size for both networks.
        output_dim: Embedding dimension for both networks.
        learning_rate: Learning rate for the predictor optimizer.
        device: Torch device.
    """

    def __init__(self, input_dim, hidden_dim=256, output_dim=64,
                 learning_rate=1e-3, device="cpu"):
        self.device = device

        # Fixed random target network (never trained)
        self.target = RNDNetwork(input_dim, hidden_dim, output_dim).to(device)
        for param in self.target.parameters():
            param.requires_grad = False

        # Trainable predictor network
        self.predictor = RNDNetwork(input_dim, hidden_dim, output_dim).to(device)

        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=learning_rate
        )

        # Running normalization for observations fed to RND
        self.obs_rms = RunningMeanStd(shape=(input_dim,))
        # Running normalization for intrinsic rewards
        self.reward_rms = RunningMeanStd(shape=())

    def _normalize_obs(self, obs):
        """Normalize observations using running statistics."""
        mean = torch.tensor(self.obs_rms.mean, dtype=torch.float32, device=self.device)
        std = torch.tensor(
            np.sqrt(self.obs_rms.var + 1e-8), dtype=torch.float32, device=self.device
        )
        return (obs - mean) / std

    def compute_intrinsic_reward(self, obs):
        """
        Compute intrinsic reward for a batch of observations.

        Args:
            obs: Tensor of shape (batch_size, input_dim).

        Returns:
            Numpy array of normalized intrinsic rewards, shape (batch_size,).
        """
        # Update observation running stats
        self.obs_rms.update(obs.cpu().numpy())

        with torch.no_grad():
            obs_normalized = self._normalize_obs(obs)
            target_features = self.target(obs_normalized)
            predictor_features = self.predictor(obs_normalized)
            # Per-sample MSE across the output dimension
            intrinsic_reward = (
                (target_features - predictor_features).pow(2).mean(dim=1)
            )

        reward_np = intrinsic_reward.cpu().numpy()

        # Normalize intrinsic reward using running variance
        self.reward_rms.update(reward_np)
        normalized_reward = reward_np / np.sqrt(self.reward_rms.var + 1e-8)

        return normalized_reward

    def update(self, obs):
        """
        Train the predictor to match the target on the given observations.

        Args:
            obs: Tensor of shape (batch_size, input_dim).

        Returns:
            float: The predictor loss (MSE).
        """
        obs_normalized = self._normalize_obs(obs)

        with torch.no_grad():
            target_features = self.target(obs_normalized)

        predictor_features = self.predictor(obs_normalized)
        loss = (target_features - predictor_features).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def state_dict(self):
        """Return state for checkpointing."""
        return {
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
            "reward_rms_mean": self.reward_rms.mean,
            "reward_rms_var": self.reward_rms.var,
            "reward_rms_count": self.reward_rms.count,
        }

    def load_state_dict(self, state):
        """Restore state from checkpoint."""
        self.predictor.load_state_dict(state["predictor"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.obs_rms.mean = state["obs_rms_mean"]
        self.obs_rms.var = state["obs_rms_var"]
        self.obs_rms.count = state["obs_rms_count"]
        self.reward_rms.mean = state["reward_rms_mean"]
        self.reward_rms.var = state["reward_rms_var"]
        self.reward_rms.count = state["reward_rms_count"]
