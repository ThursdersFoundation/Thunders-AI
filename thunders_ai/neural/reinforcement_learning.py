"""Reinforcement Learning Module for Thunders AI.

Implements RL agents supporting DQN, PPO, and SAC algorithms with
replay buffer management and both continuous and discrete action spaces.
"""

from __future__ import annotations

import random
import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from thunders_ai.config import get_config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)

ALGORITHM_TYPES = {"dqn", "ppo", "sac"}


class _ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms."""

    def __init__(self, capacity: int = 10000) -> None:
        self.buffer: Deque[Tuple[Any, ...]] = deque(maxlen=capacity)

    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[Any, ...]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


if HAS_TORCH:

    class _QNetwork(nn.Module):
        """Simple Q-network for DQN."""

        def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
            )

        def forward(self, state: "torch.Tensor") -> "torch.Tensor":
            return self.net(state)


    class _PolicyNetwork(nn.Module):
        """Policy network for PPO with continuous or discrete action spaces."""

        def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256, continuous: bool = False) -> None:
            super().__init__()
            self.continuous = continuous
            self.base = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            if continuous:
                self.mean_head = nn.Linear(hidden_size, action_dim)
                self.log_std = nn.Parameter(torch.zeros(action_dim))
            else:
                self.action_head = nn.Linear(hidden_size, action_dim)

        def forward(self, state: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            features = self.base(state)
            if self.continuous:
                mean = self.mean_head(features)
                std = torch.exp(self.log_std.clamp(-20, 2))
                dist = torch.distributions.Normal(mean, std)
                action = dist.rsample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                logits = self.action_head(features)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            return action, log_prob

        def evaluate(self, state: "torch.Tensor", action: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            features = self.base(state)
            if self.continuous:
                mean = self.mean_head(features)
                std = torch.exp(self.log_std.clamp(-20, 2))
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(action).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
            else:
                logits = self.action_head(features)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
            return log_prob, entropy


    class _ValueNetwork(nn.Module):
        """Value network for PPO/SAC."""

        def __init__(self, state_dim: int, hidden_size: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, state: "torch.Tensor") -> "torch.Tensor":
            return self.net(state).squeeze(-1)


class RLAgent:
    """Reinforcement Learning agent supporting multiple algorithms.

    Supports DQN (discrete), PPO (discrete/continuous), and SAC (continuous)
    with replay buffer management and configurable hyperparameters.

    Args:
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        algorithm: RL algorithm ('dqn', 'ppo', or 'sac').
        continuous: Whether the action space is continuous.
        learning_rate: Optimizer learning rate.
        gamma: Discount factor.
        buffer_size: Replay buffer capacity.
        batch_size: Training batch size.
        device: Device for computation.

    Example::

        agent = RLAgent(state_dim=4, action_dim=2, algorithm="dqn")
        action = agent.act(state)
        agent.learn()
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        algorithm: str = "ppo",
        continuous: bool = False,
        learning_rate: Optional[float] = None,
        gamma: Optional[float] = None,
        buffer_size: int = 10000,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch and NumPy are required for RLAgent. "
                "Install them with: pip install torch numpy"
            )

        app_cfg = get_config()
        cfg = app_cfg.neural
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm.lower()
        if self.algorithm not in ALGORITHM_TYPES:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from {ALGORITHM_TYPES}")
        self.continuous = continuous
        self.learning_rate = learning_rate or cfg.rl_learning_rate
        self.gamma = gamma or cfg.rl_gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device or app_cfg.device

        self._replay_buffer = _ReplayBuffer(capacity=buffer_size)
        self._total_steps = 0
        self._build_networks()
        logger.info(
            "RLAgent initialized: algo=%s, state_dim=%d, action_dim=%d, continuous=%s",
            self.algorithm, self.state_dim, self.action_dim, self.continuous,
        )

    def _build_networks(self) -> None:
        """Construct networks based on the selected algorithm."""
        hidden = 256
        if self.algorithm == "dqn":
            self._q_network = _QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
            self._target_network = _QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
            self._target_network.load_state_dict(self._q_network.state_dict())
            self._optimizer = torch.optim.Adam(self._q_network.parameters(), lr=self.learning_rate)
            self._epsilon = 1.0
            self._epsilon_min = 0.01
            self._epsilon_decay = 0.995
        elif self.algorithm == "ppo":
            self._policy = _PolicyNetwork(self.state_dim, self.action_dim, hidden, self.continuous).to(self.device)
            self._value = _ValueNetwork(self.state_dim, hidden).to(self.device)
            self._policy_optimizer = torch.optim.Adam(self._policy.parameters(), lr=self.learning_rate)
            self._value_optimizer = torch.optim.Adam(self._value.parameters(), lr=self.learning_rate)
            self._clip_epsilon = 0.2
            self._ppo_epochs = 4
        elif self.algorithm == "sac":
            self._policy = _PolicyNetwork(self.state_dim, self.action_dim, hidden, continuous=True).to(self.device)
            self._q1 = _QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
            self._q2 = _QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
            self._target_q1 = _QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
            self._target_q2 = _QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
            self._target_q1.load_state_dict(self._q1.state_dict())
            self._target_q2.load_state_dict(self._q2.state_dict())
            self._policy_optimizer = torch.optim.Adam(self._policy.parameters(), lr=self.learning_rate)
            self._q_optimizer = torch.optim.Adam(
                list(self._q1.parameters()) + list(self._q2.parameters()), lr=self.learning_rate
            )
            self._alpha = 0.2
            self._target_entropy = -self.action_dim

    def act(self, state: Any, deterministic: bool = False) -> Any:
        """Select an action given the current state.

        Args:
            state: Current environment state (numpy array or tensor).
            deterministic: If True, select the best action without exploration.

        Returns:
            Selected action.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            if self.algorithm == "dqn":
                q_values = self._q_network(state)
                if deterministic or random.random() > self._epsilon:
                    action = q_values.argmax(dim=-1).item()
                else:
                    action = random.randrange(self.action_dim)
            elif self.algorithm in ("ppo", "sac"):
                action, _ = self._policy(state)
                if self.continuous:
                    action = action.squeeze(0).cpu().numpy()
                else:
                    action = action.item() if not deterministic else self._policy(state)[0].argmax().item()

        self._total_steps += 1
        if self.algorithm == "dqn" and not deterministic:
            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

        return action

    def learn(self) -> Dict[str, float]:
        """Perform a learning update from replay buffer or collected experience.

        Returns:
            Dictionary of training metrics (loss, etc.).
        """
        if self.algorithm == "dqn":
            return self._learn_dqn()
        elif self.algorithm == "ppo":
            return self._learn_ppo()
        elif self.algorithm == "sac":
            return self._learn_sac()
        return {}

    def _learn_dqn(self) -> Dict[str, float]:
        """DQN learning step."""
        if not self._replay_buffer.is_ready(self.batch_size):
            return {"loss": 0.0}
        states, actions, rewards, next_states, dones = self._replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        q_values = self._q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q = self._target_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        loss = F.mse_loss(q_values, target_q)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        if self._total_steps % 100 == 0:
            self._target_network.load_state_dict(self._q_network.state_dict())
        return {"loss": loss.item()}

    def _learn_ppo(self) -> Dict[str, float]:
        """PPO learning step (simplified single-batch update)."""
        if not self._replay_buffer.is_ready(self.batch_size):
            return {"policy_loss": 0.0, "value_loss": 0.0}
        states, actions, rewards, next_states, dones = self._replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            values = self._value(states)
            next_values = self._value(next_states)
            returns = rewards + self.gamma * next_values * (1 - dones)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actions_t = torch.FloatTensor(np.array(actions)).to(self.device) if self.continuous else torch.LongTensor(actions).to(self.device)
        old_log_probs, _ = self._policy.evaluate(states, actions_t)
        old_log_probs = old_log_probs.detach()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        for _ in range(self._ppo_epochs):
            log_probs, entropy = self._policy.evaluate(states, actions_t)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self._clip_epsilon, 1 + self._clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            values_pred = self._value(states)
            value_loss = F.mse_loss(values_pred, returns)

            self._policy_optimizer.zero_grad()
            policy_loss.backward()
            self._policy_optimizer.step()

            self._value_optimizer.zero_grad()
            value_loss.backward()
            self._value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return {
            "policy_loss": total_policy_loss / self._ppo_epochs,
            "value_loss": total_value_loss / self._ppo_epochs,
        }

    def _learn_sac(self) -> Dict[str, float]:
        """SAC learning step (simplified)."""
        if not self._replay_buffer.is_ready(self.batch_size):
            return {"loss": 0.0}
        states, actions, rewards, next_states, dones = self._replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self._policy.evaluate(next_states, self._policy(next_states)[0])
            next_q1 = self._target_q1(next_states).gather(1, next_actions.unsqueeze(-1).long() if not self.continuous else next_actions)
            next_q2 = self._target_q2(next_states).gather(1, next_actions.unsqueeze(-1).long() if not self.continuous else next_actions)
            next_q = torch.min(next_q1, next_q2) - self._alpha * next_log_probs.unsqueeze(-1)
            target_q = rewards + self.gamma * (1 - dones) * next_q

        q1_values = self._q1(states).gather(1, actions_t.long().unsqueeze(-1) if not self.continuous else actions_t)
        q2_values = self._q2(states).gather(1, actions_t.long().unsqueeze(-1) if not self.continuous else actions_t)
        q_loss = F.mse_loss(q1_values, target_q) + F.mse_loss(q2_values, target_q)
        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        new_actions, log_probs = self._policy.evaluate(states, self._policy(states)[0])
        q1_new = self._q1(states).gather(1, new_actions.unsqueeze(-1).long() if not self.continuous else new_actions)
        policy_loss = (self._alpha * log_probs.unsqueeze(-1) - q1_new).mean()
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        if self._total_steps % 100 == 0:
            self._target_q1.load_state_dict(self._q1.state_dict())
            self._target_q2.load_state_dict(self._q2.state_dict())

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item()}

    def update_policy(self) -> Dict[str, float]:
        """Explicitly trigger a policy update (alias for learn).

        Returns:
            Dictionary of training metrics.
        """
        return self.learn()

    def get_value(self, state: Any) -> float:
        """Estimate the value of a given state.

        Args:
            state: Environment state.

        Returns:
            Estimated state value.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            if self.algorithm == "dqn":
                return self._q_network(state).max().item()
            elif self.algorithm in ("ppo", "sac"):
                if hasattr(self, "_value"):
                    return self._value(state).item()
                return 0.0
        return 0.0

    def store_transition(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """Store a transition in the replay buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether the episode ended.
        """
        self._replay_buffer.push(state, action, reward, next_state, done)

    def load_pretrained(self, path: str) -> None:
        """Load pretrained agent weights from disk.

        Args:
            path: File path to the saved model.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            if self.algorithm == "dqn":
                self._q_network.load_state_dict(checkpoint.get("q_network", checkpoint))
            elif self.algorithm == "ppo":
                self._policy.load_state_dict(checkpoint.get("policy", checkpoint))
                if "value" in checkpoint:
                    self._value.load_state_dict(checkpoint["value"])
            elif self.algorithm == "sac":
                self._policy.load_state_dict(checkpoint.get("policy", checkpoint))
            logger.info("Loaded pretrained RL agent from %s", path)
        except FileNotFoundError:
            logger.error("Model file not found: %s", path)
            raise

    def save(self, path: str) -> None:
        """Save agent weights to disk.

        Args:
            path: Destination file path.
        """
        checkpoint: Dict[str, Any] = {}
        if self.algorithm == "dqn":
            checkpoint["q_network"] = self._q_network.state_dict()
        elif self.algorithm == "ppo":
            checkpoint["policy"] = self._policy.state_dict()
            checkpoint["value"] = self._value.state_dict()
        elif self.algorithm == "sac":
            checkpoint["policy"] = self._policy.state_dict()
            checkpoint["q1"] = self._q1.state_dict()
            checkpoint["q2"] = self._q2.state_dict()
        torch.save(checkpoint, path)
        logger.info("RL agent saved to %s", path)

    def get_config_info(self) -> Dict[str, Any]:
        """Return agent configuration as a dictionary."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "algorithm": self.algorithm,
            "continuous": self.continuous,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "device": self.device,
        }
