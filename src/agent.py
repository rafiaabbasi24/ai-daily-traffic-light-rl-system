import copy
import random
from collections import deque
from typing import Deque, Tuple, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_device() -> torch.device:
    """
    Returns the best available torch device (GPU if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    Policy network mapping states to actions.
    Uses tanh activation on the output to keep actions in [-1, 1].
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], action_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using uniform distribution as in the original DDPG paper
        fan_in = self.fc1.weight.data.size()[0]
        lim = 1.0 / np.sqrt(fan_in)
        nn.init.uniform_(self.fc1.weight, -lim, lim)
        nn.init.uniform_(self.fc1.bias, -lim, lim)

        fan_in = self.fc2.weight.data.size()[0]
        lim = 1.0 / np.sqrt(fan_in)
        nn.init.uniform_(self.fc2.weight, -lim, lim)
        nn.init.uniform_(self.fc2.bias, -lim, lim)

        # Final layer init to small values to avoid saturation
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Output bounded in [-1, 1] (will be scaled later by env action range)
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """
    Q‑network estimating the value of a state‑action pair.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, int] = (256, 256)):
        super().__init__()
        # First layer processes state only
        self.fcs1 = nn.Linear(state_dim, hidden_dims[0])
        # Second layer processes concatenated [state_hidden, action]
        self.fc2 = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.fcs1.weight.data.size()[0]
        lim = 1.0 / np.sqrt(fan_in)
        nn.init.uniform_(self.fcs1.weight, -lim, lim)
        nn.init.uniform_(self.fcs1.bias, -lim, lim)

        fan_in = self.fc2.weight.data.size()[0]
        lim = 1.0 / np.sqrt(fan_in)
        nn.init.uniform_(self.fc2.weight, -lim, lim)
        nn.init.uniform_(self.fc2.bias, -lim, lim)

        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xs = F.relu(self.fcs1(state))
        x = torch.cat([xs, action], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OUNoise:
    """
    Ornstein‑Uhlenbeck process for temporally correlated exploration noise.
    """

    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.size = size
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state = self.state + dx
        return self.state


class ReplayBuffer:
    """
    Fixed‑size buffer to store experience tuples.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a new transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.stack(states), dtype=torch.float32)
        actions = torch.as_tensor(np.stack(actions), dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class MultiAgentDDPG:
    """
    Multi‑agent DDPG where all agents share a single actor and critic.
    Each agent interacts with the environment independently, but gradients
    are aggregated across agents before a single optimizer step.
    """

    def __init__(
        self,
        env: gym.Env,
        n_agents: int,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 5e-3,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        hidden_dims: Tuple[int, int] = (256, 256),
        seed: int = 42,
    ):
        """
        Initialise the multi‑agent DDPG learner.

        Args:
            env: Gymnasium environment (used to infer state/action dimensions).
            n_agents: Number of traffic‑light agents.
            actor_lr: Learning rate for the policy network.
            critic_lr: Learning rate for the Q‑network.
            gamma: Discount factor.
            tau: Soft update coefficient for target networks.
            buffer_capacity: Maximum number of transitions stored.
            batch_size: Number of samples per training step.
            hidden_dims: Hidden layer sizes for both networks.
            seed: Random seed for reproducibility.
        """
        self.env = env
        self.n_agents = n_agents
        self.device = get_device()
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Extract dimensions from the environment
        sample_obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        if isinstance(sample_obs, (list, tuple)):
            # Assume each agent receives its own observation vector
            state_dim = sample_obs[0].shape[0]
        else:
            state_dim = sample_obs.shape[0]

        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Box):
            action_dim = action_space.shape[0]
            self.action_low = torch.tensor(action_space.low, dtype=torch.float32, device=self.device)
            self.action_high = torch.tensor(action_space.high, dtype=torch.float32, device=self.device)
        else:
            raise ValueError("DDPG requires a continuous Box action space.")

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared networks
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)

        # Exploration noise (one per agent)
        self.noise = [OUNoise(action_dim) for _ in range(n_agents)]

        # Seed everything for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def select_action(self, states: List[np.ndarray], noise_scale: float = 0.1) -> List[np.ndarray]:
        """
        Compute actions for all agents given their current states.

        Args:
            states: List of state vectors, one per agent.
            noise_scale: Multiplicative factor for exploration noise.

        Returns:
            List of actions (numpy arrays) clipped to the environment's action bounds.
        """
        self.actor.eval()
        actions = []
        with torch.no_grad():
            for i, state in enumerate(states):
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                raw_action = self.actor(state_tensor).cpu().numpy().flatten()
                # Add OU noise
                noise = self.noise[i].sample() * noise_scale
                noisy_action = raw_action + noise
                # Clip to env bounds
                clipped = np.clip(noisy_action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
                actions.append(clipped)
        self.actor.train()
        return actions

    def store_transition(
        self,
        state: List[np.ndarray],
        action: List[np.ndarray],
        reward: List[float],
        next_state: List[np.ndarray],
        done: List[bool],
    ) -> None:
        """
        Store a joint transition (one per agent) in the shared replay buffer.
        Each agent's tuple is stored individually.
        """
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            self.replay_buffer.push(s, a, r, ns, d)

    def train(self) -> None:
        """
        Perform a single training step using a batch sampled from the replay buffer.
        Updates both actor and critic networks and performs soft updates on target networks.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough data to train

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ---------- Critic update ----------
        with torch.no_grad():
            # Target actions from target actor
            next_actions = self.actor_target(next_states)
            # Target Q-values from target critic
            q_next = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * q_next

        # Current Q estimate
        q_current = self.critic(states, actions)
        critic_loss = F.mse_loss(q_current, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---------- Actor update ----------
        # Actor loss = -E[Q(s, μ(s))]
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---------- Soft updates ----------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """
        Perform Polyak averaging (soft update) of target network parameters.
        target = tau * source + (1 - tau) * target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def reset_noise(self) -> None:
        """Reset the OU noise processes for all agents (e.g., at episode start)."""
        for n in self.noise:
            n.reset()

    def save(self, path: str) -> None:
        """
        Save model parameters and optimizer states.
        """
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model parameters and optimizer states.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        # Ensure target networks are in sync after loading
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


# Example usage (will only run when this file is executed directly)
if __name__ == "__main__":
    # Create a dummy continuous environment for sanity checking
    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            next_obs = np.random.uniform(-1, 1, size=4).astype(np.float32)
            reward = float(np.random.randn())
            done = random.random() < 0.05
            return next_obs, reward, done, False, {}

    env = DummyEnv()
    n_agents = 3
    agent = MultiAgentDDPG(env, n_agents)

    # Simulate a single interaction loop
    obs = [env.reset()[0] for _ in range(n_agents)]
    actions = agent.select_action(obs, noise_scale=0.0)  # deterministic actions
    next_obs, rewards, dones, _, _ = zip(*[env.step(a) for a in actions])
    agent.store_transition(obs, actions, rewards, next_obs, dones)
    agent.train()
    print("Single training step completed successfully.")