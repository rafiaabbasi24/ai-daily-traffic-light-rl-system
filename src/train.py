#!/usr/bin/env python3
"""
src/train.py

Training script for the Dynamic Traffic Light Control project using a
multi‑agent reinforcement learning (RL) approach.

The script handles:
* Argument parsing and reproducibility settings.
* Environment creation (Gymnasium compatible).
* Multi‑agent policy/network initialization.
* The main training loop with optional learning after each step.
* Logging of episode statistics to console and a log file.
* Periodic checkpointing of model parameters and optimizer state.
* Post‑training plot of episode returns.

The script is intentionally self‑contained; it only relies on
standard scientific Python packages and project‑specific modules that
are expected to be present in the ``src`` package:

    - ``src.env``   : definition of ``TrafficEnv`` (Gymnasium environment)
    - ``src.agent`` : definition of ``MultiAgent`` (policy + learning logic)

Both modules must expose the classes with the signatures used below.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Project‑specific imports – these must exist in the repository.
# pylint: disable=import-error
from src.agent import MultiAgent  # type: ignore
from src.env import TrafficEnv   # type: ignore
# pylint: enable=import-error


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across ``random``, ``numpy`` and ``torch``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior where possible (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logging(log_dir: Path) -> logging.Logger:
    """Create a logger that writes to both console and a file."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_dir / "training.log", mode="a")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Avoid duplicate handlers if configure_logging is called multiple times
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(
    checkpoint_dir: Path,
    episode: int,
    agent: MultiAgent,
    optimizer: torch.optim.Optimizer,
    stats: Dict[str, Any],
) -> None:
    """
    Save a checkpoint containing model weights, optimizer state and training metadata.

    Args:
        checkpoint_dir: Directory where the checkpoint will be stored.
        episode: Current episode number (used in the filename).
        agent: MultiAgent instance whose state_dict will be saved.
        optimizer: Optimizer associated with the agent.
        stats: Dictionary of additional information (e.g., average return).
    """
    checkpoint_path = checkpoint_dir / f"checkpoint_ep{episode:05d}.pth"
    torch.save(
        {
            "episode": episode,
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": stats,
        },
        checkpoint_path,
    )
    # Also keep a copy of the latest checkpoint for easy resume
    latest_path = checkpoint_dir / "latest.pth"
    checkpoint_path.replace(latest_path)


def plot_returns(
    returns: List[float],
    save_path: Path,
    title: str = "Episode Returns",
) -> None:
    """
    Plot the episode returns over time and save the figure.

    Args:
        returns: List of total returns per episode.
        save_path: Destination path for the PNG image.
        title: Plot title.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(returns, label="Return")
    plt.xlabel("Episode")
    plt.ylabel("Total Return")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Train Multi‑Agent RL for traffic lights")
    parser.add_argument(
        "--env-name",
        type=str,
        default="TrafficEnv-v0",
        help="Gymnasium environment ID (must be registered).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to store logs and checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save a checkpoint every N episodes.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print training statistics every N episodes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run training on.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the training script."""
    args = parse_args()

    # Prepare directories
    log_dir = Path(args.log_dir).expanduser().resolve()
    checkpoint_dir = log_dir / "checkpoints"
    plot_dir = log_dir / "plots"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(log_dir)
    logger.info("Starting training with configuration: %s", json.dumps(vars(args), indent=2))

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --------------------------------------------------------------------- #
    # Environment creation
    # --------------------------------------------------------------------- #
    try:
        # If the environment is already registered under ``args.env_name``
        env: gym.Env = gym.make(args.env_name, render_mode=None)
    except gym.error.Error:
        # Fallback to the explicit class import (project specific)
        env = TrafficEnv(render_mode=None)

    # Ensure the environment follows the multi‑agent API:
    #   obs, info = env.reset()
    #   next_obs, rewards, terminated, truncated, info = env.step(actions)
    #   where ``obs`` and ``rewards`` are dictionaries keyed by agent IDs.
    obs_space = env.observation_space
    act_space = env.action_space

    # --------------------------------------------------------------------- #
    # Agent initialization
    # --------------------------------------------------------------------- #
    # Assume MultiAgent can be instantiated with the following signature:
    # MultiAgent(state_dim, action_dim, num_agents, lr, gamma, device)
    # ``state_dim`` and ``action_dim`` are inferred from the spaces.
    # For simplicity we treat them as flat vectors.
    if isinstance(obs_space, gym.spaces.Dict):
        # Concatenate all sub‑spaces into a single dimension
        state_dim = sum(
            int(np.prod(space.shape)) for space in obs_space.spaces.values()
        )
    else:
        state_dim = int(np.prod(obs_space.shape))

    if isinstance(act_space, gym.spaces.Dict):
        action_dim = sum(
            int(np.prod(space.n)) if isinstance(space, gym.spaces.Discrete) else int(np.prod(space.shape))
            for space in act_space.spaces.values()
        )
    elif isinstance(act_space, gym.spaces.Discrete):
        action_dim = int(act_space.n)
    else:
        action_dim = int(np.prod(act_space.shape))

    # Number of agents is derived from the observation dictionary keys,
    # otherwise we assume a single agent.
    num_agents = len(obs) if isinstance(obs, dict) else 1

    agent = MultiAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        lr=args.lr,
        gamma=args.gamma,
        device=device,
    )
    optimizer = agent.optimizer  # type: ignore[attr-defined]

    # --------------------------------------------------------------------- #
    # Training loop
    # --------------------------------------------------------------------- #
    episode_returns: List[float] = []
    episode_lengths: List[int] = []

    for episode in tqdm(range(1, args.num_episodes + 1), desc="Training"):
        obs, info = env.reset()
        # Convert observation dict to a flat NumPy array per agent
        # Helper function for flattening
        def flatten_observation(o: Any) -> np.ndarray:
            if isinstance(o, dict):
                return np.concatenate([flatten_observation(v) for v in o.values()], dtype=np.float32)
            return np.asarray(o, dtype=np.float32).reshape(-1)

        # Initial state for each agent
        state = {aid: flatten_observation(o) for aid, o in obs.items()} if isinstance(obs, dict) else {"0": flatten_observation(obs)}

        total_reward = 0.0
        step = 0
        done = False

        while not done and step < args.max_steps:
            # Agent selects actions for all agents
            actions = agent.select_actions(state)  # expects dict[agent_id] -> np.ndarray or int

            # Step the environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            # Determine termination condition
            done = terminated or truncated

            # Flatten next observations
            next_state = (
                {aid: flatten_observation(o) for aid, o in next_obs.items()}
                if isinstance(next_obs, dict)
                else {"0": flatten_observation(next_obs)}
            )

            # Store transition in the agent's replay buffer (implementation‑specific)
            agent.store_transition(state, actions, rewards, next_state, done)

            # Perform a learning update (could be every step or every few steps)
            agent.learn()

            # Accumulate statistics
            total_reward += sum(rewards.values()) if isinstance(rewards, dict) else float(rewards)
            state = next_state
            step += 1

        episode_returns.append(total_reward)
        episode_lengths.append(step)

        # Logging
        if episode % args.log_interval == 0:
            avg_ret = np.mean(episode_returns[-args.log_interval :])
            avg_len = np.mean(episode_lengths[-args.log_interval :])
            logger.info(
                "Episode %04d | Avg Return: %.2f | Avg Length: %.1f | Steps: %d",
                episode,
                avg_ret,
                avg_len,
                step,
            )

        # Checkpointing
        if episode % args.checkpoint_interval == 0:
            stats = {
                "average_return_last_interval": np.mean(
                    episode_returns[-args.checkpoint_interval :]
                ),
                "average_length_last_interval": np.mean(
                    episode_lengths[-args.checkpoint_interval :]
                ),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                episode=episode,
                agent=agent,
                optimizer=optimizer,
                stats=stats,
            )
            logger.info("Saved checkpoint for episode %d", episode)

    # --------------------------------------------------------------------- #
    # Post‑training visualisation
    # --------------------------------------------------------------------- #
    plot_path = plot_dir / "episode_returns.png"
    plot_returns(episode_returns, plot_path)
    logger.info("Training complete. Return plot saved to %s", plot_path)

    # Final checkpoint
    final_stats = {
        "average_return": float(np.mean(episode_returns)),
        "average_length": float(np.mean(episode_lengths)),
        "total_episodes": args.num_episodes,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        episode=args.num_episodes,
        agent=agent,
        optimizer=optimizer,
        stats=final_stats,
    )
    logger.info("Final checkpoint saved.")


if __name__ == "__main__":
    main()
