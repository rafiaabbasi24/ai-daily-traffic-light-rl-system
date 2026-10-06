import pytest
import numpy as np
import gymnasium
from gymnasium import Env
from typing import Any

# Attempt to import the environment class directly.
# The actual project may expose the environment under a different name or registration.
# Adjust the import path if necessary.
try:
    # Common pattern: the environment class is defined in a module named `env` or `traffic_env`.
    from env import TrafficEnv  # type: ignore
except ImportError:
    try:
        from traffic_env import TrafficEnv  # type: ignore
    except ImportError:
        TrafficEnv = None  # Fallback to gymnasium registration later.


def _make_env() -> Env:
    """
    Helper to create an instance of the traffic environment.

    Returns
    -------
    Env
        An instantiated Gymnasium environment ready for interaction.
    """
    if TrafficEnv is not None:
        env = TrafficEnv()
    else:
        # Try to create the environment via gymnasium registration.
        # The registration name is guessed; replace with the correct one if different.
        try:
            env = gymnasium.make("DynamicTraffic-v0")
        except gymnasium.error.Error as exc:
            raise RuntimeError(
                "Could not locate the traffic environment. Ensure that either "
                "`TrafficEnv` can be imported or the environment is registered "
                "with Gymnasium under the name `DynamicTraffic-v0`."
            ) from exc
    # Seed for reproducibility
    env.reset(seed=42)
    return env


def test_environment_instantiation():
    """Test that the environment can be instantiated and follows the Gymnasium API."""
    env = _make_env()
    assert isinstance(env, Env), "The created object is not a Gymnasium Env."
    # Check that observation and action spaces are defined
    assert hasattr(env, "observation_space"), "Env missing observation_space."
    assert hasattr(env, "action_space"), "Env missing action_space."
    # Basic sanity checks on the spaces
    assert isinstance(env.observation_space, gymnasium.spaces.Space), "observation_space is not a gymnasium Space."
    assert isinstance(env.action_space, gymnasium.spaces.Space), "action_space is not a gymnasium Space."
    env.close()


def test_reset_returns_valid_observation():
    """Validate that `reset` returns an observation matching the declared space."""
    env = _make_env()
    observation, info = env.reset()
    # Observation must be compatible with the observation space
    assert env.observation_space.contains(observation), (
        f"Reset observation {observation} not contained in observation_space {env.observation_space}"
    )
    # Info should be a dict (Gymnasium convention)
    assert isinstance(info, dict), "Info returned by reset should be a dict."
    env.close()


def test_step_returns_correct_types():
    """Check that `step` returns a tuple of (obs, reward, terminated, truncated, info)."""
    env = _make_env()
    # Use a valid random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Observation type/shape check
    assert env.observation_space.contains(obs), "Step observation not in observation_space."

    # Reward must be a scalar float or int
    assert isinstance(reward, (float, int, np.floating, np.integer)), "Reward is not a numeric scalar."

    # Termination flags must be booleans
    assert isinstance(terminated, bool), "terminated flag is not a bool."
    assert isinstance(truncated, bool), "truncated flag is not a bool."

    # Info must be a dict
    assert isinstance(info, dict), "Info returned by step should be a dict."

    env.close()


def test_reward_is_finite_and_non_nan():
    """Ensure that the reward returned by the environment is a finite number."""
    env = _make_env()
    action = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)

    # Convert to float for NumPy checks
    reward_val = float(reward)
    assert np.isfinite(reward_val), f"Reward {reward_val} is not finite."
    env.close()


def test_multiple_steps_do_not_crash():
    """
    Run a short episode (10 steps) to verify that the environment
    remains stable over multiple interactions.
    """
    env = _make_env()
    observation, _ = env.reset()
    for step_idx in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # Basic sanity checks each step
        assert env.observation_space.contains(observation), f"Step {step_idx}: observation out of bounds."
        assert isinstance(reward, (float, int, np.floating, np.integer)), f"Step {step_idx}: reward not numeric."
        assert isinstance(terminated, bool) and isinstance(truncated, bool), f"Step {step_idx}: termination flags not bool."

        if terminated or truncated:
            # If the episode ends early, break out of the loop.
            break
    env.close()


def test_deterministic_reward_given_fixed_seed():
    """
    Verify that with a fixed random seed the sequence of rewards is reproducible.
    This checks that the environment's stochastic components are correctly seeded.
    """
    # First run
    env1 = _make_env()
    obs1, _ = env1.reset(seed=123)
    rewards1 = []
    for _ in range(5):
        action = env1.action_space.sample()
        _, r, _, _, _ = env1.step(action)
        rewards1.append(float(r))
    env1.close()

    # Second run with the same seed
    env2 = _make_env()
    obs2, _ = env2.reset(seed=123)
    rewards2 = []
    for _ in range(5):
        action = env2.action_space.sample()
        _, r, _, _, _ = env2.step(action)
        rewards2.append(float(r))
    env2.close()

    # The two reward sequences should match exactly
    assert np.allclose(rewards1, rewards2, atol=1e-7), (
        f"Reward sequences differ across runs with the same seed.\n"
        f"Run1: {rewards1}\nRun2: {rewards2}"
    )
    # Also ensure that the initial observations are identical
    assert np.array_equal(obs1, obs2), "Initial observations differ despite identical seeds."
