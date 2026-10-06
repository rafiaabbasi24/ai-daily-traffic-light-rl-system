import unittest
import importlib
import inspect
import numpy as np
import torch
import gymnasium
from gymnasium import spaces


def _import_class(module_names, class_name):
    """
    Try to import ``class_name`` from a list of possible module names.
    Returns the class if found, otherwise raises ImportError.
    """
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, class_name)
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        f"Unable to import class '{class_name}' from any of: {module_names}"
    )


def _get_environment():
    """
    Attempt to create a Gymnasium environment that matches the project.
    If the environment cannot be created, fall back to a dummy env with
    a reasonable observation and action space.
    """
    possible_ids = [
        "TrafficLight-v0",
        "DynamicTrafficLight-v0",
        "MultiAgentTrafficLight-v0",
    ]
    for env_id in possible_ids:
        try:
            env = gymnasium.make(env_id)
            return env
        except Exception:
            continue

    # Fallback dummy environment – 10‑dimensional continuous observation,
    # 4 discrete actions. Adjust if the real project uses different sizes.
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    act_space = spaces.Discrete(4)

    class DummyEnv:
        observation_space = obs_space
        action_space = act_space

    return DummyEnv()


class TestAgent(unittest.TestCase):
    """Test suite for the traffic‑light RL agent and its neural network."""

    @classmethod
    def setUpClass(cls):
        # Resolve the network and agent classes from the project.
        cls.NetworkClass = _import_class(
            ["model", "models", "network", "networks"], "TrafficNet"
        )
        cls.AgentClass = _import_class(
            ["agent", "agents", "rl_agent"], "TrafficLightAgent"
        )

        # Build (or fake) an environment to obtain observation/action dimensions.
        env = _get_environment()
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                "Observation space must be a Box; got "
                f"{type(env.observation_space)}"
            )
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError(
                "Action space must be Discrete; got "
                f"{type(env.action_space)}"
            )

        cls.obs_dim = int(np.prod(env.observation_space.shape))
        cls.action_dim = int(env.action_space.n)

        # Initialise a deterministic network for reproducibility.
        torch.manual_seed(0)
        cls.network = cls.NetworkClass(
            input_dim=cls.obs_dim, output_dim=cls.action_dim
        )

        # Initialise the agent.  Most implementations accept a network and an
        # optional epsilon for epsilon‑greedy policies.  We try a few common
        # signatures to stay compatible with different code bases.
        try:
            # Preferred signature: (network, epsilon)
            cls.agent = cls.AgentClass(cls.network, epsilon=0.0)
        except TypeError:
            try:
                # Some agents expose a config dict or kwargs.
                sig = inspect.signature(cls.AgentClass)
                if "network" in sig.parameters:
                    cls.agent = cls.AgentClass(network=cls.network, epsilon=0.0)
                else:
                    cls.agent = cls.AgentClass(cls.network)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to instantiate {cls.AgentClass.__name__}: {exc}"
                ) from exc

    def test_network_forward_pass(self):
        """The network must return a tensor of shape (1, action_dim)."""
        # Create a random observation and add a batch dimension.
        obs_np = np.random.randn(self.obs_dim).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)  # shape: (1, obs_dim)

        # Forward pass through the network.
        with torch.no_grad():
            output = self.network(obs_tensor)

        # Verify output type and shape.
        self.assertIsInstance(output, torch.Tensor, "Network output is not a torch.Tensor")
        expected_shape = (1, self.action_dim)
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Network output shape {output.shape} != expected {expected_shape}",
        )

    def test_agent_action_selection(self):
        """Agent.select_action should return a valid discrete action."""
        # Generate a random observation matching the environment spec.
        state_np = np.random.randn(self.obs_dim).astype(np.float32)

        # Call the agent's action selection method.
        # The signature varies across implementations; we handle the two most
        # common patterns.
        try:
            action = self.agent.select_action(state_np, epsilon=0.0)
        except TypeError:
            # Fallback to a single‑argument version.
            action = self.agent.select_action(state_np)

        # The action must be an integer within the valid range.
        self.assertIsInstance(
            action,
            (int, np.integer),
            f"Action type is {type(action)}; expected int or numpy integer",
        )
        self.assertGreaterEqual(
            action,
            0,
            f"Action {action} is less than the minimum valid value 0",
        )
        self.assertLess(
            action,
            self.action_dim,
            f"Action {action} exceeds the maximum valid value {self.action_dim - 1}",
        )


if __name__ == "__main__":
    unittest.main()