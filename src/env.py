import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any, Optional


class TrafficGridEnv(gym.Env):
    """
    Custom Gymnasium environment that simulates a grid of traffic intersections.
    Each intersection is controlled by an independent agent that selects one of two
    traffic light phases:

    * Phase 0 – North‑South green, East‑West red.
    * Phase 1 – East‑West green, North‑South red.

    The state of an intersection consists of the queue lengths for the four
    incoming directions (North, South, East, West).  The environment is fully
    deterministic apart from stochastic vehicle arrivals.

    The environment follows the multi‑agent API used by many RL libraries:
    observations, actions, rewards, terminations and truncations are dictionaries
    keyed by agent identifiers (``"<row>_<col>"``).

    Parameters
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    max_queue : int, optional
        Maximum number of vehicles that can wait in a single lane.  Queue lengths
        are clipped to this value.  Default is ``20``.
    arrival_prob : float, optional
        Probability that a new vehicle arrives at a lane each step.  Must be in
        ``[0, 1]``.  Default is ``0.1``.
    capacity : int, optional
        Maximum number of vehicles that can pass through an intersection in a
        single green phase per direction.  Default is ``5``.
    max_steps : int, optional
        Episode length limit.  After ``max_steps`` steps the episode is
        automatically terminated.  Default is ``500``.
    seed : Optional[int], optional
        Random seed for reproducibility.
    """

    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(
        self,
        rows: int,
        cols: int,
        max_queue: int = 20,
        arrival_prob: float = 0.1,
        capacity: int = 5,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if rows <= 0 or cols <= 0:
            raise ValueError("Grid dimensions must be positive integers.")
        if not (0.0 <= arrival_prob <= 1.0):
            raise ValueError("arrival_prob must be between 0 and 1.")

        self.rows = rows
        self.cols = cols
        self.max_queue = max_queue
        self.arrival_prob = arrival_prob
        self.capacity = capacity
        self.max_steps = max_steps
        self.current_step = 0

        # Agent identifiers are strings "r_c" where r=row, c=col
        self.agent_ids: List[str] = [
            f"{r}_{c}" for r in range(self.rows) for c in range(self.cols)
        ]

        # Observation: queue lengths for 4 directions per intersection
        single_obs_space = spaces.Box(
            low=0,
            high=self.max_queue,
            shape=(4,),
            dtype=np.int32,
        )
        self.observation_space = spaces.Dict(
            {aid: single_obs_space for aid in self.agent_ids}
        )

        # Action: 0 (NS green) or 1 (EW green)
        single_act_space = spaces.Discrete(2)
        self.action_space = spaces.Dict(
            {aid: single_act_space for aid in self.agent_ids}
        )

        # Internal state: shape (rows, cols, 4) – queues per direction
        self._queues: np.ndarray = np.zeros((self.rows, self.cols, 4), dtype=np.int32)

        self._rng = np.random.default_rng(seed)

        # Rendering helpers
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def _reset_queues(self) -> None:
        """Reset all queues to zero."""
        self._queues.fill(0)

    def _sample_arrivals(self) -> None:
        """Add stochastic vehicle arrivals to each lane."""
        arrivals = self._rng.random(self._queues.shape) < self.arrival_prob
        self._queues = np.minimum(
            self._queues + arrivals.astype(np.int32), self.max_queue
        )

    def _move_vehicles(self, actions: Dict[str, int]) -> None:
        """
        Move vehicles according to the chosen traffic light phases.

        Parameters
        ----------
        actions : dict
            Mapping from agent identifier to chosen phase (0 or 1).
        """
        # Prepare a copy to accumulate incoming vehicles from neighbours
        incoming = np.zeros_like(self._queues)

        for aid, phase in actions.items():
            r, c = map(int, aid.split("_"))
            # Directions: 0=N, 1=S, 2=E, 3=W
            if phase == 0:  # NS green
                dirs = [0, 1]  # N and S can move
            else:  # EW green
                dirs = [2, 3]  # E and W can move

            for d in dirs:
                queue_len = self._queues[r, c, d]
                moving = min(queue_len, self.capacity)

                # Reduce the local queue
                self._queues[r, c, d] -= moving

                # Determine target cell and target direction
                if d == 0:  # north‑bound (moving south)
                    nr, nc = r + 1, c
                    target_dir = 0  # enters the north queue of the cell below
                elif d == 1:  # south‑bound (moving north)
                    nr, nc = r - 1, c
                    target_dir = 1
                elif d == 2:  # east‑bound (moving west)
                    nr, nc = r, c - 1
                    target_dir = 2
                else:  # d == 3, west‑bound (moving east)
                    nr, nc = r, c + 1
                    target_dir = 3

                # If the target cell is inside the grid, add to its queue,
                # otherwise the vehicle leaves the system.
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    incoming[nr, nc, target_dir] += moving

        # Apply incoming vehicles, respecting max_queue
        self._queues = np.minimum(self._queues + incoming, self.max_queue)

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Return a dict of observations keyed by agent id."""
        obs = {}
        for aid in self.agent_ids:
            r, c = map(int, aid.split("_"))
            obs[aid] = self._queues[r, c].copy()
        return obs

    def _compute_rewards(self) -> Dict[str, float]:
        """
        Simple reward: negative sum of queue lengths for each intersection.
        This encourages agents to minimize waiting vehicles.
        """
        rewards = {}
        for aid in self.agent_ids:
            r, c = map(int, aid.split("_"))
            rewards[aid] = -float(self._queues[r, c].sum())
        return rewards

    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Returns
        -------
        observation : dict
            Initial observations for each agent.
        info : dict
            Empty dictionary (placeholder for compatibility).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_step = 0
        self._reset_queues()
        self._sample_arrivals()  # optional initial traffic

        return self._get_observations(), {}

    def step(
        self, action: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """
        Execute one time step.

        Parameters
        ----------
        action : dict
            Mapping from agent id to chosen phase (0 or 1).

        Returns
        -------
        observation : dict
            New observations for each agent.
        reward : dict
            Reward for each agent.
        terminated : dict
            Whether each agent's episode has terminated (always ``False`` here).
        truncated : dict
            Whether each agent's episode was truncated due to time limit.
        info : dict
            Empty dictionary (placeholder for compatibility).
        """
        if not isinstance(action, dict):
            raise TypeError("Action must be a dict mapping agent ids to integers.")

        # Validate actions
        for aid, a in action.items():
            if aid not in self.agent_ids:
                raise ValueError(f"Invalid agent id '{aid}'.")
            if a not in (0, 1):
                raise ValueError(f"Invalid action {a} for agent '{aid}'. Must be 0 or 1.")

        self._move_vehicles(action)
        self._sample_arrivals()

        self.current_step += 1
        terminated = {aid: False for aid in self.agent_ids}
        truncated = {
            aid: self.current_step >= self.max_steps for aid in self.agent_ids
        }

        obs = self._get_observations()
        rewards = self._compute_rewards()
        info = {}

        return obs, rewards, terminated, truncated, info

    def render(self) -> None:
        """
        Render the current state using Matplotlib.
        Intersections are drawn as squares; arrows indicate the current phase,
        and the queue length for each direction is shown as text.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(self.cols * 2, self.rows * 2))
        self._ax.clear()
        self._ax.set_aspect("equal")
        self._ax.set_xticks(np.arange(self.cols + 1) - 0.5, minor=False)
        self._ax.set_yticks(np.arange(self.rows + 1) - 0.5, minor=False)
        self._ax.grid(True, which="both", color="gray", linewidth=1)

        # Draw each intersection
        for r in range(self.rows):
            for c in range(self.cols):
                # Background square
                self._ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, self.rows - r - 1 - 0.5),
                        1,
                        1,
                        facecolor="#e0e0e0",
                        edgecolor="black",
                    )
                )
                # Queues as text
                queues = self._queues[r, c]
                directions = ["N", "S", "E", "W"]
                offsets = [(0, 0.35), (0, -0.35), (0.35, 0), (-0.35, 0)]
                for q, d, (dx, dy) in zip(queues, directions, offsets):
                    self._ax.text(
                        c + dx,
                        self.rows - r - 1 + dy,
                        f"{d}:{q}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="blue",
                    )

        self._ax.set_xlim(-0.5, self.cols - 0.5)
        self._ax.set_ylim(-0.5, self.rows - 0.5)
        self._ax.invert_yaxis()
        self._ax.set_title(f"Traffic Grid – Step {self.current_step}")
        plt.pause(0.001)

    def close(self) -> None:
        """Close the rendering window."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    # --------------------------------------------------------------------- #
    # Utility methods
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(rows={self.rows}, cols={self.cols}, "
            f"max_queue={self.max_queue}, arrival_prob={self.arrival_prob}, "
            f"capacity={self.capacity}, max_steps={self.max_steps})"
        )