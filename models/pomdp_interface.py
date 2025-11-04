import numpy as np
from typing import Tuple, Dict, Any

class InitialBelief:
    """Base class for initial belief models."""

    def sample(self, env, num_samples: int) -> np.ndarray:
        """
        Sample a batch of initial states from the belief distribution.

        Args:
            env: The MultiDroneEnvironment instance containing the map and configuration data.
            num_samples: Number of state samples to draw.

        Returns:
            np.ndarray: Array of shape (num_samples, N, 3), where N is the number of drones.
                        Each sample represents the 3D positions of all drones in world coordinates.
        """
        raise NotImplementedError


class TransitionModel:
    """Base class for transition models."""

    def step(self, env, state: np.ndarray, action: int) -> Tuple[np.ndarray, Dict]:
        """
        Propagate the system state forward under a given action.

        Args:
            env: The MultiDroneEnvironment instance providing configuration and obstacle data.
            state: Current state of shape (N, 3), representing all drone positions in world coordinates.
            action: Integer-encoded joint action representing one move per drone.

        Returns:
            Tuple[np.ndarray, Dict]:
                - next_state: Array of shape (N, 3), representing the next positions of all drones.
                - info: Dictionary containing diagnostic or auxiliary information
                        (e.g., collisions, goals reached, out-of-bounds).
        """
        raise NotImplementedError

    def num_actions(self, num_drones: int) -> int:
        """
        Compute the total number of possible joint actions.

        Args:
            num_drones: Number of drones controlled by the agent.

        Returns:
            int: Total number of discrete joint actions across all drones.
        """
        raise NotImplementedError


class ObservationModel:
    """Base class for observation models."""

    def observe(self, env, state: np.ndarray) -> int:
        """
        Generate an integer-encoded observation symbol given the current state.

        Args:
            env: The MultiDroneEnvironment instance (contains map, obstacles, and localization cells).
            state: Current state of shape (N, 3), representing drone positions.

        Returns:
            int: Integer-encoded joint observation symbol representing sensor readings for all drones.
        """
        raise NotImplementedError

    def likelihood(self, env, observation: int, state: np.ndarray) -> float:
        """
        Compute the observation likelihood P(o | s).

        Args:
            env: The MultiDroneEnvironment instance.
            observation: Integer-encoded observation symbol.
            state: State of shape (N, 3).

        Returns:
            float: The likelihood of receiving the given observation in the provided state.
        """
        raise NotImplementedError


class Task:
    """Base class defining the reward and termination logic for a POMDP task."""

    def reward(
        self,
        env,
        prev_state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Dict[str, Any] | None = None,
    ) -> float:
        """
        Compute the reward for a given transition (s, a, s').

        Args:
            env: The MultiDroneEnvironment instance.
            prev_state: Previous state of shape (N, 3).
            action: Integer-encoded joint action taken by the agent(s).
            next_state: Next state of shape (N, 3).
            info: Optional dictionary with auxiliary data (e.g., collisions, goal status).

        Returns:
            float: Scalar reward value corresponding to this transition.
        """
        raise NotImplementedError

    def done(
        self,
        env,
        prev_state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Dict[str, Any] | None = None,
    ) -> bool:
        """
        Determine whether the episode has reached a terminal state.

        Args:
            env: The MultiDroneEnvironment instance.
            prev_state: Previous state of shape (N, 3).
            action: Integer-encoded joint action.
            next_state: Next state of shape (N, 3).
            info: Optional dictionary with auxiliary transition details.

        Returns:
            bool: True if the episode should terminate, False otherwise.
        """
        raise NotImplementedError