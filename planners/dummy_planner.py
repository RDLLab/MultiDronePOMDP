import numpy as np
from multi_drone_environment import MultiDroneEnvironment

class DummyPlanner:
    def __init__(self, env: MultiDroneEnvironment, a_param: float = 1.0, b_param: int = 1.0):
        self._env = env
        self._a_param = a_param
        self._b_param = b_param
        self._num_actions = env.num_actions() # Total number of actions available in the environment

    def plan(self, belief_state, planning_time_per_step: float) -> int:
        # This doesn't do anything useful. It simply returns the action 
        # representen by integer 0.
        return 0
