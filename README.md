
# MultiDrone POMDP Simulator
This repository contains Python code for a compact MultiDrone POMDP simulator with multiple, centrally controlled drones operating in a 3D world, populated by static obstacles and goal regions, while being subject to uncertainties in the outcome of actions and perceiving observations. 

## Requirements
The following Python libraries are required:
 - [NumPy](https://numpy.org/) (version >= 2.3.4)
 - [SciPy](https://scipy.org) (version >= 1.16.3)
 - [PyYAML](https://pypi.org/project/PyYAML/) (version >= 6.0)
 - [SciPy](https://scipy.org/) (version >= 1.16.3)
 - [Vedo](https://pypi.org/project/vedo/) (version >= 2025.5.4)

## The MultiDrone POMDP Problem
The Multi-Drone POMDP models a cooperative navigation problem where multiple drones operate in a bounded 3D environment populated by static obstacles. Each drone starts from a noisy initial position and must navigate to its own goal region (rewarded by a goal reward) while avoiding collisions—with both obstacles and other drones—or leaving the environment (penalized by a collision penalty). Additionally, each step incurs a small motion cost per drone.

The **state** of this problem is the joint configuration of all drones in world coordinates:

`s_t = [x1, y1, z1, ..., xN, yN, zN] ∈ R^{3N}`,

where each component corresponds to the 3D position of one drone.

The **actions**  represent discrete 3D motion commands for the drones. Each drone can move in one of 26 possible 3D directions by one unit length. At every step, the joint action for all drones is formed by combining their individual choices, resulting in a total of `26^N` (with `N` being the number of drones) possible joint actions. Given the current position `p_current` of a drone and an action, its next position is thereby defined by 

`p_next = p_current + u + noise`,

where `u` is the chosen motion vector of the drone corresponding to the action, and `noise` is a random zero-mean Gaussian perturbation vector that models uncertainties in the motion.

Each drone has access to two types of sensors: The first is a noisy range sensor, which provides the drone with an approximate distance to the nearest obstacle in the environment. The second is a localization sensor, which activates when the drone enters designated 3D cells that serve as known reference points. Together, these sensors allow the drones to estimate their position within the environment, combining coarse range information with occasional precise localization cues.

## MultiDrone POMDP Implementation
The above MultiDrone POMDP problem is implemented as a configurable `MultiDroneEnvironment` in class in `multi_drone_environment.py`, which integrates four modular **generative POMDP model components** that define the stochastic behavior of the simulator, located in `models/multi_drone_model.py`:

- Transition Model (`MultiDroneTransitionModel`): Simulates the drone motion in 3D space given a joint action. It applies drone position updates with process noise and checks for collisions, out-of-bounds moves, and goal completion.
- Observation Model (`MultiDroneObservationModel`): Produces sensor readings for each drone. This includes noisy range measurements to nearby obstacles and localization cues when a drone enters predefined reference cells. It also computes the likelihood `p(o | s)`, that is, likelihood of seeing an observation, given a state.
- Initial Belief (`MultiDroneInitialBelief`): Samples initial drone positions around predefined start locations with configurable uniform noise, representing uncertainty in the starting state.
- Task (`MultiDroneTask`): Defines the reward function and termination conditions.

The `MultiDroneEnvironment` class assumes that actions and observations are encoded as integers, and states are represented as Numpy `ndarray` objects, containing the 3D positions of all drones. `MultiDroneEnvironment` provides the following main functions: 

 - `reset() -> np.ndarray`:  
Resets the environment and returns the initial state.
 - `step(action_int: int) -> Tuple[np.ndarray, int, float, bool, Dict]:`: 
Given an integer representing an action, simulates the environment forward by one time step. Internally, this function calls `simulate()` to compute the next state, observation, reward, and termination flag, then updates the environment’s internal state and visualization. It returns:
        - The next state of the environment.
        - An integer-encoded observation.
        - An immediate reward.
        - A boolean termination flag, indicating whether the environment has entered a terminal state.
        - An `info` dictionary containing diagnostic details (e.g., collisions, goal reached, out-of-bounds).
 - `simulate(state: np.ndarray, action_int: int)`: 
Given a state and an action, this function simulates a single transition without altering the internal environment state. It calls the individual POMDP model components in sequence:
        - The transition model to compute the next state,
        - The observation model to generate an observation,
        - The task to compute the reward and check for termination.
- `update_plot(belief_particles: Optional[np.ndarray] = None)`: Updates the 3D visualization of the environment. This function refreshes the positions of all drones and can optionally render a set of belief particles as transparent “ghost” drones for visualizing uncertainty. It is typically called after each `step()` to animate the drones’ movement and maintain an up-to-date scene.
- `show()`: Enters the interactive 3D viewing mode using Vedo. This allows the user to freely inspect the current environment, obstacles, goals, and drone positions after a simulation run has finished.

## Setting Up the MultiDroneEnvironment
To use the `MultiDroneEnvironment` class, we first define a 3D environment inside a YAML configuration file (e.g. `configs/config_simple.yaml`). To define the 3D environment, we can use the following parameters (an example is provided in `configs/config_hard.yaml`):

  - `environment_size`:  The size of the 3D environment
  - `num_controlled_drones`: The number of drones operating inside the environment
  - `obstacle_positions`: A list of 3D obstacle positions in the environment. An obstacle is defined as a simple unit cute.
  - `goal_positions`: A list of 3D positions of the drone's goal areas, one for each drone
  - `goal_radius`: The radius of each goal area (we assume that goal areas are modeled as spheres)

In the same YAML file, we can configure the initial belief, transition, observation, and reward parameters:

- `initial_drone_positions`: A list of 3D coordinates specifying each drone’s nominal starting position.
- `initial_drone_uncertainty`: Uncertainty in the initial drone positions. The initial position of each drone will be sampled uniformly between `initial_drone_position - 0.5 * initial_drone_uncertainty` and `start_position + 0.5 * initial_drone_uncertainty` along each axis
- `transition_noise_std`: The standard deviation of the Gaussian transition error distribution
- `use_distance_sensor`: Whether the drones should receive observations from their range sensors
- `localize_cells`: A list of 3D coordinates specifying the positions of localization cells (specified as unit cubes). Each cell acts as a unique reference point that provides a distinct integer observation when a drone enters it.
- `obs_correct_prob`: The probability that the observation model returns the correct observation. With probability `1 - obs_correct_prob`, a random incorrect observation is returned to simulate sensor noise.
- `goal_reward`: The reward a drone receives when entering its goal area.
- `collision_penalty`: The penalty a drone receives when colliding with an obstacle, another drone, or leaves the environment.
- `step_cost`: A small negative reward incurred at every step to encourage the drones to reach their goals efficiently.
- `discount_factor`: The discount factor (gamma) of the POMDP.
- `max_num_steps`: The maximum number of steps before the environment terminates

## Using the MultiDroneEnvironment
The snippet below demonstrates how to load a YAML configuration, instantiate the Multi-Drone POMDP model components, create the `MultiDroneEnvironment`, and step through a simple episode (provided in ``example_usage.py``).

```
import argparse
import numpy as np

# Import the MultiDroneEnvironment class and the MultiDrone POMDP model components
from multi_drone_environment import MultiDroneEnvironment
from models.multi_drone_model import (
    MultiDroneTransitionModel,
    MultiDroneObservationModel,
    MultiDroneInitialBelief,
    MultiDroneTask
)

# Setup argparse to load a YAML file from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="Path to the yaml configuration file")
args = parser.parse_args()

# Instantiate the POMDP model components
transition_model = MultiDroneTransitionModel()
observation_model = MultiDroneObservationModel()
initial_belief_model = MultiDroneInitialBelief()
task_model = MultiDroneTask()

# Instantiate the MultiDroneEnvironment
env = MultiDroneEnvironment(
    args.config,
    transition_model,
    observation_model,
    initial_belief_model,
    task_model,
)

# Reset environment
state = env.reset()
done = False

while not done:
    # Pick a random action
    action = np.random.choice(env.num_actions())

    # Apply action to the environment and step it forward
    next_state, observation, reward, done, info = env.step(action)

    # Update visualization
    env.update_plot()

print("Done!")    

# Enter interactive 3D viewer
env.show()
```

This can be run via

	python example_usage.py --config configs/config_hard.yaml

## Implementing a Planner
	
To add your own decision-making logic, you can implement a custom **planner** class. A planner is responsible for selecting the next action based on the current **belief state** (i.e., a distribution over possible drone states) and a given planning time budget per step. Every planner must implement the following method:

```
def plan(self, belief_state, planning_time_per_step: float) -> int:
    """
    Select the next action to execute given the current belief.

    Args:
        belief_state: An instance of BeliefState representing the current belief over states.
        planning_time_per_step: The time budget (in seconds) available for planning before
                                the next action must be returned.

    Returns:
        An integer-encoded action to be executed in the environment.
    """
```

The returned integer corresponds to the action across to be exectued in the environment by the main planning loop.
**Important notes**
- The planner **must not** call `MultiDroneEnvironment.step()` or `MultiDroneEnvironment.reset()`. These functions are managed externally by the main simulation loop, which handles the interaction between the planner, belief state, and environment.
- Instead, planners should use `MultiDroneEnvironment.simulate(state, action)`, which performs a single transition **without modifying** the environment’s internal state. This function is essential for sampling-based online planners.

**Example:**
A minimal example of a valid planner is provided in `planners/dummy_planner.py`

```
import numpy as np
from multi_drone_environment import MultiDroneEnvironment

class DummyPlanner:
    def __init__(self, env: MultiDroneEnvironment, a_param: float = 1.0, b_param: int = 1.0):
        self._env = env
        self._a_param = a_param
        self._b_param = b_param
        self._num_actions = env.num_actions() # Number of actions

    def plan(self, belief_state, planning_time_per_step: float) -> int:
        # This planner does not do anything useful and always returns
        # the integer 0, corresponding to the first available action.
        return 0
```

This example serves as a starting point. You can replace the simple `return 0` logic with any planning strategy you prefer—as long as the planner follows the interface described above.

## Using a Planner Together with the MultiDrone Simulator

To solve the MultiDrone POMDP problem using your planner, you can use a simple planning loop (given in `run_planner.py`): 
```
import argparse
from multi_drone_environment import MultiDroneEnvironment
from models.multi_drone_model import (
    MultiDroneTransitionModel,    
    MultiDroneObservationModel,
    MultiDroneInitialBelief,    
    MultiDroneTask
)
from belief_state import BeliefState

# Replace this with your own online planner
from planners.dummy_planner import DummyPlanner

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="Path to the yaml configuration file")
args = parser.parse_args()

def run(env, planner, planning_time_per_step=1.0):
    # Set the simulator to the initial state
    current_state = env.reset() # Set the simulator to the initial state

    belief_state = BeliefState(env) # Initialize a belief state

    num_steps = 0
    total_discounted_reward = 0.0
    history = []

    while True:
        # Use MCTS to plan an action from the current state
        action = planner.plan(belief_state, planning_time_per_step=planning_time_per_step)

        # Apply the action to the environment
        next_state, observation, reward, done, info = env.step(action) 

        print(f"next state: {next_state}, action {action}, observation: {observation}, reward: {reward}, done: {done}")       

        # Accumulate discounted reward
        total_discounted_reward += (env.get_config().discount_factor ** num_steps) * reward

        # Log trajectory
        history.append((current_state, action, observation, reward, next_state, done, info))

        # Update the belief with the action exectued and the observation perceived
        updated_belief = belief_state.update(action, observation, 1000) 

        # Update visualization
        if belief_state.belief_particles is not None:
            # Plot 10 belief particles
            env.update_plot(belief_state.belief_particles[:10])
        else:
            env.update_plot()

        # Move forward
        current_state = next_state
        num_steps += 1        

        if updated_belief is False:
            print("Couldn't update belief")
            break

        # Quit if we reached a terminal state or reached a maximum number of steps
        if done or num_steps >= env.get_config().max_num_steps:
            break

    return total_discounted_reward, history, num_steps

# Instantiate the POMDP model components
transition_model = MultiDroneTransitionModel()
observation_model = MultiDroneObservationModel()
initial_belief = MultiDroneInitialBelief()
task = MultiDroneTask()

# Instantiate the MultiDroneEnvironment with the POMDP models components
env = MultiDroneEnvironment(
    args.config,
    transition_model,
    observation_model,    
    initial_belief,
    task,
)

# Instantiate the planner
planner = DummyPlanner(env)

# Run the planning loop
total_discounted_reward, history, num_steps = run(env, planner, planning_time_per_step=2.0)
print(f"success: {history[-1][6]['all_reached']}, Total discounted reward: {total_discounted_reward}, num_steps: {num_steps}")
env.show()
```

The line `from planners.dummy_planner import DummyPlanner` should be replaced with an import of your own planner. This planning loop handles updates of belief states automatically, via the `BeliefState` class. This class implements a particle representation of beliefs that are updated via a Sequential Importance Resampling particle filter. Your planner has access to the particles that represent the current belief via the `BeliefState.belief_particles` attribute.

## Implementing Custom POMDP Components
You can extend the simulator by **plugging in your own POMDP components**. Each component is a small Python class that implements a tiny interface, defined in `models/pomdp_interface.py`:

- InitialBelief
  -   `InitialBelief.sample(env, num_samples) -> np.ndarray`
    
- TransitionModel
  -   `TransitionModel.step(env, state, action) -> (next_state, info)`    
  -   `TransitionModel.num_actions(num_drones) -> int`
    
- ObservationModel
  -   `ObservationModel.observe(env, state) -> int`    
  -   `ObservationModel.likelihood(env, observation, state) -> float`
    
- TaskModel
  -   `Task.reward(env, prev_state, action, next_state, info) -> float`    
  -   `Task.done(env, prev_state, action, next_state, info) -> bool`

**Conventions (important!)**
States are NumPy arrays of shape `(N, 3)`, while actions and observations are represented as integers. 

**Example: Minimal Custom TransitionModel:**
Below is the a simple transition model compatible with the `MultiDroneEnvironment`.  
It moves all drones by the same unit step along +X, +Y, or +Z, with no noise, collision or boundary checks.

```
class MyTransitionModel(TransitionModel):
    """
    Simplest possible transition model for the MultiDroneEnvironment.

    Each action moves *all drones* by the same unit step along +X, +Y, or +Z.
    No process noise, no per-drone decoding, no collision or boundary logic.
    Just pure deterministic motion — perfect for testing and tutorials.
    """

    def __init__(self):
        # Define three basic motion directions
        self.action_vectors = np.array([
            [1.0, 0.0, 0.0],  # move along +X
            [0.0, 1.0, 0.0],  # move along +Y
            [0.0, 0.0, 1.0],  # move along +Z
        ], dtype=np.float32)

    def num_actions(self, num_drones: int) -> int:
        # Three discrete actions total
        return len(self.action_vectors)

    def step(self, env, state: np.ndarray, action_int: int):
        """
        Apply the same deterministic motion to all drones.

        Args:
            env: MultiDroneEnvironment
            state: (N, 3) world coordinates
            action_int: integer index in [0, 2]
        Returns:
            next_state: (N, 3)
            info: simple dict
        """
        assert 0 <= action_int < len(self.action_vectors)
        direction = self.action_vectors[action_int]

        # Move each drone along the chosen axis
        next_state = state + direction        
        return next_state, {}
```

To use a custom model, instantiate it, and pass it to the constructor of `MultiDroneEnvironment`:

```
my_transition_model = MyTransitionModel()
env = MultiDroneEnvironment(
	...,
	my_transition_model,
	...
)
```





