# MultiDroneUnc Simulator

The `MultiDroneUnc` class provides a simple, self-contained MDP formulation for multiple drones operating in a bounded 3D grid with static obstacles and goal regions, while being subject to uncertainties in the outcome of actions. The drones can take actions to move to one of their adjacent cells: If drones are not allowed to change their altitude, each drone can move to one of its `8` neighbor cells in the XY-plane (up, down, left, right, and diagonals). Otherwise, each drone can move to one of its `26` neighbors in 3D space (all adjacent cells except staying in place).


## Requirements
The following Python libraries are required:
 - [NumPy](https://numpy.org/)
 - [PyYAML](https://pypi.org/project/PyYAML/)
 - [SciPy](https://scipy.org/)
 - [Vedo](https://pypi.org/project/vedo/)

## Usage
### Defining environments
To use the ```MultiDroneUnc``` class in ```multi_drone.py``` we have to define a 3D grid environment and set the MDP model parameters.  This is defined in a YAML file, for example ```environment.yaml```. It consists of the following parameters:

- `grid_size`: The size of the 3D grid, i.e., the number of cells in each XYZ dimension
- `start_positions`: The start position of each drones
- `goal_positions`: The goal position of each drone 
- `obstacle_cells`: A list of cells inside the grid that are considered obstacles
- `change_altitude`: A boolean that indicates whether drones are allowed to change their altitudes

The MDP model parameters are the following:

- `step_cost`: A penalty the drones receive at every step
- `collision_penalty`: The penality a drone receives upon colliding with an obstacle cell
- `goal_reward`: The reward a drone receives when reaching its goal
- `max_num_step`: The maximum number of steps before a run terminates
- `alpha`: Action uncertainty parameter (see [Action uncertainty parameter alpha](#action-uncertainty-parameter-alpha)) 

An example of a yaml configuration file is provided in ```example_environment.yaml```:
```
grid_size: [10, 10, 5] # The size of the 3D grid
start_positions: # The start positions of the drones inside the grid
  - [0, 2, 2]
  - [0, 7, 2]
  
goal_positions: # The goal positions for each drone
  - [9, 3, 2]
  - [9, 6, 2]

obstacle_cells: # The list of cells inside the grid that are considered obstacles
  - [5, 7, 2]
  - [5, 3, 2]

change_altitude: False # Only operate in the XY-plane

# MDP model parameters
step_cost: -1.0
collision_penalty: -50.0
goal_reward: 100.0
discount_factor: 0.98
max_num_steps: 100 # The maximum number of steps before the problem terminates
alpha: 0.5 # Action uncertainty parameter
```
Note that the number of start positions must be equal to the number of goal positions. The number of start positions determines the number of drones in the environment.

If `change_altitude` is `False`, each drone can move to one of its `8` neighbor cells in the XY-plane (up, down, left, right, and diagonals). Otherwise, each drone can move to one of its `26` neighbors in 3D space (all adjacent cells except staying in place). An action is represented as an integer in the range `[0, A^N - 1]`, where `A` is the number of per-drone actions (`8` or `26`) and `N` is the number of drones. The action integer is decoded into per-drone moves during simulation.

#### Action representation
Actions are represented as integers in the range `[0, A^N - 1]`, where `A` is the number of per-drone actions (`8` or `26`) and `N` is the number of drones. The action integer is decoded into per-drone moves during simulation. For example, with `N = 2` drones restricted to the XY-plane (`A = 8`), there are `8^2 = 64` possible joint actions. If the encoded integer is `17`, this decodes to Drone 0 selecting action `1` (e.g., East) and Drone 1 selecting action `2` (e.g., North-West), meaning both drones move simultaneously in their respective directions.

#### Action uncertainty parameter alpha  

The parameter `alpha` > 0 controls the uncertainty of drone movements. When a drone selects an intended action, it only succeeds with a probability that depends on the distance from its current cell to the nearest obstacle. The success probability is given by:

`p_succ = d / (d + alpha)`

where `d` is the obstacle distance at the drone’s current location. With probability `1 - p_succ`, the intended move fails and the drone instead executes a different random valid action. A smaller `alpha` makes the motion model closer to deterministic (high success probability even near obstacles), while a larger `alpha` increases stochasticity, reflecting greater motion uncertainty near obstacles.

### Running a planning loop
We provide a Python script template in ```run_planner.py``` that instantiates the ```MultiDroneUc``` environment, a dummy planner (**this must be replaced by your own planner**),  and runs a planning loop:

```
import argparse
from multi_drone import MultiDroneUnc

# Replace this with your own online planner
from dummy_planner import DummyPlanner

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="Path to the yaml configuration file")
args = parser.parse_args()

def run(env, planner, planning_time_per_step=1.0):
    # Set the simulator to the initial state
    current_state = env.reset()
    num_steps = 0
    total_discounted_reward = 0.0
    history = []

    while True:
        # Use MCTS to plan an action from the current state
        action = planner.plan(current_state, planning_time_per_step)

        # Apply the action to the environment
        next_state, reward, done, info = env.step(action)        

        # Accumulate discounted reward
        total_discounted_reward += (env.get_config().discount_factor ** num_steps) * reward

        # Log trajectory
        history.append((current_state, action, reward, next_state, done, info))

        # Move forward
        current_state = next_state
        num_steps += 1

        if done or num_steps >= env.get_config().max_num_steps:
            break

    return total_discounted_reward, history

# Instantiate the environment with the given config
env = MultiDroneUnc(args.config)

# Instantiate the planner
planner = DummyPlanner(env, a_param=1.0, b_param=2)

# Run the planning loop
total_discounted_reward, history = run(env, planner, planning_time_per_step=1.0)
print(f"success: {history[-1][5]['success']}, Total discounted reward: {total_discounted_reward}")
env.show()
```

This can be run via
```
python run_planner.py --config <path to yaml config>
```
e.g.,
```
python run_planner.py --config example_environment.yaml
```

###  Implementing and using your an online planner
To implement your own planner, you must provide a Python class which implements the following functions:
1. ```def __init__(env: MultiDroneUnc)```: This is a constructor that takes an instance of the MultiDroneUnc environment as an argument. You can also specify additional parameters for your planner here, e.g., ```def __init__(env: MultiDroneUnc, a_param: float = 1.0, b_param: int = 2)```.

2. ```def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int```: This function should plan an action for a given state within the given ```planning_time_per_step``` (in seconds), and return an integer that represents the action.

To use your planner within the planning loop template above, replace the line ```from dummy_planner import DummyPlanner``` with the import of your own planner, e.g., ```from my_planner import MyPlanner```. Additionally, replace the line ```planner = DummyPlanner(env, a_param=1.0, b_param=1.0)``` with ```planner = MyPlanner(env, a_param=1.0, b_param=1.0)``` to instantiate your planner.

###  Usage of the MultiEnvUnc class
After implementing the constructor of your planner as above, your planner has access to an instance of the MultiEnvUnc class. This class provides the following functions that your planner can use:

- ```MultiDroneUnc.simulate(state: np.ndarray, action: int)```: This function implements the **generative model** of the underlying MDP. It takes a state, an action, and simulates a next state, reward and a terminal signal for one step. This is returned as a ```Tuple[np.ndarray, float, bool, Dict]```. The last entry of the tuple is a dictionary containing useful information, e.g., if a collision occured during the one-step simulation.
- ```MultiDroneUnc.get_config()```: Returns the configuration data class defined in ```multi_done_config.py```. This provides you with accessible configuration attributes. For instance, ```MultiDroneUnc.get_config().discount_factor``` gives the discount factor of the problem.
- ```MultiDroneUnc.num_actions()```: The number of actions of the underlying MDP. The number of actions is automatically computed from the yaml config file

Your online planner should **only** rely on the above function. Additionally, the class provides functions for resetting the simulator, applying an action, and visualisation. These functions are used by the planning loop.
## Preparing your submission
To prepare your submission, create a copy of the ```run_planner.py``` script in the same folder, and rename it to ```run_UID.py```, where UID is your student ID. Inside this script, modify the import and instantiation of your planner as described in [Implementing and using your an online planner](#implementing-and-using-your-an-online-planner). This script has to be runnable via
```
python run_UID.py --config <environment_yaml_file>
```



