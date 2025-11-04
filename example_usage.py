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