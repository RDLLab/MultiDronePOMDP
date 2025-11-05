import argparse
from multi_drone_environment import MultiDroneEnvironment
from models.multi_drone_model import (
    MultiDroneTransitionModel,    
    MultiDroneObservationModel,
    MultiDroneInitialBelief,
    MultiDroneMixtureInitialBelief,
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