import argparse
from multi_drone_pomdp import MultiDroneUnc

# Replace this with your own online planner
from po_uct import POUCT

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
        action = planner.plan(planning_budget=planning_time_per_step)

        # Apply the action to the environment
        next_state, observation, reward, done, info = env.step(action) 

        print(f"next state: {next_state}, action {action}, observation: {observation}")       

        # Accumulate discounted reward
        total_discounted_reward += (env.get_config().discount_factor ** num_steps) * reward

        # Log trajectory
        history.append((current_state, action, observation, reward, next_state, done, info))

        updated_belief = planner.update_belief(action, observation, 1000)

        belief_particles = planner._current_belief
        print("belief_particles", belief_particles.shape)

        env.update_plot(belief_particles[:10])

        # Move forward
        current_state = next_state
        num_steps += 1

        if updated_belief is False:
            print("Couldn't update belief")
            break

        if done or num_steps >= env.get_config().max_num_steps:
            break

    return total_discounted_reward, history, num_steps

# Instantiate the environment with the given config
env = MultiDroneUnc(args.config)

# Instantiate the planner
#planner = MCTS(env, c_param=20.0)
planner = POUCT(env, c_param=10.0)

# Run the planning loop
total_discounted_reward, history, num_steps = run(env, planner, planning_time_per_step=2.0)
print(f"success: {history[-1][6]['all_reached']}, Total discounted reward: {total_discounted_reward}, num_steps: {num_steps}")
#env.reset()
env.show()