import numpy as np

class BeliefState:
    def __init__(self, env):
        self._env = env
        self.belief_particles = None # We will sample & update the initial belief

    def sample(self, num_samples: int) -> np.ndarray:
        """Sample states from the current belief or from the initial belief if uninitialized."""
        if self.belief_particles is None:
            return self._env.sample_from_initial_belief(num_samples=num_samples)
        else:
            indices = np.random.randint(0, len(self.belief_particles), size=num_samples)
            return self.belief_particles[indices]

    def update(self, action: int, observation: int, num_particles: int) -> bool:
            """
            Perform one belief update using a simple Sequential Importance Resampling (SIR) particle filter.
            
            Steps:
              1. Propagate each sampled state forward using the transition model (via env.simulate()).
              2. Compute observation likelihood weights P(o | s').
              3. Normalize weights and resample according to them to form the new belief.
            
            Returns:
                True if the belief update succeeded (non-zero total weight), False otherwise.
            """
            next_states = []
            weights = []

            attempts = 0
            max_attempts = num_particles * 50
            while len(next_states) < num_particles and attempts < max_attempts:
                attempts += 1
                if self.belief_particles is None:
                    state = self._env.sample_from_initial_belief(num_samples=1)[0]
                else:
                    idx = np.random.randint(0, len(self.belief_particles))
                    state = self.belief_particles[idx]

                next_state, _, _, done, _ = self._env.simulate(state, action)
                if done:
                    continue

                w = self._env.likelihood(observation, next_state)
                next_states.append(next_state)
                weights.append(w)

            # Abort if no valid particles
            if len(next_states) == 0:
                return False

            weights = np.asarray(weights, dtype=np.float64)
            w_sum = weights.sum()

            # Abort weights are too small
            if w_sum <= 0.0:
                return False
            probs = weights / w_sum

            # Resample according to weights
            indices = np.random.choice(len(next_states), size=num_particles, replace=True, p=probs)
            self.belief_particles = np.stack([next_states[i] for i in indices], axis=0).astype(np.float32)
            return True
