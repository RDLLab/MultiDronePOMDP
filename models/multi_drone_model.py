from typing import Tuple, Dict, Any
from scipy.ndimage import distance_transform_edt
import numpy as np
from .pomdp_interface import InitialBelief, TransitionModel, ObservationModel, Task

def _make_action_vectors() -> np.ndarray:
    # 26 directions in 3D (no zero move)
    vecs = np.array(
        [[dx, dy, dz]
         for dx in (-1, 0, 1)
         for dy in (-1, 0, 1)
         for dz in (-1, 0, 1)
         if not (dx == 0 and dy == 0 and dz == 0)],
        dtype=np.float32
    )
    return vecs

# ---------- Transition model ----------
class MultiDroneTransitionModel(TransitionModel):
    """
    Transition model for multiple controlled drones moving in a 3D environment.

    This implements a simple kinematic step:
      - Decode the joint action into per-drone motion directions.
      - Normalize directions to unit vectors (no motion if zero).
      - Apply one step of motion plus Gaussian transition noise.
      - Keep drones that are already at goal fixed (no further motion).
      - Compute bookkeeping info: out-of-bounds, obstacle hits, same-cell collisions,
        and goal reach flags.
    """
    def __init__(self):
        # Make raw directional action vectors
        self._action_vectors = _make_action_vectors()

    def num_actions(self, num_drones: int) -> int:
        """
        Return the total number of possible *joint actions* for a given number of drones.

        Each drone selects one of K discrete motion directions (K = len(self._action_vectors)).
        The joint action space is the Cartesian product of all per-drone actions, giving:

            |A_joint| = K ** num_drones

        Args:
            num_drones (int): Number of independently controlled drones (N).

        Returns:
            int: Total number of unique joint actions (K ** N).
        """
        return (self._action_vectors.shape[0]) ** num_drones


    def step(self, env, state: np.ndarray, action_int: int) -> Tuple[np.ndarray, Dict]:
        assert state.shape == (env.cfg.num_controlled_drones, 3), f"state must be (N,3), got {state.shape}"

        grid_size_x, grid_size_y, grid_size_z = env.cfg.environment_size
        transition_noise_std: float = float(env.cfg.transition_noise_std)
        goal_radius_m: float = float(env.cfg.goal_radius)

        previous_positions_world = state.astype(np.float32, copy=False)        # (N, 3)
        goal_centers_world = env.goal_centers                                # (N, 3)

        # Drones already at goal remain fixed during this step.
        at_goal_prev = (np.linalg.norm(previous_positions_world - goal_centers_world, axis=1) <= goal_radius_m)

        # Decode joint action into per-drone directions (N, 3) and normalize to unit length.
        motion_directions = self._decode_action(env, action_int)                     # (N, 3)
        direction_norms = np.linalg.norm(motion_directions, axis=1, keepdims=True)
        unit_directions = np.where(direction_norms > 0.0, motion_directions / direction_norms, motion_directions)

        # Apply process noise
        noise_world = env.rng.normal(0.0, transition_noise_std, size=(env.cfg.num_controlled_drones, 3)).astype(np.float32)

        # Compute next drone positions.
        next_state_world = previous_positions_world + unit_directions + noise_world

        # Freeze drones already at goal (no further motion).
        if np.any(at_goal_prev):
            next_state_world = next_state_world.copy()
            next_state_world[at_goal_prev] = previous_positions_world[at_goal_prev]

        next_state_world = next_state_world.astype(np.float32, copy=False)

        # Convert to integer grid cells by rounding to nearest cell center.
        prop_cells = np.floor(next_state_world + 0.5).astype(np.int32)

        # Out-of-bounds mask for proposed cells.
        out_of_bounds = (
            (prop_cells[:, 0] < 0) | (prop_cells[:, 0] >= grid_size_x) |
            (prop_cells[:, 1] < 0) | (prop_cells[:, 1] >= grid_size_y) |
            (prop_cells[:, 2] < 0) | (prop_cells[:, 2] >= grid_size_z)
        )
        in_bounds = ~out_of_bounds

        # Obstacle hits (only check in-bounds proposals).
        obstacle_hits = np.zeros(env.cfg.num_controlled_drones, dtype=bool)
        if np.any(in_bounds):
            cx, cy, cz = prop_cells[in_bounds, 0], prop_cells[in_bounds, 1], prop_cells[in_bounds, 2]
            obstacle_hits[in_bounds] = env.obstacles[cx, cy, cz]

        num_out_of_bounds = int(out_of_bounds.sum())
        num_obstacle_collisions = int(obstacle_hits.sum())

        # Same-cell collisions among in-bounds proposals.
        num_drone_collisions = 0
        if np.any(in_bounds):
            cells_in_bounds = prop_cells[in_bounds]
            _, counts = np.unique(cells_in_bounds, axis=0, return_counts=True)
            num_drone_collisions = int(counts[counts > 1].sum())

        # Goal reach flags at the next state.
        at_goal_next = (np.linalg.norm(next_state_world - goal_centers_world, axis=1) <= goal_radius_m)
        just_reached = (~at_goal_prev) & at_goal_next

        # Check if all drones reached their goals
        all_reached = bool(at_goal_next.all())

        info = dict(
            num_oob=num_out_of_bounds,
            num_obstacle_collisions=num_obstacle_collisions,
            num_drone_collisions=int(num_drone_collisions),
            at_goal_prev=at_goal_prev,
            at_goal_next=at_goal_next,
            just_reached=just_reached,
            all_reached=all_reached,
            prop_cells=prop_cells,  # integer (x, y, z) cells for each drone at next state
        )
        return next_state_world, info

    def _decode_action(self, env, action_int: int) -> np.ndarray:
        """Return per-drone (N,3) float32 directions"""
        N = env.cfg.num_controlled_drones
        num_per_vehicle = self._action_vectors.shape[0]
        idx = np.empty(N, dtype=np.int32)
        x = int(action_int)
        for i in range(N):
            idx[i] = x % num_per_vehicle
            x //= num_per_vehicle
        return self._action_vectors[idx].astype(np.float32, copy=False)

# ---------- Observation model ----------
class MultiDroneObservationModel(ObservationModel):
    """
    Centralized observation model with two channels:
      1) Localize-ID: if a drone is on a cell that has an integer ID (>=0),
         that per-drone symbol is the ID's index in the sorted list of IDs.
      2) Distance: if no ID is present,
         we take the Euclidean distance-to-obstacle (EDT) at the drone's cell,
         bit-cast it to uint32, and use its index in the sorted set of unique bits.

    The joint observation is an integer in base-K with K = (#ID-symbols + #distance-symbols).
    Out-of-bounds or obstacle cells yield a single terminal observation (-1).
    """
    def __init__(self):
        self._obs_lut_ready = False

    # ----------------------------- LUT build ----------------------------- #
    def _ensure_obs_lut(self, env) -> None:
        if self._obs_lut_ready:
            return
        if not hasattr(env, "obstacles") or not hasattr(env, "localize_ids"):
            raise ValueError("env.obstacles and env.localize_ids are required.")

        # Distance field (free=True → 1 for EDT input)
        free_space_mask_uint8 = (~env.obstacles).astype(np.uint8)
        self.dist_field = distance_transform_edt(free_space_mask_uint8)

        # Sorted unique Localize-IDs (>=0)
        localize_ids_flat = env.localize_ids[env.localize_ids >= 0].astype(np.int32)
        self.localize_ids_sorted = (
            np.unique(localize_ids_flat) if localize_ids_flat.size > 0 else np.empty((0,), dtype=np.int32)
        )
        num_localize_ids = int(self.localize_ids_sorted.size)

        # Distance symbols (if used)
        if env.cfg.use_distance_sensor:
            if not np.any(env.obstacles):
                # Single distance symbol (all zeros)
                self.dist_field = self.dist_field.astype(np.float32, copy=False)
                self.distance_bits_sorted = np.array([0], dtype=np.uint32)
            else:
                edt_bits = self.dist_field.astype(np.float32, copy=False).view(np.uint32)
                unique_bits = np.unique(edt_bits)
                self.distance_bits_sorted = unique_bits if unique_bits.size > 0 else np.array([0], dtype=np.uint32)
            num_distance_symbols = int(self.distance_bits_sorted.size)
        else:
            self.distance_bits_sorted = np.array([], dtype=np.uint32)
            num_distance_symbols = 0

        # ---------------- Symbol layout & base K ----------------
        # IDs-only mode: reserve per-drone symbol 0 for "no ID"
        #   per-drone symbols: 0 (no-ID), 1..num_localize_ids (IDs)
        #   K = max(1 + num_localize_ids, 1)  -> if no IDs exist anywhere, K=1 (only symbol 0)
        if not env.cfg.use_distance_sensor:
            self.id_offset = 1  # IDs start at symbol 1
            self.distance_symbol_offset = None
            self.obs_base_K = max(1 + num_localize_ids, 1)
        else:
            # Distance-enabled mode (unchanged layout):
            #   IDs: 0 .. num_localize_ids-1
            #   Dist: num_localize_ids .. num_localize_ids + num_distance_symbols - 1
            self.id_offset = 0
            self.distance_symbol_offset = num_localize_ids
            self.obs_base_K = num_localize_ids + num_distance_symbols
            if self.obs_base_K < 1:
                # no IDs and no distance symbols (shouldn't happen, but guard anyway)
                self.obs_base_K = 1  # yields only code 0 as non-terminal

        self._obs_lut_ready = True

    # ---------------------- Deterministic mapping h(s) ------------------- #
    def _deterministic_observation(self, env, state_world_m: np.ndarray) -> int | None:
        """
        Returns the joint base-K code for non-terminal states.
        Returns None only if terminal (OOB, obstacle, or ALL drones at goal).
        """
        self._ensure_obs_lut(env)

        grid_size_x, grid_size_y, grid_size_z = env.cfg.environment_size
        num_drones = int(env.cfg.num_controlled_drones)

        # ---- NEW: terminal if all drones are currently inside their goal areas ----
        if hasattr(env, "goal_centers") and hasattr(env.cfg, "goal_radius"):
            goal_centers_world = env.goal_centers  # shape (N,3)
            goal_radius_m = float(env.cfg.goal_radius)
            # Guard shapes lightly
            if goal_centers_world.shape == (num_drones, 3):
                at_goal_now = (np.linalg.norm(state_world_m - goal_centers_world, axis=1) <= goal_radius_m)
                if bool(np.all(at_goal_now)):
                    return None
        # ---------------------------------------------------------------------------

        # nearest GRID cell (voxel_size == 1)
        cell_indices = np.floor(state_world_m + 0.5).astype(np.int32)
        if cell_indices.shape != (num_drones, 3):
            raise ValueError(f"`state_world_m` must be shape ({num_drones}, 3).")

        # Terminal checks: OOB or obstacle → None
        oob = (
            (cell_indices[:, 0] < 0) | (cell_indices[:, 0] >= grid_size_x) |
            (cell_indices[:, 1] < 0) | (cell_indices[:, 1] >= grid_size_y) |
            (cell_indices[:, 2] < 0) | (cell_indices[:, 2] >= grid_size_z)
        )
        if np.any(oob):
            return None
        cx, cy, cz = cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2]
        if np.any(env.obstacles[cx, cy, cz]):
            return None

        # Per-drone symbol assembly
        per_drone_symbols = np.empty(num_drones, dtype=np.int32)
        localize_ids_here = env.localize_ids[cx, cy, cz]  # -1 where none
        has_id = (localize_ids_here >= 0)

        if env.cfg.use_distance_sensor:
            # -------- Distance-enabled mode (as before) --------
            # IDs band (0..S_ID-1)
            if np.any(has_id):
                if self.localize_ids_sorted.size == 0:
                    # shouldn't happen, but if LUT empty, treat as non-ID (fallback to distance)
                    has_id = np.zeros_like(has_id, dtype=bool)
                else:
                    id_idx = np.searchsorted(self.localize_ids_sorted, localize_ids_here[has_id])
                    if not np.all(self.localize_ids_sorted[id_idx] == localize_ids_here[has_id]):
                        # stale LUT or map changed -> treat as non-ID (fallback to distance)
                        mask = has_id.copy()
                        mask[has_id] = False
                        has_id = mask
                    else:
                        per_drone_symbols[has_id] = (self.id_offset + id_idx.astype(np.int32))

            # Distance band (S_ID .. S_ID+num_dist-1)
            no_id = ~has_id
            if np.any(no_id):
                d = self.dist_field[cx[no_id], cy[no_id], cz[no_id]].astype(np.float32, copy=False)
                d_bits = d.view(np.uint32)
                d_idx = np.searchsorted(self.distance_bits_sorted, d_bits)
                # guard against mismatch
                if not np.all(self.distance_bits_sorted[d_idx] == d_bits):
                    # if mismatch (shouldn't happen), just clamp to first distance symbol
                    d_idx = np.zeros_like(d_idx, dtype=np.int32)
                per_drone_symbols[no_id] = (self.distance_symbol_offset + d_idx.astype(np.int32))

        else:
            # -------- IDs-only mode with default=0 (non-terminal) --------
            # Default per-drone symbol = 0 when there is no ID at the cell.
            per_drone_symbols.fill(0)
            if np.any(has_id) and self.localize_ids_sorted.size > 0:
                id_idx = np.searchsorted(self.localize_ids_sorted, localize_ids_here[has_id])
                # only accept exact matches; otherwise keep default 0
                exact = (self.localize_ids_sorted[id_idx] == localize_ids_here[has_id])
                per_drone_symbols[has_id] = 0  # start from default
                if np.any(exact):
                    # IDs start at symbol 1 (reserve 0 for default)
                    per_drone_symbols[has_id.nonzero()[0][exact]] = (self.id_offset + id_idx[exact].astype(np.int32))

        # Base-K pack
        joint_code = 0
        K = int(self.obs_base_K)
        for symbol in per_drone_symbols.astype(np.int64):
            joint_code = joint_code * K + int(symbol)
        return int(joint_code)

    # ------------------------------ Observe ------------------------------ #
    def observe(self, env, state: np.ndarray) -> int:
        """
        Sample an observation:
          - Terminal: -1
          - Non-terminal: h(s) with prob p, otherwise a different code uniformly in [0, K^N - 1] \ {h(s)}.
        """
        p = float(np.clip(env.cfg.obs_correct_prob, 0.0, 1.0))
        det_code = self._deterministic_observation(env, state)
        if det_code is None:
            return -1  # only terminal emits -1

        num_drones = int(env.cfg.num_controlled_drones)
        joint_space = self.obs_base_K ** num_drones

        if joint_space <= 1 or p >= 1.0:
            return det_code

        rng = getattr(env, "rng", np.random.default_rng())
        if rng.random() < p:
            return det_code

        draw = int(rng.integers(0, joint_space - 1))
        return draw if draw < det_code else draw + 1

    # ----------------------------- Likelihood ---------------------------- #
    def likelihood(self, env, observation: int, state_world_m: np.ndarray) -> float:
        """
        Exact P(o|s) for the same joint-level noise model.
        """
        p = float(np.clip(env.cfg.obs_correct_prob, 0.0, 1.0))
        det_code = self._deterministic_observation(env, state_world_m)
        obs_code = int(observation)

        if det_code is None:
            return 1.0 if obs_code == -1 else 0.0

        num_drones = int(env.cfg.num_controlled_drones)
        joint_space = self.obs_base_K ** num_drones

        if joint_space <= 1:
            return 1.0 if obs_code == det_code else 0.0
        if obs_code == -1:
            return 0.0
        if obs_code == det_code:
            return p
        return (1.0 - p) / float(joint_space - 1)



# ---------- Initial belief ----------
class MultiDroneInitialBelief(InitialBelief):
    """
    Initial belief sampler: uniform box noise around provided start positions.

    - Uses `env.cfg.initial_drone_positions[:num_controlled_drones]` for the nominal initial drone positions.
    - Adds uniform noise within a cube of side-length `start_noise_range`.
    - Clips to environment bounds.
    """

    def sample(self, env, num_samples: int) -> np.ndarray:
        # Nominal starts for the controlled drones
        starts_world = np.array(
            env.cfg.initial_drone_positions[:env.cfg.num_controlled_drones], dtype=np.float32
        )  # (N, 3)

        noise_range_m: float = float(getattr(env.cfg, "start_noise_range", 1.0))
        noise = env.rng.uniform(
            low=-0.5 * noise_range_m, high=0.5 * noise_range_m, size=(num_samples, *starts_world.shape)
        ).astype(np.float32)

        samples_world = starts_world[None, :, :] + noise

        # Clip to environment bounds in
        if hasattr(env.cfg, "environment_size"):
            grid_size_x, grid_size_y, grid_size_z = env.cfg.environment_size            
            world_extent_xyz = np.array([grid_size_x, grid_size_y, grid_size_z], dtype=np.float32)
            samples_world = np.clip(samples_world, 0.0, world_extent_xyz - 1e-3)

        return samples_world.astype(np.float32, copy=False)

class MultiDroneMixtureInitialBelief(InitialBelief):
    """
    Initial belief sampler for N drones with per-drone bimodal (2-component) uniform mixtures.

    Config expectations (env.cfg):
      - initial_drone_positions: List[List[float]] of length N
          Each entry is a 6-vector: [x1, y1, z1, x2, y2, z2]
          where (x1,y1,z1) is the center of component 0 and (x2,y2,z2) is the center of component 1
          for that specific drone.
      - initial_drone_uncertainty: float or [3]
          Half-width(s) of the uniform cube noise added around the chosen component center (meters).
          If a float, the same half-width is used for x/y/z. If [3], it’s per-axis.
      - environment_size: (X, Y, Z)
          Used to clip samples to [0, size - 1e-3].

    Behavior:
      - For each sample and each drone, we choose component 0 or 1 with equal probability (50/50).
      - We add uniform cube noise in [-half_width, +half_width] per axis.
      - We clip to environment bounds.
      - Returns an array of shape (num_samples, N, 3) in world coordinates (floats).
    """

    def sample(self, env, num_samples: int) -> np.ndarray:
        # ----- Parse and validate per-drone 2-component centers -----
        raw_positions = np.asarray(env.cfg.initial_drone_positions, dtype=np.float32)  # expected (N, 6)
        assert raw_positions.ndim == 2 and raw_positions.shape[1] == 6, \
            f"`initial_drone_positions` must be (N, 6); got {raw_positions.shape}"

        num_drones = raw_positions.shape[0]
        # Reshape to (N, 2, 3): per-drone [component, xyz]
        centers_per_drone = raw_positions.reshape(num_drones, 2, 3)

        # ----- Noise half-width handling (float -> isotropic; [3] -> per-axis) -----
        noise_half_width = np.asarray(getattr(env.cfg, "initial_drone_uncertainty", 0.5), dtype=np.float32)
        if noise_half_width.ndim == 0:
            noise_half_width = np.full((3,), float(noise_half_width), dtype=np.float32)
        assert noise_half_width.shape == (3,), \
            "`initial_drone_uncertainty` must be a float or a length-3 iterable"

        # ----- Equal-probability component choice per (sample, drone) -----
        # comp_choices[s, d] ∈ {0, 1}
        comp_choices = env.rng.integers(0, 2, size=(num_samples, num_drones), dtype=np.int32)

        # Gather chosen centers into (num_samples, N, 3)
        centers_expanded = centers_per_drone[None, :, :, :]         # (1, N, 2, 3)
        comp_expanded    = comp_choices[:, :, None, None]           # (S, N, 1, 1)
        base_positions   = np.take_along_axis(centers_expanded, comp_expanded, axis=2).squeeze(2)  # (S, N, 3)

        # ----- Uniform cube noise per sample/drone/axis -----
        low  = (-noise_half_width)[None, None, :]  # (1,1,3)
        high = ( noise_half_width)[None, None, :]  # (1,1,3)
        noise = env.rng.uniform(low=low, high=high, size=(num_samples, num_drones, 3)).astype(np.float32)

        samples_world = base_positions + noise  # (S, N, 3)

        # ----- Clip to environment bounds -----
        world_extent_xyz = np.asarray(env.cfg.environment_size, dtype=np.float32)  # (3,)
        samples_world = np.clip(samples_world, 0.0, world_extent_xyz - 1e-3)

        return samples_world.astype(np.float32, copy=False)

# ---------- Task (reward + termination) ----------
class MultiDroneTask(Task):
    """
    Goal-reaching task with penalties for out-of-bounds, obstacle hits, and drone-drone collisions.

    Reward
    ------
    reward = step_cost * num_drones
           + collision_penalty * (num_out-of-bounds + num_obstacle_collisions + num_drone_collisions)
           + goal_reward * (# drones that newly reached the goal this step)

    Termination
    -----------
    Episode ends when:
      - any drone is OOB / in obstacle / collides with another drone (same cell), or
      - all drones are within the goal radius.
    """

    # ---- internal: compute the standard step-info fields once ----
    def _compute_step_info(self, env, prev_state: np.ndarray, next_state: np.ndarray) -> Dict[str, Any]:
        # Round to nearest cell (voxel_size == 1.0 by design in this env)
        prop_cells = np.floor(next_state + 0.5).astype(np.int32)

        gx, gy, gz = env.cfg.environment_size
        x, y, z = prop_cells[:, 0], prop_cells[:, 1], prop_cells[:, 2]

        # Out-of-bounds
        oob = (x < 0) | (x >= gx) | (y < 0) | (y >= gy) | (z < 0) | (z >= gz)

        # Obstacles (in-bounds only)
        obstacle_hits = np.zeros_like(oob)
        inb = ~oob
        if np.any(inb):
            obstacle_hits[inb] = env.obstacles[x[inb], y[inb], z[inb]]

        # Same-cell collisions among valid proposals
        num_drone_collisions = 0
        valid = inb & (~obstacle_hits)
        if np.any(valid):
            cells_valid = prop_cells[valid]
            _, counts = np.unique(cells_valid, axis=0, return_counts=True)
            num_drone_collisions = int(counts[counts > 1].sum())

        # Goal flags (world distance; voxel_size == 1.0)
        goal_centers = env.goal_centers
        r = float(env.cfg.goal_radius)
        at_goal_prev = (np.linalg.norm(prev_state - goal_centers, axis=1) <= r)
        at_goal_next = (np.linalg.norm(next_state - goal_centers, axis=1) <= r)
        just_reached = (~at_goal_prev) & at_goal_next
        all_reached = bool(at_goal_next.all())

        return dict(
            num_oob=int(oob.sum()),
            num_obstacle_collisions=int(obstacle_hits.sum()),
            num_drone_collisions=int(num_drone_collisions),
            at_goal_prev=at_goal_prev,
            at_goal_next=at_goal_next,
            just_reached=just_reached,
            all_reached=all_reached,
            prop_cells=prop_cells,
        )

    # ---- internal: fill missing keys only (do not overwrite provided ones) ----
    def _fill_missing_info(
        self, env, prev_state: np.ndarray, next_state: np.ndarray, info: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        if info is None:
            info = {}
        # If any required key is missing, compute all, then fill only the gaps.
        required = (
            "num_oob", "num_obstacle_collisions", "num_drone_collisions",
            "at_goal_prev", "at_goal_next", "just_reached", "all_reached", "prop_cells"
        )
        if not all(k in info for k in required):
            computed = self._compute_step_info(env, prev_state, next_state)
            for k, v in computed.items():
                info.setdefault(k, v)
        return info


    def reward(
        self,
        env,
        prev_state: np.ndarray,
        action_int: int,
        next_state: np.ndarray,
        info: Dict[str, Any] | None = None
    ) -> float:
        # Ensure info has all expected fields (transition may have omitted them)
        info = self._fill_missing_info(env, prev_state, next_state, info)

        step_cost = float(env.cfg.step_cost)
        collision_penalty = float(env.cfg.collision_penalty)
        goal_reward = float(env.cfg.goal_reward)

        reward_value = step_cost * env.cfg.num_controlled_drones
        reward_value += collision_penalty * (
            info["num_oob"] + info["num_obstacle_collisions"] + info["num_drone_collisions"]
        )
        reward_value += goal_reward * int(np.asarray(info["just_reached"]).sum())
        return float(reward_value)

    def done(
        self,
        env,
        prev_state: np.ndarray,
        action_int: int,
        next_state: np.ndarray,
        info: Dict[str, Any] | None = None
    ) -> bool:
        # Ensure info has all expected fields
        info = self._fill_missing_info(env, prev_state, next_state, info)

        return bool(
            (info["num_oob"] > 0) or
            (info["num_obstacle_collisions"] > 0) or
            (info["num_drone_collisions"] > 0) or
            info["all_reached"]
        )
