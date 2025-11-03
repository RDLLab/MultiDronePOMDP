import yaml
import numpy as np
from scipy.ndimage import distance_transform_edt
from multi_drone_config import MultiDroneConfig
from typing import Optional, Tuple, Dict
from vedo import Plotter, Box, Sphere, Cylinder, Line, Axes

def _make_action_vectors(change_altitude: bool) -> np.ndarray:
    if change_altitude:
        # 26 directions in 3D (no zero move)
        return np.array(
            [[dx, dy, dz]
             for dx in (-1, 0, 1)
             for dy in (-1, 0, 1)
             for dz in (-1, 0, 1)
             if not (dx == 0 and dy == 0 and dz == 0)],
            dtype=np.int32
        )
    else:
        # 8 directions in XY plane (dz=0, no zero move)
        return np.array(
            [[dx, dy, 0]
             for dx in (-1, 0, 1)
             for dy in (-1, 0, 1)
             if not (dx == 0 and dy == 0)],
            dtype=np.int32
        )

class MultiDroneUnc:    
    def __init__(self, path: str):
        self.cfg = self._load_config(path)

        assert self.cfg.start_positions is not None, "start_positions can't be empty"
        self.N = len(self.cfg.start_positions)       
        self.rng = np.random.default_rng(self.cfg.seed)

        self.positions = np.zeros((self.N, 3), dtype=np.float32)
        self.reached = np.zeros((self.N,), dtype=bool)
        self.t = 0

        self.obstacles = np.zeros(self.cfg.grid_size, dtype=bool)
        self.dist_field = np.zeros(self.cfg.grid_size, dtype=np.float32)
        self.localize_ids = np.full(self.cfg.grid_size, fill_value=-1, dtype=np.int32)

        # vedo visualization members
        self._plotter = None
        self._uuv_visuals = []
        self._goal_mesh = None
        self._action_vectors = _make_action_vectors(self.cfg.change_altitude)
    
    def get_config(self):
        return self.cfg

    @property
    def num_actions(self) -> int:        
        return (self._action_vectors.shape[0]) ** self.N

    # ---------- Core API ----------
    def reset(self) -> np.ndarray:
        X, Y, Z = self.cfg.grid_size

        # Obstacles (config lists cell CENTERS as integer triples)
        self.obstacles.fill(False)
        if self.cfg.obstacle_cells is not None:
            for (x, y, z) in self.cfg.obstacle_cells:
                assert 0 <= x < X and 0 <= y < Y and 0 <= z < Z, \
                    f"Obstacle {(x, y, z)} is outside environment bounds {self.cfg.grid_size}"
                # NOTE: obstacle index == center index
                self.obstacles[x, y, z] = True

        # Localization cells (auto-ID; config lists CENTERS [x,y,z])
        if not hasattr(self, "localize_ids") or self.localize_ids.shape != self.obstacles.shape:
            self.localize_ids = np.full(self.cfg.grid_size, fill_value=-1, dtype=np.int32)
        else:
            self.localize_ids.fill(-1)

        if getattr(self.cfg, "localize_cells", None) is not None:
            for cid, (x, y, z) in enumerate(self.cfg.localize_cells):
                assert 0 <= x < X and 0 <= y < Y and 0 <= z < Z, \
                    f"Localize cell {(x, y, z)} out of bounds {self.cfg.grid_size}"
                self.localize_ids[x, y, z] = int(cid)

        # Distance field (over grid indices/centers)
        free = (~self.obstacles).astype(np.uint8)
        self.dist_field = distance_transform_edt(free).astype(np.float32)

        # Start & goal in WORLD coordinates (no +0.5 shift)
        assert self.cfg.start_positions is not None
        assert self.cfg.goal_positions  is not None
        self.positions     = np.array(self.cfg.start_positions, dtype=np.float32)  # (N,3) world
        self._init_positions = self.positions.copy()

        # Store goal centers in WORLD coords directly
        vs = float(getattr(self.cfg, "voxel_size", 1.0))
        self.goal_centers = np.array(self.cfg.goal_positions, dtype=np.float32) * vs  # world coords

        # Reached flags (world distance to goal centers)
        goal_tol = float(self.cfg.goal_tol)
        self.reached = (np.linalg.norm(self.positions - self.goal_centers, axis=1) <= goal_tol)

        # Time
        self.t = 0

        # Vedo
        self._init_plot()
        return self.positions.copy()

    def sample_from_initial_belief(self, num_samples: int) -> np.ndarray:
        """
        Return a batch of state samples from the (simple) initial belief.
        For now, every sample is identical to the initial positions set in reset().

        Shape: (num_samples, N, 3), dtype float32
        """
        assert hasattr(self, "_init_positions"), \
            "Call reset() before sampling from the initial belief."
        return np.repeat(self._init_positions[None, :, :], num_samples, axis=0).astype(np.float32, copy=False)


    def step(self, action_int: int) -> Tuple[np.ndarray, int, float, bool, Dict]:
        """
        Execute one environment step (stateful):
          - Uses continuous dynamics (with internal RNG)
          - Updates self.positions, self.reached, and self.t
          - Returns a single centralized observation integer
        """
        # Current state
        state = self.positions.astype(np.float32, copy=False)

        # Transition
        next_state, obs, reward, done, info = self.simulate(state, action_int)

        # Update internal state
        self.positions = next_state
        self.reached = info.get("at_goal_next", np.zeros(self.N, dtype=bool))
        self.t += 1

        # Max-step termination
        if self.t >= self.cfg.max_num_steps:
            done = True
            info["max_steps_reached"] = True

        # Observation (returns terminal token if done=True)
        observation = int(self.observe(self.positions))

        # Optional visualization

        return self.positions.copy(), observation, float(reward), bool(done), info


    def simulate(self,
                 state: np.ndarray,
                 action: int) -> Tuple[np.ndarray, int, float, bool, Dict]:
        # Stateless rollout: no writes to env members.
        next_state, info = self._transition_dynamics(state, action)
        obs = self.observe(next_state)
        reward = self._get_reward(state, action, next_state, info)
        done   = self._is_terminal(state, action, next_state, info)
        return next_state, obs, reward, done, info

    def _transition_dynamics(
        self,
        state: np.ndarray,
        action_int: int,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Samples the next state in WORLD coordinates given (state, action) and
        computes all discrete-grid/goal metadata needed downstream.

        Returns:
            next_state: (N,3) world coordinates
            info: {
                num_oob, num_obstacle_collisions, num_vehicle_collisions (ints),
                at_goal_prev (N,), at_goal_next (N,), just_reached (N,), all_reached (bool),
                prop_cells (N,3) int indices
            }
        """
        assert state.shape == (self.N, 3), f"state must be (N,3), got {state.shape}"

        # --- Config shortcuts ---
        X, Y, Z   = self.cfg.grid_size
        vs        = float(getattr(self.cfg, "voxel_size", 1.0))
        step_sz   = float(self.cfg.step_size)
        noise_sd  = float(self.cfg.process_noise_std)
        goal_tol  = float(self.cfg.goal_tol)

        prev_pos = state.astype(np.float32, copy=False)  # (N,3) world
        goal_centers = self.goal_centers                 # (N,3) world

        # --- Previous "at goal" (WORLD) ---
        at_goal_prev = (np.linalg.norm(prev_pos - goal_centers, axis=1) <= goal_tol)

        # --- Decode joint action -> per-drone unit directions ---
        actions = self._decode_action(action_int)                       # (N,)
        dirs    = self._action_vectors[actions].astype(np.float32)      # (N,3)
        norms   = np.linalg.norm(dirs, axis=1, keepdims=True)
        unit    = np.where(norms > 0.0, dirs / norms, dirs)             # keep no-op as zeros

        # --- Noise (uses self.rng deliberately) ---
        noise    = self.rng.normal(0.0, noise_sd, size=(self.N, 3)).astype(np.float32)
        proposal = prev_pos + step_sz * unit + noise                    # (N,3) world

        # Freeze already-at-goal drones
        if np.any(at_goal_prev):
            proposal = proposal.copy()
            proposal[at_goal_prev] = prev_pos[at_goal_prev]

        next_state = proposal.astype(np.float32, copy=False)

        # --- Discrete-grid interactions (nearest-center indexing) ---
        prop_cells = np.floor((next_state / vs) + 0.5).astype(np.int32)

        # OOB and obstacles
        oob = (
            (prop_cells[:, 0] < 0) | (prop_cells[:, 0] >= X) |
            (prop_cells[:, 1] < 0) | (prop_cells[:, 1] >= Y) |
            (prop_cells[:, 2] < 0) | (prop_cells[:, 2] >= Z)
        )
        inb = ~oob

        obstacle_hits = np.zeros(self.N, dtype=bool)
        if np.any(inb):
            cx, cy, cz = prop_cells[inb, 0], prop_cells[inb, 1], prop_cells[inb, 2]
            obstacle_hits[inb] = self.obstacles[cx, cy, cz]

        num_oob = int(oob.sum())
        num_obs = int(obstacle_hits.sum())

        # Vehicle collisions (same-cell occupancy among in-bounds proposals)
        veh_collisions = 0
        if np.any(inb):
            cells_inb = prop_cells[inb]
            _, counts = np.unique(cells_inb, axis=0, return_counts=True)
            veh_collisions = int(counts[counts > 1].sum())

        # --- Goals after move (WORLD) ---
        at_goal_next = (np.linalg.norm(next_state - goal_centers, axis=1) <= goal_tol)
        just_reached = (~at_goal_prev) & at_goal_next
        all_reached  = bool(at_goal_next.all())

        info = dict(
            num_oob=num_oob,
            num_obstacle_collisions=num_obs,
            num_vehicle_collisions=int(veh_collisions),
            at_goal_prev=at_goal_prev,
            at_goal_next=at_goal_next,
            just_reached=just_reached,
            all_reached=all_reached,
            prop_cells=prop_cells,
        )
        return next_state, info

    def _get_reward(
        self,
        state: np.ndarray,
        action_int: int,
        next_state: np.ndarray,
        info: Dict,
    ) -> float:
        """
        Computes the scalar reward for (state, action, next_state) using `info`
        from `_transition_dynamics` (no recomputation).
        """
        step_cost = float(self.cfg.step_cost)
        col_pen   = float(self.cfg.collision_penalty)
        goal_r    = float(self.cfg.goal_reward)

        reward = step_cost * self.N
        reward += col_pen * (info["num_oob"] + info["num_obstacle_collisions"] + info["num_vehicle_collisions"])
        reward += goal_r  * int(info["just_reached"].sum())
        return float(reward)

    def _is_terminal(
        self,
        state: np.ndarray,
        action_int: int,
        next_state: np.ndarray,
        info: Dict,
    ) -> bool:
        """
        Checks termination using `info`:
          - any OOB, any obstacle cell, any vehicle same-cell collision, or all goals reached.
        """
        return bool(
            (info["num_oob"] > 0) or
            (info["num_obstacle_collisions"] > 0) or
            (info["num_vehicle_collisions"] > 0) or
            info["all_reached"]
        )

    def _ensure_obs_lut(self) -> None:
        if getattr(self, "_obs_bits_sorted", None) is not None:
            return

        # EDT symbols (as before)
        df32 = self.dist_field.astype(np.float32, copy=False)
        bits = df32.view(np.uint32)
        self._obs_bits_sorted = np.unique(bits).astype(np.uint32)  # (M,)

        # Localize IDs (collect unique non-negative)
        loc_ids = self.localize_ids[self.localize_ids >= 0]
        if loc_ids.size > 0:
            self._localize_ids_sorted = np.unique(loc_ids).astype(np.int32)  # (L,)
        else:
            self._localize_ids_sorted = np.array([], dtype=np.int32)

        # Offsets and alphabet size
        self._num_loc_symbols = int(self._localize_ids_sorted.shape[0])     # L
        self._num_edt_symbols = int(self._obs_bits_sorted.shape[0])         # M
        self._dist_symbol_offset = self._num_loc_symbols                     # EDT symbols start after IDs
        self._obs_num_symbols = self._num_loc_symbols + self._num_edt_symbols

    def _encode_obs_joint(self, obs_per_drone: np.ndarray) -> int:
        """
        obs_per_drone: (N,) int array, each in [0, M-1]
        Returns a Python int equal to sum_i obs[i] * M**i  (little-endian by drone index).
        """
        M = self._obs_num_symbols
        out = 0
        base = 1
        # little-endian: drone 0 in least-significant "digit"
        for o in obs_per_drone.tolist():
            out += int(o) * base
            base *= M
        return out  # Python int (arbitrary precision)

    def observe(self, state_continuous: np.ndarray) -> int:
        """
        Map continuous positions (N,3) -> ONE centralized observation integer.

        Per drone:
          - If the drone is inside a localize cell (ID >= 0), observe that cell's ID
            (mapped to a compact index 0..L-1).
          - Otherwise, observe the distance-to-obstacle symbol as before
            (mapped to indices L..L+M-1).

        Terminal case (any OOB or obstacle cell): return a single reserved token (default -1).
        """
        assert state_continuous.shape == (self.N, 3), f"expected (N,3), got {state_continuous.shape}"
        self._ensure_obs_lut()  # prepares: _obs_bits_sorted, _localize_ids_sorted, _num_loc_symbols, _dist_symbol_offset, _obs_num_symbols

        X, Y, Z = self.cfg.grid_size
        vs = float(getattr(self.cfg, "voxel_size", 1.0))
        terminal_token = int(getattr(self.cfg, "terminal_obs_value", -1))

        # Continuous -> (unclipped) cell indices
        cells = np.floor((state_continuous / vs) + 0.5).astype(np.int32)

        # Terminal detection: OOB
        oob = (
            (cells[:, 0] < 0) | (cells[:, 0] >= X) |
            (cells[:, 1] < 0) | (cells[:, 1] >= Y) |
            (cells[:, 2] < 0) | (cells[:, 2] >= Z)
        )
        if np.any(oob):
            return terminal_token

        # In-bounds: obstacle collision?
        cx, cy, cz = cells[:, 0], cells[:, 1], cells[:, 2]
        if np.any(self.obstacles[cx, cy, cz]):
            return terminal_token

        # Build per-drone symbols: prefer localize ID if present, else EDT symbol (offset by L)
        symbols = np.empty(self.N, dtype=np.int32)

        # Localize ID path
        loc_here = self.localize_ids[cx, cy, cz]  # (N,), -1 if none
        has_id = (loc_here >= 0)
        if np.any(has_id):
            L = int(getattr(self, "_num_loc_symbols", 0))
            if L <= 0:
                # Shouldn't happen if has_id=True; defensive fallback
                return terminal_token
            id_idx = np.searchsorted(self._localize_ids_sorted, loc_here[has_id])
            # Safety: exact match
            if not np.all(self._localize_ids_sorted[id_idx] == loc_here[has_id]):
                return terminal_token
            symbols[has_id] = id_idx.astype(np.int32)

        # EDT path for drones not in a localize cell
        no_id = ~has_id
        if np.any(no_id):
            d = self.dist_field[cx[no_id], cy[no_id], cz[no_id]].astype(np.float32, copy=False)
            d_bits = d.view(np.uint32)
            idx = np.searchsorted(self._obs_bits_sorted, d_bits)  # (k,)
            # Safety: exact match
            if not np.all(self._obs_bits_sorted[idx] == d_bits):
                return terminal_token
            symbols[no_id] = (self._dist_symbol_offset + idx.astype(np.int32))

        # Centralize: base-(L+M) encode into one integer
        return self._encode_obs_joint(symbols.astype(np.int32))


    def likelihood(self, observation: int, state_continuous: np.ndarray) -> float:
        """
        P(o | state): deterministic sensor matching `observe(state)`.
        Returns 1.0 if the encoded observation equals `observation`, else 0.0.

        Args:
            observation: centralized observation int (may be terminal token)
            state_continuous: (N, 3) float32 array
        """
        assert state_continuous.shape == (self.N, 3), f"expected (N,3), got {state_continuous.shape}"
        o = int(self.observe(state_continuous))
        return 1.0 if o == int(observation) else 0.0

    def _load_config(self, path: str) -> MultiDroneConfig:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        if "grid_size" in data and isinstance(data["grid_size"], list):
            data["grid_size"] = tuple(data["grid_size"])

        return MultiDroneConfig(**data)

    # ---------- Action encoding ----------
    def _decode_action(self, action_int: int) -> np.ndarray:        
        num_per_vehicle = self._action_vectors.shape[0]
        actions = np.zeros(self.N, dtype=np.int32)
        x = action_int
        for i in range(self.N):
            actions[i] = x % num_per_vehicle
            x //= num_per_vehicle
        return actions

    def _encode_action(self, actions: np.ndarray) -> int:        
        num_per_vehicle = self._action_vectors.shape[0]
        base = 1
        out = 0
        for a in actions:
            out += int(a) * base
            base *= num_per_vehicle
        return out

    # ---------- Visualization ----------
    def _grid_to_world(self, pos: np.ndarray, vs: float = 1.0) -> np.ndarray:
        return pos * vs

    def _init_plot(self):
        self._plotter = Plotter(interactive=False)
        self._uuv_visuals = []
        self._traj_lines = []
        self._obstacle_meshes = []
        self._goal_meshes = []
        self._particle_ghosts = []  # list of vedo actors for belief particle drones
        goal_tol = float(self.cfg.goal_tol)

        X, Y, Z = self.cfg.grid_size
        vs = 1.0        

        # obstacles (center each cube on the cell center)
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    if self.obstacles[x, y, z]:
                        center = self._grid_to_world(np.array([x, y, z]), vs)
                        cube = Box(pos=center.tolist(), length=vs, width=vs, height=vs).c("gray").alpha(0.5)
                        self._obstacle_meshes.append(cube)

        # goals: green cubes centered on cell centers
        for c in self.goal_centers:   # world coords
            goal_sphere = Sphere(pos=c.tolist(), r=goal_tol).c("green").alpha(0.35)
            self._goal_meshes.append(goal_sphere)

        # localize cells: yellow cubes centered on cell centers
        self._localize_meshes = []
        if np.any(self.localize_ids >= 0):
            loc_coords = np.argwhere(self.localize_ids >= 0)
            for (x, y, z) in loc_coords:
                center = self._grid_to_world(np.array([x, y, z]), vs)
                loc_box = Box(
                    pos=center.tolist(),
                    length=vs, width=vs, height=vs
                ).c("yellow").alpha(0.6)
                self._localize_meshes.append(loc_box)

                domain = Box(
                    pos=[(X*vs)/2, (Y*vs)/2, (Z*vs)/2],
                    length=X*vs, width=Y*vs, height=Z*vs,
                ).wireframe().c("gray3").alpha(0.3)

        # drones
        for i in range(self.N):
            pos = self._grid_to_world(self.positions[i], vs)
            body = Sphere(pos=pos.tolist(), r=vs*0.3).c("cyan")
            arm1 = Cylinder(pos=[pos + np.array([-0.5, 0, 0])*vs,
                                 pos + np.array([ 0.5, 0, 0])*vs], r=vs*0.05).c("black")
            arm2 = Cylinder(pos=[pos + np.array([0, -0.5, 0])*vs,
                                 pos + np.array([0,  0.5, 0])*vs], r=vs*0.05).c("black")
            traj = Line([pos]).c("blue").lw(2)
            self._uuv_visuals.append((body, arm1, arm2))
            self._traj_lines.append(traj)

        # Create a dummy box actor that defines the bounds
        bounds = [0, X*vs, 0, Y*vs, 0, Z*vs]
        axes = Axes(bounds, xtitle="X", ytitle="Y", ztitle="Z")

        axes = dict(
            xrange=(bounds[0], bounds[1]),
            yrange=(bounds[2], bounds[3]),
            zrange=(bounds[4], bounds[5]),
            xygrid=True,    
        )

        self._plotter.show(
            *self._obstacle_meshes, 
            *self._goal_meshes,
            *self._localize_meshes,
            *[v for triple in self._uuv_visuals for v in triple],
            *self._traj_lines,
            axes=axes,
            viewup="z", interactive=False,
        )

    def update_plot(self, belief_particles: np.ndarray = None):
        if getattr(self, "_plotter", None) is None:
            return

        vs = 1.0

        # --- Update true-state visuals and trajectories (as before) ---
        for i in range(self.N):
            pos = self._grid_to_world(self.positions[i], vs)
            body, arm1, arm2 = self._uuv_visuals[i]

            # update body
            body.pos(pos)

            # replace arms (simple update of endpoints)
            self._plotter.remove(arm1)
            self._plotter.remove(arm2)
            arm1 = Cylinder(pos=[pos + np.array([-0.5, 0, 0])*vs,
                                 pos + np.array([ 0.5, 0, 0])*vs], r=vs*0.05).c("black")
            arm2 = Cylinder(pos=[pos + np.array([0, -0.5, 0])*vs,
                                 pos + np.array([0,  0.5, 0])*vs], r=vs*0.05).c("black")
            self._uuv_visuals[i] = (body, arm1, arm2)

            # update trajectory
            '''if callable(self._traj_lines[i].points):
                old_pts = self._traj_lines[i].points()
            else:
                old_pts = self._traj_lines[i].points
            new_pts = np.vstack([old_pts, pos.reshape(1, -1)])
            self._plotter.remove(self._traj_lines[i])
            self._traj_lines[i] = Line(new_pts).c("blue").lw(2)
            self._plotter.add(self._traj_lines[i])'''

            # re-add updated arms
            self._plotter.add(arm1)
            self._plotter.add(arm2)

        # --- Remove previous particle ghosts (if any) ---
        if self._particle_ghosts:
            for actor in self._particle_ghosts:
                self._plotter.remove(actor)
            self._particle_ghosts = []

        # --- Draw belief particles as transparent drone geometries ---
        if belief_particles is not None:
            # Expect (N_particles, K, 3) where K == self.N
            assert belief_particles.ndim == 3 and belief_particles.shape[1] == self.N and belief_particles.shape[2] == 3, \
                f"belief_particles must be (N_particles, {self.N}, 3), got {belief_particles.shape}"

            # Choose per-drone colors (same palette as true drones if you like)
            drone_colors = ["cyan", "magenta", "yellow", "orange", "lime", "tomato", "deepskyblue", "purple"]

            # Transparency for ghosts
            alpha = 0.18  # nice & subtle

            P = belief_particles.shape[0]
            for p in range(P):
                for i in range(self.N):
                    pos = belief_particles[p, i, :]  # world coords
                    # ghost body
                    g_body = Sphere(pos=pos.tolist(), r=vs*0.3).c(drone_colors[i % len(drone_colors)]).alpha(alpha)
                    # ghost arms
                    g_arm1 = Cylinder(pos=[pos + np.array([-0.5, 0, 0])*vs,
                                           pos + np.array([ 0.5, 0, 0])*vs], r=vs*0.05).c("black").alpha(alpha)
                    g_arm2 = Cylinder(pos=[pos + np.array([0, -0.5, 0])*vs,
                                           pos + np.array([0,  0.5, 0])*vs], r=vs*0.05).c("black").alpha(alpha)

                    # add & remember so we can remove next update
                    self._plotter.add(g_body)
                    self._plotter.add(g_arm1)
                    self._plotter.add(g_arm2)
                    self._particle_ghosts.extend([g_body, g_arm1, g_arm2])

        self._plotter.render()


    def show(self):
        """Enter interactive mode to explore scene."""
        self._plotter.interactive()