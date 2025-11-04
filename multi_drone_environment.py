import yaml
import numpy as np
from multi_drone_config import MultiDroneConfig
from typing import Tuple, Dict
from vedo import Plotter, Box, Sphere, Cylinder

class MultiDroneEnvironment:
    def __init__(
        self, 
        config_path: str,
        transition_model,
        observation_model,        
        initial_belief_model,
        task,):

        self.cfg = self._load_config(config_path)
        self.transition_model = transition_model
        self.observation_model = observation_model        
        self.initial_belief_model = initial_belief_model
        self.task = task

        assert self.cfg.initial_drone_positions is not None, "initial_drone_positions can't be empty"        
        self.rng = np.random.default_rng(self.cfg.seed)        

        # vedo visualization members
        self._plotter = None
        self._drone_visuals = []        

    def get_config(self):
        return self.cfg
    
    def num_actions(self) -> int:
        return self.transition_model.num_actions(self.cfg.num_controlled_drones)

    # ---------- Core API ----------
    def reset(self) -> np.ndarray:
        X, Y, Z = self.cfg.environment_size

        # Obstacles (config lists cell CENTERS as integer triples)
        self.obstacles = np.zeros(self.cfg.environment_size, dtype=bool)
        self.obstacles.fill(False)
        if self.cfg.obstacle_positions is not None:
            for (x, y, z) in self.cfg.obstacle_positions:
                assert 0 <= x < X and 0 <= y < Y and 0 <= z < Z, \
                    f"Obstacle {(x, y, z)} is outside environment bounds {self.cfg.environment_size}"
                self.obstacles[x, y, z] = True

        # Localization cells (auto-ID; config lists CENTERS [x,y,z])
        if not hasattr(self, "localize_ids") or self.localize_ids.shape != self.obstacles.shape:
            self.localize_ids = np.full(self.cfg.environment_size, fill_value=-1, dtype=np.int32)
        else:
            self.localize_ids.fill(-1)

        if getattr(self.cfg, "localize_cells", None) is not None:
            for cid, (x, y, z) in enumerate(self.cfg.localize_cells):
                assert 0 <= x < X and 0 <= y < Y and 0 <= z < Z, \
                    f"Localize cell {(x, y, z)} out of bounds {self.cfg.environment_size}"                
                self.localize_ids[x, y, z] = int(cid)

        # Initial drone positions       
        self.positions = self.initial_belief_model.sample(self, 1)[0]        

        # Store goal centers in WORLD coords directly
        self._voxel_size = 1.0
        self.goal_centers = np.array(self.cfg.goal_positions, dtype=np.float32) * self._voxel_size  # world coords

        # Time
        self.t = 0

        # Vedo
        self._init_plot()
        return self.positions.copy()

    def sample_from_initial_belief(self, num_samples: int) -> np.ndarray:
        """
        Return a batch of initial state samples drawn from a uniform 3D distribution
        centered around each drone's nominal start position in self.cfg.initial_drone_positions.

        Each drone i ~ Uniform(
            start_pos_i - noise_range/2, 
            start_pos_i + noise_range/2
        )

        Args:
            num_samples: number of belief particles (B)

        Returns:
            samples: (B, N, 3) float32 array of world coordinates
        """
        return self.initial_belief_model.sample(self, num_samples)


    def step(self, action_int: int) -> Tuple[np.ndarray, int, float, bool, Dict]:        
        state = self.positions.astype(np.float32, copy=False)

        # Transition
        next_state, observation_int, reward, done, info = self.simulate(state, action_int)

        # Update internal state
        self.positions = next_state        
        self.t += 1        

        # Max-step termination
        if self.t >= self.cfg.max_num_steps:
            done = True
            info["max_steps_reached"] = True

        return self.positions.copy(), observation_int, reward, done, info

    def simulate(self, state: np.ndarray, action_int: int) -> Tuple[np.ndarray, int, float, bool, Dict]:
        next_state, info = self.transition_model.step(self, state, action_int)
        observation_int   = self.observation_model.observe(self, next_state)
        reward   = self.task.reward(self, state, action_int, next_state, info=info)
        done  = self.task.done(self, state, action_int, next_state, info=info)
        return next_state, observation_int, reward, done, info

    def likelihood(self, observation: int, state: np.ndarray) -> float:
        return self.observation_model.likelihood(self, observation, state)

    def _load_config(self, path: str) -> MultiDroneConfig:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return MultiDroneConfig(**data)
    
    # ---------- Visualization ----------
    def _init_plot(self):
        self._plotter = Plotter(interactive=False)
        self._drone_visuals = []
        self._obstacle_meshes = []
        self._goal_meshes = []
        self._localize_meshes = []
        self._particle_ghosts = []  # list of vedo actors for belief particle drones
        goal_radius = float(self.cfg.goal_radius)

        X, Y, Z = self.cfg.environment_size        

        # obstacles (center each cube on the cell center)
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    if self.obstacles[x, y, z]:
                        center = np.array([x, y, z])
                        cube = Box(
                            pos=center.tolist(), 
                            length=self._voxel_size, 
                            width=self._voxel_size, 
                            height=self._voxel_size
                        ).c("gray").alpha(0.8)
                        self._obstacle_meshes.append(cube)

        # goals: green spheres centered at world goal centers
        for c in self.goal_centers:
            goal_sphere = Sphere(pos=c.tolist(), r=goal_radius).c("green").alpha(0.35)
            self._goal_meshes.append(goal_sphere)

        # localize cells: yellow cubes centered on cell centers
        if np.any(self.localize_ids >= 0):
            loc_coords = np.argwhere(self.localize_ids >= 0)
            for (x, y, z) in loc_coords:
                center = np.array([x, y, z])
                loc_box = Box(
                    pos=center.tolist(), 
                    length=self._voxel_size, 
                    width=self._voxel_size, 
                    height=self._voxel_size).c("yellow").alpha(0.6)
                self._localize_meshes.append(loc_box)

        # drones
        for i in range(self.cfg.num_controlled_drones):
            pos = self.positions[i]
            body = Sphere(pos=pos.tolist(), r=self._voxel_size*0.3).c("cyan")
            arm1 = Cylinder(pos=[pos + np.array([-0.5, 0, 0])*self._voxel_size,
                                 pos + np.array([ 0.5, 0, 0])*self._voxel_size], r=self._voxel_size*0.05).c("black")
            arm2 = Cylinder(pos=[pos + np.array([0, -0.5, 0])*self._voxel_size,
                                 pos + np.array([0,  0.5, 0])*self._voxel_size], r=self._voxel_size*0.05).c("black")
            self._drone_visuals.append((body, arm1, arm2))

        # Axes ranges for the scene
        bounds = [0, X*self._voxel_size, 0, Y*self._voxel_size, 0, Z*self._voxel_size]
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
            *[v for triple in self._drone_visuals for v in triple],
            axes=axes,
            viewup="z", interactive=False,
        )

    def update_plot(self, belief_particles: np.ndarray = None):
        if getattr(self, "_plotter", None) is None:
            return       

        # --- Update true-state visuals (body + arms) ---
        for i in range(self.cfg.num_controlled_drones):
            pos = self.positions[i]
            body, arm1, arm2 = self._drone_visuals[i]

            body.pos(pos)

            self._plotter.remove(arm1)
            self._plotter.remove(arm2)
            arm1 = Cylinder(pos=[pos + np.array([-0.5, 0, 0])*self._voxel_size,
                                 pos + np.array([ 0.5, 0, 0])*self._voxel_size], r=self._voxel_size*0.05).c("black")
            arm2 = Cylinder(pos=[pos + np.array([0, -0.5, 0])*self._voxel_size,
                                 pos + np.array([0,  0.5, 0])*self._voxel_size], r=self._voxel_size*0.05).c("black")
            self._drone_visuals[i] = (body, arm1, arm2)

            self._plotter.add(arm1)
            self._plotter.add(arm2)

        # --- Remove previous particle ghosts (if any) ---
        if self._particle_ghosts:
            for actor in self._particle_ghosts:
                self._plotter.remove(actor)
            self._particle_ghosts = []

        # --- Draw belief particles as transparent drone geometries ---
        if belief_particles is not None:
            assert belief_particles.ndim == 3 and belief_particles.shape[1] == self.cfg.num_controlled_drones and belief_particles.shape[2] == 3, \
                f"belief_particles must be (N_particles, {self.cfg.num_controlled_drones}, 3), got {belief_particles.shape}"

            drone_colors = ["cyan", "magenta", "yellow", "orange", "lime", "tomato", "deepskyblue", "purple"]
            alpha = 0.18

            P = belief_particles.shape[0]
            for p in range(P):
                for i in range(self.cfg.num_controlled_drones):
                    pos = belief_particles[p, i, :]
                    g_body = Sphere(pos=pos.tolist(), r=self._voxel_size*0.3).c(drone_colors[i % len(drone_colors)]).alpha(alpha)
                    g_arm1 = Cylinder(pos=[pos + np.array([-0.5, 0, 0])*self._voxel_size,
                                           pos + np.array([ 0.5, 0, 0])*self._voxel_size], r=self._voxel_size*0.05).c("black").alpha(alpha)
                    g_arm2 = Cylinder(pos=[pos + np.array([0, -0.5, 0])*self._voxel_size,
                                           pos + np.array([0,  0.5, 0])*self._voxel_size], r=self._voxel_size*0.05).c("black").alpha(alpha)

                    self._plotter.add(g_body);  self._plotter.add(g_arm1);  self._plotter.add(g_arm2)
                    self._particle_ghosts.extend([g_body, g_arm1, g_arm2])

        self._plotter.render()

    def show(self):
        """Enter interactive mode to explore scene."""
        self._plotter.interactive()
