from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass
class MultiDroneConfig:
    # Environment size and config
    grid_size: Tuple[int, int, int] 
    change_altitude: bool
    start_positions: list[int]
    goal_positions: list[int]
    goal_tol: float
    step_size: float
    process_noise_std: float

    # MDP parameters
    discount_factor: float
    step_cost: float
    collision_penalty: float
    goal_reward: float
    max_num_steps: int

    # Optional obstacle cells
    obstacle_cells: Optional[list] = None
    localize_cells: Optional[list] = None    
    seed: Optional[int] = None