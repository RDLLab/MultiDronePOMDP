from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

@dataclass
class MultiDroneConfig:
    # Environment size and config
    environment_size: Tuple[int, int, int] 
    num_controlled_drones: int
    initial_drone_positions: list[int]
    initial_drone_uncertainty: float
    goal_positions: list[int]
    goal_radius: float
    transition_noise_std: float  
    obs_correct_prob: float 
    use_distance_sensor: bool 

    # MDP parameters
    discount_factor: float
    step_cost: float
    collision_penalty: float
    goal_reward: float
    max_num_steps: int

    # Optional obstacle cells
    obstacle_positions: Optional[list] = None
    localize_cells: Optional[list] = None    
    seed: Optional[int] = None

    # Optional user-defined parameters
    initial_belief_params: Dict[str, Any] = field(default_factory=dict)
    transition_params: Dict[str, Any] = field(default_factory=dict)
    observation_params: Dict[str, Any] = field(default_factory=dict)
    task_params: Dict[str, Any] = field(default_factory=dict)