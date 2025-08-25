"""
Pathfinding subpackage: A*, CBS, constraints, visualize, loader 등 
"""
# --- A* / Heuristics / Utils ---
from .Astar import (
    a_star_single,
    a_star_multi,
    manhattan_heuristics,
    multi_manhattan_heuristics,
    Dijkstra_heuristics,
    get_sum_of_cost,
    get_location,
)

# --- Constraints helpers ---
from .constraints import (
    is_constrained,
    violates_pos_constraint,
    future_constraint_exists,
    build_constraint_table,
)

# --- Collision / Splitting ---
from .collision import (
    detect_collision,
    detect_collisions,
    standard_splitting,
    disjoint_splitting,
    type_priority_splitting,
)

# --- Map loader / Visualize ---
from .MapLoader import load_map
from .visualize import (
    plot_map_with_paths,
    plot_single_path,
    animate_paths,
    plot_multi_paths,
    animate_multi_paths,
)

CBSSolver = None
try:
    from .cbs_solver import CBSSolver  
except Exception:
    try:
        from ..CBSSolver import CBSSolver  
    except Exception:
        CBSSolver = None  

__all__ = [
    # A*
    "a_star_single", "a_star_multi",
    "manhattan_heuristics", "multi_manhattan_heuristics", "Dijkstra_heuristics",
    "get_sum_of_cost", "get_location",
    # Constraints
    "is_constrained", "violates_pos_constraint",
    "future_constraint_exists", "build_constraint_table",
    # Collision
    "detect_collision", "detect_collisions",
    "standard_splitting", "disjoint_splitting", "type_priority_splitting",
    # Loader / Viz
    "load_map",
    "plot_map_with_paths", "plot_single_path", "animate_paths",
    "plot_multi_paths", "animate_multi_paths",
    # CBS
    "CBSSolver",
]
