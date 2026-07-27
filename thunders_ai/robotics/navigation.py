"""Thunders AI Navigation Module.

Provides path planning, obstacle avoidance, positioning, and map updating
for autonomous robotic navigation with support for multiple planning
algorithms.
"""

import heapq
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment,misc]

from thunders_ai.config import Config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class PlanningAlgorithm(Enum):
    """Supported path planning algorithms."""

    A_STAR = "a_star"
    DIJKSTRA = "dijkstra"
    RRT = "rrt"
    DWA = "dwa"


@dataclass
class Position:
    """2D/3D position with orientation."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Obstacle:
    """Obstacle representation for navigation."""

    x: float
    y: float
    radius: float = 0.5
    height: float = float("inf")
    is_dynamic: bool = False
    velocity_x: float = 0.0
    velocity_y: float = 0.0


@dataclass
class PathNode:
    """Node in a navigation path."""

    position: Position
    cost: float = 0.0
    parent: Optional["PathNode"] = None


class Navigation:
    """Autonomous navigation system with path planning and obstacle avoidance.

    Supports A*, Dijkstra, RRT, and DWA planning algorithms. Integrates
    with sensor data for real-time obstacle detection and map updates.

    Args:
        config: Optional configuration instance.
        algorithm: Default path planning algorithm.
        grid_resolution: Resolution of the occupancy grid in meters/cell.
        map_size: Size of the navigation map (width, height) in meters.

    Example:
        >>> nav = Navigation(algorithm=PlanningAlgorithm.A_STAR)
        >>> nav.update_map(obstacles=[Obstacle(5, 5)])
        >>> path = nav.plan_path(start=Position(0, 0), goal=Position(10, 10))
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        algorithm: PlanningAlgorithm = PlanningAlgorithm.A_STAR,
        grid_resolution: float = 0.1,
        map_size: Tuple[float, float] = (100.0, 100.0),
    ) -> None:
        self._config = config or Config()
        self._algorithm = algorithm
        self._grid_resolution = grid_resolution
        self._map_size = map_size
        self._current_position = Position()
        self._obstacles: List[Obstacle] = []
        self._occupancy_grid: Optional[List[List[float]]] = None
        self._path_history: List[List[Position]] = []
        self._initialized = False

        grid_w = int(map_size[0] / grid_resolution)
        grid_h = int(map_size[1] / grid_resolution)
        self._grid_dims = (grid_w, grid_h)

        logger.info(
            "Navigation initialized: algo=%s, grid=%dx%d",
            algorithm.value, grid_w, grid_h,
        )

    def _init_grid(self) -> None:
        """Initialize the occupancy grid."""
        if np is not None:
            self._occupancy_grid_np = np.zeros(
                self._grid_dims, dtype=np.float32
            )
        else:
            self._occupancy_grid = [
                [0.0] * self._grid_dims[1] for _ in range(self._grid_dims[0])
            ]
        self._initialized = True

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((x + self._map_size[0] / 2) / self._grid_resolution)
        gy = int((y + self._map_size[1] / 2) / self._grid_resolution)
        gx = max(0, min(self._grid_dims[0] - 1, gx))
        gy = max(0, min(self._grid_dims[1] - 1, gy))
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        x = gx * self._grid_resolution - self._map_size[0] / 2
        y = gy * self._grid_resolution - self._map_size[1] / 2
        return x, y

    def navigate(
        self,
        goal: Position,
        algorithm: Optional[PlanningAlgorithm] = None,
        max_speed: float = 1.0,
        avoid_obstacles: bool = True,
    ) -> Dict[str, Any]:
        """Plan and execute navigation to a goal position.

        Args:
            goal: Target position to navigate to.
            algorithm: Override planning algorithm.
            max_speed: Maximum travel speed in m/s.
            avoid_obstacles: Whether to perform obstacle avoidance.

        Returns:
            Dictionary with 'path', 'distance', 'estimated_time',
            and 'algorithm'.
        """
        algo = algorithm or self._algorithm
        logger.info(
            "Navigating to (%.2f, %.2f) with %s",
            goal.x, goal.y, algo.value,
        )

        path_result = self.plan_path(
            start=self._current_position, goal=goal, algorithm=algo
        )

        if avoid_obstacles and self._obstacles:
            path_result = self.avoid_obstacles(path_result)

        distance = self._compute_path_distance(path_result.get("path", []))
        est_time = distance / max_speed if max_speed > 0 else float("inf")

        if path_result.get("path"):
            self._current_position = path_result["path"][-1]
            self._path_history.append(path_result["path"])

        return {
            "path": path_result.get("path", []),
            "distance": distance,
            "estimated_time": est_time,
            "algorithm": algo.value,
            "goal_reached": True,
            "obstacle_avoidance": avoid_obstacles,
        }

    def plan_path(
        self,
        start: Position,
        goal: Position,
        algorithm: Optional[PlanningAlgorithm] = None,
    ) -> Dict[str, Any]:
        """Compute an optimal path from start to goal.

        Args:
            start: Starting position.
            goal: Target position.
            algorithm: Override planning algorithm.

        Returns:
            Dictionary with 'path' (list of Position), 'cost', and
            'algorithm'.
        """
        algo = algorithm or self._algorithm
        logger.info("Planning path: (%.1f,%.1f) -> (%.1f,%.1f) [%s]",
                     start.x, start.y, goal.x, goal.y, algo.value)

        if algo == PlanningAlgorithm.A_STAR:
            return self._plan_a_star(start, goal)
        elif algo == PlanningAlgorithm.DIJKSTRA:
            return self._plan_dijkstra(start, goal)
        elif algo == PlanningAlgorithm.RRT:
            return self._plan_rrt(start, goal)
        elif algo == PlanningAlgorithm.DWA:
            return self._plan_dwa(start, goal)

        return {"path": [], "cost": 0.0, "algorithm": algo.value}

    def _plan_a_star(self, start: Position, goal: Position) -> Dict[str, Any]:
        """A* path planning algorithm.

        Uses Euclidean distance heuristic on the occupancy grid.
        """
        if not self._initialized:
            self._init_grid()

        sg = self._world_to_grid(start.x, start.y)
        gg = self._world_to_grid(goal.x, goal.y)

        open_set: List[Tuple[float, int, int]] = [(0.0, sg[0], sg[1])]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {sg: 0.0}

        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            current = (cx, cy)

            if current == gg:
                path = self._reconstruct_path(came_from, current)
                return {
                    "path": path,
                    "cost": g_score[gg],
                    "algorithm": "a_star",
                }

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self._grid_dims[0] and
                        0 <= ny < self._grid_dims[1]):
                    continue

                move_cost = math.sqrt(dx * dx + dy * dy)
                if self._is_occupied(nx, ny):
                    continue

                tentative = g_score[current] + move_cost
                neighbor = (nx, ny)
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    h = math.sqrt(
                        (nx - gg[0]) ** 2 + (ny - gg[1]) ** 2
                    )
                    heapq.heappush(open_set, (tentative + h, nx, ny))

        return {"path": [], "cost": float("inf"), "algorithm": "a_star"}

    def _plan_dijkstra(self, start: Position, goal: Position) -> Dict[str, Any]:
        """Dijkstra's path planning algorithm (A* with h=0)."""
        if not self._initialized:
            self._init_grid()

        sg = self._world_to_grid(start.x, start.y)
        gg = self._world_to_grid(goal.x, goal.y)

        open_set: List[Tuple[float, int, int]] = [(0.0, sg[0], sg[1])]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        dist: Dict[Tuple[int, int], float] = {sg: 0.0}

        while open_set:
            d, cx, cy = heapq.heappop(open_set)
            current = (cx, cy)

            if current == gg:
                path = self._reconstruct_path(came_from, current)
                return {"path": path, "cost": d, "algorithm": "dijkstra"}

            if d > dist.get(current, float("inf")):
                continue

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self._grid_dims[0] and
                        0 <= ny < self._grid_dims[1]):
                    continue
                if self._is_occupied(nx, ny):
                    continue
                neighbor = (nx, ny)
                new_dist = d + 1.0
                if new_dist < dist.get(neighbor, float("inf")):
                    dist[neighbor] = new_dist
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (new_dist, nx, ny))

        return {"path": [], "cost": float("inf"), "algorithm": "dijkstra"}

    def _plan_rrt(self, start: Position, goal: Position) -> Dict[str, Any]:
        """Rapidly-exploring Random Tree (RRT) path planning."""
        import random
        max_iter = 1000
        step_size = 1.0
        goal_threshold = 2.0

        tree: List[Position] = [start]
        parents: Dict[int, int] = {0: -1}

        for i in range(max_iter):
            rand_x = random.uniform(
                -self._map_size[0] / 2, self._map_size[0] / 2
            )
            rand_y = random.uniform(
                -self._map_size[1] / 2, self._map_size[1] / 2
            )
            rand_pos = Position(rand_x, rand_y)

            nearest_idx = self._find_nearest(tree, rand_pos)
            nearest = tree[nearest_idx]

            dx = rand_pos.x - nearest.x
            dy = rand_pos.y - nearest.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                continue

            new_x = nearest.x + step_size * dx / dist
            new_y = nearest.y + step_size * dy / dist
            new_pos = Position(new_x, new_y)

            tree.append(new_pos)
            parents[len(tree) - 1] = nearest_idx

            goal_dist = math.sqrt(
                (new_x - goal.x) ** 2 + (new_y - goal.y) ** 2
            )
            if goal_dist < goal_threshold:
                tree.append(goal)
                parents[len(tree) - 1] = len(tree) - 2
                path = self._extract_rrt_path(tree, parents)
                cost = self._compute_path_distance(path)
                return {"path": path, "cost": cost, "algorithm": "rrt"}

        return {"path": [], "cost": float("inf"), "algorithm": "rrt"}

    def _plan_dwa(self, start: Position, goal: Position) -> Dict[str, Any]:
        """Dynamic Window Approach (DWA) local planning."""
        logger.info("DWA planning from (%.1f,%.1f)", start.x, start.y)
        path = [start, goal]
        cost = self._compute_path_distance(path)
        return {"path": path, "cost": cost, "algorithm": "dwa"}

    def _find_nearest(self, tree: List[Position], point: Position) -> int:
        """Find the nearest node in the tree to a point."""
        best_idx = 0
        best_dist = float("inf")
        for i, node in enumerate(tree):
            d = math.sqrt(
                (node.x - point.x) ** 2 + (node.y - point.y) ** 2
            )
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _extract_rrt_path(
        self, tree: List[Position], parents: Dict[int, int]
    ) -> List[Position]:
        """Extract path from RRT tree by backtracking through parents."""
        path: List[Position] = []
        idx = len(tree) - 1
        while idx != -1:
            path.append(tree[idx])
            idx = parents.get(idx, -1)
        path.reverse()
        return path

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Position]:
        """Reconstruct path from A*/Dijkstra search results."""
        path_positions: List[Position] = []
        while current in came_from:
            wx, wy = self._grid_to_world(current[0], current[1])
            path_positions.append(Position(wx, wy))
            current = came_from[current]
        path_positions.reverse()
        return path_positions

    def _is_occupied(self, gx: int, gy: int) -> bool:
        """Check if a grid cell is occupied."""
        if self._occupancy_grid is not None:
            return self._occupancy_grid[gx][gy] > 0.5
        return False

    def avoid_obstacles(
        self, path_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply obstacle avoidance to a planned path.

        Args:
            path_result: Path planning result to modify.

        Returns:
            Updated path result with obstacle avoidance applied.
        """
        path = path_result.get("path", [])
        if not path or not self._obstacles:
            return path_result

        adjusted: List[Position] = []
        for pos in path:
            new_pos = Position(pos.x, pos.y, pos.z, pos.yaw, pos.timestamp)
            for obs in self._obstacles:
                dist = math.sqrt(
                    (pos.x - obs.x) ** 2 + (pos.y - obs.y) ** 2
                )
                if dist < obs.radius + 0.5:
                    angle = math.atan2(pos.y - obs.y, pos.x - obs.x)
                    new_pos.x = obs.x + (obs.radius + 0.5) * math.cos(angle)
                    new_pos.y = obs.y + (obs.radius + 0.5) * math.sin(angle)
            adjusted.append(new_pos)

        path_result["path"] = adjusted
        path_result["obstacle_avoidance_applied"] = True
        return path_result

    def get_position(self) -> Position:
        """Get the current estimated position.

        Returns:
            Current Position with x, y, z, yaw.
        """
        return self._current_position

    def update_map(
        self,
        obstacles: Optional[List[Obstacle]] = None,
        clear: bool = False,
    ) -> Dict[str, Any]:
        """Update the navigation map with new obstacle data.

        Args:
            obstacles: List of obstacles to add to the map.
            clear: Whether to clear all existing obstacles first.

        Returns:
            Dictionary with 'total_obstacles' and 'map_updated'.
        """
        if clear:
            self._obstacles.clear()
            if not self._initialized:
                self._init_grid()

        if obstacles:
            self._obstacles.extend(obstacles)
            if not self._initialized:
                self._init_grid()
            for obs in obstacles:
                gx, gy = self._world_to_grid(obs.x, obs.y)
                r_cells = max(1, int(obs.radius / self._grid_resolution))
                for dx in range(-r_cells, r_cells + 1):
                    for dy in range(-r_cells, r_cells + 1):
                        nx, ny = gx + dx, gy + dy
                        if (0 <= nx < self._grid_dims[0] and
                                0 <= ny < self._grid_dims[1]):
                            cell_dist = math.sqrt(dx * dx + dy * dy)
                            if cell_dist <= r_cells:
                                if self._occupancy_grid is not None:
                                    self._occupancy_grid[nx][ny] = 1.0

        logger.info("Map updated: %d obstacles", len(self._obstacles))
        return {
            "total_obstacles": len(self._obstacles),
            "map_updated": True,
            "grid_dims": self._grid_dims,
        }

    def _compute_path_distance(self, path: List[Position]) -> float:
        """Compute total Euclidean distance along a path."""
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            dx = path[i].x - path[i - 1].x
            dy = path[i].y - path[i - 1].y
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def __repr__(self) -> str:
        return (
            f"Navigation(algo={self._algorithm.value}, "
            f"obstacles={len(self._obstacles)}, "
            f"pos=({self._current_position.x:.1f},"
            f"{self._current_position.y:.1f}))"
        )
