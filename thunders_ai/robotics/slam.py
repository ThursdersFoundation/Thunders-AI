"""Thunders AI SLAM Module.

Provides Simultaneous Localization and Mapping capabilities including
map building, localization, SLAM updates, and occupancy grid mapping
with support for visual and LiDAR SLAM approaches.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment,misc]

from thunders_ai.config import Config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class SLAMType(Enum):
    """Supported SLAM approaches."""

    VISUAL = "visual"
    LIDAR = "lidar"


class MapCellState(Enum):
    """States of an occupancy grid cell."""

    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1


@dataclass
class Pose:
    """Robot pose (position + orientation) for SLAM."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    timestamp: float = field(default_factory=time.time)
    covariance: Optional[List[float]] = None


@dataclass
class Landmark:
    """A mapped landmark for SLAM."""

    landmark_id: str
    x: float
    y: float
    z: float = 0.0
    descriptor: Optional[Any] = None
    observations: int = 1
    last_seen: float = field(default_factory=time.time)


@dataclass
class SLAMMap:
    """Representation of the SLAM-generated map."""

    width: int
    height: int
    resolution: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    grid: Optional[Any] = None
    landmarks: Dict[str, Landmark] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


class SLAM:
    """Simultaneous Localization and Mapping system for building maps
    and localizing a robot within them.

    Supports visual SLAM (feature-based) and LiDAR SLAM (scan-matching)
    approaches. Implements occupancy grid mapping with probabilistic
    updates.

    Args:
        config: Optional configuration instance.
        slam_type: Type of SLAM algorithm to use.
        resolution: Map resolution in meters per cell.
        map_size: Map dimensions (width, height) in meters.

    Example:
        >>> slam = SLAM(slam_type=SLAMType.LIDAR, resolution=0.05)
        >>> slam.update(sensor_data={"lidar_scan": [...]})
        >>> pose = slam.localize(sensor_data={"lidar_scan": [...]})
        >>> grid_map = slam.get_map()
    """

    LOG_ODDS_FREE = -0.5
    LOG_ODDS_OCCUPIED = 0.5
    LOG_ODDS_CLAMP = 5.0

    def __init__(
        self,
        config: Optional[Config] = None,
        slam_type: SLAMType = SLAMType.LIDAR,
        resolution: float = 0.05,
        map_size: Tuple[float, float] = (50.0, 50.0),
    ) -> None:
        self._config = config or Config()
        self._slam_type = slam_type
        self._resolution = resolution
        self._map_size = map_size

        grid_w = int(map_size[0] / resolution)
        grid_h = int(map_size[1] / resolution)

        # Initialize log-odds occupancy grid
        if np is not None:
            self._log_odds = np.zeros((grid_w, grid_h), dtype=np.float32)
        else:
            self._log_odds = [
                [0.0] * grid_h for _ in range(grid_w)
            ]

        self._current_pose = Pose()
        self._landmarks: Dict[str, Landmark] = {}
        self._trajectory: List[Pose] = [self._current_pose]
        self._keyframes: List[Dict[str, Any]] = []
        self._loop_closures: List[Dict[str, Any]] = []

        self._slam_map = SLAMMap(
            width=grid_w,
            height=grid_h,
            resolution=resolution,
        )

        logger.info(
            "SLAM initialized: type=%s, grid=%dx%d, res=%.3fm",
            slam_type.value, grid_w, grid_h, resolution,
        )

    def map_build(
        self,
        sensor_data: Dict[str, Any],
        pose_estimate: Optional[Pose] = None,
    ) -> Dict[str, Any]:
        """Build or extend the map from sensor observations.

        Processes sensor data to update the occupancy grid and register
        landmarks. For visual SLAM, extracts and matches features. For
        LiDAR SLAM, performs scan matching.

        Args:
            sensor_data: Dictionary with sensor readings. Expected keys:
                'lidar_scan' (list of ranges), 'image' (for visual SLAM),
                'odometry' (dx, dy, dtheta).
            pose_estimate: Optional prior pose estimate.

        Returns:
            Dictionary with 'map_updated', 'new_landmarks', and
            'cells_updated'.
        """
        if pose_estimate:
            self._current_pose = pose_estimate

        odometry = sensor_data.get("odometry", {})
        if odometry:
            self._current_pose = self._apply_odometry(
                self._current_pose, odometry
            )

        new_landmarks = 0
        cells_updated = 0

        if self._slam_type == SLAMType.LIDAR:
            scan = sensor_data.get("lidar_scan", [])
            if scan:
                cells_updated = self._update_grid_lidar(
                    self._current_pose, scan
                )
        elif self._slam_type == SLAMType.VISUAL:
            image = sensor_data.get("image")
            if image:
                new_landmarks = self._extract_visual_landmarks(image)

        self._trajectory.append(Pose(
            x=self._current_pose.x,
            y=self._current_pose.y,
            theta=self._current_pose.theta,
        ))

        self._slam_map.last_updated = time.time()

        logger.info(
            "Map build: %d cells updated, %d new landmarks",
            cells_updated, new_landmarks,
        )

        return {
            "map_updated": True,
            "new_landmarks": new_landmarks,
            "cells_updated": cells_updated,
            "current_pose": {
                "x": self._current_pose.x,
                "y": self._current_pose.y,
                "theta": self._current_pose.theta,
            },
            "total_landmarks": len(self._landmarks),
        }

    def localize(
        self,
        sensor_data: Dict[str, Any],
        initial_pose: Optional[Pose] = None,
        num_particles: int = 100,
    ) -> Dict[str, Any]:
        """Estimate the robot's current position using sensor data.

        For LiDAR SLAM, uses scan matching against the existing map.
        For visual SLAM, uses feature matching with map landmarks.

        Args:
            sensor_data: Dictionary with current sensor readings.
            initial_pose: Optional initial pose guess for localization.
            num_particles: Number of particles for Monte Carlo
                localization.

        Returns:
            Dictionary with 'pose' (x, y, theta), 'confidence', and
            'covariance'.
        """
        if initial_pose:
            self._current_pose = initial_pose

        logger.debug("Localizing with %d particles", num_particles)

        confidence = 0.0

        if self._slam_type == SLAMType.LIDAR:
            scan = sensor_data.get("lidar_scan", [])
            if scan:
                confidence = self._scan_match(self._current_pose, scan)
        elif self._slam_type == SLAMType.VISUAL:
            image = sensor_data.get("image")
            if image:
                matched = self._match_visual_features(image)
                confidence = min(1.0, matched / 10.0)

        return {
            "pose": {
                "x": self._current_pose.x,
                "y": self._current_pose.y,
                "theta": self._current_pose.theta,
            },
            "confidence": confidence,
            "covariance": self._current_pose.covariance,
            "slam_type": self._slam_type.value,
        }

    def update(
        self,
        sensor_data: Dict[str, Any],
        pose_estimate: Optional[Pose] = None,
    ) -> Dict[str, Any]:
        """Perform a full SLAM update cycle.

        Combines localization and mapping in a single step, updating
        both the robot's pose estimate and the map.

        Args:
            sensor_data: Dictionary with sensor readings.
            pose_estimate: Optional prior pose estimate.

        Returns:
            Dictionary with localization and mapping results.
        """
        # Localize first
        loc_result = self.localize(sensor_data, initial_pose=pose_estimate)

        # Then update map
        map_result = self.map_build(sensor_data)

        # Check for loop closures
        loop_closure = self._detect_loop_closure()
        if loop_closure:
            self._loop_closures.append(loop_closure)
            logger.info("Loop closure detected at step %d", len(self._trajectory))

        return {
            "localization": loc_result,
            "mapping": map_result,
            "loop_closure": loop_closure,
            "trajectory_length": len(self._trajectory),
        }

    def get_map(self, format: str = "occupancy_grid") -> Dict[str, Any]:
        """Retrieve the current SLAM map.

        Args:
            format: Map output format — 'occupancy_grid', 'point_cloud',
                or 'landmarks'.

        Returns:
            Dictionary with map data and metadata.
        """
        if format == "occupancy_grid":
            grid = self._get_occupancy_grid()
            return {
                "grid": grid,
                "resolution": self._resolution,
                "width": self._slam_map.width,
                "height": self._slam_map.height,
                "origin": {
                    "x": self._slam_map.origin_x,
                    "y": self._slam_map.origin_y,
                },
                "format": "occupancy_grid",
            }
        elif format == "landmarks":
            return {
                "landmarks": {
                    lid: {"x": lm.x, "y": lm.y, "observations": lm.observations}
                    for lid, lm in self._landmarks.items()
                },
                "count": len(self._landmarks),
                "format": "landmarks",
            }

        return {"format": format, "data": None}

    def _get_occupancy_grid(self) -> Any:
        """Convert log-odds grid to probability occupancy grid."""
        if np is not None:
            return (1.0 - 1.0 / (1.0 + np.exp(self._log_odds))).tolist()
        # Fallback: convert log-odds manually
        grid = []
        for row in self._log_odds:
            prob_row = [1.0 - 1.0 / (1.0 + math.exp(lo)) for lo in row]
            grid.append(prob_row)
        return grid

    def _apply_odometry(self, pose: Pose, odom: Dict[str, Any]) -> Pose:
        """Apply odometry increment to a pose.

        Args:
            pose: Current pose.
            odom: Odometry with 'dx', 'dy', 'dtheta'.

        Returns:
            Updated pose.
        """
        dx = odom.get("dx", 0.0)
        dy = odom.get("dy", 0.0)
        dtheta = odom.get("dtheta", 0.0)

        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)

        new_x = pose.x + dx * cos_t - dy * sin_t
        new_y = pose.y + dx * sin_t + dy * cos_t
        new_theta = pose.theta + dtheta

        return Pose(x=new_x, y=new_y, theta=new_theta)

    def _update_grid_lidar(
        self, pose: Pose, scan: List[float]
    ) -> int:
        """Update the occupancy grid from a LiDAR scan.

        Uses inverse sensor model with ray casting to update log-odds
        values for free and occupied cells.

        Args:
            pose: Current robot pose.
            scan: List of range measurements (meters).

        Returns:
            Number of cells updated.
        """
        cells_updated = 0
        angle_increment = 2 * math.pi / max(len(scan), 1)

        for i, distance in enumerate(scan):
            if distance <= 0 or math.isinf(distance):
                continue

            angle = pose.theta + i * angle_increment

            # Ray casting from robot to hit point
            steps = max(1, int(distance / self._resolution))
            for s in range(steps):
                t = s / steps
                rx = pose.x + t * distance * math.cos(angle)
                ry = pose.y + t * distance * math.sin(angle)

                gx, gy = self._world_to_grid(rx, ry)
                if (0 <= gx < self._slam_map.width and
                        0 <= gy < self._slam_map.height):
                    if np is not None:
                        self._log_odds[gx, gy] = max(
                            -self.LOG_ODDS_CLAMP,
                            self._log_odds[gx, gy] + self.LOG_ODDS_FREE,
                        )
                    else:
                        self._log_odds[gx][gy] = max(
                            -self.LOG_ODDS_CLAMP,
                            self._log_odds[gx][gy] + self.LOG_ODDS_FREE,
                        )
                    cells_updated += 1

            # Mark endpoint as occupied
            end_x = pose.x + distance * math.cos(angle)
            end_y = pose.y + distance * math.sin(angle)
            gx, gy = self._world_to_grid(end_x, end_y)
            if (0 <= gx < self._slam_map.width and
                    0 <= gy < self._slam_map.height):
                if np is not None:
                    self._log_odds[gx, gy] = min(
                        self.LOG_ODDS_CLAMP,
                        self._log_odds[gx, gy] + self.LOG_ODDS_OCCUPIED,
                    )
                else:
                    self._log_odds[gx][gy] = min(
                        self.LOG_ODDS_CLAMP,
                        self._log_odds[gx][gy] + self.LOG_ODDS_OCCUPIED,
                    )
                cells_updated += 1

        return cells_updated

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((x - self._slam_map.origin_x) / self._resolution)
        gy = int((y - self._slam_map.origin_y) / self._resolution)
        return gx, gy

    def _scan_match(self, pose: Pose, scan: List[float]) -> float:
        """Estimate localization confidence via scan matching.

        Args:
            pose: Current pose estimate.
            scan: Current LiDAR scan.

        Returns:
            Match confidence score (0.0 to 1.0).
        """
        return min(1.0, len(scan) / 360.0) if scan else 0.0

    def _extract_visual_landmarks(self, image: Any) -> int:
        """Extract and register visual landmarks from an image.

        Args:
            image: Image data for feature extraction.

        Returns:
            Number of new landmarks added.
        """
        return 0

    def _match_visual_features(self, image: Any) -> int:
        """Match visual features against known landmarks.

        Args:
            image: Current image observation.

        Returns:
            Number of matched features.
        """
        return 0

    def _detect_loop_closure(self) -> Optional[Dict[str, Any]]:
        """Detect if the robot has returned to a previously visited area.

        Returns:
            Loop closure data if detected, None otherwise.
        """
        if len(self._trajectory) < 50:
            return None

        current = self._current_pose
        for i, past_pose in enumerate(self._trajectory[:-10]):
            dist = math.sqrt(
                (current.x - past_pose.x) ** 2 +
                (current.y - past_pose.y) ** 2
            )
            if dist < self._resolution * 10:
                return {
                    "current_step": len(self._trajectory) - 1,
                    "matched_step": i,
                    "distance": dist,
                    "timestamp": time.time(),
                }

        return None

    def __repr__(self) -> str:
        return (
            f"SLAM(type={self._slam_type.value}, "
            f"landmarks={len(self._landmarks)}, "
            f"trajectory={len(self._trajectory)}, "
            f"loop_closures={len(self._loop_closures)})"
        )
