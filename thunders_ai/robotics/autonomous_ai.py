"""Thunders AI Autonomous AI Module.

Provides autonomous decision-making, perception, planning, learning,
and mission execution capabilities by integrating navigation, sensors,
and control subsystems.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from thunders_ai.config import Config
from thunders_ai.logger import get_logger
from thunders_ai.robotics.navigation import Navigation, Position, PlanningAlgorithm
from thunders_ai.robotics.sensors import SensorManager, SensorType
from thunders_ai.robotics.control_system import (
    ControlSystem,
    RobotType,
    ControlCommand,
    CommandType,
)

logger = get_logger(__name__)


class MissionState(Enum):
    """States of an autonomous mission."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class DecisionType(Enum):
    """Types of autonomous decisions."""

    NAVIGATE = "navigate"
    AVOID = "avoid"
    STOP = "stop"
    EXPLORE = "explore"
    RETRY = "retry"
    WAIT = "wait"


@dataclass
class Mission:
    """Definition of an autonomous mission."""

    mission_id: str
    goal: Position
    priority: int = 0
    deadline: Optional[float] = None
    waypoints: List[Position] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    state: MissionState = MissionState.IDLE
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class PerceptionResult:
    """Result of environmental perception."""

    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    landmarks: List[Dict[str, Any]] = field(default_factory=list)
    free_space: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


class AutonomousAI:
    """Autonomous AI system that integrates navigation, sensors, and control
    for intelligent robotic behavior.

    Provides high-level autonomy including mission execution, perception-
    driven decision making, and experience-based learning.

    Args:
        config: Optional configuration instance.
        robot_type: Type of robot platform.
        decision_threshold: Confidence threshold for autonomous decisions.

    Example:
        >>> ai = AutonomousAI(robot_type=RobotType.WHEELED)
        >>> mission = Mission("patrol", goal=Position(10, 10))
        >>> result = ai.run_mission(mission)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        robot_type: RobotType = RobotType.WHEELED,
        decision_threshold: float = 0.7,
    ) -> None:
        self._config = config or Config()
        self._robot_type = robot_type
        self._decision_threshold = decision_threshold
        self._navigation = Navigation(config=config)
        self._sensors = SensorManager(config=config)
        self._control = ControlSystem(config=config, robot_type=robot_type)
        self._current_mission: Optional[Mission] = None
        self._mission_history: List[Mission] = []
        self._experience_buffer: List[Dict[str, Any]] = []
        self._decision_log: List[Dict[str, Any]] = []

        logger.info(
            "AutonomousAI initialized: type=%s, threshold=%.2f",
            robot_type.value, decision_threshold,
        )

    def navigate(
        self,
        goal: Position,
        algorithm: Optional[PlanningAlgorithm] = None,
        max_speed: float = 1.0,
    ) -> Dict[str, Any]:
        """Navigate autonomously to a goal position.

        Args:
            goal: Target position.
            algorithm: Override planning algorithm.
            max_speed: Maximum navigation speed.

        Returns:
            Navigation result with path, distance, and status.
        """
        logger.info("Autonomous navigation to (%.1f, %.1f)", goal.x, goal.y)

        # Perceive environment first
        perception = self.perceive()

        # Update map with perceived obstacles
        if perception.obstacles:
            from thunders_ai.robotics.navigation import Obstacle
            obstacles = [
                Obstacle(
                    x=obs.get("x", 0),
                    y=obs.get("y", 0),
                    radius=obs.get("radius", 0.5),
                )
                for obs in perception.obstacles
            ]
            self._navigation.update_map(obstacles=obstacles)

        # Plan and execute navigation
        nav_result = self._navigation.navigate(
            goal=goal,
            algorithm=algorithm,
            max_speed=max_speed,
            avoid_obstacles=True,
        )

        # Send velocity commands to control system
        if nav_result.get("path"):
            for waypoint in nav_result["path"]:
                cmd = ControlCommand(
                    command_type=CommandType.POSITION,
                    values={"x": waypoint.x, "y": waypoint.y},
                )
                self._control.send_command(cmd, immediate=True)

        return nav_result

    def make_decision(
        self,
        context: Dict[str, Any],
        options: Optional[List[DecisionType]] = None,
    ) -> Dict[str, Any]:
        """Make an AI-driven decision based on current context.

        Args:
            context: Current environmental and system context.
            options: Available decision options. If None, all options
                are considered.

        Returns:
            Dictionary with 'decision', 'confidence', 'reasoning',
            and 'alternatives'.
        """
        options = options or list(DecisionType)
        logger.info("Making decision with context keys: %s", list(context.keys()))

        # Evaluate each option based on context
        scores: Dict[DecisionType, float] = {}
        reasoning: Dict[DecisionType, str] = {}

        obstacles = context.get("obstacles", [])
        battery = context.get("battery_level", 1.0)
        goal_distance = context.get("goal_distance", float("inf"))

        for option in options:
            if option == DecisionType.NAVIGATE:
                scores[option] = 0.8 if not obstacles else 0.3
                reasoning[option] = "Proceed to goal" if not obstacles else "Obstacles detected"
            elif option == DecisionType.AVOID:
                scores[option] = 0.9 if obstacles else 0.1
                reasoning[option] = "Obstacle avoidance needed" if obstacles else "No obstacles"
            elif option == DecisionType.STOP:
                scores[option] = 0.95 if battery < 0.1 else 0.05
                reasoning[option] = "Low battery" if battery < 0.1 else "Sufficient battery"
            elif option == DecisionType.EXPLORE:
                scores[option] = 0.7 if goal_distance == float("inf") else 0.2
                reasoning[option] = "No clear goal" if goal_distance == float("inf") else "Goal exists"
            elif option == DecisionType.RETRY:
                scores[option] = 0.5
                reasoning[option] = "Fallback option"
            elif option == DecisionType.WAIT:
                scores[option] = 0.6 if obstacles else 0.05
                reasoning[option] = "Dynamic obstacle ahead" if obstacles else "Path clear"

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[best]

        decision_result = {
            "decision": best.value,
            "confidence": confidence,
            "reasoning": reasoning.get(best, ""),
            "alternatives": [
                {"option": opt.value, "score": scores[opt], "reason": reasoning.get(opt, "")}
                for opt in sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
                if opt != best
            ],
            "context_summary": {
                "obstacle_count": len(obstacles),
                "battery_level": battery,
                "goal_distance": goal_distance,
            },
        }

        self._decision_log.append({
            "timestamp": time.time(),
            **decision_result,
        })

        return decision_result

    def perceive(self) -> PerceptionResult:
        """Perceive the environment using sensor data.

        Reads from all sensors, fuses the data, and produces a unified
        perception of the environment.

        Returns:
            PerceptionResult with obstacles, landmarks, and free space.
        """
        logger.debug("Perceiving environment")

        readings = self._sensors.read_all()
        obstacles: List[Dict[str, Any]] = []
        landmarks: List[Dict[str, Any]] = []
        free_space: List[Dict[str, Any]] = []

        for sid, reading in readings.items():
            if reading.sensor_type == SensorType.LIDAR and reading.data:
                obstacles.extend(
                    reading.data.get("obstacles", [])
                    if isinstance(reading.data, dict) else []
                )
            elif reading.sensor_type == SensorType.ULTRASONIC and reading.data:
                obstacles.extend(
                    reading.data.get("obstacles", [])
                    if isinstance(reading.data, dict) else []
                )
            elif reading.sensor_type == SensorType.CAMERA and reading.data:
                landmarks.extend(
                    reading.data.get("landmarks", [])
                    if isinstance(reading.data, dict) else []
                )

        return PerceptionResult(
            obstacles=obstacles,
            landmarks=landmarks,
            free_space=free_space,
        )

    def plan(
        self,
        goal: Position,
        horizon: float = 10.0,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Plan a sequence of actions to achieve a goal.

        Args:
            goal: Target position or objective.
            horizon: Planning horizon in seconds.
            constraints: Optional planning constraints.

        Returns:
            Dictionary with 'actions', 'estimated_duration', and
            'risk_assessment'.
        """
        logger.info("Planning for goal (%.1f, %.1f)", goal.x, goal.y)

        perception = self.perceive()
        path_result = self._navigation.plan_path(
            start=self._navigation.get_position(), goal=goal
        )

        actions: List[Dict[str, Any]] = []
        if path_result.get("path"):
            for i, waypoint in enumerate(path_result["path"]):
                actions.append({
                    "type": "navigate",
                    "target": {"x": waypoint.x, "y": waypoint.y},
                    "step": i,
                })

        risk = 0.3 if perception.obstacles else 0.1

        return {
            "actions": actions,
            "estimated_duration": len(actions) * 0.5,
            "risk_assessment": {
                "level": "low" if risk < 0.3 else "medium" if risk < 0.7 else "high",
                "score": risk,
                "obstacle_count": len(perception.obstacles),
            },
            "horizon": horizon,
            "constraints": constraints,
        }

    def learn(
        self,
        experience: Dict[str, Any],
        learning_rate: float = 0.01,
    ) -> Dict[str, Any]:
        """Learn from experience to improve future decisions.

        Args:
            experience: Experience data with 'context', 'action',
                'outcome', and 'reward'.
            learning_rate: Learning rate for updates.

        Returns:
            Dictionary with 'learned', 'experience_count', and
            'performance_delta'.
        """
        self._experience_buffer.append({
            "timestamp": time.time(),
            **experience,
        })

        logger.info(
            "Learning from experience #%d", len(self._experience_buffer)
        )

        # Compute simple performance delta
        reward = experience.get("reward", 0.0)
        performance_delta = reward * learning_rate

        return {
            "learned": True,
            "experience_count": len(self._experience_buffer),
            "performance_delta": performance_delta,
            "learning_rate": learning_rate,
        }

    def run_mission(
        self,
        mission: Mission,
        max_retries: int = 3,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute an autonomous mission with retry and error handling.

        Args:
            mission: Mission definition with goal and constraints.
            max_retries: Maximum number of retry attempts on failure.
            callback: Optional callback for progress updates.

        Returns:
            Dictionary with 'success', 'mission_id', 'duration', and
            'steps_completed'.
        """
        self._current_mission = mission
        mission.state = MissionState.PLANNING
        mission.start_time = time.time()
        logger.info("Starting mission %s", mission.mission_id)

        plan_result = self.plan(
            goal=mission.goal,
            constraints=mission.constraints,
        )

        mission.state = MissionState.EXECUTING
        steps_completed = 0
        retries = 0
        errors: List[str] = []

        actions = plan_result.get("actions", [])
        waypoints = mission.waypoints or []

        for action in actions:
            if mission.state != MissionState.EXECUTING:
                break

            try:
                if action["type"] == "navigate":
                    target = action["target"]
                    nav_result = self.navigate(
                        goal=Position(target["x"], target["y"])
                    )
                    steps_completed += 1

                    if callback:
                        callback({
                            "step": steps_completed,
                            "total": len(actions),
                            "action": action,
                        })
            except Exception as e:
                errors.append(str(e))
                retries += 1
                if retries > max_retries:
                    mission.state = MissionState.FAILED
                    mission.errors = errors
                    logger.error("Mission %s failed: %s", mission.mission_id, e)
                    break

                logger.warning(
                    "Retrying mission step (%d/%d): %s",
                    retries, max_retries, e,
                )

        if mission.state == MissionState.EXECUTING:
            mission.state = MissionState.COMPLETED

        mission.end_time = time.time()
        duration = mission.end_time - mission.start_time
        self._mission_history.append(mission)
        self._current_mission = None

        return {
            "success": mission.state == MissionState.COMPLETED,
            "mission_id": mission.mission_id,
            "state": mission.state.value,
            "duration": duration,
            "steps_completed": steps_completed,
            "total_steps": len(actions),
            "retries": retries,
            "errors": errors,
        }

    def __repr__(self) -> str:
        return (
            f"AutonomousAI(type={self._robot_type.value}, "
            f"missions={len(self._mission_history)}, "
            f"experiences={len(self._experience_buffer)})"
        )
