"""Thunders AI Simulation Module.

Provides simulation environment creation, stepping, rendering, object
management, and physics simulation for testing robotic algorithms and
behaviors in virtual environments.
"""

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


class PhysicsEngine(Enum):
    """Supported physics engine backends."""

    BUILTIN = "builtin"
    PYBULLET = "pybullet"
    MUJOCO = "mujoco"


class ObjectType(Enum):
    """Types of objects in the simulation."""

    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    PLANE = "plane"
    MESH = "mesh"
    ROBOT = "robot"


@dataclass
class SimObject:
    """An object in the simulation environment."""

    object_id: str
    object_type: ObjectType
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.3
    is_static: bool = False
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicsState:
    """State of the physics simulation at a given time step."""

    step: int = 0
    sim_time: float = 0.0
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    dt: float = 0.01
    object_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class Simulation:
    """Simulation engine for creating and running virtual robotic
    environments with physics simulation.

    Provides environment creation, time stepping, object management,
    rendering, and data recording for algorithm testing and validation.

    Args:
        config: Optional configuration instance.
        physics_engine: Physics backend to use.
        time_step: Simulation time step in seconds.
        gravity: Gravitational acceleration vector (x, y, z).

    Example:
        >>> sim = Simulation(time_step=0.01)
        >>> sim.create_env(name="warehouse", size=(20, 20, 5))
        >>> sim.add_object("box1", ObjectType.BOX, position=(5, 5, 0))
        >>> result = sim.step(num_steps=100)
        >>> sim.render()
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        physics_engine: PhysicsEngine = PhysicsEngine.BUILTIN,
        time_step: float = 0.01,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        self._config = config or Config()
        self._physics_engine = physics_engine
        self._time_step = time_step
        self._gravity = gravity
        self._objects: Dict[str, SimObject] = {}
        self._state = PhysicsState(
            dt=time_step,
            gravity=gravity,
        )
        self._env_name: Optional[str] = None
        self._env_size: Tuple[float, float, float] = (10.0, 10.0, 3.0)
        self._recording: List[Dict[str, Any]] = []
        self._is_recording = False
        self._render_enabled = True

        logger.info(
            "Simulation initialized: engine=%s, dt=%.4f, gravity=%s",
            physics_engine.value, time_step, gravity,
        )

    def create_env(
        self,
        name: str = "default",
        size: Tuple[float, float, float] = (10.0, 10.0, 3.0),
        add_floor: bool = True,
        ambient_light: float = 0.6,
    ) -> Dict[str, Any]:
        """Create a new simulation environment.

        Args:
            name: Environment name identifier.
            size: Environment dimensions (width, depth, height) in meters.
            add_floor: Whether to add a ground plane.
            ambient_light: Ambient light intensity (0.0 to 1.0).

        Returns:
            Dictionary with environment configuration details.
        """
        self._env_name = name
        self._env_size = size
        self._objects.clear()
        self._state = PhysicsState(dt=self._time_step, gravity=self._gravity)

        if add_floor:
            floor = SimObject(
                object_id="floor",
                object_type=ObjectType.PLANE,
                position=(0.0, 0.0, 0.0),
                is_static=True,
                size=(size[0], size[1], 0.01),
                color=(0.4, 0.4, 0.4),
            )
            self._objects["floor"] = floor

        logger.info(
            "Created environment %s: size=%s, floor=%s",
            name, size, add_floor,
        )

        return {
            "name": name,
            "size": size,
            "floor_added": add_floor,
            "ambient_light": ambient_light,
            "physics_engine": self._physics_engine.value,
            "time_step": self._time_step,
        }

    def step(self, num_steps: int = 1) -> Dict[str, Any]:
        """Advance the simulation by a number of time steps.

        Args:
            num_steps: Number of simulation steps to execute.

        Returns:
            Dictionary with 'step', 'sim_time', and 'object_states'.
        """
        if not self._objects:
            logger.warning("No objects in simulation; step has no effect")

        for _ in range(num_steps):
            self._state.step += 1
            self._state.sim_time += self._time_step

            # Update physics for each dynamic object
            for obj_id, obj in self._objects.items():
                if obj.is_static:
                    continue
                self._update_object_physics(obj)

            # Record if active
            if self._is_recording:
                self._record_frame()

        logger.debug(
            "Simulated %d steps: step=%d, time=%.3f",
            num_steps, self._state.step, self._state.sim_time,
        )

        return {
            "step": self._state.step,
            "sim_time": self._state.sim_time,
            "num_steps_executed": num_steps,
            "object_states": self._get_all_object_states(),
        }

    def reset(self, keep_objects: bool = False) -> Dict[str, Any]:
        """Reset the simulation to its initial state.

        Args:
            keep_objects: Whether to retain objects in the environment.

        Returns:
            Dictionary confirming the reset.
        """
        self._state = PhysicsState(dt=self._time_step, gravity=self._gravity)
        self._recording.clear()

        if not keep_objects:
            self._objects.clear()

        logger.info("Simulation reset: keep_objects=%s", keep_objects)

        return {
            "reset": True,
            "step": 0,
            "sim_time": 0.0,
            "objects_remaining": len(self._objects),
        }

    def render(
        self,
        mode: str = "rgb_array",
        camera_position: Optional[Tuple[float, float, float]] = None,
        camera_target: Optional[Tuple[float, float, float]] = None,
        width: int = 640,
        height: int = 480,
    ) -> Dict[str, Any]:
        """Render the current simulation state for visualization.

        Args:
            mode: Render mode — 'rgb_array', 'depth', or 'human'.
            camera_position: Camera position (x, y, z).
            camera_target: Camera look-at target (x, y, z).
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Dictionary with 'image', 'mode', and dimensions.
        """
        cam_pos = camera_position or (
            self._env_size[0] / 2,
            self._env_size[1] / 2,
            self._env_size[2] * 2,
        )
        cam_target = camera_target or (
            self._env_size[0] / 2,
            self._env_size[1] / 2,
            0.0,
        )

        logger.debug(
            "Rendering: mode=%s, step=%d, objects=%d",
            mode, self._state.step, len(self._objects),
        )

        image_data: Optional[Any] = None
        if mode == "rgb_array" and np is not None:
            image_data = np.zeros((height, width, 3), dtype=np.uint8)
            image_data[:, :] = [180, 200, 220]  # sky blue
        elif mode == "depth" and np is not None:
            image_data = np.ones((height, width), dtype=np.float32)

        return {
            "image": image_data,
            "mode": mode,
            "width": width,
            "height": height,
            "camera_position": cam_pos,
            "camera_target": cam_target,
            "sim_time": self._state.sim_time,
        }

    def add_object(
        self,
        object_id: str,
        object_type: ObjectType,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        mass: float = 1.0,
        friction: float = 0.5,
        restitution: float = 0.3,
        is_static: bool = False,
        size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add an object to the simulation environment.

        Args:
            object_id: Unique identifier for the object.
            object_type: Type of the object.
            position: Initial position (x, y, z).
            orientation: Initial orientation as quaternion (x, y, z, w).
            mass: Mass in kilograms.
            friction: Surface friction coefficient.
            restitution: Bounciness coefficient (0.0 to 1.0).
            is_static: Whether the object is immovable.
            size: Object dimensions (length, width, height).
            color: RGB color (0.0 to 1.0 per channel).
            velocity: Initial velocity (vx, vy, vz).
            metadata: Additional object metadata.

        Returns:
            Dictionary with added object details.
        """
        if object_id in self._objects:
            raise ValueError(f"Object {object_id!r} already exists")

        obj = SimObject(
            object_id=object_id,
            object_type=object_type,
            position=position,
            orientation=orientation,
            velocity=velocity,
            mass=mass,
            friction=friction,
            restitution=restitution,
            is_static=is_static,
            size=size,
            color=color,
            metadata=metadata or {},
        )
        self._objects[object_id] = obj

        logger.info(
            "Added object %s (%s) at (%.1f, %.1f, %.1f)",
            object_id, object_type.value, position[0], position[1], position[2],
        )

        return {
            "object_id": object_id,
            "type": object_type.value,
            "position": position,
            "mass": mass,
            "is_static": is_static,
        }

    def remove_object(self, object_id: str) -> Dict[str, Any]:
        """Remove an object from the simulation.

        Args:
            object_id: ID of the object to remove.

        Returns:
            Dictionary confirming removal.
        """
        if object_id not in self._objects:
            raise KeyError(f"Object {object_id!r} not found")

        del self._objects[object_id]
        logger.info("Removed object %s", object_id)
        return {"removed": object_id}

    def start_recording(self) -> Dict[str, Any]:
        """Start recording simulation data.

        Returns:
            Dictionary confirming recording started.
        """
        self._is_recording = True
        self._recording.clear()
        logger.info("Recording started at step %d", self._state.step)
        return {
            "recording": True,
            "start_step": self._state.step,
            "start_time": self._state.sim_time,
        }

    def stop_recording(self) -> Dict[str, Any]:
        """Stop recording and return recorded data.

        Returns:
            Dictionary with recorded frames and metadata.
        """
        self._is_recording = False
        logger.info(
            "Recording stopped: %d frames captured", len(self._recording)
        )
        return {
            "recording": False,
            "frames": len(self._recording),
            "duration": self._state.sim_time,
            "data": self._recording,
        }

    def _update_object_physics(self, obj: SimObject) -> None:
        """Update an object's physics state for one time step.

        Applies gravity, updates velocity and position using Euler
        integration, and handles basic floor collision.

        Args:
            obj: SimObject to update.
        """
        vx, vy, vz = obj.velocity
        gz = self._gravity[2]

        # Apply gravity
        vz += gz * self._time_step

        # Update position
        px, py, pz = obj.position
        px += vx * self._time_step
        py += vy * self._time_step
        pz += vz * self._time_step

        # Floor collision (simple bounce)
        half_height = obj.size[2] / 2.0 if obj.object_type != ObjectType.SPHERE else obj.size[0] / 2.0
        if pz - half_height < 0.0:
            pz = half_height
            vz = -vz * obj.restitution
            if abs(vz) < 0.01:
                vz = 0.0

        obj.position = (px, py, pz)
        obj.velocity = (vx, vy, vz)

    def _get_all_object_states(self) -> Dict[str, Dict[str, Any]]:
        """Get the current state of all objects."""
        states: Dict[str, Dict[str, Any]] = {}
        for oid, obj in self._objects.items():
            states[oid] = {
                "position": obj.position,
                "orientation": obj.orientation,
                "velocity": obj.velocity,
                "type": obj.object_type.value,
                "is_static": obj.is_static,
            }
        return states

    def _record_frame(self) -> None:
        """Record a single frame of simulation data."""
        frame: Dict[str, Any] = {
            "step": self._state.step,
            "sim_time": self._state.sim_time,
            "objects": self._get_all_object_states(),
        }
        self._recording.append(frame)

    def __repr__(self) -> str:
        return (
            f"Simulation(engine={self._physics_engine.value}, "
            f"step={self._state.step}, "
            f"time={self._state.sim_time:.2f}, "
            f"objects={len(self._objects)})"
        )
