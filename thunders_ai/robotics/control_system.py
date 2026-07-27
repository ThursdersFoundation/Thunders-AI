"""Thunders AI Control System Module.

Provides motor control, PID regulation, command dispatch, state monitoring,
and emergency stop capabilities for multiple robot types.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from thunders_ai.config import Config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class RobotType(Enum):
    """Supported robot platform types."""

    WHEELED = "wheeled"
    DRONE = "drone"
    ARM = "arm"


class CommandType(Enum):
    """Types of control commands."""

    VELOCITY = "velocity"
    POSITION = "position"
    TORQUE = "torque"
    ANGLE = "angle"
    STOP = "stop"


class SystemState(Enum):
    """Control system operating states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"


@dataclass
class PIDGains:
    """PID controller gains and limits."""

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -float("inf")
    output_max: float = float("inf")
    integral_min: float = -float("inf")
    integral_max: float = float("inf")


@dataclass
class ControlCommand:
    """A control command to be sent to actuators."""

    command_type: CommandType
    values: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0


@dataclass
class SystemStatus:
    """Current system status and telemetry."""

    state: SystemState = SystemState.IDLE
    position: Dict[str, float] = field(default_factory=dict)
    velocity: Dict[str, float] = field(default_factory=dict)
    torque: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)


class PIDController:
    """Proportional-Integral-Derivative controller.

    Implements a standard PID control loop with anti-windup protection
    and configurable output limits.

    Args:
        gains: PIDGains configuration.
        dt: Time step for the controller (seconds).
    """

    def __init__(self, gains: PIDGains, dt: float = 0.01) -> None:
        self._gains = gains
        self._dt = dt
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._initialized = False

    def compute(self, setpoint: float, measurement: float) -> float:
        """Compute PID control output.

        Args:
            setpoint: Desired target value.
            measurement: Current measured value.

        Returns:
            Control output value.
        """
        error = setpoint - measurement

        # Proportional term
        p_term = self._gains.kp * error

        # Integral term with anti-windup
        self._integral += error * self._dt
        self._integral = max(
            self._gains.integral_min,
            min(self._gains.integral_max, self._integral),
        )
        i_term = self._gains.ki * self._integral

        # Derivative term
        d_term = 0.0
        if self._initialized:
            d_term = self._gains.kd * (error - self._prev_error) / self._dt
        self._prev_error = error
        self._initialized = True

        output = p_term + i_term + d_term
        output = max(self._gains.output_min, min(self._gains.output_max, output))
        return output

    def reset(self) -> None:
        """Reset the PID controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False


class ControlSystem:
    """Robotic control system for managing actuators and motion.

    Supports wheeled robots, drones, and robotic arms with PID control,
    command queuing, and safety mechanisms.

    Args:
        config: Optional configuration instance.
        robot_type: Type of robot platform.
        control_rate_hz: Control loop frequency in Hz.

    Example:
        >>> ctrl = ControlSystem(robot_type=RobotType.WHEELED)
        >>> ctrl.pid_control(target={"x": 5.0, "theta": 0.0})
        >>> ctrl.send_command(ControlCommand(CommandType.VELOCITY, {"v": 1.0}))
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        robot_type: RobotType = RobotType.WHEELED,
        control_rate_hz: float = 100.0,
    ) -> None:
        self._config = config or Config()
        self._robot_type = robot_type
        self._control_rate_hz = control_rate_hz
        self._dt = 1.0 / control_rate_hz
        self._status = SystemStatus()
        self._pid_controllers: Dict[str, PIDController] = {}
        self._command_queue: List[ControlCommand] = []
        self._emergency = False
        self._motor_limits = self._default_motor_limits()

        logger.info(
            "ControlSystem initialized: type=%s, rate=%.1f Hz",
            robot_type.value, control_rate_hz,
        )

    def _default_motor_limits(self) -> Dict[str, Dict[str, float]]:
        """Get default motor limits for the robot type."""
        limits: Dict[str, Dict[str, float]] = {
            RobotType.WHEELED.value: {
                "max_linear_velocity": 2.0,
                "max_angular_velocity": 3.14,
                "max_acceleration": 1.0,
            },
            RobotType.DRONE.value: {
                "max_linear_velocity": 5.0,
                "max_angular_velocity": 2.0,
                "max_altitude": 50.0,
                "max_climb_rate": 2.0,
            },
            RobotType.ARM.value: {
                "max_joint_velocity": 1.57,
                "max_joint_torque": 50.0,
                "max_reach": 1.5,
            },
        }
        return limits.get(self._robot_type.value, {})

    def execute(
        self,
        commands: List[ControlCommand],
        blocking: bool = True,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Execute a list of control commands.

        Args:
            commands: List of ControlCommand instances to execute.
            blocking: Whether to wait for completion.
            timeout: Maximum execution time in seconds.

        Returns:
            Dictionary with 'executed_count', 'success', and 'duration'.
        """
        if self._emergency:
            raise RuntimeError(
                "Cannot execute commands: emergency stop is active"
            )

        start_time = time.time()
        executed = 0
        errors: List[str] = []

        for cmd in commands:
            if self._emergency:
                logger.warning("Execution halted by emergency stop")
                break

            if time.time() - start_time > timeout:
                errors.append("Timeout exceeded")
                break

            try:
                self._apply_command(cmd)
                executed += 1
            except Exception as e:
                errors.append(str(e))
                logger.error("Command execution error: %s", e)

        duration = time.time() - start_time
        success = executed == len(commands) and not errors

        return {
            "executed_count": executed,
            "total_commands": len(commands),
            "success": success,
            "duration": duration,
            "errors": errors,
        }

    def pid_control(
        self,
        target: Dict[str, float],
        current: Optional[Dict[str, float]] = None,
        gains: Optional[Dict[str, PIDGains]] = None,
    ) -> Dict[str, Any]:
        """Apply PID control to reach target values.

        Args:
            target: Target values keyed by axis (e.g., 'x', 'theta').
            current: Current measured values. If None, uses internal state.
            gains: Optional per-axis PID gains. If None, creates defaults.

        Returns:
            Dictionary with 'outputs', 'errors', and 'converged'.
        """
        if self._emergency:
            raise RuntimeError("PID control unavailable: emergency stop active")

        current = current or self._status.position
        outputs: Dict[str, float] = {}
        errors: Dict[str, float] = {}
        converged = True

        for axis, setpoint in target.items():
            if axis not in self._pid_controllers:
                pid_gains = (gains or {}).get(axis, PIDGains())
                self._pid_controllers[axis] = PIDController(pid_gains, self._dt)

            measurement = current.get(axis, 0.0)
            output = self._pid_controllers[axis].compute(setpoint, measurement)
            outputs[axis] = output
            errors[axis] = setpoint - measurement

            if abs(errors[axis]) > 0.01:
                converged = False

        self._status.state = SystemState.RUNNING
        logger.debug("PID outputs: %s", outputs)

        return {
            "outputs": outputs,
            "errors": errors,
            "converged": converged,
            "target": target,
        }

    def send_command(
        self,
        command: ControlCommand,
        immediate: bool = True,
    ) -> Dict[str, Any]:
        """Send a control command to the actuators.

        Args:
            command: ControlCommand to send.
            immediate: If True, execute immediately; otherwise queue.

        Returns:
            Dictionary with 'sent', 'command_type', and 'values'.
        """
        if self._emergency and command.command_type != CommandType.STOP:
            raise RuntimeError("Cannot send command: emergency stop active")

        if immediate:
            self._apply_command(command)
        else:
            self._command_queue.append(command)

        return {
            "sent": True,
            "command_type": command.command_type.value,
            "values": command.values,
            "immediate": immediate,
        }

    def _apply_command(self, command: ControlCommand) -> None:
        """Apply a command to the system state.

        Args:
            command: The control command to apply.
        """
        if command.command_type == CommandType.STOP:
            self._status.velocity = {
                k: 0.0 for k in self._status.velocity
            }
            self._status.state = SystemState.IDLE
            logger.info("Stop command applied")
        elif command.command_type == CommandType.VELOCITY:
            for k, v in command.values.items():
                limit = self._motor_limits.get(f"max_{k}", float("inf"))
                self._status.velocity[k] = max(-limit, min(limit, v))
            self._status.state = SystemState.RUNNING
        elif command.command_type == CommandType.POSITION:
            self._status.position.update(command.values)
        elif command.command_type == CommandType.TORQUE:
            self._status.torque.update(command.values)

        self._status.timestamp = time.time()

    def get_state(self) -> SystemStatus:
        """Get the current system state and telemetry.

        Returns:
            SystemStatus with position, velocity, torque, and state.
        """
        self._status.timestamp = time.time()
        return self._status

    def emergency_stop(self, reason: str = "Manual trigger") -> Dict[str, Any]:
        """Trigger an emergency halt.

        Immediately stops all motion and prevents further command execution
        until explicitly reset.

        Args:
            reason: Description of why the emergency stop was triggered.

        Returns:
            Dictionary with 'stopped', 'reason', and 'timestamp'.
        """
        self._emergency = True
        self._status.state = SystemState.EMERGENCY_STOP
        self._status.velocity = {k: 0.0 for k in self._status.velocity}
        self._status.errors.append(reason)
        self._command_queue.clear()

        for pid in self._pid_controllers.values():
            pid.reset()

        logger.critical("EMERGENCY STOP: %s", reason)
        return {
            "stopped": True,
            "reason": reason,
            "timestamp": time.time(),
        }

    def reset_emergency(self) -> Dict[str, Any]:
        """Reset the emergency stop state.

        Returns:
            Dictionary confirming the reset.
        """
        self._emergency = False
        self._status.state = SystemState.IDLE
        self._status.errors.clear()
        logger.info("Emergency stop reset")
        return {"reset": True, "timestamp": time.time()}

    def __repr__(self) -> str:
        return (
            f"ControlSystem(type={self._robot_type.value}, "
            f"state={self._status.state.value}, "
            f"emergency={self._emergency})"
        )
