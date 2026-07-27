"""Thunders AI Robotics Module.

Provides comprehensive robotics capabilities including sensor management,
navigation, control systems, autonomous AI, SLAM, and simulation for
intelligent robotic applications.
"""

from thunders_ai.robotics.sensors import SensorManager
from thunders_ai.robotics.navigation import Navigation
from thunders_ai.robotics.control_system import ControlSystem
from thunders_ai.robotics.autonomous_ai import AutonomousAI
from thunders_ai.robotics.slam import SLAM
from thunders_ai.robotics.simulation import Simulation

__all__ = [
    "SensorManager",
    "Navigation",
    "ControlSystem",
    "AutonomousAI",
    "SLAM",
    "Simulation",
]

__version__ = "1.0.0"
