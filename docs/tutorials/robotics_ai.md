# Robotics AI Tutorial

This tutorial guides you through building autonomous robotic systems with Thunders AI. You'll learn how to set up navigation, implement SLAM (Simultaneous Localization and Mapping), control drones, and integrate AI-powered decision-making into physical robots. Whether you're building a warehouse robot, a delivery drone, or a research platform, Thunders AI's robotics module provides the tools for perception, planning, and control.

## What You'll Build

An autonomous navigation system that:
- Uses SLAM to build maps of unknown environments
- Plans obstacle-free paths between waypoints
- Processes sensor data (LiDAR, cameras, IMU) for perception
- Controls robot actuators with smooth, safe motion
- Integrates LLM-based reasoning for high-level decision making

## Prerequisites

- Python 3.9+
- Thunders AI with robotics extras: `pip install thunders-ai[robotics]`
- A robot simulator (Gazebo or Webots recommended for testing without hardware)
- Optional: Physical robot with ROS2-compatible sensors and actuators

## Step 1: Setting Up the Navigation Stack

The navigation stack is the core of any mobile robot. It combines localization, mapping, and path planning:

```python
from thunders_ai.robotics import NavigationStack, RobotConfig

# Configure your robot
robot_config = RobotConfig(
    name="thunder_bot",
    wheel_base=0.5,        # Distance between wheels (meters)
    max_speed=1.0,         # Maximum linear speed (m/s)
    max_angular_speed=2.0, # Maximum rotation speed (rad/s)
    safety_margin=0.3,     # Obstacle avoidance distance (meters)
)

# Initialize the navigation stack
nav = NavigationStack(config=robot_config)

# Set a goal position
nav.set_goal(x=5.0, y=3.0, theta=0.0)

# Start autonomous navigation
nav.start()
```

## Step 2: Simultaneous Localization and Mapping (SLAM)

SLAM allows the robot to build a map of its environment while simultaneously tracking its own position within that map:

```python
from thunders_ai.robotics import SLAMEngine, SensorConfig

# Configure sensors
sensor_config = SensorConfig(
    lidar_topic="/scan",
    camera_topic="/camera/image_raw",
    imu_topic="/imu/data",
    odometry_topic="/odom",
)

# Initialize SLAM
slam = SLAMEngine(sensor_config=sensor_config)

# Start mapping
slam.start_mapping()

# The SLAM engine continuously processes sensor data
# and updates the map. Access the current map:
current_map = slam.get_map()
print(f"Map size: {current_map.width}x{current_map.height}")
print(f"Explored area: {current_map.explored_percentage:.1f}%")

# Get the robot's current pose
pose = slam.get_pose()
print(f"Position: ({pose.x:.2f}, {pose.y:.2f}), Heading: {pose.theta:.2f} rad")

# Save the map for later use
slam.save_map("warehouse_map.yaml")
```

## Step 3: Path Planning and Obstacle Avoidance

Plan efficient, collision-free paths through the mapped environment:

```python
from thunders_ai.robotics import PathPlanner, PlannerConfig

planner_config = PlannerConfig(
    algorithm="a_star",    # Options: a_star, rrt, rrt_star, dijkstra
    resolution=0.05,       # Grid resolution in meters
    inflation_radius=0.3,  # Inflate obstacles by this radius
)

planner = PathPlanner(config=planner_config, map=slam.get_map())

# Plan a path from current position to a goal
path = planner.plan(start=slam.get_pose(), goal=(10.0, 5.0))

if path.is_valid:
    print(f"Path found: {len(path.waypoints)} waypoints, {path.length:.1f}m")
    for i, wp in enumerate(path.waypoints):
        print(f"  Waypoint {i}: ({wp.x:.2f}, {wp.y:.2f})")
else:
    print("No valid path found!")

# Dynamic replanning when obstacles are detected
def on_obstacle_detected(obstacle):
    print(f"New obstacle at {obstacle.position}, replanning...")
    new_path = planner.replan(
        start=slam.get_pose(),
        goal=nav.current_goal,
        new_obstacle=obstacle,
    )
    nav.update_path(new_path)
```

## Step 4: Drone Control

Thunders AI also supports aerial robotics. Here's how to control a drone:

```python
from thunders_ai.robotics import DroneController, DroneConfig

drone_config = DroneConfig(
    vehicle_type="quadrotor",
    max_altitude=50.0,    # Maximum altitude (meters)
    max_speed=10.0,       # Maximum horizontal speed (m/s)
    geofence_radius=100,  # Stay within this radius (meters)
)

drone = DroneController(config=drone_config)

# Pre-flight checks
if drone.pre_flight_check():
    # Take off
    drone.takeoff(altitude=5.0)

    # Navigate to waypoints
    drone.goto(latitude=37.7749, longitude=-122.4194, altitude=10.0)

    # Execute a survey pattern
    drone.survey_pattern(
        center=(37.7749, -122.4194),
        radius=50.0,
        altitude=15.0,
        pattern="lawnmower",  # or "spiral", "crosshatch"
    )

    # Return to launch point and land
    drone.return_to_launch()
else:
    print("Pre-flight check failed!")
```

## Step 5: AI-Powered Decision Making

Combine the robotics stack with Thunders AI's LLM for intelligent, context-aware decision making:

```python
from thunders_ai import ThundersAI

ai = ThundersAI()

# Describe the robot's situation in natural language
situation = """
The robot is at position (5.2, 3.1) in the warehouse.
Three pallets are blocking the main corridor.
The left aisle is clear but 20m longer.
Battery level is at 45%.
The delivery deadline is in 15 minutes.
"""

decision = ai.chat(
    f"You are controlling a warehouse robot. Based on this situation, "
    f"what should the robot do? Choose the optimal path considering "
    f"battery and time constraints.\n\n{situation}"
)

print(f"AI Decision: {decision}")
# Example output: "Take the left aisle. The extra 20m adds ~30 seconds, 
# but the blocked corridor would require waiting for pallet removal 
# which could take 5+ minutes. Battery at 45% is sufficient for the detour."
```

## Step 6: Putting It All Together

Here's a complete autonomous robot controller:

```python
"""Autonomous robot controller using Thunders AI."""
from thunders_ai.robotics import NavigationStack, SLAMEngine, PathPlanner
from thunders_ai import ThundersAI, ThundersConfig

class AutonomousRobot:
    def __init__(self, robot_config, sensor_config):
        self.slam = SLAMEngine(sensor_config=sensor_config)
        self.planner = PathPlanner(map=self.slam.get_map())
        self.nav = NavigationStack(config=robot_config)
        self.ai = ThundersAI(config=ThundersConfig(
            system_prompt="You are an autonomous robot decision maker."
        ))

    def run_mission(self, waypoints):
        """Navigate through a list of waypoints autonomously."""
        self.slam.start_mapping()

        for i, goal in enumerate(waypoints):
            print(f"\n--- Heading to waypoint {i+1}/{len(waypoints)} ---")
            path = self.planner.plan(start=self.slam.get_pose(), goal=goal)
            self.nav.follow_path(path)

            while not self.nav.has_reached_goal():
                # Check for obstacles and replan if needed
                obstacles = self.slam.detect_new_obstacles()
                if obstacles:
                    new_path = self.planner.replan(
                        start=self.slam.get_pose(), goal=goal,
                        new_obstacles=obstacles,
                    )
                    self.nav.update_path(new_path)

        self.slam.save_map("mission_map.yaml")
        print("Mission complete!")

if __name__ == "__main__":
    robot = AutonomousRobot(
        robot_config=RobotConfig(name="scout"),
        sensor_config=SensorConfig(lidar_topic="/scan"),
    )
    waypoints = [(5, 0), (5, 5), (0, 5), (0, 0)]
    robot.run_mission(waypoints)
```

## Simulation vs. Hardware

| Feature | Simulation (Gazebo) | Real Hardware |
|---------|---------------------|---------------|
| Setup | Quick, no hardware needed | Requires physical robot |
| Safety | No damage risk | Implement safety stops |
| Sensors | Perfect data | Noisy, needs filtering |
| Latency | Predictable | Variable |
| Cost | Free | Hardware costs |

Always test thoroughly in simulation before deploying to real robots.

## Best Practices

- **Always implement safety stops** — hardware emergency stops are mandatory for physical robots
- **Test in simulation first** — validate all navigation logic before deploying on hardware
- **Use sensor fusion** — combine LiDAR, camera, and IMU data for robust perception
- **Monitor battery levels** — plan return-to-base paths when battery is low
- **Log all sensor data** — invaluable for debugging and improving navigation algorithms

## Next Steps

- Add **computer vision** for object recognition and manipulation
- Implement **multi-robot coordination** for fleet management
- Build **custom simulation environments** for your specific use case
- Integrate with **cloud services** for remote monitoring and fleet analytics
