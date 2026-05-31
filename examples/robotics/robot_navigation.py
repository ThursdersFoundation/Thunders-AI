"""Robot navigation example using Thunders AI Robotics.

Demonstrates how to set up a robot with sensors, plan a navigation path,
and execute the navigation with obstacle avoidance and progress monitoring.
"""

import time
from thunders_ai.robotics import Robot, NavigationPlanner, SensorConfig


def main() -> None:
    """Run the robot navigation example."""
    # --- Step 1: Configure the robot with sensors ---
    # Set up LiDAR, camera, and IMU sensors
    sensor_config = SensorConfig(
        lidar_enabled=True,
        lidar_range_meters=30.0,
        camera_enabled=True,
        imu_enabled=True,
        ultrasonic_enabled=True,
    )
    robot = Robot(
        name="ThunderBot-01",
        sensor_config=sensor_config,
    )
    print(f"Robot '{robot.name}' initialized with sensors:")
    print(f"  LiDAR:  {'ON' if sensor_config.lidar_enabled else 'OFF'} "
          f"(range: {sensor_config.lidar_range_meters}m)")
    print(f"  Camera: {'ON' if sensor_config.camera_enabled else 'OFF'}")
    print(f"  IMU:    {'ON' if sensor_config.imu_enabled else 'OFF'}")
    print()

    # --- Step 2: Connect to the robot ---
    robot.connect()
    print(f"Robot connected. Status: {robot.status}")
    print()

    # --- Step 3: Define navigation goals ---
    # Set the target position and orientation
    target = {"x": 10.0, "y": 5.0, "z": 0.0, "yaw": 1.57}
    print(f"Navigation target: x={target['x']}, y={target['y']}, yaw={target['yaw']}")
    print()

    # --- Step 4: Plan the navigation path ---
    # Use A* path planner with obstacle avoidance
    planner = NavigationPlanner(
        algorithm="a_star",
        resolution=0.1,          # Grid resolution in meters
        obstacle_margin=0.5,     # Safety margin around obstacles
        smooth_path=True,        # Smooth the planned path
    )

    # Get current sensor data for planning
    sensor_data = robot.get_sensor_readings()
    print(f"Sensor data received: {len(sensor_data)} data points")

    # Plan the path
    path = planner.plan(
        start=robot.position,
        goal=target,
        occupancy_map=sensor_data.get("lidar_map"),
    )
    print(f"Path planned: {len(path.waypoints)} waypoints, "
          f"{path.total_distance:.2f}m total distance, "
          f"~{path.estimated_time:.1f}s estimated time")
    print()

    # --- Step 5: Execute navigation ---
    # Navigate with real-time progress updates
    print("Starting navigation...")
    robot.navigate(path, speed=1.0, avoid_obstacles=True)

    # Monitor navigation progress
    while robot.is_navigating:
        progress = robot.get_navigation_progress()
        print(f"  Progress: {progress.percent_complete:.0f}% | "
              f"Distance remaining: {progress.distance_remaining:.2f}m | "
              f"Speed: {progress.current_speed:.2f}m/s")
        time.sleep(0.5)

    print(f"Navigation complete! Final position: {robot.position}")
    print()

    # --- Step 6: Return to home position ---
    print("Planning return path to origin...")
    return_path = planner.plan(
        start=robot.position,
        goal={"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        occupancy_map=sensor_data.get("lidar_map"),
    )
    robot.navigate(return_path, speed=0.8)
    print("Robot returning to home position...")

    # --- Step 7: Disconnect ---
    robot.disconnect()
    print("Robot disconnected.")


if __name__ == "__main__":
    main()
