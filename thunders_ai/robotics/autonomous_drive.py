"""Autonomous driving example using Thunders AI Robotics.

Demonstrates how to set up an autonomous driving system with sensor fusion,
perception, planning, and control for a self-driving vehicle workflow.
"""

import time
from thunders_ai.robotics import AutonomousAI, VehicleConfig, SensorFusion


def main() -> None:
    """Run the autonomous driving example."""
    # --- Step 1: Configure the autonomous vehicle ---
    # Set up vehicle parameters and sensor suite
    vehicle_config = VehicleConfig(
        max_speed_mps=15.0,       # Maximum speed: 15 m/s (~54 km/h)
        min_speed_mps=0.5,        # Minimum speed: 0.5 m/s
        safety_margin_meters=2.0, # Safety buffer around obstacles
        steering_max_angle=35.0,  # Maximum steering angle in degrees
    )

    autonomous = AutonomousAI(
        vehicle_config=vehicle_config,
        perception_model="thunders-drive-v2",
        planning_model="thunders-nav",
    )
    print("Autonomous driving system initialized.")
    print(f"  Max speed: {vehicle_config.max_speed_mps} m/s")
    print(f"  Safety margin: {vehicle_config.safety_margin_meters} m")
    print(f"  Perception model: {autonomous.perception_model}")
    print()

    # --- Step 2: Set up sensor fusion ---
    # Combine data from multiple sensors for robust perception
    fusion = SensorFusion(
        cameras=3,            # Front, left, right cameras
        lidar_points=64,      # 64-channel LiDAR
        radar_enabled=True,   # Long-range radar
        gps_enabled=True,     # GPS positioning
        imu_enabled=True,     # Inertial measurement unit
        fusion_rate_hz=20,    # Sensor fusion at 20 Hz
    )
    autonomous.set_sensor_fusion(fusion)
    print("Sensor fusion configured:")
    print(f"  Cameras: {fusion.cameras}")
    print(f"  LiDAR: {fusion.lidar_points}-channel")
    print(f"  Radar: {'ON' if fusion.radar_enabled else 'OFF'}")
    print(f"  Fusion rate: {fusion.fusion_rate_hz} Hz")
    print()

    # --- Step 3: Initialize and calibrate ---
    autonomous.initialize()
    autonomous.calibrate_sensors()
    print("System calibrated and ready.")
    print()

    # --- Step 4: Define waypoints for the route ---
    # Set up a multi-waypoint route through an urban environment
    waypoints = [
        {"x": 0.0, "y": 0.0, "label": "Start"},
        {"x": 50.0, "y": 0.0, "label": "Straight section"},
        {"x": 50.0, "y": 30.0, "label": "Right turn"},
        {"x": 100.0, "y": 30.0, "label": "Main avenue"},
        {"x": 100.0, "y": 60.0, "label": "Destination"},
    ]
    print("Route defined with waypoints:")
    for wp in waypoints:
        print(f"  ({wp['x']}, {wp['y']}) — {wp['label']}")
    print()

    # --- Step 5: Start autonomous driving ---
    # The system handles perception, planning, and control
    autonomous.set_route(waypoints)
    autonomous.start_driving()
    print("Autonomous driving started!")

    # --- Step 6: Monitor driving progress ---
    try:
        while autonomous.is_driving:
            state = autonomous.get_vehicle_state()
            perception = autonomous.get_perception()

            # Display current driving state
            print(f"  Speed: {state.speed:.1f} m/s | "
                  f"Steering: {state.steering_angle:.1f}° | "
                  f"Position: ({state.x:.1f}, {state.y:.1f}) | "
                  f"Obstacles: {len(perception.obstacles)}")

            # Check for safety alerts
            if perception.safety_alert:
                print(f"  ⚠ SAFETY ALERT: {perception.alert_message}")
                autonomous.emergency_slow()

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nEmergency stop requested!")
        autonomous.emergency_stop()

    # --- Step 7: Trip summary ---
    summary = autonomous.get_trip_summary()
    print(f"\nTrip completed!")
    print(f"  Distance: {summary.distance_traveled:.1f} m")
    print(f"  Duration: {summary.duration_seconds:.1f} s")
    print(f"  Average speed: {summary.avg_speed:.1f} m/s")
    print(f"  Safety events: {summary.safety_events}")


if __name__ == "__main__":
    main()
