"""AI drone example using Thunders AI Robotics.

Demonstrates how to initialize a drone, perform flight control commands,
and execute autonomous navigation missions with waypoint following.
"""

import time
from thunders_ai.robotics import Drone, DroneConfig, FlightPlanner


def main() -> None:
    """Run the AI drone navigation example."""
    # --- Step 1: Initialize the drone ---
    # Configure drone hardware and safety parameters
    drone_config = DroneConfig(
        max_altitude_meters=120.0,    # Regulatory altitude limit
        max_speed_mps=15.0,           # Maximum flight speed
        max_distance_meters=500.0,    # Maximum distance from home
        geofence_enabled=True,        # Enforce geofence boundary
        return_to_home_on_low_battery=True,
        battery_warning_percent=20.0,
    )
    drone = Drone(
        name="ThunderDrone-01",
        config=drone_config,
    )
    print(f"Drone '{drone.name}' initialized.")
    print(f"  Max altitude: {drone_config.max_altitude_meters}m")
    print(f"  Max speed: {drone_config.max_speed_mps} m/s")
    print(f"  Geofence: {'ON' if drone_config.geofence_enabled else 'OFF'}")
    print()

    # --- Step 2: Connect and pre-flight checks ---
    drone.connect()
    print(f"Drone connected. Battery: {drone.battery_percent:.0f}%")

    # Run pre-flight safety checks
    checks = drone.pre_flight_check()
    print("Pre-flight checks:")
    for check_name, passed in checks.items():
        status_icon = "✓" if passed else "✗"
        print(f"  {status_icon} {check_name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        print("Pre-flight checks FAILED. Aborting.")
        return
    print()

    # --- Step 3: Take off ---
    target_altitude = 30.0  # meters
    drone.takeoff(altitude=target_altitude)
    print(f"Taking off to {target_altitude}m...")

    # Wait for stable hover
    while not drone.is_hovering:
        print(f"  Altitude: {drone.altitude:.1f}m, Status: {drone.flight_status}")
        time.sleep(0.5)
    print(f"Hovering at {drone.altitude:.1f}m. Ready for navigation.")
    print()

    # --- Step 4: Plan autonomous flight path ---
    # Define waypoints for a survey mission
    planner = FlightPlanner(
        altitude=30.0,
        speed=5.0,
        overlap_percent=60,       # For aerial survey overlap
        obstacle_avoidance=True,
    )
    waypoints = [
        {"x": 0.0, "y": 0.0, "z": 30.0, "action": "hover"},
        {"x": 50.0, "y": 0.0, "z": 30.0, "action": "capture"},
        {"x": 50.0, "y": 50.0, "z": 30.0, "action": "capture"},
        {"x": 0.0, "y": 50.0, "z": 30.0, "action": "capture"},
        {"x": 0.0, "y": 0.0, "z": 30.0, "action": "return"},
    ]
    mission = planner.plan_mission(waypoints)
    print(f"Mission planned: {len(mission.waypoints)} waypoints, "
          f"{mission.total_distance:.1f}m total distance")
    print()

    # --- Step 5: Execute autonomous flight ---
    drone.start_mission(mission)
    print("Autonomous mission started!")

    while drone.is_flying and drone.mission_progress < 1.0:
        progress = drone.mission_progress
        current_wp = drone.current_waypoint
        battery = drone.battery_percent
        print(f"  Progress: {progress:.0%} | "
              f"Waypoint: {current_wp}/{len(mission.waypoints)} | "
              f"Battery: {battery:.0f}% | "
              f"Alt: {drone.altitude:.1f}m")

        # Check battery safety
        if battery < drone_config.battery_warning_percent:
            print("  ⚠ Low battery! Returning to home.")
            drone.return_to_home()
            break
        time.sleep(1.0)

    # --- Step 6: Land the drone ---
    if drone.is_flying:
        drone.land()
        print("Landing...")

    while drone.is_flying:
        print(f"  Descending... Altitude: {drone.altitude:.1f}m")
        time.sleep(0.5)

    print(f"Landed safely. Remaining battery: {drone.battery_percent:.0f}%")

    # --- Step 7: Post-flight summary ---
    flight_log = drone.get_flight_log()
    print(f"\nFlight summary:")
    print(f"  Duration: {flight_log.duration_seconds:.0f}s")
    print(f"  Distance: {flight_log.distance_traveled:.1f}m")
    print(f"  Max altitude: {flight_log.max_altitude:.1f}m")
    print(f"  Max speed: {flight_log.max_speed:.1f} m/s")
    drone.disconnect()
    print("Drone disconnected.")


if __name__ == "__main__":
    main()
