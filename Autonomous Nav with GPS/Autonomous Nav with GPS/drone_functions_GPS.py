from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
import time
from pymavlink import mavutil
import math
import socket
import argparse
import geopy.distance
import numpy as np
import cv2
import AprilTag_Detect as atag

vehicle = None
prev_x, prev_y = None, None
alpha = 0.3  # Smoothing factor (adjust between 0.1 and 0.5)


#connect to drone
def connect_to_vehicle():
    global master
    parser = argparse.ArgumentParser(description="Connect to a MAVLink vehicle")
    parser.add_argument('--connect', required=True,
                        help="Connection string ('/dev/ttyUSB0', 'udp:127.0.0.1:14550')")
    args = parser.parse_args()
    connect_string = args.connect

    print(f"Connecting to vehicle on {connect_string}...")
    master = mavutil.mavlink_connection(connect_string, baud=57600)

    # Wait for heartbeat to confirm connection
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print("Heartbeat received! Connection successful.")

    return master


def send_velocity(vehicle, vx, vy, vz, duration):
    """
    Send velocity command to drone.
    vx, vy, vz are in m/s (Body frame: forward, right, downward)
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,  # Time boot ms, target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # Body frame
        0b0000111111000111,  # Only velocity control
        0, 0, 0,  # No position setpoints
        vx, vy, vz,  # Velocity (m/s)
        0, 0, 0,  # No acceleration
        0, 0)  # No yaw control
    for i in range(int(duration * 10)):  # Send for 'duration' seconds
        vehicle.send_mavlink(msg)
        time.sleep(0.1)

    print(f"Velocity command executed: vx={vx}, vy={vy}, vz={vz} for {duration}s")


# arm and takeoff to meters
def arm_and_takeoff(vehicle, TargetAltitude):
    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    send_velocity(vehicle, 0, 0, -0.5, 0.5)

    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt < TargetAltitude * 0.95:
            for Vx, Vz in atag.keep_tag_center():  # Iterate over the generator
                if Vx == 0 and Vz == 0:
                    continue
                else:
                    send_velocity(vehicle, Vx, 0, Vz, 1)
        else:
            print("Reached target altitude")
            break
        time.sleep(1)


def land(vehicle):
    print("Landing...")
    send_velocity(vehicle, 0, 0, 0.5, 0.5)  # Descend at 0.5 m/s

    while True:
        current_alt = vehicle.rangefinder.distance
        print(f"Descending: {current_alt:.2f}m")
        if current_alt > 0.2:
            for Vx, Vz in atag.keep_tag_center():  # Iterate over the generator
                if Vx == 0 and Vz == 0:
                    continue
                else:
                    send_velocity(vehicle, Vx, 0, Vz, 1)
        else:
            print("Reached target altitude")
            break
        time.sleep(1)

        time.sleep(0.5)
    vehicle.armed = False
    print("Disarmed and landed.")


def hover(vehicle, duration: int):
    print(f"[INFO] Hovering for {duration} seconds...")

    start_time = time.time()
    target_altitude = vehicle.location.global_relative_frame.alt  # Current altitude

    while time.time() - start_time < duration:
        current_altitude = vehicle.location.global_relative_frame.alt
        altitude_error = target_altitude - current_altitude

        if abs(altitude_error) > 0.1:  # If altitude drifts, correct it
            print(f"[INFO] Adjusting altitude: Target {target_altitude}m, Current {current_altitude}m")
            vehicle.simple_goto(vehicle.location.global_relative_frame, airspeed=0)

        time.sleep(1)  # Check every second

    print("[INFO] Hover complete!")


def move_front(vehicle, distance):
    speed = 3  # m/s
    duration = distance / speed  # Time required to cover the distance
    send_velocity(vehicle, speed, 0, 0, duration)
    print(f"Moving front: {distance}m at {speed}m/s")


def move_back(vehicle, distance):
    speed = 3  # m/s
    duration = distance / speed  # Time required to cover the distance
    send_velocity(vehicle, -speed, 0, 0, duration)
    print(f"Moving back: {distance}m at {speed}m/s")


def move_left(vehicle, distance):
    speed = 3  # m/s
    duration = distance / speed  # Time required to cover the distance
    send_velocity(vehicle, 0, -speed, 0, duration)
    print(f"Moving left: {distance}m at {speed}m/s")


def move_right(vehicle, distance):
    speed = 3  # m/s
    duration = distance / speed  # Time required to cover the distance
    send_velocity(vehicle, 0, speed, 0, duration)
    print(f"Moving right: {distance}m at {speed}m/s")


def find_safe_spot(pipeline, grid_size=4):
    global best_spot, best_x, best_y
    while True:  # Continuous loop for live feed
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # Validate the depth frame
        if not depth_frame:
            print("No depth frame received, retrying...")
            continue  # Skip this frame and retry

        # Convert to NumPy array
        depth_image = np.asanyarray(depth_frame.get_data()).astype(float)

        # Handle zero-depth values (invalid pixels)
        depth_image[depth_image == 0] = np.nan  # Ignore zero values in calculations

        # Get image dimensions
        height, width = depth_image.shape
        h_step, w_step = height // grid_size, width // grid_size  # Grid step size

        # Dictionary to store standard deviation of each section
        grid_variations = {}

        for i in range(grid_size):
            for j in range(grid_size):
                # Extract section
                section = depth_image[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]

                # Compute standard deviation, ignoring NaN values
                variation = np.nanstd(section)

                # Store in dictionary with coordinates
                grid_variations[(i, j)] = variation

        # Find the flattest section (safest spot)
        best_spot = min(grid_variations, key=grid_variations.get)

        # Convert grid coordinates to pixel coordinates (center of the section)
        best_x = (best_spot[1] * w_step) + (w_step // 2)
        best_y = (best_spot[0] * h_step) + (h_step // 2)

        # Convert depth image to 8-bit grayscale for visualization
        depth_visual = cv2.convertScaleAbs(depth_image, alpha=0.03)

        # Draw grid on the image
        for i in range(1, grid_size):
            cv2.line(depth_visual, (0, i * h_step), (width, i * h_step), (255, 255, 255), 1)  # Horizontal
            cv2.line(depth_visual, (i * w_step, 0), (i * w_step, height), (255, 255, 255), 1)  # Vertical

        # Mark the safest spot
        cv2.circle(depth_visual, (best_x, best_y), 10, (0, 0, 255), -1)  # Red circle

        # Display details on screen
        info_text = f"Safe Spot: Grid {best_spot}, Pixel ({best_x}, {best_y})"
        cv2.putText(depth_visual, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the image
        cv2.imshow("Safe Spot Detection", depth_visual)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return {"grid_position": best_spot, "pixel_coordinates": (best_x, best_y)}


def detect_flat_surface(pipeline, grid_size=4, std_threshold=5):
    global prev_x, prev_y

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("Waiting for frames...")
        return None

    # Convert frames to NumPy arrays
    depth_image = np.asanyarray(depth_frame.get_data()).astype(float)
    color_image = np.asanyarray(color_frame.get_data())

    # Replace 0 depth values with NaN (invalid points)
    depth_image[depth_image == 0] = np.nan

    # Image dimensions
    height, width = depth_image.shape
    h_step, w_step = height // grid_size, width // grid_size

    # Find the flattest region using standard deviation
    grid_variations = {}
    for i in range(grid_size):
        for j in range(grid_size):
            section = depth_image[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
            variation = np.nanstd(section)  # Standard deviation (ignore NaN)
            grid_variations[(i, j)] = variation

    # Find the best (flattest) spot
    best_spot = min(grid_variations, key=grid_variations.get)
    best_x = (best_spot[1] * w_step) + (w_step // 2)
    best_y = (best_spot[0] * h_step) + (h_step // 2)
    best_variation = grid_variations[best_spot]

    # Apply exponential smoothing to stabilize the box
    if prev_x is None or prev_y is None:
        prev_x, prev_y = best_x, best_y  # Initialize with first value
    else:
        prev_x = alpha * best_x + (1 - alpha) * prev_x
        prev_y = alpha * best_y + (1 - alpha) * prev_y

    # ✅ Mark the area as flat if the variation is low
    if best_variation < std_threshold:
        color = (0, 255, 0)  # Green box for flat area
        label = "Flat Surface"
    else:
        color = (0, 0, 255)  # Red box for uneven area
        label = "Uneven Surface"

    # Compute stabilized box coordinates
    x1, y1 = int(prev_x - w_step // 2), int(prev_y - h_step // 2)
    x2, y2 = int(prev_x + w_step // 2), int(prev_y + h_step // 2)

    # Draw box on color image
    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ✅ Process and visualize the depth image
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Draw stabilized box on depth image
    cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), color, 3)
    cv2.putText(depth_colormap, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the images
    cv2.imshow("Flat Surface Detection - RGB", color_image)
    cv2.imshow("Flat Surface Detection - Depth", depth_colormap)
    cv2.waitKey(1)

    return {"grid_position": best_spot, "pixel_coordinates": (prev_x, prev_y)}


# To run in rpi in terminal python drone_functions_GPS.py --connect /dev/serial0
if __name__ == "__main__":
    connect_to_vehicle()
