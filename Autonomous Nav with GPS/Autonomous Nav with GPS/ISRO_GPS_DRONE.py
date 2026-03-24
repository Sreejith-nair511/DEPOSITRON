import drone_functions_GPS as dfg
import AprilTag_Detect as atag
import pyrealsense2 as rs
import cv2

# Drone parameters
target_altitude = 1.0  # Meters
hover_time = 30  # Seconds

# Initialize AprilTag detect
detector = atag.detector

# Initialize RealSense pipeline
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe.start(cfg)

# Get frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X, CENTER_Y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2

# Connect to drone
vehicle = dfg.connect_to_vehicle()

if vehicle is not None and vehicle.is_armable:
    print("[INFO] Vehicle connected and ready.")
    print("[INFO] Taking off...")
    dfg.arm_and_takeoff(vehicle, target_altitude)

    print(f"[INFO] Hovering for {hover_time} seconds...")
    dfg.hover(vehicle, hover_time)

    print("[INFO] Searching for AprilTag...")
    tag_position = atag.keep_tag_center()  # Returns (x_offset, y_offset, distance)

    if tag_position:
        print(f"[INFO] AprilTag detected at {tag_position}. Adjusting position...")
        dfg.land(vehicle)  # Implement precision landing using tag_position
    else:
        print("[WARNING] AprilTag not detected! Landing normally...")
        dfg.land(vehicle)

    print("[INFO] Mission completed!")
else:
    print("[ERROR] Failed to connect to the vehicle.")

print("[INFO] Detecting flat surfaces using RealSense...")
try:
    while True:
        result = dfg.detect_flat_surface(pipe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()

print("[INFO] Script execution completed.")
