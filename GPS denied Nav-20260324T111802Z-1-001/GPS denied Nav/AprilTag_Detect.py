import cv2
import numpy as np
from pupil_apriltags import Detector
import pyrealsense2 as rs

# Initialize AprilTag Detector
detector = Detector(families="tag36h11")

# Initialize RealSense Camera
pipe = rs.pipeline()
cfg = rs.config()

# Enable the infrared stream (IR stream 1)
cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # IR stream is grayscale
pipe.start(cfg)

# Get frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X, CENTER_Y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2  # Image center


def keep_tag_center():
    def get_drone_movement(tag_x, tag_y):
        """
        Determine drone movement based on the tag's position.
        """
        move_x = tag_x - CENTER_X  # X offset
        move_y = tag_y - CENTER_Y  # Y offset

        # Define a threshold to avoid unnecessary minor movements
        THRESHOLD = 30
        vx, vz = 0, 0

        if abs(move_x) > THRESHOLD:
            vx = -0.1 if move_x < 0 else 0.1
        if abs(move_y) > THRESHOLD:
            vz = -0.1 if move_y < 0 else 0.1
        return vx, vz

    try:
        while True:  # Run in a loop until 'q' is pressed
            frames = pipe.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)  # Get infrared stream (IR 1)
            if not ir_frame:
                continue

            ir_img = np.asanyarray(ir_frame.get_data())  # IR is already grayscale

            tags = detector.detect(ir_img)
            total_vx, total_vz = 0, 0
            tag_count = len(tags)

            for tag in tags:
                ptA, ptB, ptC, ptD = tag.corners
                tag_x = int((ptA[0] + ptC[0]) / 2)  # Compute AprilTag center
                tag_y = int((ptA[1] + ptC[1]) / 2)

                # Draw bounding box and center point
                cv2.polylines(ir_img, [np.array(tag.corners, dtype=np.int32)], True, (255, 255, 255), 2)
                cv2.circle(ir_img, (tag_x, tag_y), 5, (255, 255, 255), -1)  # White dot at tag center
                cv2.putText(ir_img, f"ID: {tag.tag_id}", (tag_x - 20, tag_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

                # Draw line from image center to AprilTag center
                cv2.line(ir_img, (CENTER_X, CENTER_Y), (tag_x, tag_y), 255, 2)

                vx, vz = get_drone_movement(tag_x, tag_y)
                total_vx += vx
                total_vz += vz

            # Draw image center for reference
            cv2.circle(ir_img, (CENTER_X, CENTER_Y), 5, 255, -1)  # White dot at image center
            cv2.imshow("AprilTag Tracking (Infrared)", ir_img)

            if tag_count > 0:
                print(f"Drone Movement: vx = {total_vx / tag_count}, vz = {total_vz / tag_count}")
            else:
                print("No tag detected. Staying in position.")

            yield total_vx, total_vz
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Stopping pipeline and closing windows...")
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    for vx, vz in keep_tag_center():
        print(f"vx: {vx}, vz: {vz}")
