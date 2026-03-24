import cv2
import numpy as np
from pupil_apriltags import Detector
import pyrealsense2 as rs

# Initialize AprilTag Detector with better detection settings
detector = Detector(families="tag36h11", quad_decimate=1.0, refine_edges=True)

# Initialize RealSense Depth Camera
pipe = rs.pipeline()
cfg = rs.config()

# Enable only depth stream
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe.start(cfg)

# Frame size
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X, CENTER_Y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2


def preprocess_depth_image(depth_frame):
    """ Convert depth frame to grayscale and enhance contrast for better AprilTag detection. """
    depth_image = np.asanyarray(depth_frame.get_data())

    # Normalize depth image to 8-bit range (0-255)
    depth_gray = cv2.convertScaleAbs(depth_image, alpha=255.0 / np.max(depth_image))

    # Apply histogram equalization to improve contrast
    depth_gray = cv2.equalizeHist(depth_gray)

    # Apply Gaussian blur to reduce noise
    depth_gray = cv2.GaussianBlur(depth_gray, (5, 5), 0)

    return depth_gray


def keep_tag_center():
    try:
        while True:
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # Preprocess depth frame
            depth_gray = preprocess_depth_image(depth_frame)

            # Detect AprilTags
            tags = detector.detect(depth_gray)

            # Convert depth to color map for visualization
            depth_colormap = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)

            for tag in tags:
                ptA, ptB, ptC, ptD = tag.corners
                tag_x = int((ptA[0] + ptC[0]) / 2)
                tag_y = int((ptA[1] + ptC[1]) / 2)

                # Draw bounding box and center point
                cv2.polylines(depth_colormap, [np.array(tag.corners, dtype=np.int32)], True, (0, 255, 0), 2)
                cv2.circle(depth_colormap, (tag_x, tag_y), 5, (0, 0, 255), -1)

                cv2.putText(depth_colormap, f"ID: {tag.tag_id}", (tag_x - 20, tag_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show depth map with detected AprilTags
            cv2.imshow("AprilTag Detection on Depth Map", depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Stopping pipeline and closing windows...")
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    keep_tag_center()
