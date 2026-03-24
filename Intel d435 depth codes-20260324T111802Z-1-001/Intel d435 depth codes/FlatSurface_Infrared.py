import pyrealsense2 as rs
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor, LinearRegression
from pykalman import KalmanFilter

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
cfg = rs.config()

# Configure the pipeline to stream both infrared and depth
cfg.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)  # Infrared stream
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream

# Start streaming
pipeline.start(cfg)

# Kalman Filter for stabilizing bounding box
kf = KalmanFilter(
    initial_state_mean=[320, 240, 0, 0],
    transition_matrices=[[1, 0, 0.5, 0], [0, 1, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]],
    observation_matrices=[[1, 0, 0, 0], [0, 1, 0, 0]],
    transition_covariance=np.eye(4) * 0.02)

state_means = np.array([320, 240, 0, 0])
state_covariances = np.eye(4)

# Real-world size constraints
desired_size_m = 1.5  # 1.5m x 1.5m
focal_length = 615  # Approximate focal length for D435 in pixels
scaling_factor = 0.001  # Convert mm to meters

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        ir_frame = frames.get_infrared_frame()  # Get infrared frame

        if not depth_frame or not ir_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data()).astype(float) * scaling_factor
        depth_image = cv2.normalize(depth_image, None, 0, 1, cv2.NORM_MINMAX)

        ir_image = np.asanyarray(ir_frame.get_data())
        height, width = depth_image.shape

        # Create a point cloud (X, Y, Z) using depth values
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        x3d = (xx - width // 2) * depth_image / focal_length
        y3d = (yy - height // 2) * depth_image / focal_length
        z3d = depth_image

        max_distance = 3.0  # Limit to 3 meters
        points = np.column_stack((x3d.flatten(), y3d.flatten(), z3d.flatten()))
        filtered_points = points[points[:, 2] < max_distance]  # Remove far points
        points = points[(~np.isnan(points).any(axis=1)) & (points[:, 2] < max_distance)]

        # Apply RANSAC plane fitting
        if len(points) > 1000:
            ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=0.02)
            ransac.fit(points[:, :2], points[:, 2])
            inliers = ransac.inlier_mask_

            # Get bounding box for inliers
            inlier_points = points[inliers]
            min_x, max_x = np.percentile(inlier_points[:, 0], [10, 90])
            min_y, max_y = np.percentile(inlier_points[:, 1], [10, 90])

            # Convert real-world meters to pixels
            epsilon = 1e-6  # Small constant to prevent division by zero
            depth_center = depth_image[int(height / 2), int(width / 2)]

            # Handle NaN or zero depth values
            if np.isnan(depth_center) or depth_center == 0:
                depth_center = np.nanmedian(depth_image)  # Use average depth if center is invalid

            box_x1 = int((min_x * focal_length) / (depth_center + epsilon) + width // 2)
            box_y1 = int((min_y * focal_length) / (depth_center + epsilon) + width // 2)
            box_x2 = int((max_x * focal_length) / (depth_center + epsilon) + width // 2)
            box_y2 = int((max_y * focal_length) / (depth_center + epsilon) + width // 2)

            # Kalman filter update
            state_means, state_covariances = kf.filter_update(
                state_means, state_covariances, [((box_x1 + box_x2) / 2), ((box_y1 + box_y2) / 2)]
            )
            box_x1, box_y1 = int(state_means[0] - (box_x2 - box_x1) / 2), int(state_means[1] - (box_y2 - box_y1) / 2)
            box_x2, box_y2 = int(state_means[0] + (box_x2 - box_x1) / 2), int(state_means[1] + (box_y2 - box_y1) / 2)

            # Contour Detection to check for objects inside the flat surface box (using IR image)
            blurred = cv2.GaussianBlur(ir_image, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            adaptive_thresh = cv2.adaptiveThreshold(ir_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            object_detected = False
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if box_x1 < x < box_x2 and box_y1 < y < box_y2:
                    object_detected = True
                    cv2.rectangle(ir_image, (x, y), (x + w, y + h), 255, 2)
                    cv2.putText(ir_image, "Object Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

            # Draw the bounding box for the flat surface
            if not object_detected:
                cv2.rectangle(ir_image, (box_x1, box_y1), (box_x2, box_y2), 255, 3)
                cv2.putText(ir_image, "Flat Surface Detected", (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

        # Show output
        cv2.imshow("Infrared Flat Surface Detection", ir_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
