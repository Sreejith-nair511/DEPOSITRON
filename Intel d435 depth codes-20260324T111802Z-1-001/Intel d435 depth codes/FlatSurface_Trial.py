import pyrealsense2 as rs
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
cfg = rs.config()

# Enable infrared and depth streams
cfg.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(cfg)

# Camera parameters
focal_length = 615  # Approximate focal length for D435 in pixels
scaling_factor = 0.001  # Convert mm to meters

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        ir_frame = frames.get_infrared_frame()

        if not depth_frame or not ir_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data()).astype(float) * scaling_factor
        ir_image = np.asanyarray(ir_frame.get_data())
        height, width = depth_image.shape

        # Create 3D points
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        x3d = (xx - width // 2) * depth_image / focal_length
        y3d = (yy - height // 2) * depth_image / focal_length
        z3d = depth_image

        points = np.column_stack((x3d.flatten(), y3d.flatten(), z3d.flatten()))
        points = points[~np.isnan(points).any(axis=1)]  # Remove NaNs

        # RANSAC for plane fitting
        if len(points) > 1000:
            ransac = RANSACRegressor(residual_threshold=0.02)
            ransac.fit(points[:, :2], points[:, 2])
            inliers = ransac.inlier_mask_
            inlier_points = points[inliers]

            # Get bounding box
            min_x, max_x = np.percentile(inlier_points[:, 0], [10, 90])
            min_y, max_y = np.percentile(inlier_points[:, 1], [10, 90])
            depth_center = np.nanmean(depth_image) if np.isnan(depth_image[int(height / 2), int(width / 2)]) else \
            depth_image[int(height / 2), int(width / 2)]

            box_x1 = int((min_x * focal_length) / (depth_center + 1e-6) + width // 2)
            box_y1 = int((min_y * focal_length) / (depth_center + 1e-6) + height // 2)
            box_x2 = int((max_x * focal_length) / (depth_center + 1e-6) + width // 2)
            box_y2 = int((max_y * focal_length) / (depth_center + 1e-6) + height // 2)

            # Compute centers
            center_box = ((box_x1 + box_x2) // 2, (box_y1 + box_y2) // 2)
            center_cam = (width // 2, height // 2)
            velocity_x, velocity_y = center_box[0] - center_cam[0], center_box[1] - center_cam[1]

            # Contour Detection
            blurred = cv2.GaussianBlur(ir_image, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            object_detected = any(box_x1 < x < box_x2 and box_y1 < y < box_y2 for x, y, w, h in
                                  [cv2.boundingRect(cnt) for cnt in contours])

            # Draw bounding box and center markers
            color = (0, 0, 255) if object_detected else (0, 255, 0)
            cv2.rectangle(ir_image, (box_x1, box_y1), (box_x2, box_y2), color, 2)
            cv2.circle(ir_image, center_box, 5, (255, 0, 0), -1)
            cv2.circle(ir_image, center_cam, 5, (0, 255, 255), -1)
            cv2.line(ir_image, center_box, center_cam, (255, 255, 0), 2)

            # Draw contours if object is detected
            if object_detected:
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if box_x1 < x < box_x2 and box_y1 < y < box_y2:
                        cv2.rectangle(ir_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(ir_image, "Object Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 255), 2)
            else:
                cv2.putText(ir_image, "Flat Surface Detected", (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

            # Show velocity values
            cv2.putText(ir_image, f"Vel: {velocity_x}, {velocity_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

        # Show output
        cv2.imshow("Flat Surface Detection (Infrared)", ir_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()