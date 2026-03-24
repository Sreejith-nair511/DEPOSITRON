import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time


def get_realsense_pipeline():
    """Initialize and return the RealSense pipeline with depth stream enabled."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    return pipeline


def depth_to_pointcloud(depth_frame, intrinsics):
    """Converts a depth frame into a 3D point cloud."""
    height, width = depth_frame.get_height(), depth_frame.get_width()
    points = []

    for y in range(height):
        for x in range(width):
            depth = depth_frame.get_distance(x, y)
            if 0.1 < depth < 5.0:  # Ignore very close and very far points
                point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                points.append(point)

    return np.array(points)


def capture_pointcloud(pipeline, duration=20):
    """Continuously capture depth frames and accumulate a 3D point cloud for a given duration."""
    align = rs.align(rs.stream.color)
    start_time = time.time()
    all_points = []

    while time.time() - start_time < duration:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        if not depth_frame:
            continue

        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        points = depth_to_pointcloud(depth_frame, intrinsics)

        if points.size > 0:
            all_points.append(points)

    pipeline.stop()

    if len(all_points) > 0:
        return np.vstack(all_points)  # Merge all point cloud data
    else:
        return np.array([])


if __name__ == "__main__":
    pipeline = get_realsense_pipeline()

    print("Scanning for 20 seconds...")
    pointcloud = capture_pointcloud(pipeline, duration=20)

    if pointcloud.size == 0:
        print("No point cloud captured.")
        exit()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    # Visualize the scanned 3D surface
    print("Visualizing 3D scan surface...")
    o3d.visualization.draw_geometries([pcd], window_name="3D Scan Surface")
