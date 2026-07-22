"""
LiDAR obstacle detector using 2D LaserScan clustering.

Optimized for water surface clutter:
  - Higher min_range (3.5m) to exclude boat hull & bow wake
  - DBSCAN min_samples raised to 7 to filter single-frame wave splash
  - Cluster diameter filtering (0.3m to 2.5m) to discard wave crests
"""

import json
import math
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan


def simple_dbscan(points: np.ndarray, eps: float, min_samples: int) -> list:
    n = len(points)
    if n == 0:
        return []

    visited = np.zeros(n, dtype=bool)
    labels = -np.ones(n, dtype=int)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        dists = np.linalg.norm(points - points[i], axis=1)
        neighbours = np.where(dists < eps)[0]

        if len(neighbours) < min_samples:
            continue

        labels[i] = cluster_id
        seed_set = list(neighbours)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_dists = np.linalg.norm(points - points[q], axis=1)
                q_neighbours = np.where(q_dists < eps)[0]
                if len(q_neighbours) >= min_samples:
                    seed_set.extend(q_neighbours.tolist())
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1

        cluster_id += 1

    clusters = []
    for cid in range(cluster_id):
        mask = labels == cid
        clusters.append(points[mask])
    return clusters


class LidarProcessorNode(Node):
    """Clusters 2D LaserScan data to detect floating obstacles."""

    def __init__(self):
        super().__init__('lidar_processor_node')

        self.declare_parameter('min_range', 3.5)       # Exclude boat hull & bow wake
        self.declare_parameter('max_range', 60.0)
        self.declare_parameter('cluster_eps', 1.2)     # Slightly tighter eps
        self.declare_parameter('cluster_min_pts', 7)   # Raised to 7 (wave splash has 2-4 pts)
        self.declare_parameter('max_process_hz', 10.0)

        self.min_range = self.get_parameter('min_range').value
        self.max_range = self.get_parameter('max_range').value
        self.eps = self.get_parameter('cluster_eps').value
        self.min_pts = self.get_parameter('cluster_min_pts').value
        self.max_dt = 1.0 / self.get_parameter('max_process_hz').value
        self.last_process_time = 0.0

        gz_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.create_subscription(
            LaserScan,
            '/wamv/sensors/lidars/lidar_wamv_sensor/scan',
            self._scan_cb,
            gz_qos,
        )
        self.pub_obstacles = self.create_publisher(String, '/asv/obstacles', 10)

        self.get_logger().info(
            f'LidarProcessor initialized (min_pts={self.min_pts}, eps={self.eps}m)')

    def _scan_cb(self, msg: LaserScan):
        now = time.monotonic()
        if now - self.last_process_time < self.max_dt:
            return
        self.last_process_time = now

        angles = np.arange(
            msg.angle_min,
            msg.angle_min + len(msg.ranges) * msg.angle_increment,
            msg.angle_increment,
        )
        n = min(len(angles), len(msg.ranges))
        angles = angles[:n]
        ranges = np.array(msg.ranges[:n], dtype=np.float32)

        valid = (
            np.isfinite(ranges) &
            (ranges >= self.min_range) &
            (ranges <= self.max_range)
        )
        ranges = ranges[valid]
        angles = angles[valid]

        if len(ranges) == 0:
            self._publish([])
            return

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points = np.column_stack((xs, ys))

        clusters = simple_dbscan(points, self.eps, self.min_pts)

        obstacles = []
        for cluster in clusters:
            centroid = cluster.mean(axis=0)
            span_x = cluster[:, 0].max() - cluster[:, 0].min()
            span_y = cluster[:, 1].max() - cluster[:, 1].min()
            diameter = max(span_x, span_y)

            # Buoy physical size filter: diameter must be between 0.25m and 2.5m
            # Ignores large wave lines or tiny single-point noise
            if diameter < 0.25 or diameter > 2.5:
                continue

            obstacles.append({
                'x': round(float(centroid[0]), 2),
                'y': round(float(centroid[1]), 2),
                'diameter': round(float(diameter), 2),
                'n_points': len(cluster),
                'range': round(float(np.linalg.norm(centroid)), 2),
                'bearing_deg': round(
                    float(math.degrees(math.atan2(centroid[1], centroid[0]))),
                    1),
            })

        self._publish(obstacles)

    def _publish(self, obstacles: list):
        msg = String()
        msg.data = json.dumps(obstacles)
        self.pub_obstacles.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
