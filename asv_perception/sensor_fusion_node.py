"""
Sensor fusion node — merges camera detections with LiDAR obstacles.

Matching strategy:
  Camera gives:  class label + bearing angle (no range)
  LiDAR gives:   range + bearing (no class)
  Fusion:        match by bearing within a tolerance window, combine to get
                 a classified, geo-located hazard.

Also maintains a persistent hazard catalogue (unique hazards seen so far)
for counting purposes, and transforms obstacle positions from body frame
to world frame using /wamv/pose.

Subscribes:
  /asv/detections   (std_msgs/String — camera JSON)
  /asv/obstacles    (std_msgs/String — LiDAR JSON)
  /wamv/pose        (geometry_msgs/PoseStamped)

Publishes:
  /asv/hazards          (std_msgs/String — fused JSON, world-frame)
  /asv/hazard_markers   (visualization_msgs/MarkerArray — for RViz)
  /asv/hazard_log       (std_msgs/String — running count summary)
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# RViz marker colours per class
MARKER_COLORS = {
    'red':    ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.85),
    'green':  ColorRGBA(r=0.0, g=0.8, b=0.0, a=0.85),
    'orange': ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.85),
    'white':  ColorRGBA(r=0.95, g=0.95, b=0.95, a=0.85),
    'black':  ColorRGBA(r=0.15, g=0.15, b=0.15, a=0.85),
}


class SensorFusionNode(Node):
    """Fuses camera class labels with LiDAR positions + maintains hazard catalogue."""

    def __init__(self):
        super().__init__('sensor_fusion_node')

        # ── Parameters ──────────────────────────────────────────
        self.declare_parameter('bearing_tolerance_deg', 8.0)
        self.declare_parameter('catalogue_merge_dist', 4.0)  # metres
        self.declare_parameter('fusion_hz', 10.0)

        self.bearing_tol = self.get_parameter('bearing_tolerance_deg').value
        self.merge_dist = self.get_parameter('catalogue_merge_dist').value
        fusion_dt = 1.0 / self.get_parameter('fusion_hz').value

        # ── State ───────────────────────────────────────────────
        self.latest_detections = []   # from camera
        self.latest_obstacles = []    # from LiDAR
        self.pose = None              # latest vehicle pose
        self.hazard_catalogue = []    # persistent unique hazards
        self.next_hazard_id = 0

        # ── QoS ─────────────────────────────────────────────────
        gz_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )

        # ── Subscribers ─────────────────────────────────────────
        self.create_subscription(
            String, '/asv/detections', self._det_cb, 10)
        self.create_subscription(
            String, '/asv/obstacles', self._obs_cb, 10)
        self.create_subscription(
            PoseStamped, '/wamv/pose', self._pose_cb, gz_qos)

        # ── Publishers ──────────────────────────────────────────
        self.pub_hazards = self.create_publisher(String, '/asv/hazards', 10)
        self.pub_markers = self.create_publisher(
            MarkerArray, '/asv/hazard_markers', 10)
        self.pub_log = self.create_publisher(String, '/asv/hazard_log', 10)

        # ── Fusion timer ────────────────────────────────────────
        self.create_timer(fusion_dt, self._fuse)

        self.get_logger().info(
            f'SensorFusion started — bearing_tol={self.bearing_tol}°, '
            f'catalogue_merge_dist={self.merge_dist}m')

    # ── Callbacks ───────────────────────────────────────────────────

    def _det_cb(self, msg: String):
        try:
            self.latest_detections = json.loads(msg.data)
        except json.JSONDecodeError:
            self.latest_detections = []

    def _obs_cb(self, msg: String):
        try:
            self.latest_obstacles = json.loads(msg.data)
        except json.JSONDecodeError:
            self.latest_obstacles = []

    def _pose_cb(self, msg: PoseStamped):
        self.pose = msg

    # ── Fusion logic ────────────────────────────────────────────────

    def _fuse(self):
        if self.pose is None:
            return

        px = self.pose.pose.position.x
        py = self.pose.pose.position.y
        yaw = yaw_from_quaternion(self.pose.pose.orientation)

        fused_hazards = []
        used_obstacle_indices = set()

        # ── Match camera detections to LiDAR obstacles by bearing ──
        for det in self.latest_detections:
            det_bearing = det.get('bearing_deg', 0.0)
            best_match = None
            best_diff = self.bearing_tol

            for idx, obs in enumerate(self.latest_obstacles):
                if idx in used_obstacle_indices:
                    continue
                obs_bearing = obs.get('bearing_deg', 0.0)
                diff = abs(det_bearing - obs_bearing)
                if diff < best_diff:
                    best_diff = diff
                    best_match = idx

            if best_match is not None:
                used_obstacle_indices.add(best_match)
                obs = self.latest_obstacles[best_match]

                # Transform obstacle from body frame to world frame
                # Body frame: x forward, y left of vehicle
                bx, by = obs['x'], obs['y']
                wx = px + bx * math.cos(yaw) - by * math.sin(yaw)
                wy = py + bx * math.sin(yaw) + by * math.cos(yaw)

                hazard = {
                    'class': det.get('class', 'unknown'),
                    'color': det.get('color', 'unknown'),
                    'shape': det.get('shape', 'unknown'),
                    'x': round(wx, 2),
                    'y': round(wy, 2),
                    'range': obs.get('range', 0.0),
                    'bearing_deg': det_bearing,
                    'source': 'fused',
                }
                fused_hazards.append(hazard)
            else:
                # Camera-only detection (no LiDAR match) — bearing only
                fused_hazards.append({
                    'class': det.get('class', 'unknown'),
                    'color': det.get('color', 'unknown'),
                    'shape': det.get('shape', 'unknown'),
                    'x': None,
                    'y': None,
                    'range': None,
                    'bearing_deg': det_bearing,
                    'source': 'camera_only',
                })

        # ── Unmatched LiDAR obstacles → "unknown" class ────────
        for idx, obs in enumerate(self.latest_obstacles):
            if idx in used_obstacle_indices:
                continue
            bx, by = obs['x'], obs['y']
            wx = px + bx * math.cos(yaw) - by * math.sin(yaw)
            wy = py + bx * math.sin(yaw) + by * math.cos(yaw)
            fused_hazards.append({
                'class': 'unknown_obstacle',
                'color': 'unknown',
                'shape': 'unknown',
                'x': round(wx, 2),
                'y': round(wy, 2),
                'range': obs.get('range', 0.0),
                'bearing_deg': obs.get('bearing_deg', 0.0),
                'source': 'lidar_only',
            })

        # ── Update hazard catalogue (counting) ──────────────────
        for h in fused_hazards:
            if h['x'] is None:
                continue  # can't catalogue without position
            self._catalogue_hazard(h)

        # ── Publish ─────────────────────────────────────────────
        self._publish_hazards(fused_hazards)
        self._publish_markers()
        self._publish_log()

    # ── Hazard catalogue management ─────────────────────────────────

    def _catalogue_hazard(self, hazard: dict):
        """Add to catalogue if it's a genuinely new hazard (far from known ones)."""
        hx, hy = hazard['x'], hazard['y']
        for cat in self.hazard_catalogue:
            dx = hx - cat['x']
            dy = hy - cat['y']
            if math.hypot(dx, dy) < self.merge_dist:
                # Close to existing — update position (running average)
                cat['x'] = round(0.8 * cat['x'] + 0.2 * hx, 2)
                cat['y'] = round(0.8 * cat['y'] + 0.2 * hy, 2)
                cat['sightings'] += 1
                # Update class if we got a better match (fused > lidar_only)
                if hazard['source'] == 'fused' and cat['source'] != 'fused':
                    cat['class'] = hazard['class']
                    cat['color'] = hazard['color']
                    cat['shape'] = hazard['shape']
                    cat['source'] = hazard['source']
                return

        # New hazard
        hazard['id'] = self.next_hazard_id
        hazard['sightings'] = 1
        self.next_hazard_id += 1
        self.hazard_catalogue.append(hazard)
        self.get_logger().info(
            f"New hazard #{hazard['id']}: {hazard['class']} at "
            f"({hazard['x']}, {hazard['y']})")

    # ── Publishing helpers ──────────────────────────────────────────

    def _publish_hazards(self, fused: list):
        msg = String()
        msg.data = json.dumps(fused)
        self.pub_hazards.publish(msg)

    def _publish_markers(self):
        """Publish RViz MarkerArray for all catalogued hazards."""
        ma = MarkerArray()

        # First, a DELETE_ALL marker to clear stale ones
        delete = Marker()
        delete.action = Marker.DELETEALL
        ma.markers.append(delete)

        for h in self.hazard_catalogue:
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'hazards'
            m.id = h['id']
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            m.pose.position = Point(
                x=float(h['x']), y=float(h['y']), z=0.0)
            m.pose.orientation.w = 1.0

            # Size: conical → tall cylinder, round → short cylinder
            if h.get('shape') == 'conical':
                m.scale = Vector3(x=0.6, y=0.6, z=2.0)
            else:
                m.scale = Vector3(x=1.0, y=1.0, z=0.6)

            m.color = MARKER_COLORS.get(
                h.get('color', 'unknown'),
                ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.85))

            m.lifetime = Duration(sec=0)  # persistent
            ma.markers.append(m)

            # Text label above the marker
            t = Marker()
            t.header = m.header
            t.ns = 'hazard_labels'
            t.id = h['id'] + 1000
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position = Point(
                x=float(h['x']), y=float(h['y']), z=2.5)
            t.pose.orientation.w = 1.0
            t.scale = Vector3(x=0.0, y=0.0, z=0.6)
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text = f"#{h['id']} {h['class']}"
            t.lifetime = Duration(sec=0)
            ma.markers.append(t)

        self.pub_markers.publish(ma)

    def _publish_log(self):
        """Publish a running count summary of unique hazards by class."""
        counts = {}
        for h in self.hazard_catalogue:
            cls = h.get('class', 'unknown')
            counts[cls] = counts.get(cls, 0) + 1

        parts = [f"{cls}: {n}" for cls, n in sorted(counts.items())]
        summary = f"Total: {len(self.hazard_catalogue)} | " + ", ".join(parts)

        msg = String()
        msg.data = summary
        self.pub_log.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
