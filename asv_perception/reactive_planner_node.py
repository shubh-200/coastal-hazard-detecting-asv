"""
Reactive patrol planner with potential-field obstacle avoidance.

Behavior:
  PATROL  — follow waypoints in a loop using heading-based steering
  AVOID   — when a hazard is within danger_radius, apply repulsive
             potential-field vector to steer around it
  LOITER  — hold position when all waypoints visited (or loop back)

Subscribes:
  /wamv/pose                     (geometry_msgs/PoseStamped)
  /asv/hazards                   (std_msgs/String — JSON array)

Publishes:
  /wamv/thrusters/left/thrust    (std_msgs/Float64)
  /wamv/thrusters/right/thrust   (std_msgs/Float64)
  /asv/patrol_status             (std_msgs/String)
"""

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64, String
from geometry_msgs.msg import PoseStamped


def yaw_from_quaternion(q):
    """Extract yaw (heading) from a quaternion.  Returns radians, 0 = +X."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class ReactivePlannerNode(Node):
    """Waypoint-loop patrol with potential-field hazard avoidance."""

    def __init__(self):
        super().__init__('reactive_planner_node')

        # ── Parameters (overridable from launch / YAML) ──────────────
        self.declare_parameter('waypoints', [])       # flat list [x1,y1, x2,y2, ...]
        self.declare_parameter('arrival_radius', 5.0) # metres
        self.declare_parameter('base_thrust', 12.0)   # newtons
        self.declare_parameter('danger_radius', 8.0)  # metres — triggers avoidance
        self.declare_parameter('kp', 5.0)             # heading PD gains
        self.declare_parameter('kd', 1.0)
        self.declare_parameter('avoidance_gain', 30.0)  # repulsive field strength
        self.declare_parameter('loop_patrol', True)

        self._load_params()

        # ── State ────────────────────────────────────────────────────
        self.current_wp_idx = 0
        self.pose = None           # latest PoseStamped
        self.hazards = []          # latest parsed hazard list
        self.prev_heading_err = 0.0
        self.state = 'PATROL'     # PATROL | AVOID | LOITER

        # ── QoS for Gazebo bridge topics (best-effort, volatile) ─────
        gz_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )

        # ── Subscribers ─────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, '/wamv/pose', self._pose_cb, gz_qos)
        self.create_subscription(
            TFMessage, '/wamv/pose', self._tf_pose_cb, gz_qos)
        self.create_subscription(
            String, '/asv/hazards', self._hazards_cb, 10)

        # ── Publishers ───────────────────────────────────────────────
        self.pub_left = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.pub_right = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.pub_status = self.create_publisher(String, '/asv/patrol_status', 10)

        # ── Control loop at 10 Hz ────────────────────────────────────
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'ReactivePlanner started — {len(self.waypoints)} waypoints, '
            f'danger_radius={self.danger_radius:.1f}m')

    # ── Parameter loading ────────────────────────────────────────────

    def _load_params(self):
        flat = self.get_parameter('waypoints').get_parameter_value().double_array_value
        if len(flat) == 0:
            # Fallback: a small square patrol (will be replaced by real coords)
            flat = [0.0, 0.0,  50.0, 0.0,  50.0, 50.0,  0.0, 50.0]
            self.get_logger().warn(
                'No waypoints provided — using default square patrol.  '
                'Set the "waypoints" param or load patrol_waypoints.yaml.')
        self.waypoints = [(flat[i], flat[i+1]) for i in range(0, len(flat) - 1, 2)]

        self.arrival_radius = self.get_parameter('arrival_radius').value
        self.base_thrust    = self.get_parameter('base_thrust').value
        self.danger_radius  = self.get_parameter('danger_radius').value
        self.kp             = self.get_parameter('kp').value
        self.kd             = self.get_parameter('kd').value
        self.avoid_gain     = self.get_parameter('avoidance_gain').value
        self.loop_patrol    = self.get_parameter('loop_patrol').value

    # ── Callbacks ────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped):
        self.pose = msg

    def _tf_pose_cb(self, msg: TFMessage):
        for tf in msg.transforms:
            # Match world -> wamv base link transform
            if 'base_link' in tf.child_frame_id and 'sensor' not in tf.child_frame_id:
                p = PoseStamped()
                p.header = tf.header
                p.pose.position.x = tf.transform.translation.x
                p.pose.position.y = tf.transform.translation.y
                p.pose.position.z = tf.transform.translation.z
                p.pose.orientation = tf.transform.rotation
                self.pose = p
                break

    def _hazards_cb(self, msg: String):
        try:
            self.hazards = json.loads(msg.data)
        except json.JSONDecodeError:
            self.hazards = []

    # ── Main control loop ────────────────────────────────────────────

    def _control_loop(self):
        if self.pose is None:
            return  # wait for first pose

        px = self.pose.pose.position.x
        py = self.pose.pose.position.y
        yaw = yaw_from_quaternion(self.pose.pose.orientation)

        # ── Check for nearby hazards ─────────────────────────────
        repulsive_x, repulsive_y = 0.0, 0.0
        hazard_nearby = False
        for h in self.hazards:
            hx, hy = h.get('x', 0.0), h.get('y', 0.0)
            dx, dy = hx - px, hy - py
            dist = math.hypot(dx, dy)
            if dist < self.danger_radius and dist > 0.1:
                hazard_nearby = True
                # Repulsive vector: push AWAY from hazard, strength ∝ 1/dist²
                strength = self.avoid_gain / (dist * dist)
                repulsive_x -= strength * (dx / dist)
                repulsive_y -= strength * (dy / dist)

        # ── Determine desired heading ────────────────────────────
        if self.state == 'LOITER':
            self._publish_thrust(0.0, 0.0)
            self._publish_status('LOITER — patrol complete')
            return

        if self.current_wp_idx >= len(self.waypoints):
            if self.loop_patrol:
                self.current_wp_idx = 0
                self.get_logger().info('Patrol loop restart')
            else:
                self.state = 'LOITER'
                return

        # Attractive vector toward current waypoint
        wp_x, wp_y = self.waypoints[self.current_wp_idx]
        attract_x = wp_x - px
        attract_y = wp_y - py
        dist_to_wp = math.hypot(attract_x, attract_y)

        # Check waypoint arrival
        if dist_to_wp < self.arrival_radius:
            self.get_logger().info(
                f'Reached waypoint {self.current_wp_idx} '
                f'({wp_x:.1f}, {wp_y:.1f})')
            self.current_wp_idx += 1
            return

        # Combine attractive + repulsive
        goal_x = attract_x + repulsive_x
        goal_y = attract_y + repulsive_y

        desired_heading = math.atan2(goal_y, goal_x)

        if hazard_nearby:
            self.state = 'AVOID'
        else:
            self.state = 'PATROL'

        # ── PD heading controller ────────────────────────────────
        heading_err = normalize_angle(desired_heading - yaw)
        d_err = heading_err - self.prev_heading_err
        self.prev_heading_err = heading_err

        steer = self.kp * heading_err + self.kd * d_err

        # Reduce thrust when turning hard (avoid overshooting)
        speed_factor = max(0.3, 1.0 - abs(heading_err) / math.pi)
        base = self.base_thrust * speed_factor

        left_thrust  = base + steer
        right_thrust = base - steer

        # Clamp to VRX thruster range
        left_thrust  = max(-100.0, min(100.0, left_thrust))
        right_thrust = max(-100.0, min(100.0, right_thrust))

        self._publish_thrust(left_thrust, right_thrust)
        self._publish_status(
            f'{self.state} — wp {self.current_wp_idx}/{len(self.waypoints)} '
            f'dist={dist_to_wp:.1f}m heading_err={math.degrees(heading_err):.1f}°')

    # ── Helpers ──────────────────────────────────────────────────────

    def _publish_thrust(self, left: float, right: float):
        l_msg = Float64()
        l_msg.data = left
        r_msg = Float64()
        r_msg.data = right
        self.pub_left.publish(l_msg)
        self.pub_right.publish(r_msg)

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReactivePlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
