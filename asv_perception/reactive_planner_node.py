"""
Perception-driven reactive planner — no hardcoded waypoints.

Navigation strategy:
  1. GATE NAVIGATION — find any two buoys ahead that form a pair (gate).
     Steer toward the midpoint between them to pass through safely.
     Gates can be ANY color combination (red-green, black-white, etc.).

  2. OBSTACLE AVOIDANCE — all buoys are obstacles. If no gate pair is
     found, or if a buoy is dangerously close, apply repulsive steering.

  3. EXPLORATION — when no buoys are visible, cruise forward slowly.

All buoys are counted by the sensor fusion node regardless of navigation
behavior. This planner focuses on safe passage and avoiding collisions.

Subscribes:
  /asv/detections                    (std_msgs/String — camera JSON)
  /asv/hazards                       (std_msgs/String — fused JSON)
  /wamv/sensors/imu/imu/data         (sensor_msgs/Imu)

Publishes:
  /wamv/thrusters/left/thrust        (std_msgs/Float64)
  /wamv/thrusters/right/thrust       (std_msgs/Float64)
  /asv/patrol_status                 (std_msgs/String)
"""

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64, String
from sensor_msgs.msg import Imu


def yaw_from_quaternion(q):
    """Extract yaw (heading) from a quaternion."""
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
    """Perception-driven navigation through buoy gate courses.

    Finds pairs of buoys ahead (any color), steers through the gap
    between them, avoids all buoys, and counts them via fusion.
    """

    def __init__(self):
        super().__init__('reactive_planner_node')

        # ── Parameters ──────────────────────────────────────────
        self.declare_parameter('base_thrust', 8.0)       # thrust when gate-seeking
        self.declare_parameter('explore_thrust', 5.0)    # thrust when exploring
        self.declare_parameter('danger_radius', 8.0)     # metres — avoid zone
        self.declare_parameter('kp', 5.0)
        self.declare_parameter('kd', 1.0)
        self.declare_parameter('avoidance_gain', 30.0)
        # Gate pair constraints (degrees)
        self.declare_parameter('min_gate_spread', 3.0)   # buoys too close = not a gate
        self.declare_parameter('max_gate_spread', 60.0)  # buoys too far = not a pair
        self.declare_parameter('max_gate_bearing', 50.0) # ignore gates too far off-center

        self.base_thrust = self.get_parameter('base_thrust').value
        self.explore_thrust = self.get_parameter('explore_thrust').value
        self.danger_radius = self.get_parameter('danger_radius').value
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.avoid_gain = self.get_parameter('avoidance_gain').value
        self.min_gate_spread = self.get_parameter('min_gate_spread').value
        self.max_gate_spread = self.get_parameter('max_gate_spread').value
        self.max_gate_bearing = self.get_parameter('max_gate_bearing').value

        # ── State ───────────────────────────────────────────────
        self.detections = []
        self.hazards = []
        self.yaw = None
        self.prev_heading_err = 0.0
        self.state = 'EXPLORE'

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
            String, '/asv/hazards', self._hazards_cb, 10)
        self.create_subscription(
            Imu, '/wamv/sensors/imu/imu/data', self._imu_cb, gz_qos)

        # ── Publishers ──────────────────────────────────────────
        self.pub_left = self.create_publisher(
            Float64, '/wamv/thrusters/left/thrust', 10)
        self.pub_right = self.create_publisher(
            Float64, '/wamv/thrusters/right/thrust', 10)
        self.pub_status = self.create_publisher(
            String, '/asv/patrol_status', 10)

        # ── Control loop at 10 Hz ───────────────────────────────
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            'ReactivePlanner started — perception-driven, all buoys = obstacles')

    # ── Callbacks ────────────────────────────────────────────────

    def _det_cb(self, msg: String):
        try:
            self.detections = json.loads(msg.data)
        except json.JSONDecodeError:
            self.detections = []

    def _hazards_cb(self, msg: String):
        try:
            self.hazards = json.loads(msg.data)
        except json.JSONDecodeError:
            self.hazards = []

    def _imu_cb(self, msg: Imu):
        self.yaw = yaw_from_quaternion(msg.orientation)

    # ── Gate detection ───────────────────────────────────────────

    def _find_best_gate(self):
        """Find the best pair of buoys forming a gate ahead.

        A gate is any two detected buoys that are:
          - separated by min_gate_spread to max_gate_spread degrees
          - with a midpoint within max_gate_bearing of center (ahead)

        Returns the midpoint bearing (degrees) of the best gate,
        or None if no valid gate pair is found.
        """
        if len(self.detections) < 2:
            return None

        best_gate = None
        best_score = float('inf')

        for i in range(len(self.detections)):
            for j in range(i + 1, len(self.detections)):
                b_i = self.detections[i].get('bearing_deg', 0.0)
                b_j = self.detections[j].get('bearing_deg', 0.0)

                # Ensure left/right ordering
                left_b = min(b_i, b_j)
                right_b = max(b_i, b_j)
                spread = right_b - left_b
                midpoint = (left_b + right_b) / 2.0

                # Check gate constraints
                if spread < self.min_gate_spread:
                    continue
                if spread > self.max_gate_spread:
                    continue
                if abs(midpoint) > self.max_gate_bearing:
                    continue

                # Score: prefer gates closest to center and with moderate spread
                # (wider gates are safer to navigate through)
                score = abs(midpoint) + (1.0 / max(spread, 1.0)) * 10.0

                if score < best_score:
                    best_score = score
                    best_gate = midpoint

        return best_gate

    # ── Obstacle avoidance ───────────────────────────────────────

    def _compute_avoidance(self):
        """Compute repulsive steering from ALL nearby buoys/obstacles.

        Uses camera detections (bearing + pixel area as proxy for distance)
        and fused hazards (bearing + LiDAR range if available).
        Returns a steering offset in degrees.
        """
        repulsive = 0.0
        any_close = False

        # Camera-based avoidance (all detections are obstacles)
        for det in self.detections:
            bearing = det.get('bearing_deg', 0.0)
            area = det.get('area_px', 0)

            # Larger pixel area = closer buoy
            # Threshold: area > 1500px means it's getting close
            if area > 1500 and abs(bearing) < 70:
                any_close = True
                # Repulsive strength proportional to area (closer = stronger)
                strength = min((area / 3000.0) * 12.0, 25.0)
                # Push AWAY from the buoy's bearing
                if bearing > 0:
                    repulsive -= strength  # buoy on right → steer left
                else:
                    repulsive += strength  # buoy on left → steer right

        # LiDAR-based avoidance (fused hazards with range)
        for h in self.hazards:
            rng = h.get('range')
            bearing = h.get('bearing_deg', 0.0)

            if rng is not None and rng < self.danger_radius and rng > 0.5:
                if abs(bearing) < 90:
                    any_close = True
                    strength = min(self.avoid_gain / (rng * rng), 30.0)
                    if bearing > 0:
                        repulsive -= strength
                    else:
                        repulsive += strength

        return repulsive if any_close else 0.0

    # ── Main control loop ────────────────────────────────────────

    def _control_loop(self):
        if self.yaw is None:
            return

        gate_bearing = self._find_best_gate()
        avoidance = self._compute_avoidance()
        has_obstacles = abs(avoidance) > 0.1

        # ── Determine target bearing and thrust ──────────────
        if gate_bearing is not None:
            # GATE_SEEK: steer toward gap between buoy pair
            # Also blend in avoidance to not clip the gate buoys
            self.state = 'GATE_SEEK'
            target_bearing_deg = gate_bearing * 0.7 + avoidance * 0.3
            thrust = self.base_thrust

        elif has_obstacles:
            # AVOID: buoys visible but no valid gate pair — dodge them
            self.state = 'AVOID'
            target_bearing_deg = avoidance
            thrust = self.explore_thrust

        else:
            # EXPLORE: nothing visible — cruise forward
            self.state = 'EXPLORE'
            target_bearing_deg = 0.0
            thrust = self.explore_thrust

        # ── PD heading controller ────────────────────────────
        heading_err = math.radians(target_bearing_deg)
        d_err = heading_err - self.prev_heading_err
        self.prev_heading_err = heading_err

        steer = self.kp * heading_err + self.kd * d_err

        speed_factor = max(0.3, 1.0 - abs(heading_err) / math.pi)
        base = thrust * speed_factor

        left_thrust = max(-100.0, min(100.0, base + steer))
        right_thrust = max(-100.0, min(100.0, base - steer))

        self._publish_thrust(left_thrust, right_thrust)

        # ── Status ───────────────────────────────────────────
        n_det = len(self.detections)
        gate_str = f'gate={gate_bearing:.1f}°' if gate_bearing is not None else 'no_gate'
        status = (
            f'{self.state} | {gate_str} | '
            f'avoid={avoidance:.1f}° | '
            f'detections={n_det}'
        )
        self._publish_status(status)

    # ── Helpers ──────────────────────────────────────────────────

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
