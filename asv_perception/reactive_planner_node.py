"""
Reactive patrol planner with potential-field obstacle avoidance.

Uses GPS + IMU for localization (no dependency on sim ground-truth pose).
Waypoints are specified in GPS lat/lon coordinates and converted to a
local ENU (East-North-Up) frame at runtime using the first GPS fix as
the origin. This matches real-world maritime ASV practice.

Behavior:
  PATROL  — follow waypoints in a loop using heading-based steering
  AVOID   — when a hazard is within danger_radius, apply repulsive
             potential-field vector to steer around it
  LOITER  — hold position when all waypoints visited (or loop back)

Subscribes:
  /wamv/sensors/gps/gps/fix      (sensor_msgs/NavSatFix)
  /wamv/sensors/imu/imu/data     (sensor_msgs/Imu)
  /asv/hazards                   (std_msgs/String — JSON array)

Publishes:
  /wamv/thrusters/left/thrust    (std_msgs/Float64)
  /wamv/thrusters/right/thrust   (std_msgs/Float64)
  /asv/patrol_status             (std_msgs/String)
  /asv/vehicle_enu               (std_msgs/String — JSON: x, y, yaw for debugging)
"""

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64, String
from sensor_msgs.msg import NavSatFix, Imu


# ── GPS → local ENU conversion ──────────────────────────────────────
# Uses a flat-earth approximation (accurate to < 1m within a few km of
# the origin). This is standard practice for local ASV navigation.

def gps_to_enu(lat, lon, origin_lat, origin_lon):
    """Convert GPS lat/lon to local ENU (East, North) metres relative to origin.

    Uses the WGS-84 ellipsoid radii at the origin latitude.
    Returns (east, north) in metres — east = +X, north = +Y.
    """
    lat_r = math.radians(lat)
    origin_lat_r = math.radians(origin_lat)

    # Earth radii at origin latitude (WGS-84 approximation)
    R_EARTH = 6378137.0                      # equatorial radius
    E2 = 0.00669437999014                    # first eccentricity squared
    sin_lat = math.sin(origin_lat_r)
    N = R_EARTH / math.sqrt(1 - E2 * sin_lat * sin_lat)

    # Differences
    dlat = math.radians(lat - origin_lat)
    dlon = math.radians(lon - origin_lon)

    north = dlat * (N * (1 - E2) / (1 - E2 * sin_lat * sin_lat))
    east = dlon * N * math.cos(origin_lat_r)

    return east, north


def enu_to_gps(east, north, origin_lat, origin_lon):
    """Convert local ENU (east, north) metres back to GPS lat/lon.

    Inverse of gps_to_enu. Used for converting ENU waypoints to lat/lon
    if you prefer to specify waypoints in local coordinates.
    """
    origin_lat_r = math.radians(origin_lat)
    R_EARTH = 6378137.0
    E2 = 0.00669437999014
    sin_lat = math.sin(origin_lat_r)
    N = R_EARTH / math.sqrt(1 - E2 * sin_lat * sin_lat)

    dlat = north / (N * (1 - E2) / (1 - E2 * sin_lat * sin_lat))
    dlon = east / (N * math.cos(origin_lat_r))

    return origin_lat + math.degrees(dlat), origin_lon + math.degrees(dlon)


def yaw_from_quaternion(q):
    """Extract yaw (heading) from a quaternion.  Returns radians, 0 = +X (East)."""
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
    """Waypoint-loop patrol with potential-field hazard avoidance.

    Uses GPS for position and IMU for heading. Waypoints can be
    specified in either GPS lat/lon or local ENU coordinates.
    """

    def __init__(self):
        super().__init__('reactive_planner_node')

        # ── Parameters ──────────────────────────────────────────
        # Waypoints as flat list: [lat1, lon1, lat2, lon2, ...]
        # If use_enu_waypoints is True, interpret as [east1, north1, ...]
        self.declare_parameter('waypoints', [])
        self.declare_parameter('use_enu_waypoints', False)
        self.declare_parameter('arrival_radius', 5.0)
        self.declare_parameter('base_thrust', 8.0)     # keep low for WSL2 stability
        self.declare_parameter('danger_radius', 8.0)
        self.declare_parameter('kp', 5.0)
        self.declare_parameter('kd', 1.0)
        self.declare_parameter('avoidance_gain', 30.0)
        self.declare_parameter('loop_patrol', True)

        self._load_params()

        # ── State ───────────────────────────────────────────────
        self.current_wp_idx = 0
        self.gps_origin = None    # (lat, lon) — set from first GPS fix
        self.pos_enu = None       # (east, north) in metres
        self.yaw = None           # heading from IMU (radians, 0 = East)
        self.hazards = []
        self.prev_heading_err = 0.0
        self.state = 'PATROL'

        # ── QoS for Gazebo bridge topics ────────────────────────
        gz_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )

        # ── Subscribers ─────────────────────────────────────────
        self.create_subscription(
            NavSatFix,
            '/wamv/sensors/gps/gps/fix',
            self._gps_cb,
            gz_qos,
        )
        self.create_subscription(
            Imu,
            '/wamv/sensors/imu/imu/data',
            self._imu_cb,
            gz_qos,
        )
        self.create_subscription(
            String, '/asv/hazards', self._hazards_cb, 10)

        # ── Publishers ──────────────────────────────────────────
        self.pub_left = self.create_publisher(
            Float64, '/wamv/thrusters/left/thrust', 10)
        self.pub_right = self.create_publisher(
            Float64, '/wamv/thrusters/right/thrust', 10)
        self.pub_status = self.create_publisher(
            String, '/asv/patrol_status', 10)
        self.pub_enu = self.create_publisher(
            String, '/asv/vehicle_enu', 10)

        # ── Control loop at 10 Hz ───────────────────────────────
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'ReactivePlanner started — {len(self.waypoints_raw)} waypoints, '
            f'base_thrust={self.base_thrust:.1f}N, '
            f'danger_radius={self.danger_radius:.1f}m')

    # ── Parameter loading ────────────────────────────────────────

    def _load_params(self):
        flat = list(
            self.get_parameter('waypoints').get_parameter_value().double_array_value)
        self.use_enu = self.get_parameter('use_enu_waypoints').value
        self.arrival_radius = self.get_parameter('arrival_radius').value
        self.base_thrust = self.get_parameter('base_thrust').value
        self.danger_radius = self.get_parameter('danger_radius').value
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.avoid_gain = self.get_parameter('avoidance_gain').value
        self.loop_patrol = self.get_parameter('loop_patrol').value

        if len(flat) == 0:
            self.get_logger().warn(
                'No waypoints provided — will use a default local square '
                'once GPS origin is established.')
            flat = []

        # Store raw pairs — will convert to ENU once GPS origin is known
        self.waypoints_raw = [
            (flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
        self.waypoints_enu = []  # populated in _resolve_waypoints

    def _resolve_waypoints(self):
        """Convert raw waypoints to ENU once GPS origin is available."""
        if self.gps_origin is None:
            return

        if len(self.waypoints_raw) == 0:
            # Default: small square patrol in ENU (50m sides)
            self.waypoints_enu = [
                (0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)]
            self.get_logger().info(
                'Using default 50m square patrol around GPS origin')
            return

        if self.use_enu:
            # Already in ENU
            self.waypoints_enu = list(self.waypoints_raw)
        else:
            # Convert lat/lon to ENU
            origin_lat, origin_lon = self.gps_origin
            self.waypoints_enu = []
            for lat, lon in self.waypoints_raw:
                e, n = gps_to_enu(lat, lon, origin_lat, origin_lon)
                self.waypoints_enu.append((e, n))

        wp_str = ', '.join(
            f'({e:.1f}, {n:.1f})' for e, n in self.waypoints_enu)
        self.get_logger().info(f'Waypoints (ENU): {wp_str}')

    # ── Callbacks ────────────────────────────────────────────────

    def _gps_cb(self, msg: NavSatFix):
        lat, lon = msg.latitude, msg.longitude

        # Set origin on first valid fix
        if self.gps_origin is None:
            self.gps_origin = (lat, lon)
            self.get_logger().info(
                f'GPS origin set: ({lat:.6f}, {lon:.6f})')
            self._resolve_waypoints()

        e, n = gps_to_enu(lat, lon, self.gps_origin[0], self.gps_origin[1])
        self.pos_enu = (e, n)

    def _imu_cb(self, msg: Imu):
        self.yaw = yaw_from_quaternion(msg.orientation)

    def _hazards_cb(self, msg: String):
        try:
            self.hazards = json.loads(msg.data)
        except json.JSONDecodeError:
            self.hazards = []

    # ── Main control loop ────────────────────────────────────────

    def _control_loop(self):
        if self.pos_enu is None or self.yaw is None:
            return  # wait for GPS + IMU

        if len(self.waypoints_enu) == 0:
            return  # waypoints not yet resolved

        px, py = self.pos_enu
        yaw = self.yaw

        # Publish current ENU position for debugging / waypoint recording
        enu_msg = String()
        enu_msg.data = json.dumps({
            'east': round(px, 2),
            'north': round(py, 2),
            'yaw_deg': round(math.degrees(yaw), 1),
            'gps_origin': list(self.gps_origin) if self.gps_origin else None,
        })
        self.pub_enu.publish(enu_msg)

        # ── Check for nearby hazards ─────────────────────────────
        repulsive_x, repulsive_y = 0.0, 0.0
        hazard_nearby = False
        for h in self.hazards:
            hx, hy = h.get('x', 0.0), h.get('y', 0.0)
            if hx is None or hy is None:
                continue
            dx, dy = hx - px, hy - py
            dist = math.hypot(dx, dy)
            if dist < self.danger_radius and dist > 0.1:
                hazard_nearby = True
                strength = self.avoid_gain / (dist * dist)
                repulsive_x -= strength * (dx / dist)
                repulsive_y -= strength * (dy / dist)

        # ── Determine desired heading ────────────────────────────
        if self.state == 'LOITER':
            self._publish_thrust(0.0, 0.0)
            self._publish_status('LOITER — patrol complete')
            return

        if self.current_wp_idx >= len(self.waypoints_enu):
            if self.loop_patrol:
                self.current_wp_idx = 0
                self.get_logger().info('Patrol loop restart')
            else:
                self.state = 'LOITER'
                return

        wp_e, wp_n = self.waypoints_enu[self.current_wp_idx]
        attract_x = wp_e - px
        attract_y = wp_n - py
        dist_to_wp = math.hypot(attract_x, attract_y)

        if dist_to_wp < self.arrival_radius:
            self.get_logger().info(
                f'Reached waypoint {self.current_wp_idx} '
                f'({wp_e:.1f}, {wp_n:.1f})')
            self.current_wp_idx += 1
            return

        goal_x = attract_x + repulsive_x
        goal_y = attract_y + repulsive_y
        desired_heading = math.atan2(goal_y, goal_x)

        self.state = 'AVOID' if hazard_nearby else 'PATROL'

        # ── PD heading controller ────────────────────────────────
        heading_err = normalize_angle(desired_heading - yaw)
        d_err = heading_err - self.prev_heading_err
        self.prev_heading_err = heading_err

        steer = self.kp * heading_err + self.kd * d_err

        speed_factor = max(0.3, 1.0 - abs(heading_err) / math.pi)
        base = self.base_thrust * speed_factor

        left_thrust = max(-100.0, min(100.0, base + steer))
        right_thrust = max(-100.0, min(100.0, base - steer))

        self._publish_thrust(left_thrust, right_thrust)
        self._publish_status(
            f'{self.state} — wp {self.current_wp_idx}/{len(self.waypoints_enu)} '
            f'dist={dist_to_wp:.1f}m hdg_err={math.degrees(heading_err):.1f}° '
            f'pos=({px:.1f},{py:.1f})')

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
