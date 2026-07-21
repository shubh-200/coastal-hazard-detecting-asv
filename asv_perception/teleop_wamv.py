"""
Keyboard teleop for the WAM-V in VRX.

Maps WASD keys to differential thrust commands:
  W / ↑  — forward (both thrusters positive)
  S / ↓  — reverse (both thrusters negative)
  A / ←  — turn left (right thruster > left)
  D / →  — turn right (left thruster > right)
  Q      — forward-left
  E      — forward-right
  SPACE  — stop (zero thrust)
  +/-    — increase/decrease thrust level

Publishes:
  /wamv/thrusters/left/thrust   (std_msgs/Float64)
  /wamv/thrusters/right/thrust  (std_msgs/Float64)

Usage:
  ros2 run asv_perception teleop_wamv
"""

import sys
import termios
import tty
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


HELP_TEXT = """
─────────────────────────────────────────
  WAM-V Keyboard Teleop
─────────────────────────────────────────
  W / ↑    : forward
  S / ↓    : reverse
  A / ←    : turn left
  D / →    : turn right
  Q        : forward-left
  E        : forward-right
  SPACE    : stop
  + / =    : increase thrust (+5)
  - / _    : decrease thrust (-5)
  P        : print current pose topic
  Ctrl-C   : quit
─────────────────────────────────────────
  Current thrust level: {thrust:.0f} N
─────────────────────────────────────────
"""

# Key codes for arrow keys (escape sequences)
ARROW_UP = '\x1b[A'
ARROW_DOWN = '\x1b[B'
ARROW_RIGHT = '\x1b[C'
ARROW_LEFT = '\x1b[D'


def get_key(timeout=0.1):
    """Read a single keypress from stdin (non-blocking with timeout)."""
    import select
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
            # Check for escape sequence (arrow keys)
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                return ch + ch2 + ch3
            return ch
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class TeleopWamvNode(Node):

    def __init__(self):
        super().__init__('teleop_wamv')

        self.pub_left = self.create_publisher(
            Float64, '/wamv/thrusters/left/thrust', 10)
        self.pub_right = self.create_publisher(
            Float64, '/wamv/thrusters/right/thrust', 10)

        self.thrust_level = 15.0  # default thrust in Newtons
        self.left = 0.0
        self.right = 0.0

        self.get_logger().info('Teleop node started — focus this terminal and use WASD')

    def run(self):
        print(HELP_TEXT.format(thrust=self.thrust_level))

        try:
            while rclpy.ok():
                start_time = time.time()
                key = get_key(timeout=0.05)

                if key in ('w', 'W', ARROW_UP):
                    self.left = self.thrust_level
                    self.right = self.thrust_level
                elif key in ('s', 'S', ARROW_DOWN):
                    self.left = -self.thrust_level
                    self.right = -self.thrust_level
                elif key in ('a', 'A', ARROW_LEFT):
                    self.left = -self.thrust_level * 0.5
                    self.right = self.thrust_level
                elif key in ('d', 'D', ARROW_RIGHT):
                    self.left = self.thrust_level
                    self.right = -self.thrust_level * 0.5
                elif key in ('q', 'Q'):
                    self.left = self.thrust_level * 0.3
                    self.right = self.thrust_level
                elif key in ('e', 'E'):
                    self.left = self.thrust_level
                    self.right = self.thrust_level * 0.3
                elif key == ' ':
                    self.left = 0.0
                    self.right = 0.0
                elif key in ('+', '='):
                    self.thrust_level = min(100.0, self.thrust_level + 5.0)
                    print(f'\r  Thrust level: {self.thrust_level:.0f} N    ', end='')
                elif key in ('-', '_'):
                    self.thrust_level = max(5.0, self.thrust_level - 5.0)
                    print(f'\r  Thrust level: {self.thrust_level:.0f} N    ', end='')
                elif key in ('p', 'P'):
                    print('\n  Hint: run in another terminal:')
                    print('  ros2 topic echo /wamv/pose --field pose.position --rate 1\n')
                elif key == '\x03':  # Ctrl-C
                    break

                self._publish()

                # Cap published rate to ~10 Hz (0.1s period) to prevent overloading Gazebo
                elapsed = time.time() - start_time
                if elapsed < 0.1:
                    time.sleep(0.1 - elapsed)

        except KeyboardInterrupt:
            pass

        finally:
            # Stop the boat on exit
            self.left = 0.0
            self.right = 0.0
            self._publish()
            print('\n  Stopped. Thrusters zeroed.')

    def _publish(self):
        l_msg = Float64()
        l_msg.data = self.left
        r_msg = Float64()
        r_msg.data = self.right
        self.pub_left.publish(l_msg)
        self.pub_right.publish(r_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopWamvNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
