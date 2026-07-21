"""
Camera-based hazard detector using classical CV (HSV color segmentation).

Detects buoys visible in the front-left camera feed and classifies them by
color and shape:
  - Red conical, Green conical, White conical, Black conical
  - Orange round, Black round

Detection pipeline:
  1. BGR → HSV conversion
  2. Per-color mask thresholding
  3. Contour extraction + area filtering
  4. Shape classification via bounding-rect aspect ratio
     (conical buoys are taller than wide; round buoys are ~square)
  5. Bearing estimation from pixel column position

Subscribes:
  /wamv/sensors/cameras/front_left_camera_sensor/image_raw  (sensor_msgs/Image)
  /wamv/sensors/cameras/front_left_camera_sensor/camera_info (sensor_msgs/CameraInfo)

Publishes:
  /asv/detections          (std_msgs/String — JSON array)
  /asv/detection_overlay   (sensor_msgs/Image — annotated)
"""

import json
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


# ── HSV colour ranges ───────────────────────────────────────────────
# Each entry: (label, lower_hsv, upper_hsv)
# OpenCV HSV: H 0-179, S 0-255, V 0-255
#
# Red wraps around H=0/180, so we need two ranges.
# White has low saturation and high value.
# Black has very low value regardless of H/S.
COLOR_RANGES = [
    # Red  (two ranges to handle hue wrap-around)
    ('red',    np.array([  0,  80, 80]), np.array([ 10, 255, 255])),
    ('red',    np.array([170,  80, 80]), np.array([179, 255, 255])),
    # Green
    ('green',  np.array([ 35,  60, 60]), np.array([ 85, 255, 255])),
    # Orange
    ('orange', np.array([ 10,  80, 80]), np.array([ 25, 255, 255])),
    # White  (very low saturation, high value)
    ('white',  np.array([  0,   0, 180]), np.array([179,  60, 255])),
    # Black  (very low value — but filter sky/water via saturation too)
    ('black',  np.array([  0,   0,   0]), np.array([179, 255,  50])),
]

# Draw colours for bounding boxes (BGR)
DRAW_COLORS = {
    'red':    (0, 0, 255),
    'green':  (0, 200, 0),
    'orange': (0, 140, 255),
    'white':  (220, 220, 220),
    'black':  (80, 80, 80),
}

# ── Shape classification ────────────────────────────────────────────
# Conical buoys are taller than wide → aspect_ratio (h/w) > threshold
# Round buoys are roughly square  → aspect_ratio ≈ 1.0
CONICAL_ASPECT_THRESHOLD = 1.3  # h/w > 1.3 → conical

# Minimum contour area in pixels to filter noise
MIN_CONTOUR_AREA = 300
# Maximum fraction of image area (filter false positives like sky/water)
MAX_CONTOUR_FRAC = 0.25


class HazardDetectorNode(Node):
    """Detects coloured buoys in camera images via HSV thresholding."""

    def __init__(self):
        super().__init__('hazard_detector_node')

        self.bridge = CvBridge()
        self.camera_hfov = None     # horizontal field of view (radians)
        self.image_width = None

        # Adaptive frame skipping for real-time performance
        self.declare_parameter('max_process_hz', 10.0)  # cap processing rate
        self.max_dt = 1.0 / self.get_parameter('max_process_hz').value
        self.last_process_time = 0.0

        # ── QoS for Gazebo bridge (best-effort) ─────────────────
        gz_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,  # only care about latest frame
        )

        # ── Subscribers ─────────────────────────────────────────
        self.create_subscription(
            Image,
            '/wamv/sensors/cameras/front_left_camera_sensor/image_raw',
            self._image_cb,
            gz_qos,
        )
        self.create_subscription(
            CameraInfo,
            '/wamv/sensors/cameras/front_left_camera_sensor/camera_info',
            self._caminfo_cb,
            gz_qos,
        )

        # ── Publishers ──────────────────────────────────────────
        self.pub_detections = self.create_publisher(String, '/asv/detections', 10)
        self.pub_overlay = self.create_publisher(Image, '/asv/detection_overlay', 5)

        self.get_logger().info('HazardDetector started — waiting for camera frames')

    # ── Camera info callback ────────────────────────────────────────

    def _caminfo_cb(self, msg: CameraInfo):
        if self.camera_hfov is None:
            # fx is in pixels; hfov = 2 * atan(width / (2*fx))
            fx = msg.k[0]
            self.image_width = msg.width
            if fx > 0:
                import math
                self.camera_hfov = 2.0 * math.atan(msg.width / (2.0 * fx))
                self.get_logger().info(
                    f'Camera HFOV = {math.degrees(self.camera_hfov):.1f}°, '
                    f'width = {msg.width}px')

    # ── Image callback ──────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        # Adaptive throttle: skip frames to maintain real-time processing
        now = time.monotonic()
        if now - self.last_process_time < self.max_dt:
            return
        self.last_process_time = now

        # Convert ROS Image → OpenCV BGR
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        detections = self._detect(frame)

        # Publish detections as JSON
        det_msg = String()
        det_msg.data = json.dumps(detections)
        self.pub_detections.publish(det_msg)

        # Publish annotated overlay image
        overlay = self._draw_overlay(frame, detections)
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header = msg.header
            self.pub_overlay.publish(overlay_msg)
        except Exception as e:
            self.get_logger().error(f'Overlay publish error: {e}')

    # ── Core detection pipeline ─────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> list:
        """Run HSV color segmentation + shape classification on a BGR frame.

        Returns a list of detection dicts:
          [{"class": "red_conical", "bearing_deg": -12.5,
            "bbox": [x,y,w,h], "area_px": 1234, "cx": 320}, ...]
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        img_area = h * w

        # Mask out the bottom 15% of the image (hull/deck self-view)
        # and top 30% (sky — reduces white/black false positives)
        roi_top = int(h * 0.30)
        roi_bottom = int(h * 0.85)

        detections = []
        seen_regions = []  # track bboxes to avoid duplicate detections

        for color_name, lower, upper in COLOR_RANGES:
            mask = cv2.inRange(hsv, lower, upper)

            # Zero out non-ROI rows
            mask[:roi_top, :] = 0
            mask[roi_bottom:, :] = 0

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue
                if area > MAX_CONTOUR_FRAC * img_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)

                # De-duplicate: skip if this bbox overlaps heavily with an
                # already-detected region (handles the two red ranges)
                cx, cy = x + bw // 2, y + bh // 2
                duplicate = False
                for (sx, sy, sw, sh) in seen_regions:
                    if (abs(cx - (sx + sw // 2)) < sw * 0.5 and
                            abs(cy - (sy + sh // 2)) < sh * 0.5):
                        duplicate = True
                        break
                if duplicate:
                    continue
                seen_regions.append((x, y, bw, bh))

                # Shape classification: aspect ratio h/w
                aspect = bh / bw if bw > 0 else 1.0
                if color_name == 'orange':
                    # Orange buoys are always round per the world
                    shape = 'round'
                elif color_name in ('red', 'green', 'white'):
                    # These are always conical per the world
                    shape = 'conical'
                elif color_name == 'black':
                    # Black can be conical or round — use aspect ratio
                    shape = 'conical' if aspect > CONICAL_ASPECT_THRESHOLD else 'round'
                else:
                    shape = 'conical' if aspect > CONICAL_ASPECT_THRESHOLD else 'round'

                # Bearing angle from image center
                bearing_deg = self._pixel_to_bearing(cx)

                detections.append({
                    'class': f'{color_name}_{shape}',
                    'color': color_name,
                    'shape': shape,
                    'bearing_deg': round(bearing_deg, 1),
                    'bbox': [int(x), int(y), int(bw), int(bh)],
                    'area_px': int(area),
                    'cx': int(cx),
                    'cy': int(cy),
                })

        return detections

    # ── Bearing estimation ──────────────────────────────────────────

    def _pixel_to_bearing(self, pixel_x: int) -> float:
        """Convert a pixel x-coordinate to a bearing angle in degrees.

        0° = straight ahead, negative = left, positive = right.
        """
        if self.camera_hfov is None or self.image_width is None:
            return 0.0
        import math
        # Normalise pixel to [-0.5, 0.5] of image width
        norm_x = (pixel_x / self.image_width) - 0.5
        return math.degrees(norm_x * self.camera_hfov)

    # ── Overlay drawing ─────────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes, labels, and bearing info on the frame."""
        overlay = frame.copy()
        for det in detections:
            x, y, bw, bh = det['bbox']
            color = DRAW_COLORS.get(det['color'], (255, 255, 0))
            label = f"{det['class']} {det['bearing_deg']:+.1f}°"

            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(overlay, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # Detection count summary in top-left corner
        summary = f"Detections: {len(detections)}"
        cv2.putText(overlay, summary, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA)

        return overlay


def main(args=None):
    rclpy.init(args=args)
    node = HazardDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
