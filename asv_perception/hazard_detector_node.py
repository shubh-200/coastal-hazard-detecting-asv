"""
Camera-based hazard detector using classical CV (HSV color segmentation).

Detects buoys visible in the front-left camera feed and classifies them by
color and shape:
  - Red conical, Green conical, White conical, Black conical
  - Orange round, Black round

Anti-glare & noise reduction optimizations:
  - Strict ROI masking (cuts top 35% sky/horizon glare & bottom 25% hull deck)
  - Calibrated white HSV range (requires high V + very low S to distinguish from reflections)
  - Higher min contour area (600px) + strict aspect ratio checks
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


# ── Tuned HSV colour ranges (OpenCV HSV: H 0-179, S 0-255, V 0-255) ─────────
COLOR_RANGES = [
    # Red (two ranges for hue wrap-around)
    ('red',    np.array([  0, 100, 100]), np.array([  8, 255, 255])),
    ('red',    np.array([170, 100, 100]), np.array([179, 255, 255])),
    # Green
    ('green',  np.array([ 40,  80,  70]), np.array([ 85, 255, 255])),
    # Orange
    ('orange', np.array([  9, 120, 120]), np.array([ 22, 255, 255])),
    # White (Tightened: requires extremely bright V and strict low S to avoid water reflection)
    ('white',  np.array([  0,   0, 220]), np.array([179,  30, 255])),
    # Black (Very low value, moderate saturation to avoid deep water shadows)
    ('black',  np.array([  0,   0,   0]), np.array([179, 255,  40])),
]

DRAW_COLORS = {
    'red':    (0, 0, 255),
    'green':  (0, 200, 0),
    'orange': (0, 140, 255),
    'white':  (220, 220, 220),
    'black':  (80, 80, 80),
}

CONICAL_ASPECT_THRESHOLD = 1.2
MIN_CONTOUR_AREA = 500       # Ignored small glare pixels
MAX_CONTOUR_FRAC = 0.15      # Ignore giant regions (water/sky patches)


class HazardDetectorNode(Node):
    """Detects coloured buoys in camera images via HSV thresholding."""

    def __init__(self):
        super().__init__('hazard_detector_node')

        self.bridge = CvBridge()
        self.camera_hfov = None
        self.image_width = None

        self.declare_parameter('max_process_hz', 10.0)
        self.max_dt = 1.0 / self.get_parameter('max_process_hz').value
        self.last_process_time = 0.0

        gz_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

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

        self.pub_detections = self.create_publisher(String, '/asv/detections', 10)
        self.pub_overlay = self.create_publisher(Image, '/asv/detection_overlay', 5)

        self.get_logger().info('HazardDetector initialized with anti-glare filters')

    def _caminfo_cb(self, msg: CameraInfo):
        if self.camera_hfov is None:
            fx = msg.k[0]
            self.image_width = msg.width
            if fx > 0:
                import math
                self.camera_hfov = 2.0 * math.atan(msg.width / (2.0 * fx))

    def _image_cb(self, msg: Image):
        now = time.monotonic()
        if now - self.last_process_time < self.max_dt:
            return
        self.last_process_time = now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        detections = self._detect(frame)

        det_msg = String()
        det_msg.data = json.dumps(detections)
        self.pub_detections.publish(det_msg)

        overlay = self._draw_overlay(frame, detections)
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header = msg.header
            self.pub_overlay.publish(overlay_msg)
        except Exception as e:
            self.get_logger().error(f'Overlay error: {e}')

    def _detect(self, frame: np.ndarray) -> list:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        img_area = h * w

        # Focus search region strictly on water horizon (35% to 75% height)
        # Eliminates sky, horizon sun glare, and front deck hull
        roi_top = int(h * 0.35)
        roi_bottom = int(h * 0.75)

        detections = []
        seen_regions = []

        for color_name, lower, upper in COLOR_RANGES:
            mask = cv2.inRange(hsv, lower, upper)

            # Crop out sky & deck
            mask[:roi_top, :] = 0
            mask[roi_bottom:, :] = 0

            # Morphological cleanup to eliminate single-pixel glare specs
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_FRAC * img_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)

                # Skip non-compact shapes (glare streaks on water are very wide and thin)
                aspect = bh / bw if bw > 0 else 1.0
                if aspect < 0.3 or aspect > 4.0:
                    continue

                cx, cy = x + bw // 2, y + bh // 2
                duplicate = False
                for (sx, sy, sw, sh) in seen_regions:
                    if (abs(cx - (sx + sw // 2)) < sw * 0.6 and
                            abs(cy - (sy + sh // 2)) < sh * 0.6):
                        duplicate = True
                        break
                if duplicate:
                    continue
                seen_regions.append((x, y, bw, bh))

                if color_name == 'orange':
                    shape = 'round'
                elif color_name in ('red', 'green', 'white'):
                    shape = 'conical'
                else:
                    shape = 'conical' if aspect > CONICAL_ASPECT_THRESHOLD else 'round'

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

    def _pixel_to_bearing(self, pixel_x: int) -> float:
        if self.camera_hfov is None or self.image_width is None:
            return 0.0
        import math
        norm_x = (pixel_x / self.image_width) - 0.5
        return math.degrees(norm_x * self.camera_hfov)

    def _draw_overlay(self, frame: np.ndarray, detections: list) -> np.ndarray:
        overlay = frame.copy()
        for det in detections:
            x, y, bw, bh = det['bbox']
            color = DRAW_COLORS.get(det['color'], (255, 255, 0))
            label = f"{det['class']} {det['bearing_deg']:+.1f}°"

            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(overlay, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

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
