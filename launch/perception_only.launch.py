"""
Perception-only launch file.

Launches just the perception stack (camera detector + LiDAR processor +
sensor fusion) WITHOUT the reactive planner. Useful for tuning HSV ranges,
testing detection accuracy, and visualising overlays without the boat
moving around autonomously.

Assumes VRX sim is already running in another terminal:
  ros2 launch vrx_gz competition.launch.py world:=gymkhana_task
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('asv_perception')
    rviz_config = os.path.join(pkg_share, 'config', 'harborwatch.rviz')
    rviz_args = ['-d', rviz_config] if os.path.exists(rviz_config) else []

    hazard_detector = Node(
        package='asv_perception',
        executable='hazard_detector_node',
        name='hazard_detector_node',
        output='screen',
        parameters=[{'max_process_hz': 10.0}],
    )

    lidar_processor = Node(
        package='asv_perception',
        executable='lidar_processor_node',
        name='lidar_processor_node',
        output='screen',
        parameters=[{
            'min_range': 3.5,
            'max_range': 60.0,
            'cluster_eps': 1.2,
            'cluster_min_pts': 7,
            'max_process_hz': 10.0,
        }],
    )

    sensor_fusion = Node(
        package='asv_perception',
        executable='sensor_fusion_node',
        name='sensor_fusion_node',
        output='screen',
        parameters=[{
            'bearing_tolerance_deg': 8.0,
            'catalogue_merge_dist': 3.5,
            'fusion_hz': 10.0,
        }],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=rviz_args,
        output='screen',
    )

    return LaunchDescription([
        hazard_detector,
        lidar_processor,
        sensor_fusion,
        rviz,
    ])

