"""
HarborWatch full-system launch file.

Brings up:
  1. VRX Gazebo simulation (gymkhana_task by default)
  2. All 4 HarborWatch nodes (detector, lidar, fusion, planner)
  3. RViz2 for visualization

Usage (from the colcon workspace, after sourcing install/setup.bash):
  ros2 launch asv_perception harborwatch.launch.py
  ros2 launch asv_perception harborwatch.launch.py world:=navigation_task
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Paths ───────────────────────────────────────────────────
    pkg_share = get_package_share_directory('asv_perception')

    # VRX launch file — assumes vrx_gz is installed in the same workspace
    try:
        vrx_share = get_package_share_directory('vrx_gz')
        vrx_launch = os.path.join(vrx_share, 'launch', 'competition.launch.py')
    except Exception:
        vrx_launch = None

    # ── Launch arguments ────────────────────────────────────────
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='gymkhana_task',
        description='VRX world name (e.g., gymkhana_task, navigation_task)',
    )

    # ── VRX simulation ──────────────────────────────────────────
    actions = [world_arg]

    if vrx_launch and os.path.exists(vrx_launch):
        vrx_sim = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(vrx_launch),
            launch_arguments={'world': LaunchConfiguration('world')}.items(),
        )
        actions.append(vrx_sim)

    # ── HarborWatch nodes ───────────────────────────────────────
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
            'min_range': 2.0,
            'max_range': 60.0,
            'cluster_eps': 1.5,
            'cluster_min_pts': 3,
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
            'catalogue_merge_dist': 4.0,
            'fusion_hz': 10.0,
        }],
    )

    # Perception-driven planner (no waypoints needed)
    reactive_planner = Node(
        package='asv_perception',
        executable='reactive_planner_node',
        name='reactive_planner_node',
        output='screen',
        parameters=[{
            'base_thrust': 8.0,
            'explore_thrust': 5.0,
            'danger_radius': 8.0,
            'kp': 5.0,
            'kd': 1.0,
            'avoidance_gain': 30.0,
            'gate_approach_bearing_threshold': 45.0,
        }],
    )

    # ── RViz2 ───────────────────────────────────────────────────
    rviz_config = os.path.join(pkg_share, 'config', 'harborwatch.rviz')
    rviz_args = ['-d', rviz_config] if os.path.exists(rviz_config) else []

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=rviz_args,
        output='screen',
    )

    actions.extend([
        hazard_detector,
        lidar_processor,
        sensor_fusion,
        reactive_planner,
        rviz,
    ])

    return LaunchDescription(actions)
