"""
SpeckFlow Launch File

Launches flight control, mocap relay, and optionally the Speck flow node.
Records all relevant topics in a timestamped rosbag.

Usage:
  ros2 launch speckflow_control speckflow_launch.py
  ros2 launch speckflow_control speckflow_launch.py enable_speck:=true
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

import os
from datetime import datetime
from pathlib import Path


def generate_launch_description():
    # Ensure FastDDS uses PREALLOCATED_WITH_REALLOC to handle px4_msgs size mismatches
    fastdds_profile = os.path.join(
        str(Path.home()), 'speck-optical-flow', 'speckflow_ros2', 'fastdds_profile.xml')
    if os.path.exists(fastdds_profile):
        os.environ.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', fastdds_profile)

    # Data directory
    home_dir = str(Path.home())
    data_base = os.path.join(home_dir, 'Developer', 'data', 'speckflow')
    date_dir = datetime.now().strftime(f'{data_base}/%Y%m%d')
    os.makedirs(date_dir, exist_ok=True)

    bag_dir = datetime.now().strftime(
        f'{data_base}/%Y%m%d/speckflow_%Y%m%d-%H%M%S')

    # Parameter file
    config = os.path.join(
        get_package_share_directory('speckflow_control'),
        'config', 'params.yaml')

    # Topics to record
    topics = [
        '/fmu/out/vehicle_odometry',
        '/fmu/out/vehicle_local_position',
        '/fmu/out/vehicle_optical_flow',
        '/fmu/out/distance_sensor',
        '/fmu/out/hover_thrust_estimate',
        '/fmu/out/vehicle_status',
        '/vehicle_motion_capture',
        '/speck/optical_flow',
        '/fmu/in/trajectory_setpoint',
        '/fmu/in/vehicle_attitude_setpoint',
        '/fmu/in/offboard_control_mode',
        '/speckflow/state',
    ]

    return LaunchDescription([
        # ── Launch arguments ──
        DeclareLaunchArgument(
            'enable_speck', default_value='false',
            description='Enable Speck optical flow node'),

        # ── Rosbag recorder ──
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', bag_dir] + topics,
            output='screen',
            emulate_tty=True,
            name='rosbag_recorder',
        ),

        # ── Mocap relay ──
        Node(
            package='mocap_relay',
            executable='mocap_relay_node',
            name='mocap_relay',
            output='screen',
        ),

        # ── Flight control ──
        Node(
            package='speckflow_control',
            executable='flight_control',
            name='flight_control_node',
            output='screen',
            parameters=[config],
        ),

        # ── Speck flow (conditional) ──
        Node(
            package='speck_flow',
            executable='speck_flow_node',
            name='speck_flow_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_speck')),
        ),
    ])
