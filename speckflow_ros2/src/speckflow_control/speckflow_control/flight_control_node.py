"""
SpeckFlow Flight Control Node

Main control node for optical flow-based drone control. Supports three flight modes:
  - waypoint:           Position-controlled square flight (Steps 1, 4)
  - flow_hover:         Flow PI for roll/pitch + distance PI for thrust (Steps 2, 5)
  - flow_hover_thrust:  Flow PI for roll/pitch + divergence PI for thrust (Steps 3, 5)

Safety: Geofence (7x7m box + 3m ceiling relative to start), permanent hold on violation.
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    VehicleLocalPosition, VehicleStatus, VehicleOdometry,
    VehicleAttitudeSetpoint, VehicleOpticalFlow, DistanceSensor,
    HoverThrustEstimate,
)
from std_msgs.msg import Float32MultiArray, String

import numpy as np
from enum import Enum, auto


class FlightState(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    MISSION = auto()
    FLOW_HOVER = auto()
    FLOW_HOVER_THRUST = auto()
    GEOFENCE_HOLD = auto()
    LANDING = auto()
    LANDED = auto()


class WaypointPhase(Enum):
    YAW_TO_WP = auto()
    FLY_TO_WP = auto()
    AT_WP = auto()
    COMPLETE = auto()


class FlightControlNode(Node):

    def __init__(self):
        super().__init__('flight_control_node')

        # ── QoS for PX4 topics ──
        self.px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Declare all parameters ──
        self._declare_parameters()

        # ── Publishers ──
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', self.px4_qos)
        self.trajectory_sp_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', self.px4_qos)
        self.attitude_sp_pub = self.create_publisher(
            VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint',
            self.px4_qos)
        self.vehicle_cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', self.px4_qos)
        self.state_pub = self.create_publisher(
            String, '/speckflow/state', 10)

        # ── Subscribers ──
        self.odom_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self._odom_cb, self.px4_qos)
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self._status_cb, self.px4_qos)
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self._local_pos_cb, self.px4_qos)
        self.flow_sensor_sub = self.create_subscription(
            VehicleOpticalFlow, '/fmu/out/vehicle_optical_flow',
            self._flow_sensor_cb, self.px4_qos)
        self.distance_sub = self.create_subscription(
            DistanceSensor, '/fmu/out/distance_sensor',
            self._distance_cb, self.px4_qos)
        self.hover_thrust_sub = self.create_subscription(
            HoverThrustEstimate, '/fmu/out/hover_thrust_estimate',
            self._hover_thrust_cb, self.px4_qos)
        self.speck_flow_sub = self.create_subscription(
            Float32MultiArray, '/speck/optical_flow',
            self._speck_flow_cb, 10)

        # ── State variables ──
        self.state = FlightState.IDLE
        self.wp_phase = WaypointPhase.YAW_TO_WP
        self.offboard_counter = 0
        self.odom_received = False

        # PX4 messages
        self.vehicle_status = VehicleStatus()
        self.vehicle_odometry = VehicleOdometry()

        # Position / orientation
        self.position = np.array([0.0, 0.0, 0.0])
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.heading = 0.0

        # Start position (captured once)
        self.start_position = None
        self.start_heading = None

        # Sensor data
        self.flow_x = 0.0
        self.flow_y = 0.0
        self.divergence = 0.0
        self.distance_m = -1.0
        self.hover_thrust_est = 0.25

        # Speck data (filtered, bias-corrected)
        self.speck_flow_x = 0.0
        self.speck_flow_y = 0.0
        self.speck_divergence = 0.0

        # Speck auto-calibration
        self.speck_calibrated = False
        self.speck_cal_buffer_x = []
        self.speck_cal_buffer_y = []
        self.speck_bias_x = 0.0
        self.speck_bias_y = 0.0

        # PI controller integrals
        self.flow_x_integral = 0.0
        self.flow_y_integral = 0.0
        self.alt_integral = 0.0

        # Waypoint navigation
        self.waypoints = []
        self.wp_index = 0

        # Geofence
        self.geofence_triggered = False
        self.hold_pos = np.array([0.0, 0.0, 0.0])
        self.hold_heading = 0.0

        # Yaw captured at flow mode entry
        self.captured_yaw = 0.0

        # ── Timers ──
        # Heartbeat at 50Hz on its own callback group (never blocked)
        self.heartbeat_cb_group = MutuallyExclusiveCallbackGroup()
        self.heartbeat_timer = self.create_timer(
            0.02, self._heartbeat_cb,
            callback_group=self.heartbeat_cb_group)

        # Control loop at 50Hz
        self.control_timer = self.create_timer(0.02, self._control_cb)

        self.get_logger().info('SpeckFlow flight control node started.')

    # ═══════════════════════════════════════════════════════════════════
    #  PARAMETERS
    # ═══════════════════════════════════════════════════════════════════

    def _declare_parameters(self):
        self.declare_parameter('flight_mode', 'waypoint')
        self.declare_parameter('flow_source', 'sensor')

        # Waypoint
        self.declare_parameter('square_size', 3.0)
        self.declare_parameter('flight_height', -1.0)
        self.declare_parameter('reach_point_margin', 0.15)
        self.declare_parameter('reach_yaw_margin', 0.174)
        self.declare_parameter('position_clamp', 0.25)
        self.declare_parameter('yaw_clamp', 0.4)
        self.declare_parameter('yaw_to_waypoint', False)

        # Geofence
        self.declare_parameter('geofence_xy', 3.5)
        self.declare_parameter('geofence_z', -3.0)

        # Flow PI (horizontal)
        self.declare_parameter('kp_flow', 0.05)
        self.declare_parameter('ki_flow', 0.1)
        self.declare_parameter('flow_integral_limit', 1.0)
        self.declare_parameter('max_roll_pitch', 0.262)

        # Altitude PI (vertical)
        self.declare_parameter('kp_alt', 0.9)
        self.declare_parameter('ki_alt', 0.12)
        self.declare_parameter('alt_integral_limit', 0.5)
        self.declare_parameter('target_distance', 1.0)
        self.declare_parameter('thrust_min', 0.12)
        self.declare_parameter('thrust_max', 0.30)
        self.declare_parameter('hover_thrust', 0.25)
        self.declare_parameter('use_px4_hover_thrust', True)
        self.declare_parameter('distance_alpha', 0.8)

        # Flow sensor filtering
        self.declare_parameter('sensor_alpha', 0.9)
        self.declare_parameter('flow_offset_x', 0.36)
        self.declare_parameter('flow_offset_y', 0.16)
        self.declare_parameter('bias_learning_rate', 0.001)
        self.declare_parameter('bias_threshold', 0.02)

        # Speck flow preprocessing
        self.declare_parameter('speck_alpha', 0.7)
        self.declare_parameter('speck_offset_x', 0.0)
        self.declare_parameter('speck_offset_y', 0.0)
        self.declare_parameter('speck_negate_x', True)
        self.declare_parameter('speck_negate_y', False)
        self.declare_parameter('speck_calibration_samples', 200)

        # Speck PI gains (separate from sensor — speck signal is ~12x larger)
        self.declare_parameter('speck_kp_flow', 0.025)
        self.declare_parameter('speck_ki_flow', 0.01)

        # Translation commands
        self.declare_parameter('target_flow_u', 0.0)
        self.declare_parameter('target_flow_v', 0.0)

    def _p(self, name):
        """Shorthand for getting a parameter value."""
        return self.get_parameter(name).value

    # ═══════════════════════════════════════════════════════════════════
    #  SUBSCRIBER CALLBACKS
    # ═══════════════════════════════════════════════════════════════════

    def _odom_cb(self, msg: VehicleOdometry):
        self.vehicle_odometry = msg
        self.position = np.array(msg.position, dtype=float)
        self.quaternion = np.array(msg.q, dtype=float)
        self.odom_received = True

    def _status_cb(self, msg: VehicleStatus):
        self.vehicle_status = msg

    def _local_pos_cb(self, msg: VehicleLocalPosition):
        pass  # Available for future use

    def _flow_sensor_cb(self, msg: VehicleOpticalFlow):
        """Process PX4 optical flow sensor data (mtf-01p)."""
        raw_u = msg.pixel_flow[0]
        raw_v = msg.pixel_flow[1]

        alpha = self._p('sensor_alpha')
        offset_x = self._p('flow_offset_x')
        offset_y = self._p('flow_offset_y')

        # Coordinate transformation (from C++ flow_control mode)
        corrected_x = -(raw_v - offset_y)
        corrected_y = (raw_u - offset_x)

        # Low-pass filter
        self.flow_x = alpha * corrected_x + (1.0 - alpha) * self.flow_x
        self.flow_y = alpha * corrected_y + (1.0 - alpha) * self.flow_y

        # Slow bias learning when stationary
        threshold = self._p('bias_threshold')
        if (abs(self.flow_x) < threshold and
                abs(self.flow_y) < threshold):
            rate = self._p('bias_learning_rate')
            # Update offsets locally (won't persist to param server)
            self._local_offset_x = (
                (1.0 - rate) * offset_x + rate * raw_u)
            self._local_offset_y = (
                (1.0 - rate) * offset_y + rate * raw_v)

    def _distance_cb(self, msg: DistanceSensor):
        if np.isfinite(msg.current_distance):
            alpha = self._p('distance_alpha')
            if self.distance_m < 0.05:
                self.distance_m = msg.current_distance
            else:
                self.distance_m = alpha * msg.current_distance + (1.0 - alpha) * self.distance_m

    def _hover_thrust_cb(self, msg: HoverThrustEstimate):
        if np.isfinite(msg.hover_thrust) and self._p('use_px4_hover_thrust'):
            self.hover_thrust_est = msg.hover_thrust

    def _speck_flow_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
            return

        raw_x = msg.data[0]
        raw_y = msg.data[1]
        raw_div = msg.data[2]

        # Auto-calibration: collect samples before flight to estimate bias
        cal_n = self._p('speck_calibration_samples')
        if not self.speck_calibrated:
            self.speck_cal_buffer_x.append(raw_x)
            self.speck_cal_buffer_y.append(raw_y)
            if len(self.speck_cal_buffer_x) >= cal_n:
                self.speck_bias_x = float(np.mean(self.speck_cal_buffer_x))
                self.speck_bias_y = float(np.mean(self.speck_cal_buffer_y))
                self.speck_calibrated = True
                self.speck_cal_buffer_x.clear()
                self.speck_cal_buffer_y.clear()
                self.get_logger().info(
                    f'Speck bias calibrated: x={self.speck_bias_x:.4f}, '
                    f'y={self.speck_bias_y:.4f} ({cal_n} samples)')
            return

        # Subtract bias (auto-cal + manual offset)
        corrected_x = raw_x - self.speck_bias_x - self._p('speck_offset_x')
        corrected_y = raw_y - self.speck_bias_y - self._p('speck_offset_y')

        # Axis sign correction
        if self._p('speck_negate_x'):
            corrected_x = -corrected_x
        if self._p('speck_negate_y'):
            corrected_y = -corrected_y

        # Low-pass filter
        alpha = self._p('speck_alpha')
        self.speck_flow_x = alpha * corrected_x + (1.0 - alpha) * self.speck_flow_x
        self.speck_flow_y = alpha * corrected_y + (1.0 - alpha) * self.speck_flow_y
        self.speck_divergence = alpha * raw_div + (1.0 - alpha) * self.speck_divergence

    # ═══════════════════════════════════════════════════════════════════
    #  HEARTBEAT (50 Hz, separate thread)
    # ═══════════════════════════════════════════════════════════════════

    def _heartbeat_cb(self):
        msg = OffboardControlMode()
        msg.timestamp = self._px4_timestamp()

        if self.state in (FlightState.FLOW_HOVER,
                          FlightState.FLOW_HOVER_THRUST):
            msg.position = False
            msg.velocity = False
            msg.acceleration = False
            msg.attitude = True
            msg.body_rate = False
        else:
            msg.position = True
            msg.velocity = False
            msg.acceleration = False
            msg.attitude = False
            msg.body_rate = False

        self.offboard_mode_pub.publish(msg)

    # ═══════════════════════════════════════════════════════════════════
    #  MAIN CONTROL LOOP (50 Hz)
    # ═══════════════════════════════════════════════════════════════════

    def _control_cb(self):
        self.offboard_counter += 1

        # Update heading
        self.heading = self._get_heading(self.quaternion)

        # Capture start position once (wait for real odometry, not default zeros)
        if self.start_position is None and self.odom_received:
            self.start_position = self.position.copy()
            self.start_heading = self.heading
            self.get_logger().info(
                f'Start: pos={self.start_position}, '
                f'heading={np.degrees(self.start_heading):.1f} deg')

        # Publish state for logging
        state_msg = String()
        phase_str = (f':{self.wp_phase.name}'
                     if self.state == FlightState.MISSION else '')
        state_msg.data = f'{self.state.name}{phase_str}'
        self.state_pub.publish(state_msg)

        # Geofence check (skip for safe states)
        if self.state not in (FlightState.IDLE, FlightState.LANDED,
                              FlightState.GEOFENCE_HOLD):
            if self._check_geofence():
                self._enter_geofence_hold()
                return

        # State machine dispatch
        if self.state == FlightState.IDLE:
            self._handle_idle()
        elif self.state == FlightState.TAKEOFF:
            self._handle_takeoff()
        elif self.state == FlightState.MISSION:
            self._handle_mission()
        elif self.state == FlightState.FLOW_HOVER:
            self._handle_flow_hover()
        elif self.state == FlightState.FLOW_HOVER_THRUST:
            self._handle_flow_hover_thrust()
        elif self.state == FlightState.GEOFENCE_HOLD:
            self._handle_geofence_hold()
        elif self.state == FlightState.LANDING:
            self._handle_landing()

    # ═══════════════════════════════════════════════════════════════════
    #  STATE HANDLERS
    # ═══════════════════════════════════════════════════════════════════

    def _handle_idle(self):
        """Pre-arm: publish setpoints, send arm/offboard, then proceed.
        Matches homingdrone pattern: send commands for a fixed period,
        then transition to TAKEOFF without waiting for VehicleStatus
        confirmation (avoids dependency on DDS message deserialization)."""
        # Need start_position before we can do anything useful
        if self.start_position is None:
            return

        # Publish position setpoint at current position (PX4 needs setpoints
        # before it will accept offboard mode)
        self._publish_trajectory(
            self.position[0], self.position[1], 0.0, self.heading)

        # Phase 1: first 50 ticks (1s) — just publish setpoints
        if self.offboard_counter < 50:
            return

        # Phase 2: ticks 50-150 (2s) — send arm/offboard every 0.2s
        if self.offboard_counter < 150:
            if self.offboard_counter % 10 == 0:
                self._engage_offboard()
                self._arm()
            if self.offboard_counter % 50 == 0:
                self.get_logger().info(
                    f'Sending arm/offboard commands... '
                    f'(tick {self.offboard_counter})')
            return

        # Phase 3: tick 150 — proceed to takeoff
        self.get_logger().info('Arm/offboard commands sent. Proceeding to TAKEOFF.')
        self.state = FlightState.TAKEOFF

    def _handle_takeoff(self):
        """Climb to flight_height at start position."""
        target_z = self._p('flight_height')
        self._publish_trajectory(
            self.start_position[0], self.start_position[1],
            target_z, self.start_heading)

        if abs(self.position[2] - target_z) <= 0.1:
            flight_mode = self._p('flight_mode')
            if flight_mode == 'waypoint':
                self._enter_mission()
            elif flight_mode == 'flow_hover':
                self._enter_flow_hover()
            elif flight_mode == 'flow_hover_thrust':
                self._enter_flow_hover_thrust()
            else:
                self.get_logger().error(
                    f'Unknown flight_mode: {flight_mode}')

    def _enter_mission(self):
        self.state = FlightState.MISSION
        yaw_enabled = self._p('yaw_to_waypoint')
        self.wp_phase = (WaypointPhase.YAW_TO_WP if yaw_enabled
                         else WaypointPhase.FLY_TO_WP)
        self.wp_index = 0
        self._generate_square_waypoints()
        self.get_logger().info(
            f'MISSION: {len(self.waypoints)} waypoints, '
            f'square_size={self._p("square_size")}m, '
            f'yaw_to_wp={yaw_enabled}')

    def _handle_mission(self):
        """Waypoint square flight: [yaw →] fly → next → land."""
        if self.wp_index >= len(self.waypoints):
            self.state = FlightState.LANDING
            self.get_logger().info('Mission complete, landing.')
            return

        target_z = self._p('flight_height')
        wx, wy = self.waypoints[self.wp_index]
        yaw_enabled = self._p('yaw_to_waypoint')

        if self.wp_phase == WaypointPhase.YAW_TO_WP:
            target_yaw = np.arctan2(
                wy - self.position[1], wx - self.position[0])
            # Hold current position while yawing
            self._publish_trajectory(
                self.position[0], self.position[1],
                target_z, self._clamp_yaw(target_yaw))

            yaw_err = abs(self._normalize_angle(target_yaw - self.heading))
            if yaw_err < self._p('reach_yaw_margin'):
                self.wp_phase = WaypointPhase.FLY_TO_WP

        elif self.wp_phase == WaypointPhase.FLY_TO_WP:
            if yaw_enabled:
                target_yaw = np.arctan2(
                    wy - self.position[1], wx - self.position[0])
                yaw_cmd = self._clamp_yaw(target_yaw)
            else:
                yaw_cmd = self.start_heading

            cx, cy = self._clamp_position(wx, wy)
            self._publish_trajectory(cx, cy, target_z, yaw_cmd)

            dist = np.sqrt(
                (wx - self.position[0])**2 + (wy - self.position[1])**2)
            if dist < self._p('reach_point_margin'):
                self.wp_phase = WaypointPhase.AT_WP
                self.get_logger().info(f'Reached WP {self.wp_index}')

        elif self.wp_phase == WaypointPhase.AT_WP:
            self.wp_index += 1
            if self.wp_index < len(self.waypoints):
                self.wp_phase = (WaypointPhase.YAW_TO_WP if yaw_enabled
                                 else WaypointPhase.FLY_TO_WP)
            else:
                self.wp_phase = WaypointPhase.COMPLETE

    def _enter_flow_hover(self):
        """Enter flow hover: attitude control with distance-based thrust."""
        self.state = FlightState.FLOW_HOVER
        self.flow_x_integral = 0.0
        self.flow_y_integral = 0.0
        self.alt_integral = 0.0
        self.captured_yaw = self.heading
        self.get_logger().info(
            f'FLOW_HOVER: yaw={np.degrees(self.captured_yaw):.1f} deg, '
            f'source={self._p("flow_source")}')

    def _handle_flow_hover(self):
        """Flow PI for roll/pitch, distance PI for thrust."""
        dt = 0.02  # 50Hz
        fx, fy, _ = self._get_active_flow()

        roll_cmd = self._calc_roll(fx, dt)
        pitch_cmd = self._calc_pitch(fy, dt)
        thrust_cmd = self._calc_thrust_distance(dt)

        q = self._euler_to_quaternion(roll_cmd, pitch_cmd, self.captured_yaw)
        self._publish_attitude(q, thrust_cmd)

    def _enter_flow_hover_thrust(self):
        """Enter flow hover with divergence-based thrust."""
        self.state = FlightState.FLOW_HOVER_THRUST
        self.flow_x_integral = 0.0
        self.flow_y_integral = 0.0
        self.alt_integral = 0.0
        self.captured_yaw = self.heading
        self.get_logger().info(
            f'FLOW_HOVER_THRUST: yaw={np.degrees(self.captured_yaw):.1f} deg, '
            f'source={self._p("flow_source")}')

    def _handle_flow_hover_thrust(self):
        """Flow PI for roll/pitch, divergence PI for thrust."""
        dt = 0.02
        fx, fy, div = self._get_active_flow()

        roll_cmd = self._calc_roll(fx, dt)
        pitch_cmd = self._calc_pitch(fy, dt)
        thrust_cmd = self._calc_thrust_divergence(div, dt)

        q = self._euler_to_quaternion(roll_cmd, pitch_cmd, self.captured_yaw)
        self._publish_attitude(q, thrust_cmd)

    def _handle_landing(self):
        """Descend to ground and disarm."""
        # Move to start position and descend
        self._publish_trajectory(
            self.position[0], self.position[1],
            0.0, self.heading)

        if self.position[2] >= -0.15:
            self._land()
            self.state = FlightState.LANDED
            self.get_logger().info('Landed.')

    # ═══════════════════════════════════════════════════════════════════
    #  GEOFENCE
    # ═══════════════════════════════════════════════════════════════════

    def _check_geofence(self):
        """Return True if geofence is violated."""
        if self.start_position is None:
            return False

        gf_xy = self._p('geofence_xy')
        gf_z = self._p('geofence_z')

        dx = abs(self.position[0] - self.start_position[0])
        dy = abs(self.position[1] - self.start_position[1])

        if dx > gf_xy or dy > gf_xy:
            self.get_logger().warn(
                f'GEOFENCE XY: dx={dx:.2f}, dy={dy:.2f}, limit={gf_xy}')
            return True

        # NED: more negative = higher
        if self.position[2] < gf_z:
            self.get_logger().warn(
                f'GEOFENCE Z: z={self.position[2]:.2f}, limit={gf_z}')
            return True

        return False

    def _enter_geofence_hold(self):
        self.geofence_triggered = True
        self.state = FlightState.GEOFENCE_HOLD
        self.hold_pos = self.position.copy()
        self.hold_heading = self.heading
        self.get_logger().error(
            'GEOFENCE TRIGGERED - holding position for manual takeover')

    def _handle_geofence_hold(self):
        self._publish_trajectory(
            self.hold_pos[0], self.hold_pos[1],
            self.hold_pos[2], self.hold_heading)

    # ═══════════════════════════════════════════════════════════════════
    #  PI CONTROLLERS
    # ═══════════════════════════════════════════════════════════════════

    def _get_active_flow(self):
        """Return (flow_x, flow_y, divergence) from the active source."""
        if self._p('flow_source') == 'speck':
            return self.speck_flow_x, self.speck_flow_y, self.speck_divergence
        return self.flow_x, self.flow_y, self.divergence

    def _calc_roll(self, flow_x, dt):
        """Flow X → Roll. Positive flow_x = drifting right → negative roll."""
        if self._p('flow_source') == 'speck':
            kp = self._p('speck_kp_flow')
            ki = self._p('speck_ki_flow')
        else:
            kp = self._p('kp_flow')
            ki = self._p('ki_flow')
        limit = self._p('flow_integral_limit')
        max_angle = self._p('max_roll_pitch')
        target = self._p('target_flow_u')

        error = target - flow_x
        self.flow_x_integral += error * dt
        self.flow_x_integral = np.clip(
            self.flow_x_integral, -limit, limit)

        roll = kp * error + ki * self.flow_x_integral
        return float(np.clip(roll, -max_angle, max_angle))

    def _calc_pitch(self, flow_y, dt):
        """Flow Y → Pitch. Negated for NED convention."""
        if self._p('flow_source') == 'speck':
            kp = self._p('speck_kp_flow')
            ki = self._p('speck_ki_flow')
        else:
            kp = self._p('kp_flow')
            ki = self._p('ki_flow')
        limit = self._p('flow_integral_limit')
        max_angle = self._p('max_roll_pitch')
        target = self._p('target_flow_v')

        error = target - flow_y
        self.flow_y_integral += error * dt
        self.flow_y_integral = np.clip(
            self.flow_y_integral, -limit, limit)

        pitch = -(kp * error + ki * self.flow_y_integral)
        return float(np.clip(pitch, -max_angle, max_angle))

    def _calc_thrust_distance(self, dt):
        """Distance sensor → thrust PI. Holds target_distance meters."""
        kp = self._p('kp_alt')
        ki = self._p('ki_alt')
        limit = self._p('alt_integral_limit')
        target = self._p('target_distance')
        t_min = self._p('thrust_min')
        t_max = self._p('thrust_max')

        if self.distance_m < 0.05 or not np.isfinite(self.distance_m):
            return self.hover_thrust_est

        error = target - self.distance_m
        self.alt_integral += error * dt
        self.alt_integral = np.clip(self.alt_integral, -limit, limit)

        thrust = self.hover_thrust_est + kp * error + ki * self.alt_integral
        return float(np.clip(thrust, t_min, t_max))

    def _calc_thrust_divergence(self, divergence, dt):
        """Divergence → thrust PI. Target divergence is 0 for hover."""
        kp = self._p('kp_alt')
        ki = self._p('ki_alt')
        limit = self._p('alt_integral_limit')
        t_min = self._p('thrust_min')
        t_max = self._p('thrust_max')

        if not np.isfinite(divergence):
            return self.hover_thrust_est

        error = divergence  # target = 0
        self.alt_integral += error * dt
        self.alt_integral = np.clip(self.alt_integral, -limit, limit)

        thrust = self.hover_thrust_est + kp * error + ki * self.alt_integral
        return float(np.clip(thrust, t_min, t_max))

    # ═══════════════════════════════════════════════════════════════════
    #  WAYPOINT GENERATION
    # ═══════════════════════════════════════════════════════════════════

    def _generate_square_waypoints(self):
        """4 corners + return, centered on start, oriented along heading."""
        s = self._p('square_size') / 2.0
        theta = self.start_heading
        x0 = self.start_position[0]
        y0 = self.start_position[1]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # (forward, left) offsets in body frame
        # Body-to-NED rotation:
        #   body forward (1,0) -> NED (cos θ, sin θ)
        #   body left (0,-1 in FRD) -> NED (sin θ, -cos θ)
        offsets = [
            (s, s),     # forward-left
            (s, -s),    # forward-right
            (-s, -s),   # back-right
            (-s, s),    # back-left
            (0.0, 0.0),  # return to center
        ]

        self.waypoints = []
        for fwd, left in offsets:
            # NED position: fwd along heading, left perpendicular
            nx = x0 + fwd * cos_t + left * sin_t
            ny = y0 + fwd * sin_t - left * cos_t
            self.waypoints.append((nx, ny))
            self.get_logger().info(f'  WP: ({nx:.2f}, {ny:.2f})')

    # ═══════════════════════════════════════════════════════════════════
    #  PUBLISHING HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _publish_trajectory(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.yaw = float(yaw)
        msg.yawspeed = float('nan')
        msg.timestamp = self._px4_timestamp()
        self.trajectory_sp_pub.publish(msg)

    def _publish_attitude(self, q, thrust):
        """Publish VehicleAttitudeSetpoint. q=[w,x,y,z], thrust=[0,1]."""
        msg = VehicleAttitudeSetpoint()
        msg.q_d = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        # FRD body frame: negative Z = upward thrust for multicopter
        msg.thrust_body = [0.0, 0.0, -float(thrust)]
        msg.yaw_sp_move_rate = 0.0
        msg.reset_integral = False
        msg.timestamp = self._px4_timestamp()
        self.attitude_sp_pub.publish(msg)

    def _publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get('param1', 0.0)
        msg.param2 = params.get('param2', 0.0)
        msg.param3 = params.get('param3', 0.0)
        msg.param4 = params.get('param4', 0.0)
        msg.param5 = params.get('param5', 0.0)
        msg.param6 = params.get('param6', 0.0)
        msg.param7 = params.get('param7', 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self._px4_timestamp()
        self.vehicle_cmd_pub.publish(msg)

    def _arm(self):
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def _disarm(self):
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def _engage_offboard(self):
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info('Offboard mode command sent')

    def _land(self):
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('Land command sent')

    # ═══════════════════════════════════════════════════════════════════
    #  MATH UTILITIES
    # ═══════════════════════════════════════════════════════════════════

    def _px4_timestamp(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    @staticmethod
    def _normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _get_heading(self, q):
        """Extract heading from PX4 quaternion (matching homingdrone)."""
        # q = [q0, q1, q2, q3] from VehicleOdometry
        # homingdrone convention: get_current_heading(z=q[0], y=q[1], x=q[2], w=q[3])
        z, y, x, w = q[0], q[1], q[2], q[3]
        t1 = 2.0 * (w * z + x * y)
        t2 = 1.0 - 2.0 * (z * z + y * y)
        heading = -np.arctan2(t1, t2) + np.pi
        if heading > np.pi:
            heading -= 2.0 * np.pi
        elif heading < -np.pi:
            heading += 2.0 * np.pi
        return heading

    def _clamp_position(self, x, y):
        """Clamp target position to limit movement speed per tick."""
        clamp = self._p('position_clamp')
        dx = x - self.position[0]
        dy = y - self.position[1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist <= clamp:
            return x, y
        angle = np.arctan2(dy, dx)
        return (self.position[0] + clamp * np.cos(angle),
                self.position[1] + clamp * np.sin(angle))

    def _clamp_yaw(self, target_yaw):
        """Clamp yaw change per tick."""
        clamp = self._p('yaw_clamp')
        diff = self._normalize_angle(target_yaw - self.heading)
        if abs(diff) <= clamp:
            return target_yaw
        clamped = self.heading + clamp * np.sign(diff)
        return self._normalize_angle(clamped)

    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        """ZYX Tait-Bryan Euler angles to quaternion [w, x, y, z]."""
        cr = np.cos(roll / 2.0)
        sr = np.sin(roll / 2.0)
        cp = np.cos(pitch / 2.0)
        sp = np.sin(pitch / 2.0)
        cy = np.cos(yaw / 2.0)
        sy = np.sin(yaw / 2.0)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [w, x, y, z]


def main(args=None):
    print('Starting SpeckFlow flight control node...')
    rclpy.init(args=args)
    node = FlightControlNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
