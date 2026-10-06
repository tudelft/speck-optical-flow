import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from px4_msgs.msg import TimesyncStatus, VehicleOdometry


class MocapRelayNode(Node):
    """
    Relays motion capture data to PX4, ensuring timestamp synchronization.
    Configurable input/output topics via parameters.
    """

    def __init__(self):
        super().__init__('mocap_relay_node')

        # Declare configurable topic parameters
        self.declare_parameter('input_topic', '/dummy')
        self.declare_parameter('output_topic', '/vehicle_motion_capture')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        # QoS for PX4 sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.timesync_sub = self.create_subscription(
            TimesyncStatus,
            '/fmu/out/timesync_status',
            self.timesync_callback,
            sensor_qos
        )
        self.mocap_sub = self.create_subscription(
            VehicleOdometry,
            input_topic,
            self.mocap_callback,
            10
        )

        # Publisher
        self.odometry_pub = self.create_publisher(
            VehicleOdometry,
            output_topic,
            10
        )

        # State
        self.timesync_offset_ns = 0

        self.get_logger().info(
            f'Mocap Relay started. {input_topic} -> {output_topic}')

    def timesync_callback(self, msg: TimesyncStatus):
        ros_time_ns = self.get_clock().now().nanoseconds
        px4_time_us = msg.timestamp
        self.timesync_offset_ns = ros_time_ns - (px4_time_us * 1000)
        self.get_logger().info(
            f'Timesync offset: {self.timesync_offset_ns} ns', once=True)

    def mocap_callback(self, msg: VehicleOdometry):
        if self.timesync_offset_ns == 0:
            self.get_logger().warn(
                'No timesync received yet. Skipping mocap message.')
            return

        ros_time_ns = self.get_clock().now().nanoseconds
        px4_timestamp_us = (ros_time_ns - self.timesync_offset_ns) // 1000

        out = VehicleOdometry()
        out.timestamp = px4_timestamp_us
        out.timestamp_sample = px4_timestamp_us
        out.pose_frame = msg.pose_frame
        out.position = msg.position
        out.q = msg.q
        out.velocity_frame = msg.velocity_frame
        out.velocity = msg.velocity
        out.angular_velocity = msg.angular_velocity
        out.position_variance = msg.position_variance
        out.orientation_variance = msg.orientation_variance
        out.velocity_variance = msg.velocity_variance

        self.odometry_pub.publish(out)
        self.get_logger().debug(
            f'Relayed mocap with synced timestamp: {px4_timestamp_us}')


def main(args=None):
    rclpy.init(args=args)
    node = MocapRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
