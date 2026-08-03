# Copyright (c) 2026 BYU FROST Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import message_filters
import numpy as np
import rclpy
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    TransformStamped,
    TwistWithCovarianceStamped,
    Vector3Stamped,
)
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from coug_visual_dvl.visual_dvl import VisualDVL


class VisualDvlNode(Node):
    """
    ROS 2 wrapper for the visual odometry script, handles stereo synchronization and TFs.

    :author: Nelson Durrant & Braden Meyers
    :date: April 2026
    """

    def __init__(self) -> None:
        super().__init__("visual_dvl_node")

        self.declare_parameter("front_stereo_topic", "stereo/front/image_raw")
        self.declare_parameter("back_stereo_topic", "stereo/back/image_raw")
        self.declare_parameter("front_stereo_info_topic", "stereo/front/camera_info")
        self.declare_parameter("back_stereo_info_topic", "stereo/back/camera_info")
        self.declare_parameter("vel_topic", "dvl/twist")
        self.declare_parameter("velocity_noise_sigmas", [0.5, 0.5, 2.0])
        self.declare_parameter("vel_frame", "dvl_link")
        self.declare_parameter("front_stereo_frame", "front_stereo_link")
        self.declare_parameter("back_stereo_frame", "back_stereo_link")

        front_topic = (
            self.get_parameter("front_stereo_topic").get_parameter_value().string_value
        )
        back_topic = (
            self.get_parameter("back_stereo_topic").get_parameter_value().string_value
        )
        front_info_topic = (
            self.get_parameter("front_stereo_info_topic")
            .get_parameter_value()
            .string_value
        )
        back_info_topic = (
            self.get_parameter("back_stereo_info_topic")
            .get_parameter_value()
            .string_value
        )
        vel_topic = self.get_parameter("vel_topic").get_parameter_value().string_value
        sigmas = (
            self.get_parameter("velocity_noise_sigmas")
            .get_parameter_value()
            .double_array_value
        )
        self.covariance = [0.0] * 36
        for i, s in enumerate(sigmas[:3]):
            self.covariance[i * 7] = s * s
        self.vel_frame = (
            self.get_parameter("vel_frame").get_parameter_value().string_value
        )
        self.front_stereo_frame = (
            self.get_parameter("front_stereo_frame").get_parameter_value().string_value
        )
        self.back_stereo_frame = (
            self.get_parameter("back_stereo_frame").get_parameter_value().string_value
        )

        self.pub = self.create_publisher(
            TwistWithCovarianceStamped, vel_topic, qos_profile_system_default
        )
        self.feature_tf_pub = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.front_sub = message_filters.Subscriber(
            self, Image, front_topic, qos_profile=qos_profile_sensor_data
        )
        self.back_sub = message_filters.Subscriber(
            self, Image, back_topic, qos_profile=qos_profile_sensor_data
        )
        self.front_info_sub = message_filters.Subscriber(
            self, CameraInfo, front_info_topic, qos_profile=qos_profile_sensor_data
        )
        self.back_info_sub = message_filters.Subscriber(
            self, CameraInfo, back_info_topic, qos_profile=qos_profile_sensor_data
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.front_sub, self.back_sub, self.front_info_sub, self.back_info_sub],
            queue_size=10,
            slop=0.05,
        )
        self.ts.registerCallback(self.stereo_callback)

        self.visual_dvl = None
        self.vel_T_front = None
        self.last_time = None
        self.bridge = CvBridge()

        self.get_logger().info("Initialization complete.")

    def stereo_callback(
        self,
        front_msg: Image,
        back_msg: Image,
        front_info: CameraInfo,
        back_info: CameraInfo,
    ) -> None:
        """
        Synchronize and process stereo images to estimate velocity.

        :param front_msg: Image message from the front camera.
        :param back_msg: Image message from the back camera.
        :param front_info: CameraInfo from front camera.
        :param back_info: CameraInfo from back camera.
        """
        try:
            cv_front = self.bridge.imgmsg_to_cv2(front_msg, "bgr8")
            cv_back = self.bridge.imgmsg_to_cv2(back_msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        curr_time = rclpy.time.Time.from_msg(front_msg.header.stamp)

        if self.visual_dvl is None:
            try:
                back_T_front_tf = self.tf_buffer.lookup_transform(
                    self.back_stereo_frame, self.front_stereo_frame, rclpy.time.Time()
                )
                q = back_T_front_tf.transform.rotation
                back_R_front = (
                    Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().tolist()
                )
                back_t_front = [
                    [back_T_front_tf.transform.translation.x],
                    [back_T_front_tf.transform.translation.y],
                    [back_T_front_tf.transform.translation.z],
                ]

                calib_dict = {
                    "mtx_f": np.array(front_info.k).reshape(3, 3).tolist(),
                    "dist_f": list(front_info.d),
                    "mtx_b": np.array(back_info.k).reshape(3, 3).tolist(),
                    "dist_b": list(back_info.d),
                    "R": back_R_front,
                    "T": back_t_front,
                }

                self.get_logger().info(
                    f"Full camera calibration parameters: \n{json.dumps(calib_dict, indent=2)}"
                )
                with open("/tmp/online_stereo_calibration_params.json", "w") as f:
                    json.dump(calib_dict, f, indent=2)

                self.vel_T_front = self.tf_buffer.lookup_transform(
                    self.vel_frame, self.front_stereo_frame, rclpy.time.Time()
                )
                self.visual_dvl = VisualDVL(
                    calib_dict, (front_info.width, front_info.height)
                )
                self.last_time = curr_time
            except Exception as e:
                self.get_logger().warn(
                    f"Could not transform {self.back_stereo_frame} to {self.front_stereo_frame}: {e}",
                    throttle_duration_sec=1.0,
                )
                return
            return

        dt = (curr_time - self.last_time).nanoseconds * 1e-9
        self.last_time = curr_time

        velocities, pts_3d = self.visual_dvl.estimate_velocity(cv_front, cv_back, dt)

        tfs = []
        for i, pt_rect in enumerate(pts_3d):
            pt_front = self.visual_dvl.R1.T @ pt_rect
            tf_msg = TransformStamped()
            tf_msg.header.stamp = front_msg.header.stamp
            tf_msg.header.frame_id = self.front_stereo_frame
            tf_msg.child_frame_id = f"visual_dvl_feature_{i}"
            tf_msg.transform.translation.x = pt_front[0]
            tf_msg.transform.translation.y = pt_front[1]
            tf_msg.transform.translation.z = pt_front[2]
            tf_msg.transform.rotation.w = 1.0
            tfs.append(tf_msg)
        self.feature_tf_pub.sendTransform(tfs)

        # Rotate the rectified front stereo frame velocity into the DVL frame
        v_front = self.visual_dvl.R1.T @ velocities
        vel_vec = Vector3Stamped()
        vel_vec.header.frame_id = self.front_stereo_frame
        vel_vec.vector.x, vel_vec.vector.y, vel_vec.vector.z = (
            v_front[0],
            v_front[1],
            v_front[2],
        )
        vel_rotated = tf2_geometry_msgs.do_transform_vector3(vel_vec, self.vel_T_front)

        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header.stamp = front_msg.header.stamp
        twist_msg.header.frame_id = self.vel_frame
        twist_msg.twist.twist.linear.x = vel_rotated.vector.x
        twist_msg.twist.twist.linear.y = vel_rotated.vector.y
        twist_msg.twist.twist.linear.z = vel_rotated.vector.z
        twist_msg.twist.covariance = self.covariance

        self.pub.publish(twist_msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    visual_dvl_node = VisualDvlNode()
    try:
        rclpy.spin(visual_dvl_node)
    except KeyboardInterrupt:
        pass
    finally:
        visual_dvl_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
