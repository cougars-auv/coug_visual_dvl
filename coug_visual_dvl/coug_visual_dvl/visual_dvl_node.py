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
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, TwistWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from coug_visual_dvl.visual_dvl import VisualDvl


class VisualDvlNode(Node):
    def __init__(self) -> None:
        super().__init__("visual_dvl_node")

        self.declare_parameter("velocity_noise_sigmas", [0.5, 0.5, 2.0])
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("front_stereo_topic", "stereo/front/image_raw")
        self.declare_parameter("back_stereo_topic", "stereo/back/image_raw")
        self.declare_parameter("front_stereo_info_topic", "stereo/front/camera_info")
        self.declare_parameter("back_stereo_info_topic", "stereo/back/camera_info")
        self.declare_parameter("vel_topic", "dvl/visual")
        self.declare_parameter("vel_frame", "dvl_link")
        self.declare_parameter("front_stereo_frame", "front_stereo_link")
        self.declare_parameter("back_stereo_frame", "back_stereo_link")

        sigmas = self.get_parameter("velocity_noise_sigmas").value
        sync_slop_sec = self.get_parameter("sync_slop_sec").value
        front_topic = self.get_parameter("front_stereo_topic").value
        back_topic = self.get_parameter("back_stereo_topic").value
        front_info_topic = self.get_parameter("front_stereo_info_topic").value
        back_info_topic = self.get_parameter("back_stereo_info_topic").value
        vel_topic = self.get_parameter("vel_topic").value
        self._rect_covariance = np.diag(np.square(np.asarray(sigmas[:3], dtype=float)))
        self._covariance = [0.0] * 36
        self._vel_frame = self.get_parameter("vel_frame").value
        self._front_stereo_frame = self.get_parameter("front_stereo_frame").value
        self._back_stereo_frame = self.get_parameter("back_stereo_frame").value

        self._vel_pub = self.create_publisher(
            TwistWithCovarianceStamped, vel_topic, qos_profile_system_default
        )
        self._feature_tf_pub = TransformBroadcaster(self)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._front_sub = message_filters.Subscriber(
            self, Image, front_topic, qos_profile=qos_profile_sensor_data
        )
        self._back_sub = message_filters.Subscriber(
            self, Image, back_topic, qos_profile=qos_profile_sensor_data
        )
        self._front_info_sub = message_filters.Subscriber(
            self, CameraInfo, front_info_topic, qos_profile=qos_profile_sensor_data
        )
        self._back_info_sub = message_filters.Subscriber(
            self, CameraInfo, back_info_topic, qos_profile=qos_profile_sensor_data
        )

        self._time_sync = message_filters.ApproximateTimeSynchronizer(
            [
                self._front_sub,
                self._back_sub,
                self._front_info_sub,
                self._back_info_sub,
            ],
            queue_size=10,
            slop=sync_slop_sec,
        )
        self._time_sync.registerCallback(self._stereo_callback)

        self._visual_dvl = None
        self._vel_R_rect = None
        self._last_time = None
        self._bridge = CvBridge()

        self.get_logger().info("Initialization complete.")

    def _stereo_callback(
        self,
        front_msg: Image,
        back_msg: Image,
        front_info: CameraInfo,
        back_info: CameraInfo,
    ) -> None:
        try:
            cv_front = self._bridge.imgmsg_to_cv2(front_msg, "bgr8")
            cv_back = self._bridge.imgmsg_to_cv2(back_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        curr_time = rclpy.time.Time.from_msg(front_msg.header.stamp)

        if self._visual_dvl is None:
            try:
                back_T_front_tf = self._tf_buffer.lookup_transform(
                    self._back_stereo_frame, self._front_stereo_frame, rclpy.time.Time()
                )
            except TransformException as e:
                self.get_logger().warn(
                    f"Could not transform {self._back_stereo_frame} to {self._front_stereo_frame}: {e}",
                    throttle_duration_sec=1.0,
                )
                return

            q = back_T_front_tf.transform.rotation
            back_R_front = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().tolist()
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

            try:
                vel_T_front = self._tf_buffer.lookup_transform(
                    self._vel_frame, self._front_stereo_frame, rclpy.time.Time()
                )
            except TransformException as e:
                self.get_logger().warn(
                    f"Could not transform {self._vel_frame} to {self._front_stereo_frame}: {e}",
                    throttle_duration_sec=1.0,
                )
                return

            visual_dvl = VisualDvl(calib_dict, (front_info.width, front_info.height))

            q = vel_T_front.transform.rotation
            vel_R_front = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            self._vel_R_rect = vel_R_front @ visual_dvl.rect_R_front.T

            # Conjugate the rectified-frame covariance into the DVL frame
            covariance = self._vel_R_rect @ self._rect_covariance @ self._vel_R_rect.T
            for i in range(3):
                for j in range(3):
                    self._covariance[i * 6 + j] = float(covariance[i, j])

            self._last_time = curr_time
            self._visual_dvl = visual_dvl
            return

        dt = (curr_time - self._last_time).nanoseconds * 1e-9
        self._last_time = curr_time

        velocities, pts_3d = self._visual_dvl.estimate_velocity(cv_front, cv_back, dt)

        tfs = []
        for i, pt_rect in enumerate(pts_3d):
            pt_front = self._visual_dvl.rect_R_front.T @ pt_rect
            tf_msg = TransformStamped()
            tf_msg.header.stamp = front_msg.header.stamp
            tf_msg.header.frame_id = self._front_stereo_frame
            tf_msg.child_frame_id = f"visual_dvl_feature_{i}"
            tf_msg.transform.translation.x = pt_front[0]
            tf_msg.transform.translation.y = pt_front[1]
            tf_msg.transform.translation.z = pt_front[2]
            tf_msg.transform.rotation.w = 1.0
            tfs.append(tf_msg)
        self._feature_tf_pub.sendTransform(tfs)

        # Transform the velocity into the DVL frame
        velocity = self._vel_R_rect @ velocities

        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header.stamp = front_msg.header.stamp
        twist_msg.header.frame_id = self._vel_frame
        twist_msg.twist.twist.linear.x = velocity[0]
        twist_msg.twist.twist.linear.y = velocity[1]
        twist_msg.twist.twist.linear.z = velocity[2]
        twist_msg.twist.covariance = self._covariance

        self._vel_pub.publish(twist_msg)


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
