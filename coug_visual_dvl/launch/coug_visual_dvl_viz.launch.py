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

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def visualization_setup(context, *args, **kwargs) -> list:

    use_sim_time = LaunchConfiguration("use_sim_time")
    auv_ns_str = LaunchConfiguration("auv_ns").perform(context)

    pkg_share = get_package_share_directory("coug_visual_dvl")

    template_path = os.path.join(pkg_share, "config", "plotjuggler.xml.template")
    with open(template_path, "r") as f:
        template_content = f.read()

    config_content = template_content.replace("AUV_NS", auv_ns_str)

    temp_config = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".xml")
    temp_config.write(config_content)
    temp_config.close()

    return [
        Node(
            package="plotjuggler",
            executable="plotjuggler",
            name="plotjuggler",
            arguments=["-l", temp_config.name],
            parameters=[{"use_sim_time": use_sim_time}],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation/rosbag clock if true",
            ),
            DeclareLaunchArgument(
                "auv_ns",
                default_value="auv0",
                description="Namespace for the AUV (e.g. auv0)",
            ),
            OpaqueFunction(function=visualization_setup),
        ]
    )
