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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    use_sim_time = LaunchConfiguration("use_sim_time")
    agent_ns = LaunchConfiguration("agent_ns")

    fleet_params = PathJoinSubstitution(
        [
            EnvironmentVariable("CONFIG_DIR"),
            "fleet",
            "coug_visual_dvl_params.yaml",
        ]
    )
    agent_params = PathJoinSubstitution(
        [
            EnvironmentVariable("CONFIG_DIR"),
            PythonExpression(["'", agent_ns, "' + '_params.yaml'"]),
        ]
    )

    dvl_link_frame = PythonExpression(
        [
            "'",
            agent_ns,
            "/dvl_link' if '",
            agent_ns,
            "' != '' else 'dvl_link'",
        ]
    )

    front_stereo_link_frame = PythonExpression(
        [
            "'",
            agent_ns,
            "/front_stereo_link' if '",
            agent_ns,
            "' != '' else 'front_stereo_link'",
        ]
    )

    back_stereo_link_frame = PythonExpression(
        [
            "'",
            agent_ns,
            "/back_stereo_link' if '",
            agent_ns,
            "' != '' else 'back_stereo_link'",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation/rosbag clock if true",
            ),
            DeclareLaunchArgument(
                "agent_ns",
                default_value="auv0",
                description="Namespace for the agent (e.g. auv0)",
            ),
            Node(
                package="coug_visual_dvl",
                executable="visual_dvl",
                name="visual_dvl_node",
                parameters=[
                    fleet_params,
                    agent_params,
                    {
                        "use_sim_time": use_sim_time,
                        "vel_frame": dvl_link_frame,
                        "front_stereo_frame": front_stereo_link_frame,
                        "back_stereo_frame": back_stereo_link_frame,
                    },
                ],
            ),
        ]
    )
