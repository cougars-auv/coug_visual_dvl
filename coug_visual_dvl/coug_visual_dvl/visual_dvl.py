# Copyright 2026 BYU FROST Lab
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

import cv2
import numpy as np

MAX_CORNERS = 200
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10
MIN_FEATURE_COUNT = 150
EPIPOLAR_THRESHOLD_PX = 2.0
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 50.0
MIN_POINTS_FOR_LLS = 3


class VisualDvl:
    def __init__(self, calib_dict: dict, img_size: tuple) -> None:
        (
            self.rect_R_front,
            self._rect_R_back,
            self._front_projection,
            self._back_projection,
            self._disparity_to_depth,
            _,
            _,
        ) = cv2.stereoRectify(
            np.array(calib_dict["mtx_f"]),
            np.array(calib_dict["dist_f"]),
            np.array(calib_dict["mtx_b"]),
            np.array(calib_dict["dist_b"]),
            img_size,
            np.array(calib_dict["R"]),
            np.array(calib_dict["T"]),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        self._front_remap_x, self._front_remap_y = cv2.initUndistortRectifyMap(
            np.array(calib_dict["mtx_f"]),
            np.array(calib_dict["dist_f"]),
            self.rect_R_front,
            self._front_projection,
            img_size,
            cv2.CV_32FC1,
        )
        self._back_remap_x, self._back_remap_y = cv2.initUndistortRectifyMap(
            np.array(calib_dict["mtx_b"]),
            np.array(calib_dict["dist_b"]),
            self._rect_R_back,
            self._back_projection,
            img_size,
            cv2.CV_32FC1,
        )

        self._prev_gray_front = None
        self._prev_points_front = None

    @property
    def rect_R_front(self) -> np.ndarray:
        return self.rect_R_front

    def estimate_velocity(
        self, image_front: np.ndarray, image_back: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        rect_front = cv2.remap(
            image_front, self._front_remap_x, self._front_remap_y, cv2.INTER_LINEAR
        )
        rect_back = cv2.remap(
            image_back, self._back_remap_x, self._back_remap_y, cv2.INTER_LINEAR
        )
        gray_front = (
            cv2.cvtColor(rect_front, cv2.COLOR_BGR2GRAY)
            if len(rect_front.shape) == 3
            else rect_front
        )
        gray_back = (
            cv2.cvtColor(rect_back, cv2.COLOR_BGR2GRAY)
            if len(rect_back.shape) == 3
            else rect_back
        )

        if dt <= 0.0 or self._prev_gray_front is None:
            self._prev_gray_front = gray_front
            self._prev_points_front = cv2.goodFeaturesToTrack(
                gray_front,
                maxCorners=MAX_CORNERS,
                qualityLevel=QUALITY_LEVEL,
                minDistance=MIN_DISTANCE,
            )
            return np.array([0.0, 0.0, 0.0]), np.empty((0, 3))

        # Track features forward across frames using LK optical flow
        curr_points, prev_points = None, None
        if self._prev_points_front is not None:
            curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray_front, gray_front, self._prev_points_front, None
            )
            valid = status.ravel() == 1  # Discard untracked features
            curr_points = (
                curr_points[valid] if valid.sum() > 0 else None
            )  # Current feature positions
            prev_points = (
                self._prev_points_front[valid].reshape(-1, 2)
                if valid.sum() > 0
                else None
            )  # Previous feature positions

        # Detect new features when count drops below a threshold
        new_points = None
        if curr_points is None or len(curr_points) < MIN_FEATURE_COUNT:
            new_points = cv2.goodFeaturesToTrack(
                gray_front,
                maxCorners=MAX_CORNERS,
                qualityLevel=QUALITY_LEVEL,
                minDistance=MIN_DISTANCE,
            )

        # Check to make sure we tracked or found something
        if curr_points is None and new_points is None:
            self._prev_gray_front = gray_front
            self._prev_points_front = None
            return np.array([0.0, 0.0, 0.0]), np.empty((0, 3))
        next_points_front = np.concatenate(
            [x for x in [curr_points, new_points] if x is not None]
        )

        points_3d = np.empty((0, 3))
        velocity = np.array([0.0, 0.0, 0.0])
        if curr_points is not None:
            # Find each tracked feature in the rectified back frame using LK optical flow
            points_back, stereo_status, _ = cv2.calcOpticalFlowPyrLK(
                gray_front, gray_back, curr_points, None
            )
            epipolar_valid = (
                np.abs(curr_points[:, 0, 0] - points_back[:, 0, 0])
                <= EPIPOLAR_THRESHOLD_PX
            )
            stereo_valid = (
                stereo_status.ravel() == 1
            ) & epipolar_valid  # Discard unlocated features

            if stereo_valid.sum() < 1:
                self._prev_gray_front = gray_front
                self._prev_points_front = next_points_front.reshape(-1, 1, 2)
                return np.array([0.0, 0.0, 0.0]), np.empty((0, 3))

            # Triangulate matched feature pairs into homogeneous 3D coordinates
            points_4d = cv2.triangulatePoints(
                self._front_projection,  # Front projection matrix
                self._back_projection,  # Back projection matrix
                curr_points[stereo_valid].reshape(-1, 2).T,  # 2xN points in front image
                points_back[stereo_valid].reshape(-1, 2).T,  # 2xN points in back image
            )
            points_3d_all = (points_4d[:3] / points_4d[3]).T

            # Discard invalid points behind the camera or crazy far away
            depth_valid = (
                (points_3d_all[:, 2] > MIN_DEPTH_M)
                & (points_3d_all[:, 2] < MAX_DEPTH_M)
                & np.isfinite(points_3d_all).all(axis=1)
            )
            points_3d = points_3d_all[depth_valid]

            # Solve for camera velocity using LLS
            # du/dt = -(fx/Z)*Vx + (fx*X/Z²)*Vz + (fx*X*Y/Z²)*ωx - fx*(1+X²/Z²)*ωy + (fx*Y/Z)*ωz
            # dv/dt = -(fy/Z)*Vy + (fy*Y/Z²)*Vz + fy*(1+Y²/Z²)*ωx - (fy*X*Y/Z²)*ωy - (fy*X/Z)*ωz
            if prev_points is not None and len(points_3d) >= MIN_POINTS_FOR_LLS:
                curr_points_2d = curr_points[stereo_valid][depth_valid].reshape(
                    -1, 2
                )  # Current 2D positions
                prev_points_2d = prev_points[stereo_valid][depth_valid].reshape(
                    -1, 2
                )  # Previous 2D positions
                flow = (curr_points_2d - prev_points_2d) / dt

                fx, fy = self._front_projection[0, 0], self._front_projection[1, 1]
                point_x, point_y, point_z = (
                    points_3d[:, 0],
                    points_3d[:, 1],
                    points_3d[:, 2],
                )
                J_flow_twist = np.vstack(
                    [
                        np.column_stack(
                            [
                                -fx / point_z,
                                np.zeros_like(point_z),
                                fx * point_x / point_z**2,
                                fx * point_x * point_y / point_z**2,
                                -fx * (1 + point_x**2 / point_z**2),
                                fx * point_y / point_z,
                            ]
                        ),
                        np.column_stack(
                            [
                                np.zeros_like(point_z),
                                -fy / point_z,
                                fy * point_y / point_z**2,
                                fy * (1 + point_y**2 / point_z**2),
                                -fy * point_x * point_y / point_z**2,
                                -fy * point_x / point_z,
                            ]
                        ),
                    ]
                )
                measured_flow = np.concatenate([flow[:, 0], flow[:, 1]])
                twist_solution, _, _, _ = np.linalg.lstsq(
                    J_flow_twist, measured_flow, rcond=None
                )
                velocity = twist_solution[:3]

        self._prev_gray_front = gray_front
        self._prev_points_front = next_points_front.reshape(-1, 1, 2)

        return velocity, points_3d
