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

import cv2
import numpy as np

MAX_CORNERS = 200
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10
MIN_FEATURE_COUNT = 150
EPIPOLAR_THRESHOLD_PX = 2.0
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 50.0
MIN_PTS_FOR_LLS = 3


class VisualDVL:
    def __init__(self, calib_dict: dict, img_size: tuple) -> None:
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
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
        self.map_f1, self.map_f2 = cv2.initUndistortRectifyMap(
            np.array(calib_dict["mtx_f"]),
            np.array(calib_dict["dist_f"]),
            self.R1,
            self.P1,
            img_size,
            cv2.CV_32FC1,
        )
        self.map_b1, self.map_b2 = cv2.initUndistortRectifyMap(
            np.array(calib_dict["mtx_b"]),
            np.array(calib_dict["dist_b"]),
            self.R2,
            self.P2,
            img_size,
            cv2.CV_32FC1,
        )

        self.prev_gray_f = None
        self.prev_pts_f = None

    def estimate_velocity(
        self, cv_f: np.ndarray, cv_b: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        rect_f = cv2.remap(cv_f, self.map_f1, self.map_f2, cv2.INTER_LINEAR)
        rect_b = cv2.remap(cv_b, self.map_b1, self.map_b2, cv2.INTER_LINEAR)
        gray_f = (
            cv2.cvtColor(rect_f, cv2.COLOR_BGR2GRAY)
            if len(rect_f.shape) == 3
            else rect_f
        )
        gray_b = (
            cv2.cvtColor(rect_b, cv2.COLOR_BGR2GRAY)
            if len(rect_b.shape) == 3
            else rect_b
        )

        if dt <= 0.0 or self.prev_gray_f is None:
            self.prev_gray_f = gray_f
            self.prev_pts_f = cv2.goodFeaturesToTrack(
                gray_f,
                maxCorners=MAX_CORNERS,
                qualityLevel=QUALITY_LEVEL,
                minDistance=MIN_DISTANCE,
            )
            return np.array([0.0, 0.0, 0.0]), np.empty((0, 3))

        # Track features forward across frames using LK optical flow
        curr_pts, prev_pts = None, None
        if self.prev_pts_f is not None:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray_f, gray_f, self.prev_pts_f, None
            )
            valid = status.ravel() == 1  # Discard untracked features
            curr_pts = (
                curr_pts[valid] if valid.sum() > 0 else None
            )  # Current feature positions
            prev_pts = (
                self.prev_pts_f[valid].reshape(-1, 2) if valid.sum() > 0 else None
            )  # Previous feature positions

        # Detect new features when count drops below a threshold
        new_pts = None
        if curr_pts is None or len(curr_pts) < MIN_FEATURE_COUNT:
            new_pts = cv2.goodFeaturesToTrack(
                gray_f,
                maxCorners=MAX_CORNERS,
                qualityLevel=QUALITY_LEVEL,
                minDistance=MIN_DISTANCE,
            )

        # Check to make sure we tracked or found something
        if curr_pts is None and new_pts is None:
            self.prev_gray_f = gray_f
            self.prev_pts_f = None
            return np.array([0.0, 0.0, 0.0]), np.empty((0, 3))
        next_pts_f = np.concatenate([x for x in [curr_pts, new_pts] if x is not None])

        pts_3d = np.empty((0, 3))
        velocity = np.array([0.0, 0.0, 0.0])
        if curr_pts is not None:
            # Find each tracked feature in the rectified back frame using LK optical flow
            pts_back, stereo_status, _ = cv2.calcOpticalFlowPyrLK(
                gray_f, gray_b, curr_pts, None
            )
            epipolar_valid = (
                np.abs(curr_pts[:, 0, 0] - pts_back[:, 0, 0]) <= EPIPOLAR_THRESHOLD_PX
            )
            stereo_valid = (
                stereo_status.ravel() == 1
            ) & epipolar_valid  # Discard unlocated features

            if stereo_valid.sum() < 1:
                self.prev_gray_f = gray_f
                self.prev_pts_f = next_pts_f.reshape(-1, 1, 2)
                return np.array([0.0, 0.0, 0.0]), np.empty((0, 3))

            # Triangulate matched feature pairs into homogeneous 3D coordinates
            pts_4d = cv2.triangulatePoints(
                self.P1,  # Front projection matrix
                self.P2,  # Back projection matrix
                curr_pts[stereo_valid].reshape(-1, 2).T,  # 2xN points in front image
                pts_back[stereo_valid].reshape(-1, 2).T,  # 2xN points in back image
            )
            pts_3d_all = (pts_4d[:3] / pts_4d[3]).T

            # Discard invalid points behind the camera or crazy far away
            depth_valid = (
                (pts_3d_all[:, 2] > MIN_DEPTH_M)
                & (pts_3d_all[:, 2] < MAX_DEPTH_M)
                & np.isfinite(pts_3d_all).all(axis=1)
            )
            pts_3d = pts_3d_all[depth_valid]

            # Solve for camera velocity using LLS
            # du/dt = -(fx/Z)*Vx + (fx*X/Z²)*Vz + (fx*X*Y/Z²)*ωx - fx*(1+X²/Z²)*ωy + (fx*Y/Z)*ωz
            # dv/dt = -(fy/Z)*Vy + (fy*Y/Z²)*Vz + fy*(1+Y²/Z²)*ωx - (fy*X*Y/Z²)*ωy - (fy*X/Z)*ωz
            if prev_pts is not None and len(pts_3d) >= MIN_PTS_FOR_LLS:
                curr_2d = curr_pts[stereo_valid][depth_valid].reshape(
                    -1, 2
                )  # Current 2D positions
                prev_2d = prev_pts[stereo_valid][depth_valid].reshape(
                    -1, 2
                )  # Previous 2D positions
                flow = (curr_2d - prev_2d) / dt

                fx, fy = self.P1[0, 0], self.P1[1, 1]
                X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
                A = np.vstack(
                    [
                        np.column_stack(
                            [
                                -fx / Z,
                                np.zeros_like(Z),
                                fx * X / Z**2,
                                fx * X * Y / Z**2,
                                -fx * (1 + X**2 / Z**2),
                                fx * Y / Z,
                            ]
                        ),
                        np.column_stack(
                            [
                                np.zeros_like(Z),
                                -fy / Z,
                                fy * Y / Z**2,
                                fy * (1 + Y**2 / Z**2),
                                -fy * X * Y / Z**2,
                                -fy * X / Z,
                            ]
                        ),
                    ]
                )
                b = np.concatenate([flow[:, 0], flow[:, 1]])
                solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                velocity = solution[:3]

        self.prev_gray_f = gray_f
        self.prev_pts_f = next_pts_f.reshape(-1, 1, 2)

        return velocity, pts_3d
