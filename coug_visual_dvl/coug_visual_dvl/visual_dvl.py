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


class VisualDVL:
    """
    Core computer vision logic for AUV visual odometry using stereo cameras.

    :author: Nelson Durrant & Braden Meyers
    :date: Mar 2026
    """

    def __init__(self, calib_dict: dict, img_size: tuple):
        """
        Initializes stereo rectification and undistortion maps.

        :param calib_dict: Dictionary containing camera calibration parameters.
        :param img_size: Tuple containing the input image dimensions (width, height).
        """
        self.mtx_f = np.array(calib_dict["mtx_f"])
        self.dist_f = np.array(calib_dict["dist_f"])
        self.mtx_b = np.array(calib_dict["mtx_b"])
        self.dist_b = np.array(calib_dict["dist_b"])
        self.R = np.array(calib_dict["R"])
        self.T = np.array(calib_dict["T"])

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.mtx_f,
            self.dist_f,
            self.mtx_b,
            self.dist_b,
            img_size,
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        self.map_f1, self.map_f2 = cv2.initUndistortRectifyMap(
            self.mtx_f, self.dist_f, self.R1, self.P1, img_size, cv2.CV_32FC1
        )
        self.map_b1, self.map_b2 = cv2.initUndistortRectifyMap(
            self.mtx_b, self.dist_b, self.R2, self.P2, img_size, cv2.CV_32FC1
        )

        self.prev_gray_f = None
        self.prev_gray_b = None
        self.prev_pts_f = None
        self.prev_pts_b = None

    def estimate_velocity(
        self, cv_f: np.ndarray, cv_b: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Estimates 3D velocity (vx, vy, vz) from unrectified OpenCV arrays.

        :param cv_f: Unrectified image from the front camera.
        :param cv_b: Unrectified image from the back camera.
        :param dt: Time elapsed since the last frame in seconds.
        :return: Estimated 3D velocity as a numpy array [vx, vy, vz].
        """
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

        if self.prev_gray_f is None or dt <= 0.0:
            self.prev_gray_f = gray_f
            self.prev_gray_b = gray_b
            return np.array([0.0, 0.0, 0.0])

        # TODO: Add here

        velocity = np.array([0.0, 0.0, 0.0])

        self.prev_gray_f = gray_f
        self.prev_gray_b = gray_b

        return velocity
