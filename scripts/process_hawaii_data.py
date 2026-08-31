#!/usr/bin/env python3
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
import os
import re
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(os.environ["OVERLAY_WS"]) / "src" / "coug_visual_dvl"

sys.path.insert(0, str(BASE_DIR / "coug_visual_dvl"))
from coug_visual_dvl.visual_dvl import VisualDvl

CALIB_FILE = BASE_DIR / "scripts" / "stereo_calibration_params.json"
STEREO_PAIRS = BASE_DIR / "scripts" / "mission_stereo_pairs.json"


def _load_bayer_bmp(filepath: str) -> np.ndarray:
    with open(filepath, "rb") as f:
        f.seek(10)
        start = struct.unpack("<I", f.read(4))[0]
        f.seek(18)
        w, h = struct.unpack("<ii", f.read(8))
        f.seek(34)
        size = struct.unpack("<I", f.read(4))[0]

        f.seek(start)
        is_8bit = (size / abs(h)) == w
        fmt = "<" + ("B" if is_8bit else "H") * abs(h) * w
        img = np.array(struct.unpack(fmt, f.read())).reshape(abs(h), w)

    if not is_8bit:
        img >>= 4  # Convert 12-bit to 8-bit

    gray = cv2.cvtColor(np.uint8(img), cv2.COLOR_BayerBG2GRAY)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def main() -> None:
    with open(CALIB_FILE, "r") as f:
        calib_dict = json.load(f)

    with open(STEREO_PAIRS, "r") as f:
        stereo_pairs = json.load(f)

    estimator = None
    prev_time = None

    for pair in stereo_pairs:
        front_path = BASE_DIR / pair["front"]
        back_path = BASE_DIR / pair["back"]

        match = re.search(r"_(\d+)_raw\.bmp$", pair["front"])
        if not match:
            print(f"Could not parse timestamp from {pair['front']}. Skipping.")
            continue

        curr_time = int(match.group(1)) * 1e-9
        dt = curr_time - prev_time if prev_time is not None else 0.0
        prev_time = curr_time

        image_front = _load_bayer_bmp(str(front_path))
        image_back = _load_bayer_bmp(str(back_path))

        if estimator is None:
            estimator = VisualDvl(calib_dict, image_front.shape[::-1])

        velocity, _ = estimator.estimate_velocity(image_front, image_back, dt)
        vx, vy, vz = velocity[0], velocity[1], velocity[2]
        print(
            f"Time: {curr_time:.2f} s | dt: {dt:.3f} s | Velocity: [{vx:.3f}, {vy:.3f}, {vz:.3f}] m/s"
        )


if __name__ == "__main__":
    main()
