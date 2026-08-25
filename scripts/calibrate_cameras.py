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

# %%
import json
import os
import re
import struct
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = (
    Path(os.environ.get("HOME", os.path.expanduser("~")))
    / "cougars-dev"
    / "ros2_ws"
    / "src"
    / "coug_visual_dvl"
)

CALIBRATION_DIR = BASE_DIR / "hawaii_data" / "20240418_calibration_imgs"
CALIBRATION_FRONT_DIR = CALIBRATION_DIR / "DEV_000F314F3266"
CALIBRATION_BACK_DIR = CALIBRATION_DIR / "DEV_000F314F3269"
ALL_PAIRS_OUTPUT = BASE_DIR / "scripts" / "all_stereo_pairs.json"
GOOD_PAIRS_OUTPUT = BASE_DIR / "scripts" / "good_stereo_pairs.json"
VERIFIED_PAIRS_OUTPUT = BASE_DIR / "scripts" / "verified_stereo_pairs.json"

MISSION_DIR = (
    BASE_DIR
    / "hawaii_data"
    / "20240418-130518--2024-04-18_oahu_zblocks_4mHFB-IVER3-3100"
)
MISSION_FRONT_DIR = MISSION_DIR / "DEV_000F314F3266"
MISSION_BACK_DIR = MISSION_DIR / "DEV_000F314F3269"
MISSION_PAIRS_OUTPUT = BASE_DIR / "scripts" / "mission_stereo_pairs.json"

BOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.05
BUCKET_NS = 100_000_000  # 100 ms in nanoseconds
NS_RE = re.compile(r"_(\d+)_raw\.bmp$")


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


def _index_by_bucket(directory: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for f in sorted(directory.glob("*.bmp")):
        m = NS_RE.search(f.name)
        if not m:
            continue
        bucket = round(int(m.group(1)) / BUCKET_NS)
        mapping[bucket] = f
    return mapping


# %%
print("Matching stereo pairs by timestamp...")

front_map = _index_by_bucket(CALIBRATION_FRONT_DIR)
back_map = _index_by_bucket(CALIBRATION_BACK_DIR)

common_buckets = sorted(set(front_map) & set(back_map))

all_pairs = [
    {
        "front": str(front_map[b].relative_to(BASE_DIR)),
        "back": str(back_map[b].relative_to(BASE_DIR)),
    }
    for b in common_buckets
]

ALL_PAIRS_OUTPUT.write_text(json.dumps(all_pairs, indent=2))
print(f"Matched stereo pairs: {len(all_pairs)}")

mission_front_map = _index_by_bucket(MISSION_FRONT_DIR)
mission_back_map = _index_by_bucket(MISSION_BACK_DIR)

mission_common_buckets = sorted(set(mission_front_map) & set(mission_back_map))

mission_pairs = [
    {
        "front": str(mission_front_map[b].relative_to(BASE_DIR)),
        "back": str(mission_back_map[b].relative_to(BASE_DIR)),
    }
    for b in mission_common_buckets
]

MISSION_PAIRS_OUTPUT.write_text(json.dumps(mission_pairs, indent=2))
print(f"Matched mission stereo pairs: {len(mission_pairs)}\n")

# %%
print("Detecting chessboard corners to find good stereo pairs...")

if GOOD_PAIRS_OUTPUT.exists():
    print(f"Found existing {GOOD_PAIRS_OUTPUT.name}. Skipping.\n")
    with open(GOOD_PAIRS_OUTPUT, "r") as f:
        good_pairs = json.load(f)
else:
    good_pairs = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for i, pair in enumerate(all_pairs):
        front_path = BASE_DIR / pair["front"]
        back_path = BASE_DIR / pair["back"]

        image_front = clahe.apply(_load_bayer_bmp(str(front_path)))
        image_back = clahe.apply(_load_bayer_bmp(str(back_path)))

        found_front, corners_front = cv2.findChessboardCornersSB(
            image_front, BOARD_SIZE, None
        )
        found_back, corners_back = cv2.findChessboardCornersSB(
            image_back, BOARD_SIZE, None
        )

        if found_front and found_back:
            pair_data = pair.copy()
            pair_data["corners_f"] = corners_front.tolist()
            pair_data["corners_b"] = corners_back.tolist()
            good_pairs.append(pair_data)

        if i % 50 == 0 or i == len(all_pairs) - 1:
            print(
                f"Scanned {i}/{len(all_pairs)}... Good pairs so far: {len(good_pairs)}"
            )

    GOOD_PAIRS_OUTPUT.write_text(json.dumps(good_pairs, indent=2))
    print(f"Good stereo pairs: {len(good_pairs)}\n")

# %%
print("Displaying good stereo pairs for review...")

with open(GOOD_PAIRS_OUTPUT, "r") as f:
    loaded_good_pairs = json.load(f)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

window_name = "Stereo Calibration Review"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

verified_pairs = []
for i, pair in enumerate(loaded_good_pairs):
    front_path = BASE_DIR / pair["front"]
    back_path = BASE_DIR / pair["back"]

    image_front = clahe.apply(_load_bayer_bmp(str(front_path)))
    image_back = clahe.apply(_load_bayer_bmp(str(back_path)))

    corners_front = np.array(pair["corners_f"], dtype=np.float32)
    corners_back = np.array(pair["corners_b"], dtype=np.float32)

    display_front = cv2.cvtColor(image_front, cv2.COLOR_GRAY2BGR)
    display_back = cv2.cvtColor(image_back, cv2.COLOR_GRAY2BGR)

    cv2.drawChessboardCorners(display_front, BOARD_SIZE, corners_front, True)
    cv2.drawChessboardCorners(display_back, BOARD_SIZE, corners_back, True)

    combined_img = np.vstack((display_front, display_back))
    combined_img = cv2.resize(combined_img, (0, 0), fx=0.25, fy=0.25)

    overlay_text = (
        f"{i + 1}/{len(loaded_good_pairs)} | [y] Accept | [n] Reject | [q] Save & Quit"
    )
    cv2.putText(
        combined_img,
        overlay_text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow(window_name, combined_img)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("y"):
            verified_pairs.append(pair)
            print(f"[{i + 1}/{len(loaded_good_pairs)}] Accepted")
            break
        elif key == ord("n"):
            print(f"[{i + 1}/{len(loaded_good_pairs)}] Rejected")
            break
        elif key == ord("q"):
            break

    if key == ord("q"):
        break

cv2.destroyAllWindows()

VERIFIED_PAIRS_OUTPUT.write_text(json.dumps(verified_pairs, indent=2))
print(f"Verified stereo pairs: {len(verified_pairs)}\n")

# %%
print("Using verified stereo pairs to calibrate the cameras...")

with open(VERIFIED_PAIRS_OUTPUT) as f:
    pairs = json.load(f)

object_points_template = np.zeros((np.prod(BOARD_SIZE), 3), np.float32)
object_points_template[:, :2] = (
    np.mgrid[0 : BOARD_SIZE[0], 0 : BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
)

object_points = [object_points_template] * len(pairs)
image_points_front = [np.float32(p["corners_f"]) for p in pairs]
image_points_back = [np.float32(p["corners_b"]) for p in pairs]

img_size = _load_bayer_bmp(str(BASE_DIR / pairs[0]["front"])).shape[::-1]

rmse_front, front_camera_matrix, front_dist_coeffs, *_ = cv2.calibrateCamera(
    object_points, image_points_front, img_size, None, None
)
print(f"Front Camera RMSE (pixels): {rmse_front:.3f}")

rmse_back, back_camera_matrix, back_dist_coeffs, *_ = cv2.calibrateCamera(
    object_points, image_points_back, img_size, None, None
)
print(f"Back Camera RMSE (pixels): {rmse_back:.3f}")

(
    rmse_stereo,
    front_camera_matrix,
    front_dist_coeffs,
    back_camera_matrix,
    back_dist_coeffs,
    back_R_front,
    back_t_front,
    essential_matrix,
    fundamental_matrix,
) = cv2.stereoCalibrate(
    object_points,
    image_points_front,
    image_points_back,
    front_camera_matrix,
    front_dist_coeffs,
    back_camera_matrix,
    back_dist_coeffs,
    img_size,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=cv2.CALIB_FIX_INTRINSIC,
)
print(f"Stereo RMSE (pixels): {rmse_stereo:.3f}\n")

calib_data = {
    k: v.tolist()
    for k, v in zip(
        ["mtx_f", "dist_f", "mtx_b", "dist_b", "R", "T", "E", "F"],
        [
            front_camera_matrix,
            front_dist_coeffs,
            back_camera_matrix,
            back_dist_coeffs,
            back_R_front,
            back_t_front,
            essential_matrix,
            fundamental_matrix,
        ],
    )
}

out_file = BASE_DIR / "scripts" / "stereo_calibration_params.json"
out_file.write_text(json.dumps(calib_data, indent=2))
