#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import quaternion  # noqa # pylint: disable=unused-import


def quaternion_to_rotation(q_r, q_i, q_j, q_k):
    r"""
    ref: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    s = 1  # unit quaternion
    return np.array(
        [
            [
                1 - 2 * s * (q_j**2 + q_k**2),
                2 * s * (q_i * q_j - q_k * q_r),
                2 * s * (q_i * q_k + q_j * q_r),
            ],
            [
                2 * s * (q_i * q_j + q_k * q_r),
                1 - 2 * s * (q_i**2 + q_k**2),
                2 * s * (q_j * q_k - q_i * q_r),
            ],
            [
                2 * s * (q_i * q_k - q_j * q_r),
                2 * s * (q_j * q_k + q_i * q_r),
                1 - 2 * s * (q_i**2 + q_j**2),
            ],
        ],
        dtype=np.float32,
    )


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def compute_pixel_coverage(instance_seg, object_id):
    cand_mask = instance_seg == object_id
    return cand_mask.sum().astype(np.float64) / cand_mask.size


def get_angle(x, y):
    """
    Gets the angle between two vectors in radians.
    """
    x_norm = x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x
    y_norm = y / np.linalg.norm(y) if np.linalg.norm(y) != 0 else y
    return np.arccos(np.clip(np.dot(x_norm, y_norm), -1, 1))
