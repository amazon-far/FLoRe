"""Data transforms for the S1 policy: rotation lift, absolute->relative
EE conversion, and gripper normalization.

Rotation conventions come from ``flore.utils.rot6d`` (the single source of truth for
the whole repo) — rot6d here is the FIRST-TWO-ROWS flavor, axis-angle (rotvec) is the
on-disk UMI zarr rotation. We do NOT duplicate that math.

Absolute->relative EE conversion uses RELATIVE SE(3) semantics: each future pose is
expressed in the body frame of a reference (current-EE) pose::

    R_rel = R_ref^T @ R_a
    p_rel = R_ref^T @ (p_a - p_ref)

The caller passes pos and rot6d arrays explicitly (cleaner for the fixed UMI 3-point
layout) rather than one flat action vector sliced by config. ``delta_ee_to_absolute`` is the exact inverse (used by serving / tests).
"""

from __future__ import annotations

import numpy as np

from flore.utils.rot6d import (
    rot6d_to_rotmat,
    rotmat_to_rot6d,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
)

# Reference full-open gripper width in metres; width_m / this -> [0, 1].
GRIPPER_WIDTH_REF_M = 0.095


# ---------------------------------------------------------------------------
# Rotation lift: axis-angle (on-disk) <-> rot6d (model space)
# ---------------------------------------------------------------------------
def rotvec_to_rot6d(rotvec: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 6) rot6d (first-two-rows)."""
    return rotmat_to_rot6d(rotvec_to_rotmat(np.asarray(rotvec, dtype=np.float64)))


def rot6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) rot6d -> (..., 3) axis-angle."""
    return rotmat_to_rotvec(rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64)))


def rot6d_relative(rot6d: np.ndarray, ref_rot6d: np.ndarray) -> np.ndarray:
    """Orientation of ``rot6d`` expressed in the frame of ``ref_rot6d``: ``R_ref^T R``.

    The ROTATION-ONLY half of :func:`absolute_to_delta_ee` — no translation. Used for
    the ``wrt_start`` observation channel (a keypoint's current orientation relative to
    its episode-start orientation), mirroring UMI-on-Air's
    ``robot0_eef_rot_axis_angle_wrt_start``. Frame-invariant to the world origin, so it
    is safe to feed the model where absolute pose would be out-of-distribution.

    Parameters
    ----------
    rot6d : (..., 6) current rotation (rot6d).
    ref_rot6d : (6,) or broadcastable reference rotation (episode-start).
    """
    r = rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64))          # (..., 3, 3)
    r_ref = rot6d_to_rotmat(np.asarray(ref_rot6d, dtype=np.float64))  # (3, 3) typically
    r_rel = np.einsum("ji,...jk->...ik", r_ref, r)                    # R_ref^T @ R
    return rotmat_to_rot6d(r_rel)


# ---------------------------------------------------------------------------
# Absolute EE trajectory -> relative-to-reference (RELATIVE SE(3) math)
# ---------------------------------------------------------------------------
def absolute_to_delta_ee(
    pos: np.ndarray,
    rot6d: np.ndarray,
    ref_pos: np.ndarray,
    ref_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express absolute EE poses in the body frame of a reference pose.

    RELATIVE math per pose:
        ``R_rel = R_ref^T @ R_a``  and  ``p_rel = R_ref^T @ (p_a - p_ref)``.

    Parameters
    ----------
    pos : (..., 3) absolute positions.
    rot6d : (..., 6) absolute rotations (rot6d).
    ref_pos : (3,) or broadcastable reference position (current EE).
    ref_rot6d : (6,) or broadcastable reference rotation.

    Returns
    -------
    (rel_pos, rel_rot6d) : each same leading shape as ``pos`` / ``rot6d``.
    """
    pos = np.asarray(pos, dtype=np.float64)
    rot6d = np.asarray(rot6d, dtype=np.float64)
    r_a = rot6d_to_rotmat(rot6d)  # (..., 3, 3)
    r_ref = rot6d_to_rotmat(np.asarray(ref_rot6d, dtype=np.float64))  # (3, 3) typically
    ref_pos = np.asarray(ref_pos, dtype=np.float64)

    # R_rel = R_ref^T @ R_a
    r_rel = np.einsum("ji,...jk->...ik", r_ref, r_a)
    rel_rot6d = rotmat_to_rot6d(r_rel)

    # p_rel = R_ref^T @ (p_a - p_ref)
    dp = pos - ref_pos
    rel_pos = np.einsum("ji,...j->...i", r_ref, dp)
    return rel_pos.astype(np.float64), rel_rot6d.astype(np.float64)


def delta_ee_to_absolute(
    rel_pos: np.ndarray,
    rel_rot6d: np.ndarray,
    ref_pos: np.ndarray,
    ref_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`absolute_to_delta_ee`.

    ``R_a = R_ref @ R_rel``  and  ``p_a = R_ref @ p_rel + p_ref``.
    """
    rel_pos = np.asarray(rel_pos, dtype=np.float64)
    rel_rot6d = np.asarray(rel_rot6d, dtype=np.float64)
    r_rel = rot6d_to_rotmat(rel_rot6d)
    r_ref = rot6d_to_rotmat(np.asarray(ref_rot6d, dtype=np.float64))
    ref_pos = np.asarray(ref_pos, dtype=np.float64)

    r_a = np.einsum("ij,...jk->...ik", r_ref, r_rel)
    abs_rot6d = rotmat_to_rot6d(r_a)
    abs_pos = np.einsum("ij,...j->...i", r_ref, rel_pos) + ref_pos
    return abs_pos.astype(np.float64), abs_rot6d.astype(np.float64)


# ---------------------------------------------------------------------------
# Identity rotation
# ---------------------------------------------------------------------------
# rot6d of the identity rotation (first two rows of I3). rot6d_to_rotmat of this
# is exactly I3 (verified), so an identity base is a no-op reference pose.
IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Gripper width normalization: metres <-> [0, 1]
# ---------------------------------------------------------------------------
def gripper_width_to_norm(width_m: np.ndarray, ref_m: float = GRIPPER_WIDTH_REF_M) -> np.ndarray:
    """Absolute gripper width in metres -> normalized [0, 1] (clipped)."""
    return np.clip(np.asarray(width_m, dtype=np.float64) / ref_m, 0.0, 1.0)


def gripper_norm_to_width(norm: np.ndarray, ref_m: float = GRIPPER_WIDTH_REF_M) -> np.ndarray:
    """Normalized [0, 1] gripper -> absolute width in metres."""
    return np.asarray(norm, dtype=np.float64) * ref_m
