"""
Convert MotionGPT / HumanML3D joint trajectories to MotionData JSON compatible
with nirvana-animate-saas (rot6d, transl, keypoints3d, root_rotations_mat).
"""

from __future__ import annotations

import numpy as np

from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat

# HumanML3D joint order -> SMPL body-22 order (matches nirvana smpl.json jointOrder).
HUMANML_TO_SMPL_PERM = np.array(
    [
        0,  # root -> Pelvis
        2,  # LH -> L_Hip
        1,  # RH -> R_Hip
        3,  # BP -> Spine1
        5,  # LK -> L_Knee
        4,  # RK -> R_Knee
        6,  # BT -> Spine2
        8,  # LMrot -> L_Ankle
        7,  # RMrot -> R_Ankle
        9,  # BLN -> Spine3
        11,  # LF -> L_Foot
        10,  # RF -> R_Foot
        12,  # BMN -> Neck
        14,  # LSI -> L_Collar
        13,  # RSI -> R_Collar
        15,  # BUN -> Head
        17,  # LS -> L_Shoulder
        16,  # RS -> R_Shoulder
        19,  # LE -> L_Elbow
        18,  # RE -> R_Elbow
        21,  # LW -> L_Wrist
        20,  # RW -> R_Wrist
    ],
    dtype=np.int64,
)

NUM_JOINTS = 22
# HumanML3D / MotionGPT visualization FPS used by the SaaS motionToClip path.
DEFAULT_FPS = 20


def _rotmat_to_rot6d(R: np.ndarray) -> list[float]:
    """Match HY-Motion / motionToClip Gram-Schmidt input layout (row-major 3x2)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return [
        float(R[0, 0]),
        float(R[0, 1]),
        float(R[1, 0]),
        float(R[1, 1]),
        float(R[2, 0]),
        float(R[2, 1]),
    ]


def joints_to_motion_data(
    joints: np.ndarray,
    num_frames: int,
    fps: float = DEFAULT_FPS,
) -> dict:
    """
    joints: (T, 22, 3) global positions in HumanML3D joint order.
    num_frames: valid frame count (slice if joints is padded).
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    j = np.asarray(joints, dtype=np.float64)[:num_frames]
    if j.shape != (num_frames, NUM_JOINTS, 3):
        raise ValueError(
            f"expected joints shape ({num_frames}, {NUM_JOINTS}, 3), got {j.shape}"
        )

    # SMPL-22 order for HybrIK + client rig (smpl.json).
    j_smpl = j[:, HUMANML_TO_SMPL_PERM, :]

    # Match render.py slow path: recenter for stable IK (preserves local rotations).
    origin = j_smpl[0:1, 0:1, :]
    j_ik = j_smpl - origin

    hybrik = HybrIKJointsToRotmat()
    pose = hybrik(j_ik)  # (T, 22, 3, 3)

    rot6d: list[list[list[float]]] = []
    for t in range(num_frames):
        frame_rot: list[list[float]] = []
        for joint_idx in range(NUM_JOINTS):
            frame_rot.append(_rotmat_to_rot6d(pose[t, joint_idx]))
        rot6d.append(frame_rot)

    transl = j_smpl[:, 0, :].astype(float).tolist()
    root_rotations_mat = pose[:, 0, :, :].astype(float).tolist()
    keypoints3d = j_smpl.astype(float).tolist()

    return {
        "rot6d": rot6d,
        "transl": transl,
        "keypoints3d": keypoints3d,
        "root_rotations_mat": root_rotations_mat,
        "num_frames": int(num_frames),
        "fps": float(fps),
    }
