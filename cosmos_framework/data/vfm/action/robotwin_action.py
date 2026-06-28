# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboTwin action-space helpers.

RoboTwin LeRobot shards store 14D dual-arm joint/gripper targets::

    [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]

Supported ``action_space`` values:

* ``joint_pos``: absolute commanded joint targets (legacy default).
* ``joint_delta``: backward-anchored joint offsets on arm joints only; each step
  is ``absolute_target[k] - qpos[t0]`` (OpenPI / pi05 style). Grippers stay
  absolute at each target timestep.
* ``ee_pose_delta``: SE(3) backward-framewise deltas per arm with rot6d rotation
  in OpenCV coordinates; grippers stay absolute.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from cosmos_framework.data.vfm.action.pose_utils import (
    _delta_transform_to_pose_vector,
    pose_abs_to_rel,
    pose_rel_to_abs,
)
from cosmos_framework.data.vfm.action.robotwin_fk import fk_dual_arm_ee_poses, ik_dual_arm_from_poses

RobotTwinActionSpace = Literal["joint_pos", "joint_delta", "ee_pose_delta"]

_LEFT_ARM_SLICE = slice(0, 6)
_LEFT_GRIPPER_IDX = 6
_RIGHT_ARM_SLICE = slice(7, 13)
_RIGHT_GRIPPER_IDX = 13
_JOINT_DIM = 14
_DUAL_EE_POSE_DIM = 20
_EE_POSE_ACTION_DIM = 9


def validate_action_space(action_space: str) -> RobotTwinActionSpace:
    allowed: tuple[RobotTwinActionSpace, ...] = ("joint_pos", "joint_delta", "ee_pose_delta")
    if action_space not in allowed:
        raise ValueError(f"Unsupported RoboTwin action_space {action_space!r}; expected one of {allowed}.")
    return action_space  # type: ignore[return-value]


def raw_action_dim_for_space(action_space: RobotTwinActionSpace) -> int:
    if action_space == "ee_pose_delta":
        return _DUAL_EE_POSE_DIM
    return _JOINT_DIM


def joint_trajectory_to_backward_framewise_delta(trajectory: torch.Tensor) -> torch.Tensor:
    """Convert absolute joint trajectory ``[T, 14]`` to ``[T-1, 14]`` framewise deltas.

    Arm joints use ``q[t+1] - q[t]``. Gripper channels keep the absolute commanded
    value at the target frame ``q[t+1]``. Legacy helper; ``joint_delta`` training
    uses :func:`joint_trajectory_to_backward_anchored_delta` instead.
    """
    if trajectory.ndim != 2 or trajectory.shape[-1] != _JOINT_DIM:
        raise ValueError(f"Expected trajectory shape [T, {_JOINT_DIM}], got {tuple(trajectory.shape)}")
    if trajectory.shape[0] < 2:
        raise ValueError(f"Need at least 2 frames to compute joint deltas, got T={trajectory.shape[0]}")

    current = trajectory[:-1]
    target = trajectory[1:]
    delta = target.clone()
    delta[..., _LEFT_ARM_SLICE] = target[..., _LEFT_ARM_SLICE] - current[..., _LEFT_ARM_SLICE]
    delta[..., _RIGHT_ARM_SLICE] = target[..., _RIGHT_ARM_SLICE] - current[..., _RIGHT_ARM_SLICE]
    return delta


def joint_trajectory_to_backward_anchored_delta(trajectory: torch.Tensor) -> torch.Tensor:
    """Convert absolute joint trajectory ``[T, 14]`` to ``[T-1, 14]`` anchored offsets.

    Arm joints use ``absolute_target[k] - qpos[t0]`` for every step in the chunk.
    Gripper channels keep the absolute commanded value at each target frame.
    """
    if trajectory.ndim != 2 or trajectory.shape[-1] != _JOINT_DIM:
        raise ValueError(f"Expected trajectory shape [T, {_JOINT_DIM}], got {tuple(trajectory.shape)}")
    if trajectory.shape[0] < 2:
        raise ValueError(f"Need at least 2 frames to compute joint deltas, got T={trajectory.shape[0]}")

    anchor = trajectory[:1]
    targets = trajectory[1:]
    delta = targets.clone()
    delta[..., _LEFT_ARM_SLICE] = targets[..., _LEFT_ARM_SLICE] - anchor[..., _LEFT_ARM_SLICE]
    delta[..., _RIGHT_ARM_SLICE] = targets[..., _RIGHT_ARM_SLICE] - anchor[..., _RIGHT_ARM_SLICE]
    return delta


def joint_anchored_deltas_to_absolute(initial_qpos: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Convert anchored joint offsets back to absolute 14D joint targets ``[T, 14]``."""
    if initial_qpos.shape[-1] != _JOINT_DIM:
        raise ValueError(f"initial_qpos must have dim {_JOINT_DIM}, got {initial_qpos.shape[-1]}")
    if deltas.ndim != 2 or deltas.shape[-1] != _JOINT_DIM:
        raise ValueError(f"Expected deltas shape [T, {_JOINT_DIM}], got {tuple(deltas.shape)}")

    initial_qpos = initial_qpos.reshape(-1)[:_JOINT_DIM].to(device=deltas.device, dtype=deltas.dtype)
    absolute = deltas.clone()
    absolute[..., _LEFT_ARM_SLICE] = initial_qpos[_LEFT_ARM_SLICE] + deltas[..., _LEFT_ARM_SLICE]
    absolute[..., _RIGHT_ARM_SLICE] = initial_qpos[_RIGHT_ARM_SLICE] + deltas[..., _RIGHT_ARM_SLICE]
    return absolute


def integrate_joint_deltas(initial_qpos: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Convert ``joint_delta`` model outputs to absolute 14D joint targets ``[T, 14]``."""
    return joint_anchored_deltas_to_absolute(initial_qpos, deltas)


def _split_dual_arm_joints(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_arm = joints[..., _LEFT_ARM_SLICE]
    left_gripper = joints[..., [_LEFT_GRIPPER_IDX]]
    right_arm = joints[..., _RIGHT_ARM_SLICE]
    right_gripper = joints[..., [_RIGHT_GRIPPER_IDX]]
    return left_arm, left_gripper, right_arm, right_gripper


def _fk_dual_arm_poses_from_joints(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return fk_dual_arm_ee_poses(joints, apply_opencv=True)


def dual_arm_joints_to_absolute_ee_pose_action(joints: torch.Tensor) -> torch.Tensor:
    """Encode one or more 14D joint rows as absolute 20D ee pose actions."""
    joints_np = joints.detach().cpu().numpy().astype(np.float32, copy=False)
    if joints_np.ndim == 1:
        joints_np = joints_np.reshape(1, -1)
    left_poses, right_poses = _fk_dual_arm_poses_from_joints(joints_np)
    _, left_gripper, _, right_gripper = _split_dual_arm_joints(joints_np)
    rows = []
    for index in range(joints_np.shape[0]):
        rows.append(
            np.concatenate(
                [
                    _absolute_pose_to_action_vector(left_poses[index]),
                    left_gripper[index],
                    _absolute_pose_to_action_vector(right_poses[index]),
                    right_gripper[index],
                ],
                axis=-1,
            )
        )
    return torch.from_numpy(np.stack(rows, axis=0)).float()


def dual_arm_joints_to_ee_pose_delta_action(
    trajectory: torch.Tensor,
    *,
    pose_convention: str = "backward_framewise",
) -> torch.Tensor:
    """Build a 20D dual-arm SE(3) delta action chunk from absolute joint targets."""
    joints = trajectory.detach().cpu().numpy().astype(np.float32, copy=False)
    left_poses, right_poses = _fk_dual_arm_poses_from_joints(joints)
    left_rel = pose_abs_to_rel(left_poses, rotation_format="rot6d", pose_convention=pose_convention)
    right_rel = pose_abs_to_rel(right_poses, rotation_format="rot6d", pose_convention=pose_convention)
    _, left_gripper, _, right_gripper = _split_dual_arm_joints(joints)
    action = np.concatenate(
        [
            left_rel,
            left_gripper[-left_rel.shape[0] :],
            right_rel,
            right_gripper[-right_rel.shape[0] :],
        ],
        axis=-1,
    )
    return torch.from_numpy(action).float()


def integrate_dual_arm_ee_pose_deltas(
    initial_left_pose: torch.Tensor,
    initial_right_pose: torch.Tensor,
    deltas: torch.Tensor,
    *,
    pose_convention: str = "backward_framewise",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert 20D dual-arm SE(3) delta actions back to absolute 4x4 poses."""
    if deltas.shape[-1] != _DUAL_EE_POSE_DIM:
        raise ValueError(f"Expected ee_pose_delta dim {_DUAL_EE_POSE_DIM}, got {deltas.shape[-1]}")
    left_rel = deltas[..., :_EE_POSE_ACTION_DIM].detach().cpu().numpy()
    left_gripper = deltas[..., [_EE_POSE_ACTION_DIM]].detach().cpu().numpy()
    right_rel = deltas[..., 10:19].detach().cpu().numpy()
    right_gripper = deltas[..., [19]].detach().cpu().numpy()
    left_abs = pose_rel_to_abs(
        left_rel,
        initial_pose=initial_left_pose.detach().cpu().numpy(),
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )
    right_abs = pose_rel_to_abs(
        right_rel,
        initial_pose=initial_right_pose.detach().cpu().numpy(),
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )
    return torch.from_numpy(left_abs).float(), torch.from_numpy(right_abs).float()


def ee_pose_delta_actions_to_joint_targets(
    initial_qpos: torch.Tensor,
    deltas: torch.Tensor,
    *,
    pose_convention: str = "backward_framewise",
) -> torch.Tensor:
    """Convert denormalized ee_pose_delta actions to absolute 14D joint targets."""
    initial_qpos_np = initial_qpos.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)[:_JOINT_DIM]
    deltas_np = deltas.detach().cpu().numpy().astype(np.float32, copy=False)
    if deltas_np.ndim != 2 or deltas_np.shape[-1] != _DUAL_EE_POSE_DIM:
        raise ValueError(f"Expected deltas shape [T, {_DUAL_EE_POSE_DIM}], got {tuple(deltas_np.shape)}")

    initial_left, initial_right = _fk_dual_arm_poses_from_joints(initial_qpos_np)
    left_abs, right_abs = integrate_dual_arm_ee_pose_deltas(
        torch.from_numpy(initial_left),
        torch.from_numpy(initial_right),
        torch.from_numpy(deltas_np),
        pose_convention=pose_convention,
    )
    left_targets = left_abs[1:].detach().cpu().numpy()
    right_targets = right_abs[1:].detach().cpu().numpy()
    joint_targets = ik_dual_arm_from_poses(
        left_targets,
        right_targets,
        initial_qpos_np,
        deltas_np[:, [_EE_POSE_ACTION_DIM]],
        deltas_np[:, [19]],
    )
    return torch.from_numpy(joint_targets).float()
