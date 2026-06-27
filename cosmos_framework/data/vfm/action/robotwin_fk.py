# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Lightweight Aloha-Agilex forward kinematics for RoboTwin action encoding."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

_LEFT_ARM_JOINT_NAMES = (
    "fl_joint1",
    "fl_joint2",
    "fl_joint3",
    "fl_joint4",
    "fl_joint5",
    "fl_joint6",
)
_RIGHT_ARM_JOINT_NAMES = (
    "fr_joint1",
    "fr_joint2",
    "fr_joint3",
    "fr_joint4",
    "fr_joint5",
    "fr_joint6",
)
_LEFT_EE_LINK = "fl_link6"
_RIGHT_EE_LINK = "fr_link6"
_ROOT_LINK = "footprint"

# Matches aloha-agilex config.yml ``global_trans_matrix``.
_ALOHA_AGILEX_TO_OPENCV = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float32,
)

_DEFAULT_URDF_CANDIDATES = (
    Path(__file__).resolve().parents[4].parent
    / "third_party/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf",
    Path(
        "/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/assert/"
        "embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
    ),
)


def resolve_robotwin_aloha_urdf_path() -> Path:
    """Return the Aloha-Agilex URDF used for RoboTwin FK."""

    env_path = os.environ.get("ROBOTWIN_ALOHA_URDF")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"ROBOTWIN_ALOHA_URDF does not exist: {path}")

    for candidate in _DEFAULT_URDF_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate arx5_description_isaac.urdf. Set ROBOTWIN_ALOHA_URDF to the URDF path."
    )


def _parse_xyz_rpy(origin_element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if origin_element is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = np.asarray([float(v) for v in origin_element.attrib.get("xyz", "0 0 0").split()], dtype=np.float64)
    rpy = np.asarray([float(v) for v in origin_element.attrib.get("rpy", "0 0 0").split()], dtype=np.float64)
    return xyz, rpy


def _origin_to_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler("xyz", rpy, degrees=False).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    return R.from_rotvec(axis / norm * angle).as_matrix()


@dataclass(frozen=True)
class _UrdfJoint:
    name: str
    parent_link: str
    child_link: str
    joint_type: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


class _AlohaAgilexFk:
    """URDF chain FK for the dual-arm Aloha-Agilex model."""

    def __init__(self, urdf_path: Path) -> None:
        root = ET.parse(urdf_path).getroot()
        self._joint_by_child: dict[str, _UrdfJoint] = {}
        for joint_element in root.findall("joint"):
            parent = joint_element.find("parent")
            child = joint_element.find("child")
            if parent is None or child is None:
                continue
            origin_xyz, origin_rpy = _parse_xyz_rpy(joint_element.find("origin"))
            axis_element = joint_element.find("axis")
            axis = np.asarray(
                [float(v) for v in (axis_element.attrib.get("xyz", "0 0 1") if axis_element is not None else "0 0 1").split()],
                dtype=np.float64,
            )
            self._joint_by_child[child.attrib["link"]] = _UrdfJoint(
                name=joint_element.attrib["name"],
                parent_link=parent.attrib["link"],
                child_link=child.attrib["link"],
                joint_type=joint_element.attrib.get("type", "fixed"),
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
            )

    def _joint_transform(self, joint: _UrdfJoint, joint_values: dict[str, float]) -> np.ndarray:
        transform = _origin_to_matrix(joint.origin_xyz, joint.origin_rpy)
        if joint.joint_type in {"revolute", "continuous"}:
            angle = float(joint_values.get(joint.name, 0.0))
            rotation = np.eye(4, dtype=np.float64)
            rotation[:3, :3] = _axis_angle_matrix(joint.axis, angle)
            transform = transform @ rotation
        elif joint.joint_type == "prismatic":
            displacement = float(joint_values.get(joint.name, 0.0))
            translation = np.eye(4, dtype=np.float64)
            axis = joint.axis / max(np.linalg.norm(joint.axis), 1e-12)
            translation[:3, 3] = axis * displacement
            transform = transform @ translation
        return transform.astype(np.float32)

    def link_pose(self, link_name: str, joint_values: dict[str, float]) -> np.ndarray:
        if link_name == _ROOT_LINK:
            return np.eye(4, dtype=np.float32)
        joint = self._joint_by_child.get(link_name)
        if joint is None:
            raise KeyError(f"Unknown URDF link {link_name!r}")
        parent_pose = self.link_pose(joint.parent_link, joint_values)
        return (parent_pose @ self._joint_transform(joint, joint_values)).astype(np.float32)

    def arm_pose(
        self,
        arm_joint_angles: np.ndarray,
        *,
        arm: str,
        apply_opencv: bool = True,
    ) -> np.ndarray:
        if arm == "left":
            joint_names = _LEFT_ARM_JOINT_NAMES
            ee_link = _LEFT_EE_LINK
        elif arm == "right":
            joint_names = _RIGHT_ARM_JOINT_NAMES
            ee_link = _RIGHT_EE_LINK
        else:
            raise ValueError(f"Unsupported arm {arm!r}")

        joint_values = {name: float(angle) for name, angle in zip(joint_names, arm_joint_angles.reshape(-1)[:6])}
        pose = self.link_pose(ee_link, joint_values)
        if apply_opencv:
            pose = pose.copy()
            pose[:3, :3] = pose[:3, :3] @ _ALOHA_AGILEX_TO_OPENCV
        return pose.astype(np.float32)

    def dual_arm_poses(
        self,
        joints: np.ndarray,
        *,
        apply_opencv: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(joints, dtype=np.float32)
        if joints.ndim == 1:
            left = self.arm_pose(joints[0:6], arm="left", apply_opencv=apply_opencv)
            right = self.arm_pose(joints[7:13], arm="right", apply_opencv=apply_opencv)
            return left, right

        left_poses = np.zeros((joints.shape[0], 4, 4), dtype=np.float32)
        right_poses = np.zeros((joints.shape[0], 4, 4), dtype=np.float32)
        for index in range(joints.shape[0]):
            left_poses[index], right_poses[index] = self.dual_arm_poses(joints[index], apply_opencv=apply_opencv)
        return left_poses, right_poses


@lru_cache(maxsize=1)
def _get_fk_engine() -> _AlohaAgilexFk:
    return _AlohaAgilexFk(resolve_robotwin_aloha_urdf_path())


def fk_dual_arm_ee_poses(joints: np.ndarray, *, apply_opencv: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute OpenCV-aligned dual-arm ee poses from 14D joint vectors."""

    return _get_fk_engine().dual_arm_poses(joints, apply_opencv=apply_opencv)


def _pose_error(target_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    position_error = target_pose[:3, 3] - current_pose[:3, 3]
    rotation_error = (
        R.from_matrix(target_pose[:3, :3] @ current_pose[:3, :3].T).as_rotvec().astype(np.float32)
    )
    return np.concatenate([position_error, rotation_error], axis=0)


def _numerical_arm_jacobian(
    engine: _AlohaAgilexFk,
    arm_joint_angles: np.ndarray,
    *,
    arm: str,
    eps: float = 1e-4,
) -> np.ndarray:
    base_pose = engine.arm_pose(arm_joint_angles, arm=arm, apply_opencv=True)
    jacobian = np.zeros((6, 6), dtype=np.float64)
    for joint_index in range(6):
        perturbed = arm_joint_angles.copy()
        perturbed[joint_index] += eps
        perturbed_pose = engine.arm_pose(perturbed, arm=arm, apply_opencv=True)
        jacobian[:, joint_index] = _pose_error(perturbed_pose, base_pose) / eps
    return jacobian


def ik_single_arm(
    target_pose: np.ndarray,
    initial_joint_angles: np.ndarray,
    *,
    arm: str,
    max_iterations: int = 40,
    damping: float = 0.05,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Damped least-squares IK for one 6-DoF Aloha arm."""

    engine = _get_fk_engine()
    joint_angles = np.asarray(initial_joint_angles, dtype=np.float64).reshape(6).copy()
    target_pose = np.asarray(target_pose, dtype=np.float32)
    for _ in range(max_iterations):
        current_pose = engine.arm_pose(joint_angles, arm=arm, apply_opencv=True)
        error = _pose_error(target_pose, current_pose)
        if float(np.linalg.norm(error)) < tolerance:
            break
        jacobian = _numerical_arm_jacobian(engine, joint_angles, arm=arm)
        lhs = jacobian @ jacobian.T + (damping**2) * np.eye(6, dtype=np.float64)
        delta = jacobian.T @ np.linalg.solve(lhs, error)
        joint_angles += delta
    return joint_angles.astype(np.float32)


def ik_dual_arm_from_poses(
    left_poses: np.ndarray,
    right_poses: np.ndarray,
    initial_joints: np.ndarray,
    left_grippers: np.ndarray,
    right_grippers: np.ndarray,
) -> np.ndarray:
    """Convert batched ee poses and absolute grippers back to 14D joint targets."""

    initial_joints = np.asarray(initial_joints, dtype=np.float32).reshape(-1)[:14]
    left_poses = np.asarray(left_poses, dtype=np.float32)
    right_poses = np.asarray(right_poses, dtype=np.float32)
    left_grippers = np.asarray(left_grippers, dtype=np.float32).reshape(-1)
    right_grippers = np.asarray(right_grippers, dtype=np.float32).reshape(-1)
    if left_poses.ndim != 3 or right_poses.ndim != 3:
        raise ValueError("Expected batched poses with shape [T, 4, 4].")

    outputs: list[np.ndarray] = []
    left_seed = initial_joints[0:6].copy()
    right_seed = initial_joints[7:13].copy()
    for step in range(left_poses.shape[0]):
        left_seed = ik_single_arm(left_poses[step], left_seed, arm="left")
        right_seed = ik_single_arm(right_poses[step], right_seed, arm="right")
        outputs.append(
            np.concatenate(
                [
                    left_seed,
                    np.asarray([left_grippers[step]], dtype=np.float32),
                    right_seed,
                    np.asarray([right_grippers[step]], dtype=np.float32),
                ],
                axis=0,
            )
        )
    return np.stack(outputs, axis=0).astype(np.float32)
