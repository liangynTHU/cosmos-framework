# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compute RoboTwin LeRobot action normalization stats for Cosmos3 training.

Scans parquet episodes and aggregates action vectors into quantile / minmax /
meanstd statistics compatible with ``action_normalization.py``. Episode filters
match ``RobotTwinLeRobotDataset`` (task / episode subset / max episodes).

Example::

  python -m cosmos_framework.scripts.compute_robotwin_lerobot_action_stats \\
    --dataset-path /path/to/robotwin_lerobot \\
    --action-space joint_delta \\
    --output /path/to/run_dir/action_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from cosmos_framework.data.vfm.action.robotwin_action import (
    dual_arm_joints_to_ee_pose_delta_action,
    joint_trajectory_to_backward_anchored_delta,
    raw_action_dim_for_space,
    validate_action_space,
)
from cosmos_framework.data.vfm.action.robotwin_lerobot_episode_filters import (
    iter_filtered_episode_paths,
    resolve_task_index,
    load_tasks,
)

_DEFAULT_STATE_KEYS = (
    "observation.state",
    "observation.qpos",
    "observation.proprio",
    "observation.states.qpos",
    "observation.states.joint.position",
    "observation.state.joint_positions",
    "qpos",
    "proprio",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="RoboTwin LeRobot dataset root (contains meta/info.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cosmos_framework/data/vfm/action/datasets/stats/robotwin_lerobot_stats.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=14,
        help="Expected action dimension (RoboTwin dual-arm default: 14).",
    )
    parser.add_argument(
        "--include-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also aggregate absolute qpos rows into a separate ``state`` stats block (needed for use_state=True).",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default=None,
        help="Explicit state column name. When omitted, tries common qpos keys.",
    )
    parser.add_argument(
        "--action-space",
        type=str,
        default="joint_pos",
        choices=("joint_pos", "joint_delta", "ee_pose_delta"),
        help="Stats target action space.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on matched episodes (same as training dataset.max_episodes).",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Optional RoboTwin task_index filter.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Optional RoboTwin task name filter.",
    )
    parser.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help='Optional comma-separated episode indices, e.g. "0,1,2".',
    )
    parser.add_argument(
        "--quantile-low",
        type=float,
        default=0.01,
        help="Lower quantile for q01 (default 0.01).",
    )
    parser.add_argument(
        "--quantile-high",
        type=float,
        default=0.99,
        help="Upper quantile for q99 (default 0.99).",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=16,
        help="Action chunk length for joint_delta stats (must match training chunk_length).",
    )
    return parser.parse_args()


def _resolve_state_column(table, state_key: str | None) -> str | None:
    columns = set(table.column_names)
    if state_key is not None:
        return state_key if state_key in columns else None
    for key in _DEFAULT_STATE_KEYS:
        if key in columns:
            return key
    return None


def _stack_column(table, column: str, action_dim: int) -> np.ndarray:
    values = np.asarray(table[column].to_pylist(), dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[-1] != action_dim:
        raise ValueError(f"Column {column!r} has dim={values.shape[-1]}, expected {action_dim}.")
    return values.reshape(-1, action_dim)


def _filter_rows_by_task(table, states: np.ndarray, resolved_task_index: int | None) -> np.ndarray:
    if resolved_task_index is None or "task_index" not in table.column_names:
        return states
    task_index = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
    if task_index.shape[0] != states.shape[0]:
        raise ValueError("task_index column length does not match state rows.")
    return states[task_index == resolved_task_index]


def _compute_stats_block(
    values: np.ndarray,
    *,
    quantile_low: float,
    quantile_high: float,
) -> dict[str, Any]:
    if values.ndim != 2:
        raise ValueError(f"Expected values shape [N, D], got {values.shape}")
    return {
        "num_rows": int(values.shape[0]),
        "q01": np.quantile(values, quantile_low, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(values, quantile_high, axis=0).astype(np.float32).tolist(),
        "min": values.min(axis=0).astype(np.float32).tolist(),
        "max": values.max(axis=0).astype(np.float32).tolist(),
        "mean": values.mean(axis=0).astype(np.float32).tolist(),
        "std": values.std(axis=0).astype(np.float32).tolist(),
    }


def main() -> None:
    args = _parse_args()
    action_space = validate_action_space(args.action_space)
    action_dim = raw_action_dim_for_space(action_space) if args.action_dim == 14 else args.action_dim
    if action_space == "ee_pose_delta":
        action_dim = raw_action_dim_for_space(action_space)
    dataset_path = args.dataset_path.expanduser().resolve()
    if not (dataset_path / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing meta/info.json under {dataset_path}")

    tasks = load_tasks(dataset_path)
    resolved_task_index = resolve_task_index(tasks, args.task_index, args.task_name)
    episode_paths = iter_filtered_episode_paths(
        dataset_path,
        max_episodes=args.max_episodes,
        task_index=args.task_index,
        task_name=args.task_name,
        episode_indices=args.episode_indices,
    )
    if not episode_paths:
        raise ValueError(
            "No RoboTwin episodes matched the stats filters: "
            f"task_index={args.task_index}, task_name={args.task_name!r}, "
            f"episode_indices={args.episode_indices!r}, max_episodes={args.max_episodes}."
        )

    action_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    for data_path in episode_paths:
        table = pq.read_table(data_path, columns=None)
        if "action" not in table.column_names:
            raise KeyError(f"Episode parquet missing 'action' column: {data_path}")
        state_col = _resolve_state_column(table, args.state_key) or "observation.state"
        if state_col not in table.column_names:
            raise KeyError(f"Episode parquet missing state column {state_col!r}: {data_path}")
        states = _stack_column(table, state_col, 14)
        states = _filter_rows_by_task(table, states, resolved_task_index)
        if states.shape[0] == 0:
            continue
        if action_space == "joint_delta":
            if states.shape[0] < 2:
                continue
            action_horizon = max(1, int(args.action_horizon))
            for anchor_idx in range(states.shape[0] - 1):
                end_idx = min(states.shape[0], anchor_idx + 1 + action_horizon)
                if end_idx - anchor_idx < 2:
                    continue
                traj = torch.from_numpy(states[anchor_idx:end_idx].astype(np.float32))
                delta = joint_trajectory_to_backward_anchored_delta(traj).numpy()
                action_rows.append(delta)
            if args.include_state:
                state_rows.append(states)
            continue
        if action_space == "ee_pose_delta":
            if states.shape[0] < 2:
                continue
            ee_action = dual_arm_joints_to_ee_pose_delta_action(torch.from_numpy(states).float())
            action_rows.append(ee_action.numpy())
            continue
        action_rows.append(_stack_column(table, "action", action_dim))
        if args.include_state and action_space == "joint_pos":
            state_rows.append(states)

    if not action_rows:
        raise ValueError("No action rows collected for stats computation.")

    action_values = np.concatenate(action_rows, axis=0)
    stats: dict[str, Any] = {
        "action_dim": int(action_dim),
        "num_episodes": int(len(episode_paths)),
        "action_space": action_space,
        "filters": {
            "dataset_path": str(dataset_path),
            "task_index": args.task_index,
            "task_name": args.task_name,
            "episode_indices": args.episode_indices,
            "max_episodes": args.max_episodes,
            "include_state": args.include_state,
            "state_key": args.state_key,
            "action_horizon": args.action_horizon,
            "quantile_low": args.quantile_low,
            "quantile_high": args.quantile_high,
        },
        "actions": _compute_stats_block(
            action_values,
            quantile_low=args.quantile_low,
            quantile_high=args.quantile_high,
        ),
    }
    if state_rows:
        state_values = np.concatenate(state_rows, axis=0)
        stats["state"] = _compute_stats_block(
            state_values,
            quantile_low=args.quantile_low,
            quantile_high=args.quantile_high,
        )
    stats["num_rows"] = int(stats["actions"]["num_rows"] + stats.get("state", {}).get("num_rows", 0))

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")

    print(f"Wrote RoboTwin action stats to {output_path}")
    print(f"  episodes={stats['num_episodes']} rows={stats['num_rows']} dim={stats['action_dim']}")
    print(f"  filters={stats['filters']}")
    print(f"  actions.q01={stats['actions']['q01']}")
    print(f"  actions.q99={stats['actions']['q99']}")
    if "state" in stats:
        print(f"  state.q01={stats['state']['q01']}")
        print(f"  state.q99={stats['state']['q99']}")


if __name__ == "__main__":
    main()
