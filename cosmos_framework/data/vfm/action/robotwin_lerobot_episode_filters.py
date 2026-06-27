# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared RoboTwin LeRobot episode filtering for datasets and stats scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


def load_tasks(dataset_path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    with (dataset_path / "meta" / "tasks.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            tasks[int(item["task_index"])] = str(item["task"])
    return tasks


def resolve_task_index(
    tasks: dict[int, str],
    task_index: int | None,
    task_name: str | None,
) -> int | None:
    if task_name is None:
        return task_index
    matched = [idx for idx, name in tasks.items() if name == task_name]
    if not matched:
        examples = ", ".join(f"{idx}:{name}" for idx, name in sorted(tasks.items())[:10])
        raise ValueError(f"Unknown RoboTwin task_name {task_name!r}. Available examples: {examples}")
    resolved = matched[0]
    if task_index is not None and task_index != resolved:
        raise ValueError(
            f"RoboTwin task filter mismatch: task_index={task_index} but "
            f"task_name={task_name!r} resolves to task_index={resolved}."
        )
    return resolved


def parse_episode_indices(episode_indices: str | None) -> set[int] | None:
    if episode_indices is None:
        return None
    cleaned = episode_indices.strip().strip("[]")
    if not cleaned:
        return None
    values = {int(part.strip()) for part in cleaned.replace(" ", ",").split(",") if part.strip()}
    return values or None


def episode_has_task_rows(data_path: Path, resolved_task_index: int) -> bool:
    table = pq.read_table(data_path, columns=["task_index"])
    if table.num_rows <= 0:
        return False
    task_index = table.column("task_index").to_numpy()
    return bool((task_index == resolved_task_index).any())


def iter_filtered_episode_paths(
    dataset_path: Path,
    *,
    max_episodes: int | None = None,
    task_index: int | None = None,
    task_name: str | None = None,
    episode_indices: str | set[int] | None = None,
) -> list[Path]:
    """Return parquet paths for episodes matching the same filters as training."""

    dataset_path = dataset_path.expanduser().resolve()
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))
    total_episodes = int(info["total_episodes"])
    tasks = load_tasks(dataset_path)
    resolved_task_index = resolve_task_index(tasks, task_index, task_name)

    if isinstance(episode_indices, str):
        parsed_episode_indices = parse_episode_indices(episode_indices)
    elif episode_indices is None:
        parsed_episode_indices = None
    else:
        parsed_episode_indices = {int(index) for index in episode_indices}

    if parsed_episode_indices is not None:
        candidate_indices = sorted(index for index in parsed_episode_indices if 0 <= index < total_episodes)
    else:
        candidate_indices = list(range(total_episodes))

    paths: list[Path] = []
    for episode_index in candidate_indices:
        episode_chunk = episode_index // chunks_size
        data_path = dataset_path / info["data_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        if not data_path.exists():
            continue
        if resolved_task_index is not None and not episode_has_task_rows(data_path, resolved_task_index):
            continue
        paths.append(data_path)
        if max_episodes is not None and len(paths) >= max_episodes:
            break
    return paths
