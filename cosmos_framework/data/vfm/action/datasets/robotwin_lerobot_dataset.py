# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal RoboTwin LeRobot dataset for Cosmos Action SFT smoke tests."""

from __future__ import annotations

import json
import random
from bisect import bisect_right
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

_DEFAULT_CAMERA_KEY = "observation.images.cam_high"
_DEFAULT_WRIST_CAMERA_KEYS = ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist")
_DEFAULT_STATE_KEY_CANDIDATES = (
    "observation.state",
    "observation.qpos",
    "observation.proprio",
    "observation.states.qpos",
    "observation.states.joint.position",
    "observation.state.joint_positions",
    "qpos",
    "proprio",
)
_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the high camera. "
    "The bottom row contains two horizontally concatenated wrist camera views."
)
_MULTI_IMAGE_CAMERA_DESCRIPTION = (
    "The observation contains three camera images in fixed order: "
    "<cam_high>: image 1 is the external high camera; "
    "<cam_left_wrist>: image 2 is the left wrist camera; "
    "<cam_right_wrist>: image 3 is the right wrist camera."
)
_MODE_CHOICES = ("forward_dynamics", "policy")
_OBSERVATION_IMAGE_MODE_CHOICES = ("concat", "multi_image")


class RobotTwinLeRobotDataset(Dataset):
    """RoboTwin LeRobot v2.1 action dataset with concat or multi-image observations.

    This adapter targets datasets laid out as::

        meta/info.json
        meta/tasks.jsonl
        data/chunk-XXX/episode_XXXXXX.parquet
        videos/chunk-XXX/<camera_key>/episode_XXXXXX.mp4

    The first version intentionally keeps the action layout native to RoboTwin:
    14D dual-arm joint/gripper values. It returns samples in the same coarse
    format as the existing action dataset wrappers: ``video`` as uint8
    ``[C, T, H, W]``, ``action`` as float ``[chunk_length, 14]``, caption,
    fps, mode and domain id. When ``use_state=True``, the initial 14D
    ``observation.state`` qpos/proprio vector is prepended to the action sequence,
    yielding ``[chunk_length + 1, 14]`` with the first row treated as conditioning.
    """

    def __init__(
        self,
        root: str,
        camera_key: str = _DEFAULT_CAMERA_KEY,
        wrist_camera_keys: tuple[str, str] = _DEFAULT_WRIST_CAMERA_KEYS,
        viewpoint: str | None = None,
        observation_image_mode: str = "concat",
        fps: float | None = None,
        chunk_length: int = 16,
        mode: str = "forward_dynamics",
        tolerance_s: float = 2e-4,
        normalize_action: bool = False,
        wam_mode: str = "cosmos_default",
        future_video_frames: int | None = None,
        action_horizon: int | None = None,
        caption_prefix: str = "",
        max_episodes: int | None = None,
        task_index: int | None = None,
        task_name: str | None = None,
        dataset_repeat: int = 1,
        use_state: bool = False,
        state_key: str | None = None,
    ) -> None:
        super().__init__()
        allowed_modes = (*_MODE_CHOICES, "joint")
        if mode not in allowed_modes:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {allowed_modes}.")

        self._root = Path(root)
        self._camera_key = camera_key
        if len(wrist_camera_keys) != 2:
            raise ValueError(f"wrist_camera_keys must contain exactly two camera keys, got {wrist_camera_keys!r}.")
        if viewpoint not in (None, "concat_view"):
            raise ValueError(f"Unsupported viewpoint {viewpoint!r}; expected None or 'concat_view'.")
        if observation_image_mode not in _OBSERVATION_IMAGE_MODE_CHOICES:
            raise ValueError(
                f"Unsupported observation_image_mode {observation_image_mode!r}; "
                f"expected one of {_OBSERVATION_IMAGE_MODE_CHOICES}."
            )
        self._wrist_camera_keys = (wrist_camera_keys[0], wrist_camera_keys[1])
        self._viewpoint = viewpoint
        self._observation_image_mode = observation_image_mode
        self._wam_mode = wam_mode
        self._fastwam_action_only = wam_mode == "fastwam_action_only"
        self._action_horizon = int(action_horizon if action_horizon is not None else chunk_length)
        self._future_video_frames = int(future_video_frames if future_video_frames is not None else chunk_length)
        self._chunk_length = int(max(self._action_horizon, self._future_video_frames) if self._fastwam_action_only else chunk_length)
        self._mode = mode
        self._tolerance_s = float(tolerance_s)
        self._normalize_action = bool(normalize_action)
        self._caption_prefix = caption_prefix.strip()
        self._max_episodes = max_episodes
        self._task_index = int(task_index) if task_index is not None else None
        self._task_name = task_name.strip() if task_name else None
        self._dataset_repeat = max(1, int(dataset_repeat))
        self._use_state = bool(use_state)
        self._state_key = state_key.strip() if state_key else None
        self._domain_id = get_domain_id("robotwin_lerobot")

        self._info = json.loads((self._root / "meta" / "info.json").read_text())
        self._fps = float(fps if fps is not None else self._info.get("fps", 50.0))
        self._tasks = self._load_tasks()
        self._resolved_task_index = self._resolve_task_filter()
        self._episodes = self._load_episode_index()
        self._cumulative_sizes = self._build_cumulative_sizes()
        if not self._episodes:
            raise ValueError(
                "No RoboTwin episodes matched the dataset filters: "
                f"task_index={self._task_index}, task_name={self._task_name!r}, max_episodes={self._max_episodes}."
            )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in (*_MODE_CHOICES, "joint"):
            raise ValueError(f"Unsupported mode {value!r}; expected one of {*_MODE_CHOICES, 'joint'}.")
        self._mode = value

    @property
    def domain_id(self) -> int:
        return self._domain_id

    @property
    def action_dim(self) -> int:
        return 14

    @property
    def camera_key(self) -> str:
        return self._camera_key

    @property
    def wrist_camera_keys(self) -> tuple[str, str]:
        return self._wrist_camera_keys

    @property
    def viewpoint(self) -> str | None:
        return self._viewpoint

    @property
    def observation_image_mode(self) -> str:
        return self._observation_image_mode

    def __len__(self) -> int:
        base_len = self._cumulative_sizes[-1] if self._cumulative_sizes else 0
        return base_len * self._dataset_repeat

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode, start_frame = self._locate_index(int(idx))
        mode = self._choose_mode()
        table = pq.read_table(episode["data_path"])
        max_required_frames = self._chunk_length + 1
        max_available_frames = int(episode["num_frames"])
        raw_frame_indexes = list(range(start_frame, min(start_frame + max_required_frames, max_available_frames)))
        if len(raw_frame_indexes) < max_required_frames:
            if not self._fastwam_action_only:
                raise IndexError(f"Insufficient frames at idx={idx}: got {len(raw_frame_indexes)}, need {max_required_frames}")
            raw_frame_indexes.extend([max_available_frames - 1] * (max_required_frames - len(raw_frame_indexes)))
        rows = table.take(raw_frame_indexes).to_pylist()

        video_rows = rows[: self._future_video_frames + 1] if self._fastwam_action_only else rows
        video = self._load_video(episode, video_rows)
        action_valid_mask = torch.zeros(self._action_horizon, dtype=torch.bool)
        action_values = []
        action_start_row = 1 if self._use_state else 0
        for step in range(self._action_horizon):
            row_idx = min(action_start_row + step, len(rows) - 1)
            action_values.append(rows[row_idx]["action"])
            action_valid_mask[step] = start_frame + action_start_row + step < max_available_frames
        action = torch.tensor(action_values, dtype=torch.float32)
        if self._use_state:
            initial_state = self._extract_state(rows[0]).unsqueeze(0)
            action = torch.cat([initial_state, action], dim=0)
        if self._normalize_action:
            action = action.clamp(-1.0, 1.0)

        task = self._tasks[int(rows[0]["task_index"])]
        ai_caption = self._format_caption(task)
        sample_task_index = int(rows[0]["task_index"])

        if isinstance(video, list):
            formatted_video = [
                (item * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
                for item in video
            ]
        else:
            formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
        sample = {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "raw_action_dim": torch.tensor(self.action_dim, dtype=torch.long),
            "camera_key": self._camera_key,
            "episode_index": torch.tensor(int(episode["episode_index"]), dtype=torch.long),
            "task_index": torch.tensor(sample_task_index, dtype=torch.long),
            "frame_index": torch.tensor(start_frame, dtype=torch.long),
        }
        if self._fastwam_action_only:
            future_video_valid_mask = torch.zeros(self._future_video_frames, dtype=torch.bool)
            for step in range(self._future_video_frames):
                future_video_valid_mask[step] = start_frame + 1 + step < max_available_frames
            sample.update(
                wam_mode="fastwam_action_only",
                fastwam_action_only=True,
                instruction=ai_caption,
                obs_images=formatted_video[:, :1] if isinstance(formatted_video, torch.Tensor) else [v[:, :1] for v in formatted_video],
                future_images=(
                    formatted_video[:, 1 : 1 + self._future_video_frames]
                    if isinstance(formatted_video, torch.Tensor)
                    else [v[:, 1 : 1 + self._future_video_frames] for v in formatted_video]
                ),
                actions=action,
                action_valid_mask=action_valid_mask,
                future_video_valid_mask=future_video_valid_mask,
                episode_id=torch.tensor(int(episode["episode_index"]), dtype=torch.long),
                timestep=torch.tensor(start_frame, dtype=torch.long),
            )
        if self._observation_image_mode == "concat" or self._viewpoint == "concat_view":
            sample["viewpoint"] = "concat_view"
            sample["additional_view_description"] = _CONCAT_VIEW_DESCRIPTION
        if self._observation_image_mode == "multi_image":
            sample["ai_caption"] = f"{_MULTI_IMAGE_CAMERA_DESCRIPTION} {sample['ai_caption']}"
            sample["observation_image_mode"] = "multi_image"
        return sample

    def _load_tasks(self) -> dict[int, str]:
        tasks: dict[int, str] = {}
        with (self._root / "meta" / "tasks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                tasks[int(item["task_index"])] = str(item["task"])
        return tasks

    def _extract_state(self, row: dict[str, Any]) -> torch.Tensor:
        if self._state_key:
            candidate_keys = (self._state_key,)
        else:
            candidate_keys = _DEFAULT_STATE_KEY_CANDIDATES
        for key in candidate_keys:
            if key not in row or row[key] is None:
                continue
            state = np.asarray(row[key], dtype=np.float32).reshape(-1)
            if state.size != self.action_dim:
                raise ValueError(
                    f"RoboTwin state field {key!r} has dim={state.size}, expected {self.action_dim}. "
                    "Set dataset.state_key to a 14D qpos/proprio field or disable dataset.use_state."
                )
            return torch.from_numpy(state.copy()).float()
        available = ", ".join(sorted(str(key) for key in row.keys()))
        expected = ", ".join(candidate_keys)
        raise KeyError(
            "RoboTwin use_state=True requires a 14D qpos/proprio field. "
            f"Tried: {expected}. Available columns: {available}"
        )

    def _resolve_task_filter(self) -> int | None:
        if self._task_name is None:
            return self._task_index
        matched = [idx for idx, name in self._tasks.items() if name == self._task_name]
        if not matched:
            examples = ", ".join(f"{idx}:{name}" for idx, name in sorted(self._tasks.items())[:10])
            raise ValueError(f"Unknown RoboTwin task_name {self._task_name!r}. Available examples: {examples}")
        resolved = matched[0]
        if self._task_index is not None and self._task_index != resolved:
            raise ValueError(
                f"RoboTwin task filter mismatch: task_index={self._task_index} but "
                f"task_name={self._task_name!r} resolves to task_index={resolved}."
            )
        return resolved

    def _episode_task_index(self, data_path: Path) -> int:
        table = pq.ParquetFile(data_path).read_row_group(0, columns=["task_index"])
        if table.num_rows <= 0:
            raise ValueError(f"Empty RoboTwin episode parquet: {data_path}")
        return int(table.column("task_index")[0].as_py())

    def _valid_window_count(self, num_frames: int) -> int:
        if self._fastwam_action_only:
            return max(0, int(num_frames))
        return max(0, int(num_frames) - self._chunk_length)

    def _episode_valid_starts_for_task(self, data_path: Path, num_frames: int) -> np.ndarray:
        window_count = self._valid_window_count(num_frames)
        if window_count <= 0:
            return np.zeros(0, dtype=np.int64)
        assert self._resolved_task_index is not None
        resolved_task_index = int(self._resolved_task_index)
        table = pq.read_table(data_path, columns=["task_index"])
        task_index = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
        return np.flatnonzero(task_index[:window_count] == resolved_task_index).astype(np.int64)

    def _load_episode_index(self) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        chunks_size = int(self._info.get("chunks_size", 1000))
        total_episodes = int(self._info["total_episodes"])
        max_episodes = int(self._max_episodes) if self._max_episodes is not None else None
        for episode_index in range(total_episodes):
            episode_chunk = episode_index // chunks_size
            data_path = self._root / self._info["data_path"].format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
            num_frames = pq.read_metadata(data_path).num_rows
            valid_starts: np.ndarray | None = None
            if self._resolved_task_index is not None:
                valid_starts = self._episode_valid_starts_for_task(data_path, num_frames)
                if valid_starts.size == 0:
                    continue
                episode_task_index = int(self._resolved_task_index)
            else:
                if self._valid_window_count(num_frames) <= 0:
                    continue
                episode_task_index = self._episode_task_index(data_path)
            episodes.append(
                {
                    "episode_index": episode_index,
                    "episode_chunk": episode_chunk,
                    "data_path": data_path,
                    "num_frames": num_frames,
                    "task_index": episode_task_index,
                    "valid_starts": valid_starts,
                }
            )
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
        return episodes

    def _build_cumulative_sizes(self) -> list[int]:
        cumulative_sizes: list[int] = []
        total = 0
        for episode in self._episodes:
            valid_starts = episode.get("valid_starts")
            if valid_starts is not None:
                total += int(len(valid_starts))
            else:
                total += self._valid_window_count(int(episode["num_frames"]))
            cumulative_sizes.append(total)
        return cumulative_sizes

    def _locate_index(self, idx: int) -> tuple[dict[str, Any], int]:
        base_len = self._cumulative_sizes[-1] if self._cumulative_sizes else 0
        if idx < 0 or idx >= len(self) or base_len <= 0:
            raise IndexError(idx)
        idx = idx % base_len
        episode_idx = bisect_right(self._cumulative_sizes, idx)
        previous = 0 if episode_idx == 0 else self._cumulative_sizes[episode_idx - 1]
        episode = self._episodes[episode_idx]
        local_idx = idx - previous
        valid_starts = episode.get("valid_starts")
        if valid_starts is not None:
            return episode, int(valid_starts[local_idx])
        return episode, local_idx

    def _choose_mode(self) -> str:
        if self._mode == "joint":
            return random.choice(_MODE_CHOICES)
        return self._mode

    def _format_caption(self, task: str) -> str:
        if not self._caption_prefix:
            return task
        return f"{self._caption_prefix} {task}"

    def _load_video(self, episode: dict[str, Any], rows: list[dict[str, Any]]) -> torch.Tensor | list[torch.Tensor]:
        frame_indices = [int(row["frame_index"]) for row in rows]
        if self._observation_image_mode == "multi_image":
            return [
                self._decode_video_frames_pyav(self._video_path(episode, self._camera_key), frame_indices),
                self._decode_video_frames_pyav(self._video_path(episode, self._wrist_camera_keys[0]), frame_indices),
                self._decode_video_frames_pyav(self._video_path(episode, self._wrist_camera_keys[1]), frame_indices),
            ]
        if self._observation_image_mode != "concat" and self._viewpoint != "concat_view":
            return self._decode_video_frames_pyav(self._video_path(episode, self._camera_key), frame_indices)

        primary = self._decode_video_frames_pyav(self._video_path(episode, self._camera_key), frame_indices)
        left = self._decode_video_frames_pyav(self._video_path(episode, self._wrist_camera_keys[0]), frame_indices)
        right = self._decode_video_frames_pyav(self._video_path(episode, self._wrist_camera_keys[1]), frame_indices)
        return self._build_concat_view(primary, left, right)

    def _build_concat_view(self, primary: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        _, _, h, w = primary.shape
        half_h, half_w = h // 2, w // 2
        if half_h <= 0 or half_w <= 0:
            raise ValueError(f"Primary camera frames are too small for concat_view: H={h}, W={w}.")
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([primary, bottom], dim=-2)

    def _decode_video_frames_pyav(self, video_path: Path, frame_indices: list[int]) -> torch.Tensor:
        wanted = set(frame_indices)
        frames: dict[int, torch.Tensor] = {}
        with av.open(str(video_path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    array = frame.to_rgb().to_ndarray()
                    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float() / 255.0
                    frames[index] = tensor
                    if len(frames) == len(wanted):
                        break

        missing = [index for index in frame_indices if index not in frames]
        if missing:
            raise RuntimeError(f"Failed to decode frames {missing} from {video_path}")
        return torch.stack([frames[index] for index in frame_indices], dim=0)

    def _video_path(self, episode: dict[str, Any], camera_key: str) -> Path:
        rel = self._info["video_path"].format(
            episode_chunk=int(episode["episode_chunk"]),
            episode_index=int(episode["episode_index"]),
            video_key=camera_key,
        )
        return self._root / rel


class _TransformedRobotTwinLeRobotDataset(Dataset):
    """Apply the Action SFT transform pipeline lazily on RoboTwin samples."""

    def __init__(self, dataset: RobotTwinLeRobotDataset, transform: ActionTransformPipeline, resolution: str | None) -> None:
        self.dataset = dataset
        self.transform = transform
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]
        return self.transform(sample, self.resolution)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"dataset", "transform", "resolution"} or "dataset" not in self.__dict__:
            super().__setattr__(name, value)
        else:
            setattr(self.dataset, name, value)


def get_robotwin_lerobot_sft_dataset(
    root: str,
    camera_key: str = _DEFAULT_CAMERA_KEY,
    wrist_camera_keys: tuple[str, str] = _DEFAULT_WRIST_CAMERA_KEYS,
    viewpoint: str | None = None,
    observation_image_mode: str = "concat",
    fps: float | None = None,
    chunk_length: int = 16,
    mode: str = "forward_dynamics",
    tolerance_s: float = 2e-4,
    normalize_action: bool = False,
    wam_mode: str = "cosmos_default",
    future_video_frames: int | None = None,
    action_horizon: int | None = None,
    caption_prefix: str = "",
    max_episodes: int | None = None,
    task_index: int | None = None,
    task_name: str | None = None,
    dataset_repeat: int = 1,
    use_state: bool = False,
    state_key: str | None = None,
    resolution: str | None = "480",
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.0,
    max_action_dim: int = 64,
    action_channel_masking: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_viewpoint_info: bool = False,
    format_prompt_as_json: bool = False,
    video_temporal_downsample: int = 4,
    keep_aspect_ratio: bool = True,
) -> Dataset:
    """Create a RoboTwin dataset ready for Cosmos3 Action SFT training.

    The base dataset returns raw RoboTwin samples. This factory additionally
    applies ``ActionTransformPipeline`` so the training dataloader receives
    ``text_token_ids``, ``image_size``, ``sequence_plan`` and action tensors
    padded to ``max_action_dim``.
    """
    dataset = RobotTwinLeRobotDataset(
        root=root,
        camera_key=camera_key,
        wrist_camera_keys=wrist_camera_keys,
        viewpoint=viewpoint,
        observation_image_mode=observation_image_mode,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        tolerance_s=tolerance_s,
        normalize_action=normalize_action,
        wam_mode=wam_mode,
        future_video_frames=future_video_frames,
        action_horizon=action_horizon,
        caption_prefix=caption_prefix,
        max_episodes=max_episodes,
        task_index=task_index,
        task_name=task_name,
        dataset_repeat=dataset_repeat,
        use_state=use_state,
        state_key=state_key,
    )
    transform = ActionTransformPipeline(
        keep_aspect_ratio=keep_aspect_ratio,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        video_temporal_downsample=video_temporal_downsample,
        max_action_dim=max_action_dim,
        action_channel_masking=action_channel_masking,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        format_prompt_as_json=format_prompt_as_json,
        add_camera_tokens=observation_image_mode == "multi_image",
    )
    return _TransformedRobotTwinLeRobotDataset(dataset, transform, resolution)
