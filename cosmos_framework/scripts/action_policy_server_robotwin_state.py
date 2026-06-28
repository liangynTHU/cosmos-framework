# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboTwin Action Policy Server with state (qpos) conditioning and action denormalization.

Extends ``action_policy_server_libero.ActionModelService`` for RoboTwin eval:

- When the client sends ``qpos``, prepend it as action row 0 (``use_state=True`` training).
- Optionally normalize qpos / denormalize predicted actions via ``--action-stats-path``.
  Omit ``--action-stats-path`` to disable action normalization (raw absolute values).

Example::

  PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_robotwin_state \\
    --checkpoint-path /path/to/checkpoints/iter_000020000 \\
    --config-file /path/to/config.yaml \\
    --port 8000 --host 0.0.0.0 \\
    --raw-action-dim 14 --action-chunk-size 16 --fps 50 \\
    --action-stats-path /path/to/run/action_stats.json \\
    --action-normalization quantile
"""

from __future__ import annotations

from cosmos_framework.inference.common.init import init_script, is_rank0  # noqa: F401

init_script()

import base64
import io
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
import tyro

from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.robotwin_action import (
    dual_arm_joints_to_absolute_ee_pose_action,
    ee_pose_delta_actions_to_joint_targets,
    integrate_joint_deltas,
    validate_action_space,
)
from cosmos_framework.data.vfm.action.transforms import (
    build_sequence_plan_from_mode,
    find_closest_target_size,
    reflection_pad_to_target,
    remove_reflection_padding,
)
from cosmos_framework.inference.common.args import tyro_cli
from cosmos_framework.scripts.action_policy_server_libero import (
    ActionModelService,
    ActionNormalization,
    ActionServerArgs,
    _ActionHandler,
    _augment_prompt_with_metadata,
    _decode_request_video_chw_uint8,
    _request_dump_target,
    _save_policy_request_dump,
    _video_tensor_to_pil_images,
)
from cosmos_framework.scripts.action_policy_server_utils import get_local_ip
from cosmos_framework.utils import log
from cosmos_framework.utils.vfm.data_utils import get_vision_data_resolution


class RoboTwinStateServerArgs(ActionServerArgs):
    """RoboTwin server args with state conditioning defaults."""

    use_state: bool = True
    """Prepend client ``qpos`` as the first action row when provided."""

    action_space: str = "joint_delta"
    """RoboTwin action space: joint_pos, joint_delta, or ee_pose_delta."""

    action_stats_path: Path | None = None
    """Action stats JSON for denormalization. Omit to disable action normalization."""

    action_normalization: ActionNormalization = "quantile"
    """Action normalization method. Must match training (default: quantile)."""

class RoboTwinStateActionService(ActionModelService):
    """RoboTwin HTTP service with qpos prepend and action denormalization."""

    def __init__(self, args: RoboTwinStateServerArgs) -> None:
        self._robotwin_action_space = validate_action_space(args.action_space)
        super().__init__(args)
        self.use_state = bool(args.use_state)
        stats_path = self.cfg.action_stats_path
        log.info(
            f"[robotwin-state-server] use_state={self.use_state} "
            f"action_space={self._robotwin_action_space} "
            f"raw_action_dim={self.raw_action_dim} "
            f"action_chunk_size={self.cfg.action_chunk_size} "
            f"max_action_dim={self.cfg.max_action_dim} "
            f"action_normalization={self.action_normalization} "
            f"action_stats_path={stats_path if stats_path is not None else 'disabled'}"
        )

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["server"] = "action_policy_server_robotwin_state"
        info["use_state"] = self.use_state
        info["action_space"] = self._robotwin_action_space
        info["action_normalization"] = self.action_normalization
        return info

    def predict_policy(self, req: dict[str, Any]) -> dict[str, Any]:
        t0 = time.monotonic()

        injected_id = req.get("request_id", None)
        if isinstance(injected_id, int) and injected_id > 0:
            request_id = int(injected_id)
        else:
            with self._req_id_lock:
                self._req_id += 1
                request_id = int(self._req_id)

        images_b64 = req.get("images")
        image_b64 = req.get("image")
        has_images = isinstance(images_b64, list) and len(images_b64) > 0
        if not has_images and not isinstance(image_b64, str):
            raise ValueError("request must include 'image' or a non-empty 'images' list")

        prompt = req.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")

        domain_name = req.get("domain_name")
        if not isinstance(domain_name, str):
            raise ValueError("'domain_name' must be a string")

        image_size = req.get("image_size")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("'image_size' must be a positive integer")

        t_decode0 = time.monotonic()
        t_frames = self.cfg.action_chunk_size + 1
        video_c_t_h_w_uint8 = _decode_request_video_chw_uint8(
            req,
            image_size=image_size,
            target_frames=t_frames,
        )
        img_chw_uint8 = video_c_t_h_w_uint8[:, -1]
        _, final_h, final_w = img_chw_uint8.shape
        t_decode1 = time.monotonic()

        resolution = get_vision_data_resolution((final_h, final_w))
        target_w, target_h = find_closest_target_size(final_h, final_w, resolution)
        pad_dict: dict[str, Any] = {"video": video_c_t_h_w_uint8}
        reflection_pad_to_target(pad_dict, ["video"], True, target_w, target_h)
        video_padded = pad_dict["video"]
        padded_image_size = pad_dict["image_size"]

        qpos_list = req.get("qpos")
        has_qpos = self.use_state and isinstance(qpos_list, list) and len(qpos_list) > 0
        raw_qpos_tensor: torch.Tensor | None = None

        if has_qpos:
            raw_qpos_tensor = torch.tensor(qpos_list, dtype=torch.float32).reshape(-1)
            if self._robotwin_action_space == "ee_pose_delta":
                if int(raw_qpos_tensor.shape[0]) != 14:
                    raise ValueError(
                        f"qpos must have dim=14 for ee_pose_delta FK conditioning, got {int(raw_qpos_tensor.shape[0])}"
                    )
            elif int(raw_qpos_tensor.shape[0]) != self.raw_action_dim:
                raise ValueError(
                    f"qpos must have dim={self.raw_action_dim}, got {int(raw_qpos_tensor.shape[0])}"
                )
            if self._robotwin_action_space == "ee_pose_delta":
                qpos_tensor = self._normalize_action_input(
                    dual_arm_joints_to_absolute_ee_pose_action(raw_qpos_tensor[:14]).reshape(-1)
                )
            else:
                qpos_tensor = self._normalize_state_input(raw_qpos_tensor)
            if qpos_tensor.shape[0] < self.cfg.max_action_dim:
                qpos_tensor = torch.nn.functional.pad(
                    qpos_tensor, (0, self.cfg.max_action_dim - qpos_tensor.shape[0])
                )
            else:
                qpos_tensor = qpos_tensor[: self.cfg.max_action_dim]
            state_row = qpos_tensor.unsqueeze(0)
            noise_rows = torch.zeros(
                (self.cfg.action_chunk_size, self.cfg.max_action_dim), dtype=torch.float32
            )
            action_t_d = torch.cat([state_row, noise_rows], dim=0)
            action_length = self.cfg.action_chunk_size + 1
        else:
            action_t_d = torch.zeros(
                (self.cfg.action_chunk_size, self.cfg.max_action_dim),
                dtype=torch.float32,
            )
            action_length = self.cfg.action_chunk_size

        input_video_key = getattr(self.model, "input_video_key", None)
        if input_video_key is None:
            input_video_key = getattr(self.model, "config", None).input_video_key

        sequence_plan = build_sequence_plan_from_mode(
            mode="policy",
            video_length=self.cfg.action_chunk_size + 1,
            action_length=action_length,
            has_text=True,
        )

        augmented_prompt = _augment_prompt_with_metadata(
            prompt,
            t_frames=t_frames,
            fps=self.cfg.fps,
            height=final_h,
            width=final_w,
            append_duration_fps=self.append_duration_fps,
            append_resolution_info=self.append_resolution_info,
        )

        batch: dict[str, Any] = {
            input_video_key: [[video_padded]],
            **self._make_action_processing_batch_fields(),
            "action": [[action_t_d]],
            "mode": ["policy"],
            "ai_caption": [augmented_prompt],
            "prompt": [augmented_prompt],
            "conditioning_fps": [torch.tensor(self.cfg.fps, dtype=torch.long)],
            "image_size": padded_image_size.unsqueeze(0).to(device="cuda"),
            "domain_id": [torch.tensor(get_domain_id(domain_name), dtype=torch.long)],
            "sequence_plan": [sequence_plan],
        }

        log.info(
            f"[robotwin-state-server] request_id={request_id} mode=policy "
            f"prompt={augmented_prompt!r} domain={domain_name!r} image_size={image_size} "
            f"video={tuple(video_c_t_h_w_uint8.shape)} action={tuple(action_t_d.shape)} "
            f"use_state={has_qpos} steps={self.cfg.num_steps} guidance={self.cfg.guidance}"
        )

        t_inf0 = time.monotonic()
        with self._lock:
            with torch.inference_mode():
                samples = self.model.generate_samples_from_batch(
                    batch,
                    guidance=self.cfg.guidance,
                    seed=[self.cfg.seed],
                    num_steps=self.cfg.num_steps,
                    has_negative_prompt=False,
                )
                pred_action = samples["action"][0]
                pred_video_c_t_h_w = self.model.decode(samples["vision"][0]).squeeze(0)
                pred_video_c_t_h_w = remove_reflection_padding(pred_video_c_t_h_w, padded_image_size)
        t_inf1 = time.monotonic()

        pred_action = pred_action.float().squeeze(0)
        if has_qpos and pred_action.shape[0] > self.cfg.action_chunk_size:
            pred_action = pred_action[1:]
        pred_action = self._denormalize_action(pred_action)
        if self._robotwin_action_space == "joint_delta":
            if not has_qpos:
                raise ValueError("joint_delta inference requires client qpos for delta integration")
            # Anchor-relative offsets -> absolute targets (pi05 / OpenPI style).
            pred_action = integrate_joint_deltas(
                raw_qpos_tensor[: self.raw_action_dim].detach().cpu(),
                pred_action[..., :14].detach().cpu(),
            )
        elif self._robotwin_action_space == "ee_pose_delta":
            if not has_qpos:
                raise ValueError("ee_pose_delta inference requires client qpos for FK conditioning and IK")
            pred_action = ee_pose_delta_actions_to_joint_targets(
                raw_qpos_tensor[:14],
                pred_action[..., : self.raw_action_dim],
            )
        pred_action_list = pred_action.detach().cpu().numpy().tolist()

        pred_video_frames = _video_tensor_to_pil_images(pred_video_c_t_h_w)
        pred_video_b64: list[str] = []
        for frame in pred_video_frames:
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            pred_video_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        dump_target = _request_dump_target(req, self.cfg.dump_dir, request_id)
        if dump_target is not None and (
            isinstance(req.get("dump_episode_dir"), str) or self._should_dump(request_id)
        ):
            dump_root, dump_name = dump_target
            dump_root.mkdir(parents=True, exist_ok=True)
            try:
                log.info(f"[robotwin-state-server] request_id={request_id} dumping to {str(dump_root)}")
                _save_policy_request_dump(
                    dump_root=dump_root,
                    request_id=request_id,
                    request_json=req,
                    obs_chw_uint8=img_chw_uint8,
                    pred_action=pred_action_list,
                    pred_video_c_t_h_w=pred_video_c_t_h_w,
                    fps=int(self.cfg.fps),
                    dump_name=dump_name,
                )
            except Exception as e:
                log.error(f"[robotwin-state-server] dump failed for request_id={request_id}: {e}")

        dt_total_ms = (time.monotonic() - t0) * 1000.0
        dt_decode_ms = (t_decode1 - t_decode0) * 1000.0
        dt_inf_ms = (t_inf1 - t_inf0) * 1000.0
        log.info(
            f"[robotwin-state-server] request_id={request_id} done "
            f"action_steps={len(pred_action_list)} video_frames={len(pred_video_b64)} "
            f"ms_total={dt_total_ms:.1f} ms_decode={dt_decode_ms:.1f} ms_infer={dt_inf_ms:.1f}"
        )
        return {"action": pred_action_list, "video": pred_video_b64}


def serve(args: RoboTwinStateServerArgs) -> None:
    if args.dump_dir is not None:
        dump_root = Path(args.dump_dir).resolve()
        dump_root.mkdir(parents=True, exist_ok=True)
        log.info(f"[robotwin-state-server] dump_root={str(dump_root)} dump_every={args.dump_every}")

    service = RoboTwinStateActionService(args)

    local_ip = get_local_ip()
    log.info(
        f"[robotwin-state-server] http://{local_ip}:{int(args.port)}/ "
        f"use_state={service.use_state} domain=robotwin_lerobot"
    )
    log.info("[robotwin-state-server] Endpoints:")
    log.info("  - GET  /       : Health check")
    log.info("  - GET  /info   : Model info")
    log.info("  - POST /predict: Policy inference (image + prompt + qpos -> action)")

    httpd: ThreadingHTTPServer = ThreadingHTTPServer((args.host, int(args.port)), _ActionHandler)
    setattr(httpd, "service", service)
    httpd.serve_forever()


def main() -> None:
    args = tyro_cli(
        RoboTwinStateServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    serve(args)


if __name__ == "__main__":
    main()
