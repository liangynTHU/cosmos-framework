#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import argparse
import importlib
import time
from typing import Any

import torch


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _load_model(args: argparse.Namespace) -> Any:
    try:
        from cosmos_framework.inference.args import OmniSetupOverrides
    except Exception as exc:  # pragma: no cover - depends on runtime install
        raise RuntimeError("Failed to import Cosmos3 inference helpers. Run from cosmos-framework environment.") from exc

    overrides = OmniSetupOverrides.model_validate(
        {
            "checkpoint_path": args.checkpoint,
            "config_file": args.config,
            "output_dir": args.output_dir,
            "sampler": args.sampler,
            "experiment_overrides": [
                f"model.config.wam_mode={args.wam_mode}",
                "model.config.fastwam_action_only.enabled=true",
                "model.config.fastwam_action_only.disable_video_decode_at_inference=true",
            ],
        }
    )
    setup_args = overrides.build_setup()
    inference = setup_args.get_inference_cls().create(setup_args)
    model = inference.model
    if not hasattr(model, "predict_action_fastwam"):
        raise AttributeError("Loaded model does not provide predict_action_fastwam(...).")
    return model


def _make_env(args: argparse.Namespace) -> Any | None:
    if args.env_factory is None:
        return None
    module_name, fn_name = args.env_factory.split(":", 1)
    factory = getattr(importlib.import_module(module_name), fn_name)
    return factory(args)


def _extract_obs_and_instruction(env_obs: Any) -> tuple[torch.Tensor, str]:
    if isinstance(env_obs, dict):
        image = env_obs.get("image") or env_obs.get("obs_image") or env_obs.get("rgb")
        instruction = env_obs.get("instruction") or env_obs.get("prompt") or env_obs.get("task") or ""
    else:
        image = env_obs
        instruction = ""
    if image is None:
        raise ValueError("RobotWin observation must contain image/obs_image/rgb or be an image tensor/array.")
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        image = image[..., :3].permute(2, 0, 1).contiguous()
    return image, str(instruction)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.wam_mode != "fastwam_action_only":
        raise ValueError("This evaluator is for --wam_mode fastwam_action_only only.")
    if not args.output_action_only:
        raise ValueError("FastWAM evaluation requires --output_action_only true.")
    if args.save_generated_video:
        raise ValueError("FastWAM action-only evaluation must not save model-generated video.")

    model = _load_model(args)
    env = _make_env(args)

    if env is None:
        dummy = torch.zeros(3, args.image_height, args.image_width, dtype=torch.uint8)
        start = time.perf_counter()
        out = model.predict_action_fastwam(
            dummy,
            args.instruction,
            action_horizon=args.action_horizon,
            action_denoising_steps=args.action_denoising_steps,
            return_debug=True,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        actions = out["actions"]
        return {
            "success_rate": None,
            "average_episode_length": None,
            "inference_latency_ms": latency_ms,
            "action_chunk_shape": tuple(actions.shape),
            "denoising_steps": args.action_denoising_steps,
            "gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "used_video_decode": out.get("debug", {}).get("used_video_decode", False),
            "num_future_video_tokens": out.get("debug", {}).get("num_future_video_tokens", 0),
        }

    successes: list[float] = []
    lengths: list[int] = []
    latencies: list[float] = []
    last_shape: tuple[int, ...] | None = None
    for episode_idx in range(args.num_episodes):
        obs = env.reset()
        done = False
        ep_len = 0
        while not done and ep_len < args.max_episode_steps:
            obs_image, instruction = _extract_obs_and_instruction(obs)
            start = time.perf_counter()
            out = model.predict_action_fastwam(
                obs_image,
                instruction,
                action_horizon=args.action_horizon,
                action_denoising_steps=args.action_denoising_steps,
                return_debug=True,
            )
            latencies.append((time.perf_counter() - start) * 1000.0)
            actions = out["actions"]
            last_shape = tuple(actions.shape)
            execute_horizon = args.execute_horizon or 1
            for action in actions[:execute_horizon]:
                obs, reward, done, info = env.step(action.detach().cpu().numpy())
                ep_len += 1
                if done or ep_len >= args.max_episode_steps:
                    break
        successes.append(float(info.get("success", reward > 0) if isinstance(info, dict) else reward > 0))
        lengths.append(ep_len)
        print(f"episode={episode_idx} success={successes[-1]} length={ep_len}")

    return {
        "success_rate": sum(successes) / max(len(successes), 1),
        "average_episode_length": sum(lengths) / max(len(lengths), 1),
        "inference_latency_ms": sum(latencies) / max(len(latencies), 1),
        "action_chunk_shape": last_shape,
        "denoising_steps": args.action_denoising_steps,
        "gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "used_video_decode": False,
        "num_future_video_tokens": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RobotWin FastWAM-style Cosmos3 action-only evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wam_mode", default="fastwam_action_only")
    parser.add_argument("--output_action_only", type=_str2bool, default=True)
    parser.add_argument("--save_generated_video", type=_str2bool, default=False)
    parser.add_argument("--env_factory", default=None, help="Optional module:function returning a RobotWin env")
    parser.add_argument("--output_dir", default="/tmp/cosmos3_fastwam_robotwin_eval")
    parser.add_argument("--sampler", default="unipc", choices=["unipc", "edm"])
    parser.add_argument("--instruction", default="")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--execute_horizon", type=int, default=None)
    parser.add_argument("--action_denoising_steps", type=int, default=None)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--image_width", type=int, default=640)
    return parser.parse_args()


if __name__ == "__main__":
    metrics = evaluate(parse_args())
    for key, value in metrics.items():
        print(f"{key}: {value}")
