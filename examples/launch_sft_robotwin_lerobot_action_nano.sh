#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -uo pipefail

TOML_FILE="examples/toml/sft_config/robotwin_lerobot_action_sft_nano.toml"
: "${DATASET_PATH:=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0}"
: "${BASE_CHECKPOINT_PATH:=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models/wan22_vae/Wan2.2_VAE.pth}"
: "${NPROC_PER_NODE:=1}"
: "${MASTER_PORT:=50021}"
: "${LOG_FILENAME:=robotwin_lerobot_action_sft_nano.log}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/meta/info.json" ]] || { echo "ERROR: missing $DATASET_PATH/meta/info.json" >&2; exit 1; }'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
