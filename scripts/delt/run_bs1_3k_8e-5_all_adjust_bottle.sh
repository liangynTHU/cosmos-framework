#!/usr/bin/env bash
# =============================================================================
# adjust_bottle 单任务全量训练 (episode 0-549, 共 550 条)
# 参考: /mnt/lyn/workspace/wam/scripts/xxr_debug/run_bs32_3k_1e-4_all_adjust_bottle_xxr_debug.sh
#
# - 数据: 全部 adjust_bottle full-trajectory 窗口 (ROBOTWIN_EPISODE_INDICES=0..549)
# - 模型: Cosmos3-Nano, joint_delta + quantile norm + use_state
# - 分片: shard_map_style_dataset=true, 64 卡各看 1/64 窗口
# - global batch = 8 nodes x 8 GPU x 32 = 2048
#
# 用法 (在 cosmos-framework 根目录):
#   bash scripts/delt/run_bs32_3k_1e-4_all_adjust_bottle.sh
# =============================================================================
set -euo pipefail

# ---- 集群 ----
# EXP_IP_LIST=29.119.84.104,29.127.48.51,29.127.50.141,29.232.241.61,29.119.96.20,29.232.224.85,29.191.192.197,29.119.83.12
EXP_IP_LIST=29.191.210.123,29.127.64.8,29.127.65.89,29.119.84.187,29.127.82.110,29.191.209.108,29.119.97.119,29.119.99.50
NNODES=8
# MASTER_ADDR=29.119.84.104
MASTER_ADDR=29.191.210.123
MASTER_PORT=50221
NPROC_PER_NODE=8
COSMOS_WORKDIR=/mnt/lyn/ybw/workspace/cosmos-framework

# ---- 数据 (全部 adjust_bottle: episode 0-549) ----
ROBOTWIN_MODE=policy
OBSERVATION_IMAGE_MODE=concat
ROBOTWIN_VIEWPOINT=concat_view
ROBOTWIN_EPISODE_INDICES=$(seq -s, 0 549)
ROBOTWIN_DATASET_REPEAT=1
ROBOTWIN_USE_STATE=true
ROBOTWIN_STATE_KEY=observation.state
ROBOTWIN_ACTION_SPACE=joint_delta
COMPUTE_ACTION_STATS=true
FORCE_RECOMPUTE_ACTION_STATS=true

# ---- 超参 ----
OPTIMIZER_LR=8e-5
SCHEDULER_F_MAX=1.0
SCHEDULER_F_MIN=0.0
MAX_ITER=3000
SAVE_ITER=500
SCHEDULER_WARM_UP_STEPS=$((MAX_ITER / 10))
MAX_SAMPLES_PER_BATCH=1
GLOBAL_BATCH=$((NNODES * NPROC_PER_NODE * MAX_SAMPLES_PER_BATCH))

# ---- 输出 & W&B ----
EXP_TAG=all_adjust_bottle_${ROBOTWIN_ACTION_SPACE}_lr${OPTIMIZER_LR}_gbs${GLOBAL_BATCH}
TRAINING_NAME=$EXP_TAG
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models
OUTPUT_BASE_ROOT=/apdcephfs_gy7/share_303588738/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano
WANDB_PROJECT=cosmos3
WANDB_GROUP=robotwin_adjustbottle_${ROBOTWIN_ACTION_SPACE}_sft
WANDB_NAME=${EXP_TAG}_$(date +%m%d_%H%M)

export EXP_IP_LIST NNODES MASTER_ADDR MASTER_PORT NPROC_PER_NODE COSMOS_WORKDIR \
  ROBOTWIN_MODE OBSERVATION_IMAGE_MODE ROBOTWIN_VIEWPOINT ROBOTWIN_EPISODE_INDICES \
  ROBOTWIN_DATASET_REPEAT ROBOTWIN_USE_STATE ROBOTWIN_STATE_KEY \
  ROBOTWIN_ACTION_SPACE COMPUTE_ACTION_STATS FORCE_RECOMPUTE_ACTION_STATS \
  OPTIMIZER_LR SCHEDULER_F_MAX SCHEDULER_F_MIN SCHEDULER_WARM_UP_STEPS MAX_ITER SAVE_ITER \
  TRAINING_NAME COSMOS_MODELS_ROOT OUTPUT_BASE_ROOT \
  WANDB_PROJECT WANDB_GROUP WANDB_NAME

cd "$COSMOS_WORKDIR"

bash scripts/run_pdsh_robotwin_cosmos3nano_one.sh -- \
  dataloader_train.max_samples_per_batch=$MAX_SAMPLES_PER_BATCH \
  dataloader_train.infinite_data_stream=true \
  dataloader_train.dataloader.shard_map_style_dataset=true \
  '+dataloader_train.dataloader.shuffle=true' \
  dataloader_train.dataloader.datasets.robotwin.dataset.action_space=$ROBOTWIN_ACTION_SPACE \
  dataloader_train.dataloader.datasets.robotwin.dataset.action_normalization=quantile
