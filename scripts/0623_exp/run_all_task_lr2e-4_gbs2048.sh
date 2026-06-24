#!/usr/bin/env bash
# =============================================================================
# 实验目的:全量 RoboTwin 数据训练 (不再分 task / 不再过拟合)
#   - 全部 27500 episode,~5.64M 唯一窗口
#   - lr=2e-4, global batch=2048 (64 卡 × 每卡 32)
#   - MAX_ITER=6000 ≈ 2.18 epoch (数据过两遍左右;1 epoch≈2752 step)
#   - DistributedSampler 分片:64 卡各看 ~88k 窗口,无重叠
#
# 注意:直接调 run_pdsh_robotwin_cosmos3nano.sh(跳过 _one.sh),
#   因为 _one.sh 在 task/episode 三者都空时会强制回退到 task_index=0。
#   这里三者都不设 → task_index/episode_indices/task_name 全部 null → 全量数据。
#
# 风险:max_samples_per_batch=32 + 变长视频,单卡 packed batch token 数很大,
#   若 OOM 需调小 MAX_SAMPLES_PER_BATCH (gbs 相应下降) 或减小 resolution。
# =============================================================================
set -euo pipefail

# ---- 集群 ----
EXP_IP_LIST=29.119.84.104,29.127.48.51,29.127.50.141,29.232.241.61,29.119.96.20,29.232.224.85,29.191.192.197,29.119.83.12
NNODES=8
MASTER_ADDR=29.119.84.104
MASTER_PORT=50221
NPROC_PER_NODE=8
COSMOS_WORKDIR=/mnt/lyn/wjh/workspace/cosmos3/wam   # 含分片改动的 checkout

# ---- 数据 (全量) ----
# 不设 ROBOTWIN_TASK_INDEX / ROBOTWIN_EPISODE_INDICES / ROBOTWIN_TASK_NAME,
# 内层脚本会把它们置为 null → 使用全部 27500 episode。
ROBOTWIN_MODE=policy
OBSERVATION_IMAGE_MODE=concat
ROBOTWIN_VIEWPOINT=concat_view
ROBOTWIN_DATASET_REPEAT=1          # 全量数据无需 repeat,训练长度靠 MAX_ITER 控制
ROBOTWIN_USE_STATE=true
ROBOTWIN_STATE_KEY=observation.state

# ---- 超参 ----
OPTIMIZER_LR=2e-4
SCHEDULER_F_MAX=1.0
SCHEDULER_F_MIN=0.0
MAX_ITER=6000                     # ≈ 2.18 epoch (5.64M 窗口 / gbs2048 ≈ 2752 step/epoch)
SAVE_ITER=1000
SCHEDULER_WARM_UP_STEPS=$((MAX_ITER / 10))
MAX_SAMPLES_PER_BATCH=16          # 每卡每 step 样本数。global batch = NNODES*NPROC*该值
GLOBAL_BATCH=$((NNODES * NPROC_PER_NODE * MAX_SAMPLES_PER_BATCH))  # = 2048

# ---- 输出 & W&B ----
# TRAINING_NAME 决定 checkpoint 目录 (OUTPUT_BASE_ROOT/TRAINING_NAME),用简短语义名。
# wandb 跟踪靠 GROUP 分组 + 短 NAME;详细超参在 wandb 的 config 面板里可查。
EXP_TAG=all_task_lr${OPTIMIZER_LR}_gbs${GLOBAL_BATCH}
TRAINING_NAME=$EXP_TAG
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models
OUTPUT_BASE_ROOT=/apdcephfs_gy7/share_303588738/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano
WANDB_PROJECT=cosmos3
WANDB_GROUP=robotwin_all_task     # 同主题 run 在 wandb 自动聚到一起
WANDB_NAME=${EXP_TAG}_$(date +%m%d_%H%M)

export EXP_IP_LIST NNODES MASTER_ADDR MASTER_PORT NPROC_PER_NODE COSMOS_WORKDIR \
  ROBOTWIN_MODE OBSERVATION_IMAGE_MODE ROBOTWIN_VIEWPOINT \
  ROBOTWIN_DATASET_REPEAT ROBOTWIN_USE_STATE ROBOTWIN_STATE_KEY \
  OPTIMIZER_LR SCHEDULER_F_MAX SCHEDULER_F_MIN SCHEDULER_WARM_UP_STEPS MAX_ITER SAVE_ITER \
  TRAINING_NAME COSMOS_MODELS_ROOT OUTPUT_BASE_ROOT \
  WANDB_PROJECT WANDB_GROUP WANDB_NAME

# ---- 训练侧 OmegaConf override ----
#   直接调内层 pdsh 脚本(非 _one.sh),不注入任何 task/episode 过滤 → 全量数据。
#   shard_map_style_dataset 已是 config 显式参数 → 普通覆盖 (无 +)。
#   shuffle 走 **dataloader_kwargs,config 里不存在 → 必须 '+' 新增。
bash scripts/run_pdsh_robotwin_cosmos3nano.sh -- \
  dataloader_train.max_samples_per_batch=$MAX_SAMPLES_PER_BATCH \
  dataloader_train.infinite_data_stream=true \
  dataloader_train.dataloader.shard_map_style_dataset=true \
  '+dataloader_train.dataloader.shuffle=true'
