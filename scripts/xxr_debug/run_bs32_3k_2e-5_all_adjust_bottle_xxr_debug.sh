#!/usr/bin/env bash
# =============================================================================
# 实验目的:验证 DistributedSampler map-style 分片 (shard_map_style_dataset)
#   - 所有adjust_bottle 的 full-trajectory 过拟合
#   - 开分片后 64 卡各看 1/64 的窗口 (不再 64 卡冗余看同一份)
#   - 预期:之前固定周期的 action-loss spike 消失/改变形态
#   - repeat=64 -> 每卡 ~650 step 的唯一遍历量,MAX_ITER=500 硬停
# 关键开关:dataloader_train.dataloader.shard_map_style_dataset=true (默认即 true)
# =============================================================================
set -euo pipefail

# ---- 集群 ----
EXP_IP_LIST=29.191.210.123,29.127.64.8,29.127.65.89,29.119.84.187,29.127.82.110,29.191.209.108,29.119.97.119,29.119.99.50 \
NNODES=8
MASTER_ADDR=29.191.210.123
MASTER_PORT=50221
NPROC_PER_NODE=8
COSMOS_WORKDIR=/mnt/lyn/wjh/workspace/cosmos3/wam   # 含分片改动的 checkout

# ---- 数据 (5 episode 过拟合) ----
ROBOTWIN_MODE=policy
OBSERVATION_IMAGE_MODE=concat
ROBOTWIN_VIEWPOINT=concat_view
ROBOTWIN_EPISODE_INDICES=$(seq -s, 0 549)
ROBOTWIN_DATASET_REPEAT=1        # 全量adjust bottle数据
ROBOTWIN_USE_STATE=true
ROBOTWIN_STATE_KEY=observation.state

# ---- 超参 ----
OPTIMIZER_LR=2e-5
SCHEDULER_F_MAX=1.0
SCHEDULER_F_MIN=0.0
MAX_ITER=300         # all: 5.64M 窗口 / gbs2048 ≈ 2752 step/epoch; 单task: (5.64M 窗口/ 50) / gbs2048 ≈ 55 step/epoch
SAVE_ITER=50
SCHEDULER_WARM_UP_STEPS=$((MAX_ITER / 10))
MAX_SAMPLES_PER_BATCH=32     # 每卡每 step 样本数。global batch = NNODES*NPROC*该值
GLOBAL_BATCH=$((NNODES * NPROC_PER_NODE * MAX_SAMPLES_PER_BATCH))  # = 2048

# ---- 输出 & W&B ----
# TRAINING_NAME 决定 checkpoint 目录 (OUTPUT_BASE_ROOT/TRAINING_NAME),用简短语义名。
# wandb 跟踪靠 GROUP 分组 + 短 NAME;详细超参在 wandb 的 config 面板里可查。
EXP_TAG=all_adjust_bottle_lr${OPTIMIZER_LR}_gbs${GLOBAL_BATCH}
TRAINING_NAME=$EXP_TAG
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models
OUTPUT_BASE_ROOT=/apdcephfs_gy7/share_303588738/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano
WANDB_PROJECT=cosmos3
WANDB_GROUP=robotwin_adjustbottle_test   # 同主题 run 在 wandb 自动聚到一起
WANDB_NAME=${EXP_TAG}_$(date +%m%d_%H%M)

export EXP_IP_LIST NNODES MASTER_ADDR MASTER_PORT NPROC_PER_NODE COSMOS_WORKDIR \
  ROBOTWIN_MODE OBSERVATION_IMAGE_MODE ROBOTWIN_VIEWPOINT ROBOTWIN_EPISODE_INDICES \
  ROBOTWIN_DATASET_REPEAT ROBOTWIN_USE_STATE ROBOTWIN_STATE_KEY \
  OPTIMIZER_LR SCHEDULER_F_MAX SCHEDULER_F_MIN SCHEDULER_WARM_UP_STEPS MAX_ITER SAVE_ITER \
  TRAINING_NAME COSMOS_MODELS_ROOT OUTPUT_BASE_ROOT \
  WANDB_PROJECT WANDB_GROUP WANDB_NAME

# ---- 训练侧 OmegaConf override ----
#   shard_map_style_dataset 是 RankPartitionedDataLoader 的显式参数 → config 里已存在,
#     用普通覆盖 (无 +);加 + 会报 "An item is already at ...".
#   shuffle 走 **dataloader_kwargs,config 里不存在 → 必须 '+' 新增。
bash scripts/run_pdsh_robotwin_cosmos3nano_one.sh -- \
  dataloader_train.max_samples_per_batch=$MAX_SAMPLES_PER_BATCH \
  dataloader_train.infinite_data_stream=true \
  dataloader_train.dataloader.shard_map_style_dataset=true \
  '+dataloader_train.dataloader.shuffle=true'
