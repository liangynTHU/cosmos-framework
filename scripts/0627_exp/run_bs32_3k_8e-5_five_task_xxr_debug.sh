#!/usr/bin/env bash
# =============================================================================
# 实验目的: 5 个 task 联合训练 (adjust_bottle / Pick Diverse Bottles /
#           Place Mouse Pad / Place Object Basket / Turn Switch)
#   - 每 task 取 550 episode (共 2750 episode, ~375k 窗口)
#   - gbs=2048, lr=8e-5, MAX_ITER=3000 (~16.3 epoch, 对齐之前单 task 训练量)
#   - 64 卡 DistributedSampler 分片, 每条样本 1/64 概率被某卡看到
# 关键开关:dataloader_train.dataloader.shard_map_style_dataset=true (默认即 true)
# =============================================================================
set -euo pipefail

# ---- 集群 ----
# EXP_IP_LIST=29.119.84.104,29.127.48.51,29.127.50.141,29.232.241.61,29.119.96.20,29.232.224.85,29.191.192.197,29.119.83.12
# NNODES=8
# MASTER_ADDR=29.119.84.104
EXP_IP_LIST=29.191.210.123,29.127.64.8,29.127.65.89,29.119.84.187,29.127.82.110,29.191.209.108,29.119.97.119,29.119.99.50 \
NNODES=8
MASTER_ADDR=29.191.210.123
MASTER_PORT=50221
NPROC_PER_NODE=8
COSMOS_WORKDIR=/mnt/lyn/xxr/cosmos-framework   # 含分片改动的 checkout

# ---- 数据 (5 task x 550 episode) ----
ROBOTWIN_MODE=policy
OBSERVATION_IMAGE_MODE=concat
ROBOTWIN_VIEWPOINT=concat_view
ROBOTWIN_EPISODE_INDICES=$(seq -s, 0 549),$(seq -s, 9900 10449),$(seq -s, 17050 17599),$(seq -s, 17600 18149),$(seq -s, 26950 27499)  #0-594; 9900-10449; 17050~17599; 17600~18149; 26950-27499; 
ROBOTWIN_DATASET_REPEAT=1        # 全量adjust bottle数据
ROBOTWIN_USE_STATE=true
ROBOTWIN_STATE_KEY=observation.state
ROBOTWIN_ACTION_SPACE=joint_delta #joint_delta or joint_pos
COMPUTE_ACTION_STATS=true
FORCE_RECOMPUTE_ACTION_STATS=true

# ---- 超参 ----
OPTIMIZER_LR=8e-5
SCHEDULER_F_MAX=1.0
SCHEDULER_F_MIN=0.0
MAX_ITER=3000         # 69611+58755+74913+127117+45398+=375794; 375794/2048=184;  adjust bottle, Pick Diverse Bottles, Place Mouse Pad, Place Object Basket, turn switch
SAVE_ITER=200
SCHEDULER_WARM_UP_STEPS=$((MAX_ITER / 10))
MAX_SAMPLES_PER_BATCH=32     # 每卡每 step 样本数。global batch = NNODES*NPROC*该值
GLOBAL_BATCH=$((NNODES * NPROC_PER_NODE * MAX_SAMPLES_PER_BATCH))  # = 2048

# ---- 输出 & W&B ----
# TRAINING_NAME 决定 checkpoint 目录 (OUTPUT_BASE_ROOT/TRAINING_NAME),用简短语义名。
# wandb 跟踪靠 GROUP 分组 + 短 NAME;详细超参在 wandb 的 config 面板里可查。
EXP_TAG=five_task_lr${OPTIMIZER_LR}_gbs${GLOBAL_BATCH}_iter${MAX_ITER}
TRAINING_NAME=$EXP_TAG
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models
OUTPUT_BASE_ROOT=/apdcephfs_gy7/share_303588738/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano
WANDB_PROJECT=cosmos3
WANDB_GROUP=robotwin_multitask_${ROBOTWIN_ACTION_SPACE}_test   # 同主题 run 在 wandb 自动聚到一起
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
