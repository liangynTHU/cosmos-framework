#!/usr/bin/env bash
# =============================================================================
# 16-node 全量训练 —— 在【机器 A (head, 含全局 rank0)】上运行本脚本。
# 机器 B 上运行 run_all_task_lr2e-4_gbs2048_16node_B.sh (NODE_RANK_OFFSET=8)。
#
#   - 两台 8-node 机器拼成 16 node = 128 卡
#   - global batch = 16node × 8gpu × bs16 = 2048
#   - 全量 27500 episode,MAX_ITER=6000 ≈ 2.18 epoch
#   - 本机驱动 global rank 0-7 (NODE_RANK_OFFSET=0)
#   - MASTER_ADDR = 机器 A 第一个 IP (全局 rank0),两台机器必须用同一个
#
# 风险:跨机先试 IB。若 NCCL 跨机连不上 hang,回退纯 TCP:
#   在两台机器都额外 export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=<两机互通网卡>
# =============================================================================
set -euo pipefail

# ---- 集群:本机 (机器 A) 的 8 个 IP ----
EXP_IP_LIST=29.119.84.104,29.127.48.51,29.127.50.141,29.232.241.61,29.119.96.20,29.232.224.85,29.191.192.197,29.119.83.12
NNODES=16                         # 全局总节点数 (两台机器之和)
NODE_RANK_OFFSET=0                # 本机起始 global rank → 0-7
MASTER_ADDR=29.119.84.104         # 全局 rank0 = 机器 A 第一个 IP;机器 B 必须用同一个
MASTER_PORT=50221
NPROC_PER_NODE=8
COSMOS_WORKDIR=/mnt/lyn/wjh/workspace/cosmos3/wam   # 含分片改动的 checkout

# ---- 数据 (全量) ----
ROBOTWIN_MODE=policy
OBSERVATION_IMAGE_MODE=concat
ROBOTWIN_VIEWPOINT=concat_view
ROBOTWIN_DATASET_REPEAT=1
ROBOTWIN_USE_STATE=true
ROBOTWIN_STATE_KEY=observation.state

# ---- 超参 ----
OPTIMIZER_LR=2e-4
SCHEDULER_F_MAX=1.0
SCHEDULER_F_MIN=0.0
MAX_ITER=3000                     # ≈ 2.18 epoch (5.64M 窗口 / gbs2048 ≈ 2752 step/epoch)
SAVE_ITER=500
SCHEDULER_WARM_UP_STEPS=$((MAX_ITER / 10))
MAX_SAMPLES_PER_BATCH=16          # 每卡每 step 样本数。global batch = NNODES*NPROC*该值
GLOBAL_BATCH=$((NNODES * NPROC_PER_NODE * MAX_SAMPLES_PER_BATCH))  # = 2048

# ---- 输出 & W&B ----
EXP_TAG=all_task_lr${OPTIMIZER_LR}_gbs${GLOBAL_BATCH}_16node
TRAINING_NAME=$EXP_TAG
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models
OUTPUT_BASE_ROOT=/apdcephfs_gy7/share_303588738/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano
WANDB_PROJECT=cosmos3
WANDB_GROUP=robotwin_all_task
# 16node 两 head 是同一个 job,wandb run name 在两台机器上必须一致才能聚成一个 run。
# 用固定时间戳 (不带 $(date)) 避免两边各生成不同的 name。
WANDB_NAME=${EXP_TAG}_0624

export EXP_IP_LIST NNODES NODE_RANK_OFFSET MASTER_ADDR MASTER_PORT NPROC_PER_NODE COSMOS_WORKDIR \
  ROBOTWIN_MODE OBSERVATION_IMAGE_MODE ROBOTWIN_VIEWPOINT \
  ROBOTWIN_DATASET_REPEAT ROBOTWIN_USE_STATE ROBOTWIN_STATE_KEY \
  OPTIMIZER_LR SCHEDULER_F_MAX SCHEDULER_F_MIN SCHEDULER_WARM_UP_STEPS MAX_ITER SAVE_ITER \
  TRAINING_NAME COSMOS_MODELS_ROOT OUTPUT_BASE_ROOT \
  WANDB_PROJECT WANDB_GROUP WANDB_NAME

# ---- 训练侧 OmegaConf override (全量数据;直接调内层 pdsh,跳过 _one.sh) ----
# 切到 COSMOS_WORKDIR,确保用的是这个 checkout 的 scripts/(含 NODE_RANK_OFFSET 改动),
# 而不是执行时 cwd 所在目录的旧脚本。
cd "$COSMOS_WORKDIR"

# 把命令行看到的所有输出 (pdsh 过程 / Starting node / 报错) 同时存一份到 log。
LAUNCH_LOG_DIR=/mnt/lyn/wjh/workspace/cosmos3/wam/logs
mkdir -p "$LAUNCH_LOG_DIR"
LAUNCH_LOG="$LAUNCH_LOG_DIR/${EXP_TAG}_A_$(date +%m%d_%H%M%S).log"
echo "launcher log -> $LAUNCH_LOG"

bash "$COSMOS_WORKDIR/scripts/run_pdsh_robotwin_cosmos3nano.sh" -- \
  dataloader_train.max_samples_per_batch=$MAX_SAMPLES_PER_BATCH \
  dataloader_train.infinite_data_stream=true \
  dataloader_train.dataloader.shard_map_style_dataset=true \
  '+dataloader_train.dataloader.shuffle=true' \
  2>&1 | tee "$LAUNCH_LOG"
