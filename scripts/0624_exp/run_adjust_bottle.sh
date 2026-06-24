#!/usr/bin/env bash
# =============================================================================
# 单任务训练: adjust_bottle (RoboTwin2.0 LeRobot)
#   - 数据集无任务名标签, 任务按 episode 连续成段, 每任务 550 个 episode。
#     指令文本 + RoboTwin 官方模板交叉验证: adjust_bottle = episode 0~549 (chunk-000)。
#   - 训这 550 个 episode 的全量窗口 (非过拟合, repeat=1)。
#   - 开 map-style 分片 (shard_map_style_dataset=true): 64 卡各看 1/64 窗口。
# 换任务: 改 ROBOTWIN_TASK_INDEX (每任务连续 550 个 episode) + TASK_LABEL 即可。
# =============================================================================
set -euo pipefail

# ---- 集群 ----
EXP_IP_LIST=29.119.84.104,29.127.48.51,29.127.50.141,29.232.241.61,29.119.96.20,29.232.224.85,29.191.192.197,29.119.83.12
NNODES=8
NODE_RANK_OFFSET=0                 # 本机起始 global rank (单机 8 节点保持 0)
MASTER_ADDR=29.119.84.104
MASTER_PORT=50221
NPROC_PER_NODE=8
COSMOS_WORKDIR=/mnt/lyn/wjh/workspace/cosmos3/wam   # 含分片改动的 checkout

# ---- 数据: 两种选择模式 (二选一, 别同时设) ----
# 【模式 A】按粗任务号选: ROBOTWIN_TASK_INDEX=0   (多个: =0,2,5 取并集)
#   task 在数据里顺序排列, 每任务 550 个 episode。task_index=k 自动选 episode
#   [k*550, k*550+549]。adjust_bottle=0 -> 0~549。换任务改这个数 + TASK_LABEL 即可。
#   (每任务非 550 个时, 可加 ROBOTWIN_EPISODES_PER_TASK=<n> 覆盖。)
#
# 【模式 B】只选指定的几个 episode: ROBOTWIN_EPISODE_INDICES=3,7,42
#   只加载列出的这些 episode, 不多不少。用模式 B 时把下面 ROBOTWIN_TASK_INDEX 那行
#   注释掉 (两个都设会取交集, 不是单纯的列表)。
#   ⚠️ 不用就整行删/注释掉, 千万别写成 =None 或 =null (会被当成列表内容 -> 训练崩)。
#
# 当前: 模式 A, adjust_bottle (task 0)
ROBOTWIN_TASK_INDEX=0
# ROBOTWIN_EPISODE_INDICES=3,7,42    # 模式 B 示例: 取消注释并注释掉上面那行
TASK_LABEL=adjust_bottle           # 仅用于实验命名
ROBOTWIN_MODE=policy
OBSERVATION_IMAGE_MODE=concat
ROBOTWIN_VIEWPOINT=concat_view
ROBOTWIN_DATASET_REPEAT=1          # 550 episode 已是全量, 不 repeat
ROBOTWIN_USE_STATE=true
ROBOTWIN_STATE_KEY=observation.state

# ---- 超参 ----
OPTIMIZER_LR=2e-4
SCHEDULER_F_MAX=1.0
SCHEDULER_F_MIN=0.0
MAX_ITER=3000
SAVE_ITER=500
SCHEDULER_WARM_UP_STEPS=$((MAX_ITER / 10))
MAX_SAMPLES_PER_BATCH=1          # 每卡每 step 样本数。global batch = NNODES*NPROC*该值
GLOBAL_BATCH=$((NNODES * NPROC_PER_NODE * MAX_SAMPLES_PER_BATCH))

# ---- 输出 & W&B ----
# TRAINING_NAME 决定 checkpoint 目录 (OUTPUT_BASE_ROOT/TRAINING_NAME), 用简短语义名。
# wandb 靠 GROUP 分组 + 短 NAME; 详细超参在 wandb config 面板里可查。
# DATA_TAG: 模式 A 用 task<k>, 模式 B 用 ep<列表>; 兼容只设其中一个的情况 (set -u 安全)。
if [[ -n "${ROBOTWIN_TASK_INDEX:-}" ]]; then
  DATA_TAG=task${ROBOTWIN_TASK_INDEX}
else
  DATA_TAG=ep${ROBOTWIN_EPISODE_INDICES:-unset}
  DATA_TAG=${DATA_TAG//,/_}        # 逗号换下划线, 适合做目录名
fi
EXP_TAG=${TASK_LABEL}_${DATA_TAG}_lr${OPTIMIZER_LR}_gbs${GLOBAL_BATCH}
TRAINING_NAME=$EXP_TAG
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models
OUTPUT_BASE_ROOT=/apdcephfs_gy7/share_303588738/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano
WANDB_PROJECT=cosmos3
WANDB_GROUP=robotwin_single_task   # 同主题 run 在 wandb 自动聚到一起
WANDB_NAME=${EXP_TAG}_$(date +%m%d_%H%M)

# 两种数据模式的变量都 export (没设的填空, 桥接端会转成 null); 先用 :- 兜底, set -u 下不会因未定义报错。
ROBOTWIN_TASK_INDEX="${ROBOTWIN_TASK_INDEX:-}"
ROBOTWIN_EPISODE_INDICES="${ROBOTWIN_EPISODE_INDICES:-}"
export EXP_IP_LIST NNODES NODE_RANK_OFFSET MASTER_ADDR MASTER_PORT NPROC_PER_NODE COSMOS_WORKDIR \
  ROBOTWIN_MODE OBSERVATION_IMAGE_MODE ROBOTWIN_VIEWPOINT ROBOTWIN_TASK_INDEX ROBOTWIN_EPISODE_INDICES \
  ROBOTWIN_DATASET_REPEAT ROBOTWIN_USE_STATE ROBOTWIN_STATE_KEY \
  OPTIMIZER_LR SCHEDULER_F_MAX SCHEDULER_F_MIN SCHEDULER_WARM_UP_STEPS MAX_ITER SAVE_ITER \
  TRAINING_NAME COSMOS_MODELS_ROOT OUTPUT_BASE_ROOT \
  WANDB_PROJECT WANDB_GROUP WANDB_NAME

# ---- 训练侧 OmegaConf override (直接调内层 pdsh) ----
# 切到 COSMOS_WORKDIR, 确保用这个 checkout 的 scripts/(含分片 / NODE_RANK_OFFSET 改动)。
cd "$COSMOS_WORKDIR"

LAUNCH_LOG_DIR=/mnt/lyn/wjh/workspace/cosmos3/wam/logs
mkdir -p "$LAUNCH_LOG_DIR"
LAUNCH_LOG="$LAUNCH_LOG_DIR/${EXP_TAG}_$(date +%m%d_%H%M%S).log"
echo "launcher log -> $LAUNCH_LOG"

#   shard_map_style_dataset 是 RankPartitionedDataLoader 显式参数 → config 已存在, 普通覆盖 (无 +)。
#   shuffle 走 **dataloader_kwargs, config 不存在 → 必须 '+' 新增。
bash "$COSMOS_WORKDIR/scripts/run_pdsh_robotwin_cosmos3nano.sh" -- \
  dataloader_train.max_samples_per_batch=$MAX_SAMPLES_PER_BATCH \
  dataloader_train.infinite_data_stream=true \
  dataloader_train.dataloader.shard_map_style_dataset=true \
  '+dataloader_train.dataloader.shuffle=true' \
  2>&1 | tee "$LAUNCH_LOG"
