#!/usr/bin/env bash
# RoboTwin all-task SFT with joint_delta + quantile norm.
# Action stats are computed automatically for the training subset and saved to:
#   $OUTPUT_BASE_ROOT/$TRAINING_NAME/$WANDB_PROJECT/$WANDB_GROUP/$WANDB_NAME/action_stats.json

TRAINING_NAME=joint_delta_all_task_concat_view_global_bs32_5k_steps \
EXP_IP_LIST=29.191.210.123,29.127.64.8,29.127.65.89,29.119.84.187,29.127.82.110,29.191.209.108,29.119.97.119,29.119.99.50 \
NNODES=8 MASTER_ADDR=29.191.210.123 MASTER_PORT=50221 NPROC_PER_NODE=8 \
COSMOS_WORKDIR=/mnt/lyn/ybw/workspace/cosmos-framework \
ROBOTWIN_MODE=policy OBSERVATION_IMAGE_MODE=concat ROBOTWIN_VIEWPOINT=concat_view \
ROBOTWIN_USE_STATE=true ROBOTWIN_STATE_KEY=observation.state \
ROBOTWIN_ACTION_SPACE=joint_delta \
COMPUTE_ACTION_STATS=true \
OPTIMIZER_LR=2.0e-5 SCHEDULER_F_MAX=1.0 SCHEDULER_F_MIN=0.0 SCHEDULER_WARM_UP_STEPS=0 \
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models \
WANDB_PROJECT=cosmos3 WANDB_GROUP=robotwin_joint_delta_sft \
WANDB_NAME=cosmos3nano_robotwin_joint_delta_one_task_concat_view_bs1_$(date +%Y%m%d_%H%M%S) \
OUTPUT_BASE_ROOT=/apdcephfs_gy7/share_303588738/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano \
MAX_ITER=32000 SAVE_ITER=1000 \
ROBOTWIN_DATASET_REPEAT=5 \
ROBOTWIN_EPISODE_INDICES=0,2,5,8,17 \
FORCE_RECOMPUTE_ACTION_STATS=true \
  bash scripts/run_pdsh_robotwin_cosmos3nano.sh -- \
  dataloader_train.max_samples_per_batch=1 \
  dataloader_train.infinite_data_stream=true \
  dataloader_train.dataloader.datasets.robotwin.dataset.action_space=joint_delta \
  dataloader_train.dataloader.datasets.robotwin.dataset.action_normalization=quantile

# Optional subset training examples (uncomment as needed):
#   ROBOTWIN_TASK_NAME=beat_block_hammer \
#   MAX_EPISODES=500 \
#   ROBOTWIN_EPISODE_INDICES=0,2,5,8,17 \
#   FORCE_RECOMPUTE_ACTION_STATS=true \
