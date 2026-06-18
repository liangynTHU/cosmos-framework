TRAINING_NAME=ep0_5_full_traj_concat_view_bs64 \
EXP_IP_LIST=29.191.210.123,29.127.64.8,29.127.65.89,29.119.84.187,29.127.82.110,29.191.209.108,29.119.97.119,29.119.99.50 \
NNODES=8 MASTER_ADDR=29.191.210.123 MASTER_PORT=50221 NPROC_PER_NODE=8 \
COSMOS_WORKDIR=/mnt/lyn/workspace/wam \
ROBOTWIN_MODE=policy OBSERVATION_IMAGE_MODE=concat ROBOTWIN_VIEWPOINT=concat_view \
ROBOTWIN_EPISODE_INDICES=0,2,5,8,17 \
ROBOTWIN_DATASET_REPEAT=100 \
ROBOTWIN_USE_STATE=true ROBOTWIN_STATE_KEY=observation.state \
OPTIMIZER_LR=2.0e-4 SCHEDULER_F_MAX=1.0 SCHEDULER_F_MIN=0.0 SCHEDULER_WARM_UP_STEPS=0 \
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models \
WANDB_PROJECT=cosmos3 WANDB_GROUP=robotwin_lerobot_action_sft \
WANDB_NAME=cosmos3nano_robotwin_ep0_5_full_traj_concat_view_bs64_$(date +%Y%m%d_%H%M%S) \
MAX_ITER=1000 SAVE_ITER=200 \
bash scripts/run_pdsh_robotwin_cosmos3nano_one.sh -- \
  dataloader_train.max_samples_per_batch=32 \
  dataloader_train.infinite_data_stream=true 