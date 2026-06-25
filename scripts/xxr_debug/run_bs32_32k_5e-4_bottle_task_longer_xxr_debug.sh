TRAINING_NAME=ep0_5_full_traj_concat_view_global_bs64_32k_lr5e-4_steps_warmup1k \
EXP_IP_LIST=29.191.210.123,29.127.64.8,29.127.65.89,29.119.84.187,29.127.82.110,29.191.209.108,29.119.97.119,29.119.99.50 \
NNODES=8 MASTER_ADDR=29.191.210.123 MASTER_PORT=50221 NPROC_PER_NODE=8 \
COSMOS_WORKDIR=/mnt/lyn/workspace/wam \
ROBOTWIN_MODE=policy OBSERVATION_IMAGE_MODE=concat ROBOTWIN_VIEWPOINT=concat_view \
ROBOTWIN_EPISODE_INDICES=0,2,5,8,17 \
ROBOTWIN_DATASET_REPEAT=100000 \
ROBOTWIN_USE_STATE=true ROBOTWIN_STATE_KEY=observation.state \
OPTIMIZER_LR=5e-4 SCHEDULER_F_MAX=1.0 SCHEDULER_F_MIN=0.0 SCHEDULER_WARM_UP_STEPS=1000 \
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models \
WANDB_PROJECT=cosmos3 WANDB_GROUP=robotwin_lerobot_action_sft \
WANDB_NAME=cosmos3nano_robotwin_ep0_5_full_traj_concat_view_global_bs32_5k_steps_warmup1k_$(date +%Y%m%d_%H%M%S) \
OUTPUT_BASE_ROOT=/apdcephfs_gy7/share_303588738/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano \
MAX_ITER=5000 SAVE_ITER=100 \
bash scripts/run_pdsh_robotwin_cosmos3nano_one.sh -- \
  dataloader_train.max_samples_per_batch=32 \
  dataloader_train.infinite_data_stream=true \
  '+dataloader_train.dataloader.shuffle=true'