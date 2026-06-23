TRAINING_NAME=task1_full_traj_concat_view_global_bs64_32k_steps \
EXP_IP_LIST=29.119.84.104,29.127.48.51,29.127.50.141,29.232.241.61,29.119.96.20,29.232.224.85,29.191.192.197,29.119.83.12 \
NNODES=8 MASTER_ADDR=29.119.84.104 MASTER_PORT=50221 NPROC_PER_NODE=8 \
COSMOS_WORKDIR=/mnt/lyn/workspace/wam \
ROBOTWIN_MODE=policy OBSERVATION_IMAGE_MODE=concat ROBOTWIN_VIEWPOINT=concat_view \
ROBOTWIN_EPISODE_INDICES=550,551,552,553,554,555,556 \
ROBOTWIN_DATASET_REPEAT=100000 \
ROBOTWIN_USE_STATE=true ROBOTWIN_STATE_KEY=observation.state \
OPTIMIZER_LR=2.0e-4 SCHEDULER_F_MAX=1.0 SCHEDULER_F_MIN=0.0 SCHEDULER_WARM_UP_STEPS=0 \
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models \
WANDB_PROJECT=cosmos3 WANDB_GROUP=robotwin_lerobot_action_sft \
WANDB_NAME=cosmos3nano_robotwin_task1_full_traj_concat_view_bs64_$(date +%Y%m%d_%H%M%S) \
OUTPUT_BASE_ROOT=/apdcephfs_gy7/share_303588738/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano \
MAX_ITER=32000 SAVE_ITER=500 \
bash scripts/run_pdsh_robotwin_cosmos3nano_one.sh -- \
  dataloader_train.max_samples_per_batch=1 \
  dataloader_train.infinite_data_stream=true 



# caption: Use the medium-sized metal hammer to hammer the block.