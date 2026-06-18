#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 只取 RoboTwin LeRobot 中一个 task 的所有 episodes，并重复采样用于 overfit。
# 默认 task_index=0；也可以通过 ROBOTWIN_TASK_INDEX 或 ROBOTWIN_TASK_NAME 覆盖。
export TRAINING_NAME="${TRAINING_NAME:-robotwin_one_task_concat}"
export ROBOTWIN_TASK_INDEX="${ROBOTWIN_TASK_INDEX:-0}"
export ROBOTWIN_DATASET_REPEAT="${ROBOTWIN_DATASET_REPEAT:-1000}"
export OBSERVATION_IMAGE_MODE="${OBSERVATION_IMAGE_MODE:-concat}"
export ROBOTWIN_VIEWPOINT="${ROBOTWIN_VIEWPOINT:-concat_view}"
export ROBOTWIN_USE_STATE="${ROBOTWIN_USE_STATE:-true}"
export ROBOTWIN_STATE_KEY="${ROBOTWIN_STATE_KEY:-observation.state}"
export OPTIMIZER_LR="${OPTIMIZER_LR:-2.0e-4}"
export SCHEDULER_F_MAX="${SCHEDULER_F_MAX:-1.0}"
export SCHEDULER_F_MIN="${SCHEDULER_F_MIN:-0.0}"
export SCHEDULER_WARM_UP_STEPS="${SCHEDULER_WARM_UP_STEPS:-0}"
export MAX_ITER="${MAX_ITER:-1000}"
export SAVE_ITER="${SAVE_ITER:-200}"
export WANDB_NAME="${WANDB_NAME:-cosmos3nano_robotwin_one_task_${ROBOTWIN_TASK_INDEX}_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/run_pdsh_robotwin_cosmos3nano.sh" "$@"
