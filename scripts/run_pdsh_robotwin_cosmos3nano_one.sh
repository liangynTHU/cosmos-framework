#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 只取 RoboTwin LeRobot 中一小部分数据用于 overfit。
# 优先级（按用户显式设置）：
#   ROBOTWIN_EPISODE_INDICES  (e.g. "0,1,2,3,4,5")  -- 推荐，覆盖完整轨迹
#   ROBOTWIN_TASK_NAME        -- 按 tasks.jsonl 里的具体文本
#   ROBOTWIN_TASK_INDEX       -- 按 task_index 数字（注意：可能只筛到稀疏帧）
# 若三者都未设置，则保留旧默认 ROBOTWIN_TASK_INDEX=0 以维持向后兼容。
export TRAINING_NAME="${TRAINING_NAME:-robotwin_one_task_concat}"
if [[ -z "${ROBOTWIN_EPISODE_INDICES:-}" && -z "${ROBOTWIN_TASK_NAME:-}" && -z "${ROBOTWIN_TASK_INDEX:-}" ]]; then
    export ROBOTWIN_TASK_INDEX=0
fi
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
if [[ -n "${ROBOTWIN_EPISODE_INDICES:-}" ]]; then
    _wandb_tag="ep_${ROBOTWIN_EPISODE_INDICES//,/_}"
else
    _wandb_tag="task_${ROBOTWIN_TASK_INDEX:-0}"
fi
export WANDB_NAME="${WANDB_NAME:-cosmos3nano_robotwin_${_wandb_tag}_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/run_pdsh_robotwin_cosmos3nano.sh" "$@"
