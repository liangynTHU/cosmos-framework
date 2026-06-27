#!/usr/bin/env bash
# Compute RoboTwin action stats for the exact training data subset, then export
# ROBOTWIN_ACTION_STATS_PATH for run_train_robotwin_cosmos3nano.sh.
#
# Reads the same filter env vars as training:
#   DATASET_PATH, ROBOTWIN_TASK_INDEX, ROBOTWIN_TASK_NAME, ROBOTWIN_EPISODE_INDICES,
#   MAX_EPISODES, ROBOTWIN_STATE_KEY, ROBOTWIN_ACTION_SPACE
#
# Output default: $TRAINING_RUN_DIR/action_stats.json
#   where TRAINING_RUN_DIR=$IMAGINAIRE_OUTPUT_ROOT/$WANDB_PROJECT/$WANDB_GROUP/$WANDB_NAME
#   (same directory as config.yaml and checkpoints/).
#
# Legacy fallback: $OUTPUT_ROOT/action_stats.json (older runs).
#
# Set COMPUTE_ACTION_STATS=false to skip. Set FORCE_RECOMPUTE_ACTION_STATS=true to rebuild.

set -euo pipefail

: "${COSMOS_WORKDIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${DATASET_PATH:=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${ROBOTWIN_ACTION_SPACE:=joint_delta}"
: "${ROBOTWIN_STATE_KEY:=observation.state}"
: "${COMPUTE_ACTION_STATS:=true}"

resolve_training_run_dir() {
    local output_root="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
    local project="${WANDB_PROJECT:-cosmos3}"
    local group="${WANDB_GROUP:-robotwin_lerobot_action_sft}"
    local name="${WANDB_NAME:-}"
    if [[ -z "$name" ]]; then
        echo "ERROR: WANDB_NAME must be set before computing action stats (needed for TRAINING_RUN_DIR)." >&2
        exit 1
    fi
    printf '%s/%s/%s/%s' "$output_root" "$project" "$group" "$name"
}

TRAINING_RUN_DIR="$(resolve_training_run_dir)"
LEGACY_ROBOTWIN_ACTION_STATS_PATH="${OUTPUT_ROOT}/action_stats.json"
ROBOTWIN_ACTION_STATS_PATH="${ROBOTWIN_ACTION_STATS_PATH:-$TRAINING_RUN_DIR/action_stats.json}"
export ROBOTWIN_ACTION_STATS_PATH TRAINING_RUN_DIR

should_compute=true
if [[ "${COMPUTE_ACTION_STATS}" != "true" ]]; then
    echo "[action-stats] COMPUTE_ACTION_STATS=false, using ${ROBOTWIN_ACTION_STATS_PATH}"
    should_compute=false
elif [[ -f "$ROBOTWIN_ACTION_STATS_PATH" && "${FORCE_RECOMPUTE_ACTION_STATS:-false}" != "true" ]]; then
    echo "[action-stats] Reusing existing stats: $ROBOTWIN_ACTION_STATS_PATH"
    should_compute=false
elif [[ -f "$LEGACY_ROBOTWIN_ACTION_STATS_PATH" && "${FORCE_RECOMPUTE_ACTION_STATS:-false}" != "true" ]]; then
    echo "[action-stats] Reusing legacy stats: $LEGACY_ROBOTWIN_ACTION_STATS_PATH"
    ROBOTWIN_ACTION_STATS_PATH="$LEGACY_ROBOTWIN_ACTION_STATS_PATH"
    export ROBOTWIN_ACTION_STATS_PATH
    should_compute=false
fi

if [[ "$should_compute" == "true" ]]; then
    CONDA_SH="${CONDA_SH:-/jizhicfs/peterrao/miniconda3/etc/profile.d/conda.sh}"
    CONDA_ENV="${CONDA_ENV:-cosmos3}"

    mkdir -p "$TRAINING_RUN_DIR"

    STATS_ARGS=(
        --dataset-path "$DATASET_PATH"
        --action-space "$ROBOTWIN_ACTION_SPACE"
        --state-key "$ROBOTWIN_STATE_KEY"
        --output "$ROBOTWIN_ACTION_STATS_PATH"
    )

    if [[ -n "${MAX_EPISODES:-}" ]]; then
        STATS_ARGS+=(--max-episodes "$MAX_EPISODES")
    fi
    if [[ -n "${ROBOTWIN_TASK_INDEX:-}" ]]; then
        STATS_ARGS+=(--task-index "$ROBOTWIN_TASK_INDEX")
    fi
    if [[ -n "${ROBOTWIN_TASK_NAME:-}" ]]; then
        STATS_ARGS+=(--task-name "$ROBOTWIN_TASK_NAME")
    fi
    if [[ -n "${ROBOTWIN_EPISODE_INDICES:-}" ]]; then
        STATS_ARGS+=(--episode-indices "$ROBOTWIN_EPISODE_INDICES")
    fi
    if [[ -n "${ROBOTWIN_ACTION_HORIZON:-}" ]]; then
        STATS_ARGS+=(--action-horizon "$ROBOTWIN_ACTION_HORIZON")
    fi

    echo "[action-stats] Computing stats for training subset"
    echo "  dataset_path=$DATASET_PATH"
    echo "  action_space=$ROBOTWIN_ACTION_SPACE"
    echo "  training_run_dir=$TRAINING_RUN_DIR"
    echo "  output=$ROBOTWIN_ACTION_STATS_PATH"
    echo "  task_index=${ROBOTWIN_TASK_INDEX:-}"
    echo "  task_name=${ROBOTWIN_TASK_NAME:-}"
    echo "  episode_indices=${ROBOTWIN_EPISODE_INDICES:-}"
    echo "  max_episodes=${MAX_EPISODES:-}"

    (
        cd "$COSMOS_WORKDIR"
        export PYTHONPATH=.
        if [[ -f "$CONDA_SH" ]]; then
            # shellcheck disable=SC1090
            . "$CONDA_SH"
            conda activate "$CONDA_ENV"
        fi
        python -m cosmos_framework.scripts.compute_robotwin_lerobot_action_stats "${STATS_ARGS[@]}"
    )

    echo "[action-stats] Done: $ROBOTWIN_ACTION_STATS_PATH"
fi
