#!/usr/bin/env bash
set -euo pipefail
set -x

usage() {
    cat <<'USAGE'
Usage: bash scripts/run_train_robotwin_cosmos3nano.sh [--nnodes N] [--node-rank R] [--master-addr HOST] [--master-port PORT] [-- extra_overrides...]

Environment variables:
  DATASET_PATH              RoboTwin/LeRobot dataset root.
  COSMOS_MODELS_ROOT        Local model/cache root.
  BASE_CHECKPOINT_PATH      Cosmos3-Nano DCP checkpoint path.
  WAN_VAE_PATH              Wan VAE checkpoint path.
  QWEN_TOKENIZER_PATH       Local Qwen/Qwen3-VL-8B-Instruct tokenizer/config path.
  TRAINING_NAME             Required run directory name under OUTPUT_BASE_ROOT.
  OUTPUT_BASE_ROOT          Base output root, default /apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano.
  OUTPUT_ROOT               Resolved output root. Defaults to OUTPUT_BASE_ROOT/TRAINING_NAME.
  NPROC_PER_NODE            Number of GPUs per node.
  WANDB_PROJECT             W&B project name.
  WANDB_GROUP               W&B group name.
  WANDB_NAME                W&B run name.
  WANDB_MODE                W&B mode, default online. Use offline only for debugging.
  WANDB_API_KEY_FILE        File containing W&B API key when WANDB_API_KEY is unset.
  MAX_ITER                  Training iterations override.
  SAVE_ITER                 Checkpoint save interval override.
  MAX_EPISODES              Dataset max episodes override. Set empty or null to use all data if supported.
  ROBOTWIN_TASK_INDEX       Optional RoboTwin task_index filter. Keeps all episodes from this task.
  ROBOTWIN_TASK_NAME        Optional RoboTwin task name filter. Must match meta/tasks.jsonl exactly.
  ROBOTWIN_EPISODE_INDICES  Optional comma-separated list of episode indices, e.g. "0,1,2,3,4,5". When set, all valid windows inside each listed episode are used. Combine with ROBOTWIN_TASK_INDEX/_NAME to additionally restrict by task.
  ROBOTWIN_DATASET_REPEAT   Repeat filtered dataset length, default 1. Useful for one-task overfit.
  ROBOTWIN_USE_STATE        Whether to prepend qpos/proprio state as the first action row, default true.
  ROBOTWIN_STATE_KEY        qpos/proprio parquet field, default observation.state.
  OBSERVATION_IMAGE_MODE    concat or multi_image. concat keeps the legacy single concatenated observation.
  ROBOTWIN_VIEWPOINT        Optional dataset viewpoint override. Set to concat_view for the three-camera baseline.
  OPTIMIZER_LR              Optional optimizer base lr override, e.g. 2.0e-4.
  SCHEDULER_F_MAX           Optional scheduler f_max override, e.g. 1.0.
  SCHEDULER_F_MIN           Optional scheduler f_min override, e.g. 1.0 for fixed lr.
  SCHEDULER_WARM_UP_STEPS   Optional scheduler warm_up_steps override, e.g. 0.
USAGE
}

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-50021}"
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node-rank|--node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr|--master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port|--master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_OVERRIDES+=("$@")
            break
            ;;
        *)
            EXTRA_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

WORKDIR="${COSMOS_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$WORKDIR"

COSMOS_SITE="${COSMOS_SITE:-$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0])
PY
)}"
export LD_LIBRARY_PATH="$COSMOS_SITE/nvidia/cu13/lib:$COSMOS_SITE/nvidia/nvjitlink/lib:$COSMOS_SITE/nvidia/cudnn/lib:$COSMOS_SITE/nvidia/nccl/lib:$COSMOS_SITE/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_MODELS_ROOT="${COSMOS_MODELS_ROOT:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models}"
export HF_HOME="${HF_HOME:-$COSMOS_MODELS_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1,10.0.0.0/8,29.0.0.0/8}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_SL="${NCCL_IB_SL:-3}"
export NCCL_CHECKS_DISABLE="${NCCL_CHECKS_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_LL_THRESHOLD="${NCCL_LL_THRESHOLD:-16384}"
export NCCL_IB_CUDA_SUPPORT="${NCCL_IB_CUDA_SUPPORT:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-bond1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}"
export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
export SHARP_COLL_ENABLE_SAT="${SHARP_COLL_ENABLE_SAT:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_NET_GDR_READ="${NCCL_NET_GDR_READ:-0}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
export NCCL_IB_TC="${NCCL_IB_TC:-160}"
export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

export DATASET_PATH="${DATASET_PATH:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$COSMOS_MODELS_ROOT/Cosmos3-Nano}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-$COSMOS_MODELS_ROOT/wan22_vae/Wan2.2_VAE.pth}"
export QWEN_TOKENIZER_PATH="${QWEN_TOKENIZER_PATH:-$COSMOS_MODELS_ROOT/Qwen/Qwen3-VL-8B-Instruct}"
if [[ -z "${TRAINING_NAME:-}" ]]; then
    echo "ERROR: TRAINING_NAME is required. Example: TRAINING_NAME=multi_image bash scripts/run_train_robotwin_cosmos3nano.sh" >&2
    exit 1
fi
if [[ "$TRAINING_NAME" == */* ]]; then
    echo "ERROR: TRAINING_NAME must be a directory name, not a path: $TRAINING_NAME" >&2
    exit 1
fi
export OUTPUT_BASE_ROOT="${OUTPUT_BASE_ROOT:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_BASE_ROOT/$TRAINING_NAME}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/robotwin_lerobot_action_sft_nano.toml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-cosmos3}"
WANDB_GROUP="${WANDB_GROUP:-robotwin_lerobot_action_sft}"
WANDB_NAME="${WANDB_NAME:-cosmos3nano_robotwin_n${NNODES}x${NPROC_PER_NODE}_$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/wandb_api_key}"
LOG_FILENAME="${LOG_FILENAME:-robotwin_lerobot_action_sft_nano_n${NNODES}_rank${NODE_RANK}.log}"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/$LOG_FILENAME"
mkdir -p "$LOG_DIR"

if [[ "$WANDB_MODE" == "disabled" ]]; then
    echo "ERROR: WANDB_MODE=disabled is not allowed for this launcher." >&2
    exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" && -f "$WANDB_API_KEY_FILE" ]]; then
    export WANDB_API_KEY
    WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_API_KEY_FILE")"
fi

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_MODE=online requires WANDB_API_KEY or WANDB_API_KEY_FILE=$WANDB_API_KEY_FILE" >&2
    exit 1
fi

if [[ "$WANDB_MODE" == "online" ]]; then
    command -v wandb >/dev/null 2>&1 || { echo "ERROR: wandb is not installed." >&2; exit 1; }
    wandb login "$WANDB_API_KEY" >/dev/null
fi

[[ -f "$TOML_FILE" ]] || { echo "ERROR: TOML not found: $TOML_FILE" >&2; exit 1; }
[[ -d "$DATASET_PATH" ]] || { echo "ERROR: DATASET_PATH not found: $DATASET_PATH" >&2; exit 1; }
[[ -f "$DATASET_PATH/meta/info.json" ]] || { echo "ERROR: missing $DATASET_PATH/meta/info.json" >&2; exit 1; }
[[ -d "$BASE_CHECKPOINT_PATH" ]] || { echo "ERROR: BASE_CHECKPOINT_PATH not found: $BASE_CHECKPOINT_PATH" >&2; exit 1; }
[[ -d "$BASE_CHECKPOINT_PATH/model" ]] || { echo "ERROR: missing DCP model dir: $BASE_CHECKPOINT_PATH/model" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH" >&2; exit 1; }
[[ -d "$QWEN_TOKENIZER_PATH" ]] || { echo "ERROR: QWEN_TOKENIZER_PATH not found: $QWEN_TOKENIZER_PATH" >&2; exit 1; }
[[ -f "$QWEN_TOKENIZER_PATH/tokenizer_config.json" || -f "$QWEN_TOKENIZER_PATH/vocab.json" ]] || { echo "ERROR: QWEN_TOKENIZER_PATH does not look like a tokenizer snapshot: $QWEN_TOKENIZER_PATH" >&2; exit 1; }

TRAIN_MAX_ITER="${MAX_ITER:-10000}"
OBSERVATION_IMAGE_MODE="${OBSERVATION_IMAGE_MODE:-concat}"
if [[ "$OBSERVATION_IMAGE_MODE" != "concat" && "$OBSERVATION_IMAGE_MODE" != "multi_image" ]]; then
    echo "ERROR: OBSERVATION_IMAGE_MODE must be 'concat' or 'multi_image', got: $OBSERVATION_IMAGE_MODE" >&2
    exit 1
fi
USE_CAMERA_SPECIAL_TOKENS=false
if [[ "$OBSERVATION_IMAGE_MODE" == "multi_image" ]]; then
    USE_CAMERA_SPECIAL_TOKENS=true
fi

OVERRIDES=(
    "job.wandb_mode=$WANDB_MODE"
    "job.project=$WANDB_PROJECT"
    "job.group=$WANDB_GROUP"
    "job.name=$WANDB_NAME"
    "model.config.vlm_config.model_name=$QWEN_TOKENIZER_PATH"
    "model.config.vlm_config.tokenizer.pretrained_model_name=$QWEN_TOKENIZER_PATH"
    "trainer.max_iter=$TRAIN_MAX_ITER"
    "scheduler.cycle_lengths=[$TRAIN_MAX_ITER]"
    "checkpoint.save_iter=${SAVE_ITER:-1000}"
    "dataloader_train.dataloader.datasets.robotwin.dataset.mode=${ROBOTWIN_MODE:-policy}"
    "dataloader_train.dataloader.datasets.robotwin.dataset.observation_image_mode=$OBSERVATION_IMAGE_MODE"
    "dataloader_train.dataloader.datasets.robotwin.dataset.use_state=${ROBOTWIN_USE_STATE:-true}"
    "dataloader_train.dataloader.datasets.robotwin.dataset.state_key=${ROBOTWIN_STATE_KEY:-observation.state}"
    "model.config.use_camera_special_tokens=$USE_CAMERA_SPECIAL_TOKENS"
)

if [[ -n "${OPTIMIZER_LR:-}" ]]; then
    OVERRIDES+=("optimizer.lr=$OPTIMIZER_LR")
fi

if [[ -n "${SCHEDULER_F_MAX:-}" ]]; then
    OVERRIDES+=("scheduler.f_max=[$SCHEDULER_F_MAX]")
fi

if [[ -n "${SCHEDULER_F_MIN:-}" ]]; then
    OVERRIDES+=("scheduler.f_min=[$SCHEDULER_F_MIN]")
fi

if [[ -n "${SCHEDULER_WARM_UP_STEPS:-}" ]]; then
    OVERRIDES+=("scheduler.warm_up_steps=[$SCHEDULER_WARM_UP_STEPS]")
fi

if [[ "$OBSERVATION_IMAGE_MODE" == "multi_image" ]]; then
    OVERRIDES+=("optimizer.keys_to_select=[moe_gen,time_embedder,vae2llm,llm2vae,action,extra_weight]")
fi

if [[ -n "${MAX_EPISODES+x}" ]]; then
    OVERRIDES+=("dataloader_train.dataloader.datasets.robotwin.dataset.max_episodes=$MAX_EPISODES")
else
    OVERRIDES+=("dataloader_train.dataloader.datasets.robotwin.dataset.max_episodes=null")
fi

if [[ -n "${ROBOTWIN_VIEWPOINT:-}" ]]; then
    OVERRIDES+=("dataloader_train.dataloader.datasets.robotwin.dataset.viewpoint=$ROBOTWIN_VIEWPOINT")
fi

if [[ -n "${ROBOTWIN_TASK_INDEX:-}" ]]; then
    OVERRIDES+=("dataloader_train.dataloader.datasets.robotwin.dataset.task_index=$ROBOTWIN_TASK_INDEX")
else
    OVERRIDES+=("dataloader_train.dataloader.datasets.robotwin.dataset.task_index=null")
fi

if [[ -n "${ROBOTWIN_TASK_NAME:-}" ]]; then
    OVERRIDES+=("dataloader_train.dataloader.datasets.robotwin.dataset.task_name=$ROBOTWIN_TASK_NAME")
else
    OVERRIDES+=("dataloader_train.dataloader.datasets.robotwin.dataset.task_name=null")
fi

if [[ -n "${ROBOTWIN_EPISODE_INDICES:-}" ]]; then
    # 接受 "0,1,2" 或 "0 1 2" 或 "[0,1,2]" 多种写法，统一规范成 [0,1,2]。
    _ep_clean="${ROBOTWIN_EPISODE_INDICES//[\[\]]/}"
    _ep_clean="${_ep_clean// /,}"
    while [[ "$_ep_clean" == *",,"* ]]; do _ep_clean="${_ep_clean//,,/,}"; done
    _ep_clean="${_ep_clean#,}"
    _ep_clean="${_ep_clean%,}"
    OVERRIDES+=("dataloader_train.dataloader.datasets.robotwin.dataset.episode_indices=[${_ep_clean}]")
else
    OVERRIDES+=("dataloader_train.dataloader.datasets.robotwin.dataset.episode_indices=null")
fi

OVERRIDES+=("dataloader_train.dataloader.datasets.robotwin.dataset.dataset_repeat=${ROBOTWIN_DATASET_REPEAT:-1}")

OVERRIDES+=("${EXTRA_OVERRIDES[@]}")

echo "=========================================="
echo "Cosmos3-Nano RoboTwin multi-node training"
echo "=========================================="
echo "WORKDIR=$WORKDIR"
echo "NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC_PER_NODE=$NPROC_PER_NODE"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "DATASET_PATH=$DATASET_PATH"
echo "OBSERVATION_IMAGE_MODE=$OBSERVATION_IMAGE_MODE"
echo "ROBOTWIN_TASK_INDEX=${ROBOTWIN_TASK_INDEX:-}"
echo "ROBOTWIN_TASK_NAME=${ROBOTWIN_TASK_NAME:-}"
echo "ROBOTWIN_EPISODE_INDICES=${ROBOTWIN_EPISODE_INDICES:-}"
echo "ROBOTWIN_DATASET_REPEAT=${ROBOTWIN_DATASET_REPEAT:-1}"
echo "USE_CAMERA_SPECIAL_TOKENS=$USE_CAMERA_SPECIAL_TOKENS"
echo "COSMOS_MODELS_ROOT=$COSMOS_MODELS_ROOT"
echo "HF_HOME=$HF_HOME"
echo "BASE_CHECKPOINT_PATH=$BASE_CHECKPOINT_PATH"
echo "WAN_VAE_PATH=$WAN_VAE_PATH"
echo "QWEN_TOKENIZER_PATH=$QWEN_TOKENIZER_PATH"
echo "TOML_FILE=$TOML_FILE"
echo "TRAINING_NAME=$TRAINING_NAME"
echo "OUTPUT_BASE_ROOT=$OUTPUT_BASE_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "WANDB_PROJECT=$WANDB_PROJECT WANDB_GROUP=$WANDB_GROUP WANDB_NAME=$WANDB_NAME WANDB_MODE=$WANDB_MODE"
echo "ROBOTWIN_USE_STATE=${ROBOTWIN_USE_STATE:-true}"
echo "ROBOTWIN_STATE_KEY=${ROBOTWIN_STATE_KEY:-observation.state}"
echo "OPTIMIZER_LR=${OPTIMIZER_LR:-}"
echo "SCHEDULER_F_MAX=${SCHEDULER_F_MAX:-}"
echo "SCHEDULER_F_MIN=${SCHEDULER_F_MIN:-}"
echo "SCHEDULER_WARM_UP_STEPS=${SCHEDULER_WARM_UP_STEPS:-}"
echo "LOG_FILE=$LOG_FILE"
echo "OVERRIDES=${OVERRIDES[*]}"

env | grep -E '^(MASTER|NNODES|NODE_RANK|NPROC_PER_NODE|NCCL|UCX|CUDA|LD_LIBRARY_PATH|PYTORCH_CUDA_ALLOC_CONF|VLLM|WANDB|HF_|HUGGINGFACE|TRANSFORMERS|http_proxy|https_proxy|HTTP_PROXY|HTTPS_PROXY|no_proxy|NO_PROXY|COSMOS_MODELS_ROOT|BASE_CHECKPOINT_PATH|WAN_VAE_PATH|QWEN_TOKENIZER_PATH|TRAINING_NAME|OUTPUT_BASE_ROOT|IMAGINAIRE_OUTPUT_ROOT|OBSERVATION_IMAGE_MODE|ROBOTWIN_TASK_INDEX|ROBOTWIN_TASK_NAME|ROBOTWIN_EPISODE_INDICES|ROBOTWIN_DATASET_REPEAT|ROBOTWIN_USE_STATE|ROBOTWIN_STATE_KEY|OPTIMIZER_LR|SCHEDULER_F_MAX|SCHEDULER_F_MIN|SCHEDULER_WARM_UP_STEPS)=' | sort || true

PYTHONPATH=. torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$NPROC_PER_NODE" \
    -m cosmos_framework.scripts.train \
    --sft-toml="$TOML_FILE" \
    -- "${OVERRIDES[@]}" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit "$EXIT_CODE"
