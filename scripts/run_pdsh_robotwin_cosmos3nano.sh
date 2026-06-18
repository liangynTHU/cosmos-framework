#!/usr/bin/env bash
set -euo pipefail
set -x

normalize_node_list() {
    local raw_list="$1"
    printf '%s' "$raw_list" \
        | tr '[:space:]' ',' \
        | tr ';' ',' \
        | sed -E 's/,+/,/g; s/^,//; s/,$//' \
        | awk -F',' '
            BEGIN { out = "" }
            {
                for (i = 1; i <= NF; i++) {
                    item = $i
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", item)
                    if (item == "") {
                        continue
                    }
                    if (item ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+$/) {
                        sub(/:[0-9]+$/, "", item)
                    }
                    if (!(item in seen)) {
                        seen[item] = 1
                        out = (out == "" ? item : out "," item)
                    }
                }
            }
            END { print out }
        '
}

count_nodes() {
    local node_list="$1"
    if [[ -z "$node_list" ]]; then
        echo 0
        return
    fi
    awk -F',' '{ print NF }' <<< "$node_list"
}

usage() {
    cat <<'USAGE'
Usage:
  NODE_IP_LIST=ip1,ip2 bash scripts/run_pdsh_robotwin_cosmos3nano.sh [extra_overrides...]

This script launches Cosmos3-Nano RoboTwin/LeRobot action SFT on multiple nodes via pdsh.
It calls scripts/run_train_robotwin_cosmos3nano.sh on every node and assigns NODE_RANK automatically.

Required or auto-filled:
  NODE_IP_LIST / EXP_IP_LIST    Comma/space/semicolon separated node IPs or hostnames.
  NNODES                       Number of nodes. Defaults to node-list length.
  MASTER_ADDR                  Rank-0 node address. Defaults to first node.

Common environment variables:
  COSMOS_WORKDIR               Repo path on every node, default /mnt/lyn/workspace/wam.
  CONDA_SH                     Conda profile script.
  CONDA_ENV                    Conda env name, default cosmos3.
  TRAINING_NAME                Required run directory name under OUTPUT_BASE_ROOT.
  OUTPUT_BASE_ROOT             Base output root, default /apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano.
  NPROC_PER_NODE               GPUs per node, default 8.
  MASTER_PORT                  torchrun master port, default 50021.
  DATASET_PATH                 RoboTwin/LeRobot dataset root.
  COSMOS_MODELS_ROOT           Local model/cache root shared by all nodes.
  BASE_CHECKPOINT_PATH         Cosmos3-Nano DCP checkpoint path.
  WAN_VAE_PATH                 Wan VAE checkpoint path.
  QWEN_TOKENIZER_PATH          Local Qwen/Qwen3-VL-8B-Instruct tokenizer/config path.
  OUTPUT_ROOT                  Resolved output root. Defaults to OUTPUT_BASE_ROOT/TRAINING_NAME.
  WANDB_PROJECT/GROUP/NAME     W&B metadata.
  WANDB_MODE                   Default online. disabled is rejected by the node-local launcher.
  WANDB_API_KEY_FILE           W&B API key file when WANDB_API_KEY is unset.
  MAX_ITER/SAVE_ITER           Training length and checkpoint interval.
  MAX_EPISODES                 Dataset episode limit. Unset means use all data if supported.
  ROBOTWIN_TASK_INDEX          Optional RoboTwin task_index filter. Keeps all episodes from this task.
  ROBOTWIN_TASK_NAME           Optional RoboTwin task name filter. Must match meta/tasks.jsonl exactly.
  ROBOTWIN_EPISODE_INDICES     Optional comma-separated episode indices, e.g. "0,1,2,3,4,5". When set, all valid windows in each listed episode are used (recommended for full-trajectory overfit).
  ROBOTWIN_DATASET_REPEAT      Repeat filtered dataset length, default 1. Useful for one-task overfit.
  ROBOTWIN_USE_STATE           Whether to prepend qpos/proprio state as the first action row, default true.
  ROBOTWIN_STATE_KEY           qpos/proprio parquet field, default observation.state.
  OBSERVATION_IMAGE_MODE       concat or multi_image. concat keeps the legacy single concatenated observation.
  ROBOTWIN_VIEWPOINT           Optional dataset viewpoint override. Set to concat_view for the three-camera baseline.
  OPTIMIZER_LR                 Optional optimizer base lr override.
  SCHEDULER_F_MAX/F_MIN        Optional scheduler multiplier overrides.
  SCHEDULER_WARM_UP_STEPS      Optional scheduler warm-up override.

Any positional args are passed as extra OmegaConf overrides after the defaults.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

SCRIPT_ARGS=("$@")
SCRIPT_ARGS_STR="${SCRIPT_ARGS[*]}"

COSMOS_WORKDIR="${COSMOS_WORKDIR:-/mnt/lyn/workspace/wam}"
CONDA_SH="${CONDA_SH:-/jizhicfs/peterrao/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cosmos3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/run_train_robotwin_cosmos3nano.sh}"

if [[ -z "${NODE_IP_LIST:-}" && -z "${EXP_IP_LIST:-}" ]]; then
    if [[ -f "$COSMOS_WORKDIR/scripts/auto_node_discovery.sh" ]]; then
        source "$COSMOS_WORKDIR/scripts/auto_node_discovery.sh"
    elif [[ -f "/mnt/lyn/workspace/FastWAM/scripts/auto_node_discovery.sh" ]]; then
        source "/mnt/lyn/workspace/FastWAM/scripts/auto_node_discovery.sh"
    else
        echo "ERROR: NODE_IP_LIST is empty and no auto_node_discovery.sh was found." >&2
        echo "Example: NODE_IP_LIST=ip1,ip2 NNODES=2 MASTER_ADDR=ip1 bash scripts/run_pdsh_robotwin_cosmos3nano.sh" >&2
        exit 1
    fi
fi

NODE_IP_LIST="$(normalize_node_list "${NODE_IP_LIST:-${EXP_IP_LIST:-}}")"
EXP_IP_LIST="$(normalize_node_list "${EXP_IP_LIST:-$NODE_IP_LIST}")"
DISCOVERED_NNODES="$(count_nodes "$EXP_IP_LIST")"

if [[ -z "$EXP_IP_LIST" || "$DISCOVERED_NNODES" -eq 0 ]]; then
    echo "ERROR: node list is empty." >&2
    exit 1
fi

NNODES="${NNODES:-$DISCOVERED_NNODES}"
if [[ "$DISCOVERED_NNODES" -ne "$NNODES" ]]; then
    echo "ERROR: node-list count ($DISCOVERED_NNODES) != NNODES ($NNODES)." >&2
    echo "EXP_IP_LIST=$EXP_IP_LIST" >&2
    exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-${EXP_IP_LIST%%,*}}"
MASTER_PORT="${MASTER_PORT:-50021}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
WANDB_NAME="${WANDB_NAME:-cosmos3nano_robotwin_${NNODES}n${NPROC_PER_NODE}g_$RUN_STAMP}"

if [[ -z "${TRAINING_NAME:-}" ]]; then
    echo "ERROR: TRAINING_NAME is required. Example: TRAINING_NAME=multi_image bash scripts/run_pdsh_robotwin_cosmos3nano.sh" >&2
    exit 1
fi
if [[ "$TRAINING_NAME" == */* ]]; then
    echo "ERROR: TRAINING_NAME must be a directory name, not a path: $TRAINING_NAME" >&2
    exit 1
fi
OUTPUT_BASE_ROOT="${OUTPUT_BASE_ROOT:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_BASE_ROOT/$TRAINING_NAME}"

export NODE_IP_LIST EXP_IP_LIST NNODES MASTER_ADDR MASTER_PORT NPROC_PER_NODE WANDB_NAME TRAINING_NAME OUTPUT_BASE_ROOT OUTPUT_ROOT

echo "=========================================="
echo "Cosmos3-Nano RoboTwin pdsh launcher"
echo "=========================================="
echo "COSMOS_WORKDIR=$COSMOS_WORKDIR"
echo "EXP_IP_LIST=$EXP_IP_LIST"
echo "NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NPROC_PER_NODE=$NPROC_PER_NODE"
echo "TRAINING_NAME=$TRAINING_NAME"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "WANDB_NAME=$WANDB_NAME"
echo "EXTRA_OVERRIDES=$SCRIPT_ARGS_STR"

IFS=',' read -ra IP_ARRAY <<< "$EXP_IP_LIST"

for i in "${!IP_ARRAY[@]}"; do
    NODE_IP="${IP_ARRAY[i]}"
    NODE_RANK="$i"

    echo "Starting node $NODE_RANK ($NODE_IP)..."

    REMOTE_ENV_ARGS=(
        "COSMOS_WORKDIR=$COSMOS_WORKDIR"
        "NNODES=$NNODES"
        "NODE_RANK=$NODE_RANK"
        "MASTER_ADDR=$MASTER_ADDR"
        "MASTER_PORT=$MASTER_PORT"
        "NPROC_PER_NODE=$NPROC_PER_NODE"
        "TRAINING_NAME=$TRAINING_NAME"
        "OUTPUT_BASE_ROOT=$OUTPUT_BASE_ROOT"
        "COSMOS_MODELS_ROOT=${COSMOS_MODELS_ROOT:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models}"
        "HF_HOME=${HF_HOME:-${COSMOS_MODELS_ROOT:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models}/hf_cache}"
        "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${COSMOS_MODELS_ROOT:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models}/hf_cache}/hub}"
        "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME:-${COSMOS_MODELS_ROOT:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models}/hf_cache}/transformers}"
        "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}"
        "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
        "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}"
        "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}"
        "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}"
        "http_proxy=${http_proxy:-http://star-proxy.oa.com:3128}"
        "https_proxy=${https_proxy:-http://star-proxy.oa.com:3128}"
        "HTTP_PROXY=${HTTP_PROXY:-${http_proxy:-http://star-proxy.oa.com:3128}}"
        "HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy:-http://star-proxy.oa.com:3128}}"
        "no_proxy=${no_proxy:-localhost,127.0.0.1,::1,10.0.0.0/8,29.0.0.0/8}"
        "NO_PROXY=${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1,10.0.0.0/8,29.0.0.0/8}}"
        "DATASET_PATH=${DATASET_PATH:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0}"
        "BASE_CHECKPOINT_PATH=${BASE_CHECKPOINT_PATH:-${COSMOS_MODELS_ROOT:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models}/Cosmos3-Nano}"
        "WAN_VAE_PATH=${WAN_VAE_PATH:-${COSMOS_MODELS_ROOT:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models}/wan22_vae/Wan2.2_VAE.pth}"
        "QWEN_TOKENIZER_PATH=${QWEN_TOKENIZER_PATH:-${COSMOS_MODELS_ROOT:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models}/Qwen/Qwen3-VL-8B-Instruct}"
        "OUTPUT_ROOT=$OUTPUT_ROOT"
        "WANDB_PROJECT=${WANDB_PROJECT:-cosmos3}"
        "WANDB_GROUP=${WANDB_GROUP:-robotwin_lerobot_action_sft}"
        "WANDB_NAME=$WANDB_NAME"
        "WANDB_MODE=${WANDB_MODE:-online}"
        "WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/wandb_api_key}"
        "MAX_ITER=${MAX_ITER:-10000}"
        "SAVE_ITER=${SAVE_ITER:-1000}"
        "OBSERVATION_IMAGE_MODE=${OBSERVATION_IMAGE_MODE:-concat}"
        "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}"
        "NCCL_IB_SL=${NCCL_IB_SL:-3}"
        "NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE:-1}"
        "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}"
        "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"
        "NCCL_LL_THRESHOLD=${NCCL_LL_THRESHOLD:-16384}"
        "NCCL_IB_CUDA_SUPPORT=${NCCL_IB_CUDA_SUPPORT:-0}"
        "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}"
        "UCX_NET_DEVICES=${UCX_NET_DEVICES:-bond1}"
        "NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}"
        "NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}"
        "SHARP_COLL_ENABLE_SAT=${SHARP_COLL_ENABLE_SAT:-0}"
        "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}"
        "NCCL_NET_GDR_READ=${NCCL_NET_GDR_READ:-0}"
        "NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-0}"
        "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}"
        "NCCL_IB_TC=${NCCL_IB_TC:-160}"
        "NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-0}"
        "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
        "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
        "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
        "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
        "TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}"
        "TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}"
        "TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
        "TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
        "NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}"
    )

    if [[ -n "${MAX_EPISODES+x}" ]]; then
        REMOTE_ENV_ARGS+=("MAX_EPISODES=$MAX_EPISODES")
    fi

    if [[ -n "${ROBOTWIN_TASK_INDEX:-}" ]]; then
        REMOTE_ENV_ARGS+=("ROBOTWIN_TASK_INDEX=$ROBOTWIN_TASK_INDEX")
    fi

    if [[ -n "${ROBOTWIN_TASK_NAME:-}" ]]; then
        REMOTE_ENV_ARGS+=("ROBOTWIN_TASK_NAME=$ROBOTWIN_TASK_NAME")
    fi

    if [[ -n "${ROBOTWIN_EPISODE_INDICES:-}" ]]; then
        REMOTE_ENV_ARGS+=("ROBOTWIN_EPISODE_INDICES=$ROBOTWIN_EPISODE_INDICES")
    fi

    if [[ -n "${ROBOTWIN_DATASET_REPEAT:-}" ]]; then
        REMOTE_ENV_ARGS+=("ROBOTWIN_DATASET_REPEAT=$ROBOTWIN_DATASET_REPEAT")
    fi

    if [[ -n "${ROBOTWIN_USE_STATE:-}" ]]; then
        REMOTE_ENV_ARGS+=("ROBOTWIN_USE_STATE=$ROBOTWIN_USE_STATE")
    fi

    if [[ -n "${ROBOTWIN_STATE_KEY:-}" ]]; then
        REMOTE_ENV_ARGS+=("ROBOTWIN_STATE_KEY=$ROBOTWIN_STATE_KEY")
    fi

    if [[ -n "${ROBOTWIN_VIEWPOINT:-}" ]]; then
        REMOTE_ENV_ARGS+=("ROBOTWIN_VIEWPOINT=$ROBOTWIN_VIEWPOINT")
    fi

    if [[ -n "${OPTIMIZER_LR:-}" ]]; then
        REMOTE_ENV_ARGS+=("OPTIMIZER_LR=$OPTIMIZER_LR")
    fi

    if [[ -n "${SCHEDULER_F_MAX:-}" ]]; then
        REMOTE_ENV_ARGS+=("SCHEDULER_F_MAX=$SCHEDULER_F_MAX")
    fi

    if [[ -n "${SCHEDULER_F_MIN:-}" ]]; then
        REMOTE_ENV_ARGS+=("SCHEDULER_F_MIN=$SCHEDULER_F_MIN")
    fi

    if [[ -n "${SCHEDULER_WARM_UP_STEPS:-}" ]]; then
        REMOTE_ENV_ARGS+=("SCHEDULER_WARM_UP_STEPS=$SCHEDULER_WARM_UP_STEPS")
    fi

    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        REMOTE_ENV_ARGS+=("WANDB_API_KEY=$WANDB_API_KEY")
    fi

    REMOTE_EXPORTS="$(printf 'export %q; ' "${REMOTE_ENV_ARGS[@]}")"
    REMOTE_SCRIPT_PATH="$(printf '%q' "$TRAIN_SCRIPT")"
    REMOTE_SCRIPT_ARGS=(
        "--nnodes" "$NNODES"
        "--node-rank" "$NODE_RANK"
        "--master-addr" "$MASTER_ADDR"
        "--master-port" "$MASTER_PORT"
        "--"
        "${SCRIPT_ARGS[@]}"
    )
    REMOTE_SCRIPT_ARGS_STR="$(printf ' %q' "${REMOTE_SCRIPT_ARGS[@]}")"

    pdsh -w "$NODE_IP" \
        "cd $(printf '%q' "$COSMOS_WORKDIR"); \
         . $(printf '%q' "$CONDA_SH"); \
         conda activate $(printf '%q' "$CONDA_ENV"); \
         ${REMOTE_EXPORTS} \
         echo \"REMOTE CHECK: NODE_RANK=\$NODE_RANK NNODES=\$NNODES MASTER_ADDR=\$MASTER_ADDR MASTER_PORT=\$MASTER_PORT NPROC_PER_NODE=\$NPROC_PER_NODE TRAINING_NAME=\$TRAINING_NAME OUTPUT_ROOT=\$OUTPUT_ROOT WANDB_NAME=\$WANDB_NAME WANDB_MODE=\$WANDB_MODE ROBOTWIN_TASK_INDEX=\${ROBOTWIN_TASK_INDEX:-} ROBOTWIN_TASK_NAME=\${ROBOTWIN_TASK_NAME:-} ROBOTWIN_EPISODE_INDICES=\${ROBOTWIN_EPISODE_INDICES:-} ROBOTWIN_DATASET_REPEAT=\${ROBOTWIN_DATASET_REPEAT:-1} ROBOTWIN_USE_STATE=\${ROBOTWIN_USE_STATE:-true} ROBOTWIN_STATE_KEY=\${ROBOTWIN_STATE_KEY:-observation.state} OPTIMIZER_LR=\${OPTIMIZER_LR:-} SCHEDULER_F_MAX=\${SCHEDULER_F_MAX:-} SCHEDULER_F_MIN=\${SCHEDULER_F_MIN:-} SCHEDULER_WARM_UP_STEPS=\${SCHEDULER_WARM_UP_STEPS:-} COSMOS_MODELS_ROOT=\$COSMOS_MODELS_ROOT HF_HOME=\$HF_HOME BASE_CHECKPOINT_PATH=\$BASE_CHECKPOINT_PATH WAN_VAE_PATH=\$WAN_VAE_PATH QWEN_TOKENIZER_PATH=\$QWEN_TOKENIZER_PATH NCCL_SOCKET_IFNAME=\$NCCL_SOCKET_IFNAME NCCL_IB_HCA=\$NCCL_IB_HCA\"; \
         exec bash ${REMOTE_SCRIPT_PATH}${REMOTE_SCRIPT_ARGS_STR}" &
done

wait

echo ""
echo "✅ Cosmos3-Nano RoboTwin multi-node training finished or exited on all nodes."
echo "=========================================="
