#!/usr/bin/env bash
set -euo pipefail
set -x

usage() {
    cat <<'USAGE'
Usage: bash scripts/prefetch_robotwin_cosmos3nano_models.sh

Downloads/stages the assets required by Cosmos3-Nano RoboTwin training:
  1. Wan2.2_VAE.pth from Wan-AI/Wan2.2-TI2V-5B
  2. Cosmos3-Nano converted to PyTorch Distributed Checkpoint (DCP)
  3. Qwen/Qwen3-VL-8B-Instruct tokenizer/config files in the HF cache

Environment variables:
  COSMOS_MODELS_ROOT      Target model root. Default: /apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models
  COSMOS_WORKDIR          Cosmos framework repo path. Default: parent of this script.
  HF_TOKEN                Hugging Face token if required by gated repos.
  HF_ENDPOINT             Optional HF endpoint/mirror.
  http_proxy/https_proxy  Optional network proxy. Defaults match FastWAM.
  SKIP_COSMOS_DCP         Set to 1 to skip Cosmos3-Nano DCP conversion.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

WORKDIR="${COSMOS_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$WORKDIR"

export COSMOS_MODELS_ROOT="${COSMOS_MODELS_ROOT:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models}"
export HF_HOME="${HF_HOME:-$COSMOS_MODELS_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1,10.0.0.0/8,29.0.0.0/8}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

WAN_VAE_DIR="$COSMOS_MODELS_ROOT/wan22_vae"
WAN_VAE_PATH="$WAN_VAE_DIR/Wan2.2_VAE.pth"
BASE_CHECKPOINT_PATH="$COSMOS_MODELS_ROOT/Cosmos3-Nano"
QWEN_LINK_DIR="$COSMOS_MODELS_ROOT/Qwen"
QWEN_LINK_PATH="$QWEN_LINK_DIR/Qwen3-VL-8B-Instruct"

mkdir -p "$COSMOS_MODELS_ROOT" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$WAN_VAE_DIR" "$QWEN_LINK_DIR"

python - <<'PY'
import importlib.util
missing = [pkg for pkg in ("huggingface_hub",) if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}. Please install them in the cosmos3 env first.")
PY

if [[ ! -s "$WAN_VAE_PATH" ]]; then
    python - <<PY
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    filename="Wan2.2_VAE.pth",
    local_dir="$WAN_VAE_DIR",
    local_dir_use_symlinks=False,
    token=None,
)
print(path)
PY
else
    echo "Wan VAE already exists: $WAN_VAE_PATH"
fi

python - <<PY
from huggingface_hub import snapshot_download
import os
snap = snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    cache_dir=os.path.join("$HF_HOME", "hub"),
    ignore_patterns=[
        "*.safetensors", "*.bin", "*.pth", "*.pt", "*.ckpt",
        "model-*.safetensors", "pytorch_model*", "consolidated*",
    ],
    token=os.environ.get("HF_TOKEN") or None,
)
print(snap)
os.makedirs("$QWEN_LINK_DIR", exist_ok=True)
if os.path.islink("$QWEN_LINK_PATH") or os.path.exists("$QWEN_LINK_PATH"):
    if os.path.islink("$QWEN_LINK_PATH"):
        os.unlink("$QWEN_LINK_PATH")
else:
    pass
if not os.path.exists("$QWEN_LINK_PATH"):
    os.symlink(snap, "$QWEN_LINK_PATH")
print("QWEN_TOKENIZER_PATH=$QWEN_LINK_PATH")
PY

if [[ "${SKIP_COSMOS_DCP:-0}" != "1" ]]; then
    if [[ ! -d "$BASE_CHECKPOINT_PATH/model" ]]; then
        PYTHONPATH=. python -m cosmos_framework.scripts.convert_model_to_dcp \
            -o "$BASE_CHECKPOINT_PATH" \
            --checkpoint-path Cosmos3-Nano
    else
        echo "Cosmos3-Nano DCP already exists: $BASE_CHECKPOINT_PATH"
    fi
fi

cat > "$COSMOS_MODELS_ROOT/robotwin_cosmos3nano_env.sh" <<EOF
export COSMOS_MODELS_ROOT="$COSMOS_MODELS_ROOT"
export HF_HOME="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export HF_HUB_DISABLE_XET="1"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH"
export WAN_VAE_PATH="$WAN_VAE_PATH"
export QWEN_TOKENIZER_PATH="$QWEN_LINK_PATH"
EOF

echo "=========================================="
echo "Cosmos3-Nano RoboTwin assets are ready"
echo "=========================================="
echo "COSMOS_MODELS_ROOT=$COSMOS_MODELS_ROOT"
echo "HF_HOME=$HF_HOME"
echo "BASE_CHECKPOINT_PATH=$BASE_CHECKPOINT_PATH"
echo "WAN_VAE_PATH=$WAN_VAE_PATH"
echo "QWEN_TOKENIZER_PATH=$QWEN_LINK_PATH"
echo "Env file: $COSMOS_MODELS_ROOT/robotwin_cosmos3nano_env.sh"
