#!/usr/bin/env bash
set -euxo pipefail

# ==== 設定 ====
PROJECT_ROOT="/workspace/llm-2025"

# ==== 0. 基本パッケージ ====
sudo apt-get update
sudo apt-get install -y \
  git wget curl build-essential \
  python3-dev python3-pip \
  pkg-config nodejs npm unzip

# ==== codexのインストール ====
npm i -g @openai/codex

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws configure

aws s3 cp s3://llm-train-dev/codex/auth.json ~/.codex/auth.json

# ==== 1. uv インストール ====
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ==== 2. プロジェクトディレクトリ ====
mkdir -p "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

if [ ! -f pyproject.toml ]; then
  uv init --python 3.10 .
fi

# ==== 3. pyproject.toml を上書き（コンペ用構成・グループ分け） ====
cat > pyproject.toml << 'EOF'
[project]
name = "llmcomp"
version = "0.1.0"
description = "Matsuo Lab LLM Competition Pipeline"
requires-python = ">=3.10"

# 共通依存（SFT / GRPO / Eval で共通）
dependencies = [
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "bitsandbytes",
    "sentencepiece",
    "evaluate",
    "wandb",
    "tiktoken",
    "scikit-learn",
    "numpy",
    "einops",
    "setuptools",
    "sympy",
    "unsloth",
]

# グループ依存（uv の新仕様）
[dependency-groups]
# SFT: 今後 SFT専用をここに追加
sft = [
]

# GRPO / RL / 推論
grpo = [
    "transformers==4.56.2",
    "trl==0.22.2",
    "vllm==0.10.2",
    "math-verify[antlr4_13_2]",
]

# 評価系（math-verify 等）
eval = [
]

# 開発用
dev = [
    "pytest",
    "ipykernel",
]
EOF

# ==== 4. 依存インストール（Torch 以外） ====
uv sync --group sft --group grpo --group dev

# ==== 5. PyTorch (CUDA 12.1 wheel) ====
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "torchaudio==2.4.0"

# ==== 6. ディレクトリ構成 ====
mkdir -p ../logs ../checkpoints ../configs ../data ../model

# ==== 7. 動作確認 ====
uv run python - << 'PYCODE'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
PYCODE

echo "=== setup done. ==="
echo "次回以降は:"
echo "  cd ${PROJECT_ROOT}"
echo "  uv run python your_script.py"

