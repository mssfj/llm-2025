#!/usr/bin/env bash
set -euxo pipefail

# ==== 設定 ====
PROJECT_ROOT="/workspace/llm-2025"  # 好きなら変えていい

# ==== 0. 基本パッケージ ====
sudo apt-get update
sudo apt-get install -y \
  git wget curl build-essential \
  python3-dev python3-pip \
  pkg-config

# ==== 1. uv インストール ====
# ~/.local/bin に入るので PATH を通す
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ==== 2. プロジェクトディレクトリ ====
mkdir -p "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

# 既にpyprojectがあれば壊さないように一応確認
if [ ! -f pyproject.toml ]; then
  uv init --python 3.10 .
fi

# ==== 3. pyproject.toml を上書き（コンペ用構成） ====
cat > pyproject.toml << 'EOF'
[project]
name = "llmcomp"
version = "0.1.0"
description = "Matsuo Lab LLM Competition Pipeline"
requires-python = ">=3.10"

dependencies = [
    "transformers",
    "accelerate",
    "datasets",
    "trl",
    "peft",
    "bitsandbytes",
    "sentencepiece",
    "evaluate",
    "wandb",
    "tiktoken",
    "scikit-learn",
    "numpy",
    "einops",
    "unsloth",
    "setuptools",
]

[tool.uv]
dev-dependencies = [
    "pytest",
    "ipykernel",
]
EOF

# ==== 4. 依存インストール（Torch 以外） ====
# .venv を作ってそこに全部入る
uv sync

# ==== 5. PyTorch (CUDA 対応 wheel) ====
# ドライバが CUDA 12.4 でも、cu121 wheel は普通に動く
source .venv/bin/activate

# 必要に応じてバージョンは変えていいが、cu121 wheel を使うこと
pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "torchaudio==2.4.0"

# ==== 6. vLLM 等が必要ならここで追加 ====
# いったんコメントアウト。使うと決めた時に外せ。
# uv add vllm

# ==== 7. ディレクトリ構成 ====
mkdir -p logs checkpoints configs scripts data

# ==== 8. 動作確認 ====
python - << 'PYCODE'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
PYCODE

echo "=== setup done. ==="
echo "次回以降は:"
echo "  cd ${PROJECT_ROOT}"
echo "  source .venv/bin/activate"

