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

# ==== 3. pyproject.toml を上書き（コンペ用構成・グループ分け） ====
cat > pyproject.toml << 'EOF'
[project]
name = "llmcomp"
version = "0.1.0"
description = "Matsuo Lab LLM Competition Pipeline"
requires-python = ">=3.10"

# 共通で使うライブラリ（SFT/GRPO/Eval/Serve 全体で共有）
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

# 用途ごとのグループ
[project.optional-dependencies]
# SFT専用（Unsloth 等）
sft = [
]

# GRPO / 推論系（TRL + vLLM 等）
grpo = [
    "trl",
    "vllm",
]

# 評価系はここに追加していく想定（例: math-verify など）
eval = [
]

# 開発ツール
dev = [
    "pytest",
    "ipykernel",
]
EOF

# ==== 4. 依存インストール（Torch 以外） ====
# .venv を作ってそこに全部入る
# ベース + SFT + GRPO + 開発ツールを最初から入れておく
uv sync --group sft --group grpo --group dev

# ==== 5. PyTorch (CUDA 対応 wheel) ====
# ドライバが CUDA 12.4 でも、cu121 wheel は普通に動く
# uv 経由でプロジェクト環境(.venv)にインストールする
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "torchaudio==2.4.0"

# ==== 6. vLLM 等をあとから追加したい場合 ====
# 例:
#   uv add --group grpo "flash-attn"
#   uv add --group eval "math-verify"

# ==== 7. ディレクトリ構成 ====
mkdir -p ../logs ../checkpoints ../configs ../data ../model

# ==== 8. 動作確認 ====
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
echo "  # 例: uv run python scripts/grpo_train.py --config configs/grpo/qwen3_8b.yaml"

