#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-GPU GRPO on GSM8K with Qwen3-8B-Instruct + Unsloth

前提:
  uv add unsloth trl "datasets>=2.19" accelerate transformers peft
  GPU: A100 40GB or RTX 5090 32GB クラス
"""

import re
from decimal import Decimal, InvalidOperation
from typing import List

import sys
import types
import torch
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer


# ===== 1. Dataset: GSM8K を [prompt, answer] 形式にする =====

SYSTEM_PROMPT = """You are a helpful math tutor.
Solve the problem step by step and give the final answer as a number.

Format:
Reasoning:
  (your reasoning here)
Final answer: <number>
"""

def extract_hash_answer(text: str) -> str:
    # GSM8K: ".... #### 12" の形式
    if "####" not in text:
        return text.strip()
    return text.split("####")[-1].strip()

def get_gsm8k_questions(split: str = "train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


# ===== 2. Reward 関数: 最後に出てきた数値 == 正解なら +2.0 =====

num_pattern = re.compile(r"-?\d+(?:\.\d+)?")

def parse_number(s: str) -> Decimal | None:
    m = num_pattern.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return Decimal(m.group(0))
    except InvalidOperation:
        return None

def correctness_reward_func(
    prompts,
    completions,
    answer: List[str],
    **kwargs,
) -> List[float]:
    """
    TRL/Unsloth-GRPO 仕様に合わせた reward 関数。
    completions: List[List[{"content": str, "role": "assistant"}]]
    answer: gold 数値 (文字列, GSM8K から)
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards: List[float] = []

    for resp, gold_str in zip(responses, answer):
        pred_num = parse_number(resp)
        gold_num = parse_number(gold_str)

        if pred_num is not None and gold_num is not None and pred_num == gold_num:
            rewards.append(2.0)
        else:
            rewards.append(0.0)

    return rewards


# ===== 3. モデル: Qwen3-8B-Instruct を Unsloth 4bit + LoRA でロード =====

def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen3-8B",
    max_seq_length: int = 1024,
    lora_rank: int = 32,
):
    """
    Unsloth FastLanguageModel + GRPO パッチ
    """
    # GRPO 用に FastLanguageModel をパッチ
    PatchFastRL("GRPO", FastLanguageModel)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,            # QLoRA
        fast_inference=False,         # まずは vLLM 無しでシンプルに
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,   # OOM したら 0.5 以下に落とす
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    return model, tokenizer


# ===== 4. GRPO 設定 & Trainer =====

def main():
    torch.cuda.set_device(0)

    max_seq_length = 1024
    max_prompt_length = 256
    max_completion_length = max_seq_length - max_prompt_length

    # データ (まずは train の先頭 100 件くらいで動作確認するのが現実的)
    raw_dataset = get_gsm8k_questions(split="train")
    dataset = raw_dataset.select(range(100))   # 動いたらここを増やす

    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen3-8B",
        max_seq_length=max_seq_length,
        lora_rank=32,
    )

    training_args = GRPOConfig(
        # 典型的な設定 (Unsloth / HF LLM Course をベース)
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",

        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,

        num_generations=4,  # 1問につき 4 サンプル (OOM したら 2 に下げる)
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,

        max_steps=200,      # まずは 200 step 程度で挙動確認
        save_steps=200,
        max_grad_norm=0.1,
        report_to="none",   # wandb は切る。Unsloth のログで十分
        output_dir="qwen3-8b-gsm8k-grpo",

        # 最初は vLLM は使わない。安定してから use_vllm=True を検討。
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # LoRA だけ保存
    model.save_lora("qwen3-8b-gsm8k-grpo-lora")


if __name__ == "__main__":
    main()

