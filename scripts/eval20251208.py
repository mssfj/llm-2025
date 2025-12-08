#!/usr/bin/env python
# eval.py
"""
Unsloth 4bit Base Model + LoRA + vLLM で openai/gsm8k を評価するスクリプト。

変更点:
  - quantization="bitsandbytes" を追加 (Unsloth 4bit対応)
  - load_format="bitsandbytes" を追加
"""

import argparse
import json
from collections import Counter
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from mymath_verify import verify_math_answer, MathVerifyConfig, MathVerifyResult


def extract_gsm8k_gold_answer(answer_text: str) -> str:
    lines = [ln.strip() for ln in answer_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "####" in ln:
            after = ln.split("####", 1)[1].strip()
            return after
    return lines[-1] if lines else ""

def build_prompt(question: str) -> str:
    return (
        "You are a careful mathematical problem solver.\n"
        "Solve the following problem step by step.\n"
        "Then on the last line, output only the final answer in the format:\n"
        " <number>\n\n"
        f"Problem:\n{question}\n"
    )

def evaluate_gsm8k_with_vllm(
    model_name: str,
    lora_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    # データ読み込み
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Loaded GSM8K test split: {len(ds)} samples")
    print(f"Loading 4-bit Quantized Base Model: {model_name}")
    
    if lora_path:
        print(f"Enabling LoRA with adapter: {lora_path}")

    # ★修正箇所: 4bit (bitsandbytes) 設定を追加
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        
        # ★Unsloth 4bit対応の肝
        quantization="bitsandbytes", 
        load_format="bitsandbytes",

        enforce_eager=True,

        # 4bit化でメモリに余裕ができるため、0.9でもOOMしにくいですが、
        # 安全を見て0.8程度にしておくと他のプロセスと共存しやすいです
        gpu_memory_utilization=0.3,
        
        enable_lora=(lora_path is not None),
        max_lora_rank=64 if lora_path else 16,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
    )
    
    prompts: List[str] = []
    gold_answers: List[str] = []
    raw_questions: List[str] = []

    for ex in ds:
        q = ex["question"]
        gold_full = ex["answer"]
        gold = extract_gsm8k_gold_answer(gold_full)
        raw_questions.append(q)
        gold_answers.append(gold)
        prompts.append(build_prompt(q))

    print("Running vLLM generation...")
    
    lora_request = None
    if lora_path:
        lora_request = LoRARequest("adapter", 1, lora_path)

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # --- 以下評価ロジックは同じ ---
    config = MathVerifyConfig(use_exact=True, use_numeric=True, use_sympy=True)
    num_correct = 0
    num_total = len(outputs)
    reason_counter: Counter = Counter()
    detailed_results: List[Dict[str, Any]] = []

    for i, (out, q, gold) in enumerate(zip(outputs, raw_questions, gold_answers)):
        if not out.outputs:
            pred_text = ""
        else:
            pred_text = out.outputs[0].text

        res: MathVerifyResult = verify_math_answer(pred_text, gold, config=config)
        if res.is_correct:
            num_correct += 1
        reason_counter[res.reason] += 1

        detailed_results.append({
            "index": i, "question": q, "gold_answer": gold, "model_output": pred_text,
            "extracted_pred_answer": res.pred_answer, "is_correct": res.is_correct, "reason": res.reason
        })

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{num_total} samples")

    em = num_correct / max(num_total, 1)
    print(f"\n==== Evaluation Result ====")
    print(f"Base Model (4bit): {model_name}")
    print(f"LoRA Path: {lora_path}")
    print(f"EM: {em:.4f}")

    result_summary = {
        "model_name": model_name, "lora_path": lora_path, "num_samples": num_total,
        "num_correct": num_correct, "em": em, "reason_counts": dict(reason_counter),
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for row in detailed_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(output_path + ".summary.json", "w", encoding="utf-8") as f:
            json.dump(result_summary, f, ensure_ascii=False, indent=2)

    return result_summary

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-name",
        type=str,
        # ★重要: ここにはUnslothの「4bit版」モデル名を指定します
        # 例: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
        default="unsloth/Qwen3-4B-bnb-4bit",
        help="Hugging Face 4-bit base model name.",
    )
    p.add_argument(
        "--lora-path",
        type=str,
        default="/workspace/model/qwen3_4b_dapo_sft_lora/",
        #default="/workspace/llm-2025/model/grpo_saved_lora",
        help="Path to the LoRA adapter.",
    )
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output-path", type=str, default="/workspace/outputs/gsm8k_eval.jsonl")
    return p.parse_args()

def main():
    args = parse_args()
    evaluate_gsm8k_with_vllm(
        model_name=args.model_name,
        lora_path=args.lora_path,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        batch_size=args.batch_size,
        output_path=args.output_path,
    )

if __name__ == "__main__":
    main()
