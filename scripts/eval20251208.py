#!/usr/bin/env python
# eval.py
"""
Unsloth 4bit Base Model + LoRA + vLLM で openai/gsm8k を評価するスクリプト。
"""

import argparse
import json
import os
from collections import Counter
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from mymath_verify import verify_math_answer, MathVerifyConfig, MathVerifyResult

from transformers import AutoTokenizer

WANDB_PROJECT = "qwen3-4b-gsm8k-100"
WANDB_ENTITY = "mssfj-1"
WANDB_RUNNAME = "qwen3-4b-openmathinst2-sft-10000"

MODEL_NAME = "unsloth/Qwen3-4B-bnb-4bit"

def extract_gsm8k_gold_answer(answer_text: str) -> str:
    lines = [ln.strip() for ln in answer_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "####" in ln:
            after = ln.split("####", 1)[1].strip()
            return after
    return lines[-1] if lines else ""

# チャットテンプレートを適用するためトークナイザを定義する
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def build_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a careful mathematical problem solver."},
        {"role": "user", "content": f"Solve the following problem step by step.\nProblem:\n{question}\nOutput the answer in the format: Final Answer: <number>"}
    ]
    # トークナイザーのテンプレートを適用
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def evaluate_gsm8k_with_vllm(
    model_name: str,
    lora_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    output_path: Optional[str] = None,
    wandb_run=None,
    wandb_log_artifacts: bool = False,
) -> Dict[str, Any]:
    
    # --- 修正1: ここでトークナイザーをロードします ---
    print(f"Loading Tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # ----------------------------------------------

    # データ読み込み
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Loading 4-bit Quantized Base Model: {model_name}")

    if lora_path:
        print(f"Enabling LoRA with adapter: {lora_path}")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        enable_lora=(lora_path is not None),
        max_lora_rank=32 if lora_path else 16,
    )

    # ★重要: Stop Tokenの設定変更（前回の指摘事項）
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=None, # "Final Answer:" で止まらないように削除
    )
    
    gold_answers: List[str] = []
    raw_questions: List[str] = []
    prompts: List[str] = []  # ★ここも初期化が必要です（前回の指摘事項）

    for ex in ds:
        q = ex["question"]
        gold_full = ex["answer"]
        gold = extract_gsm8k_gold_answer(gold_full)
        raw_questions.append(q)
        gold_answers.append(gold)
        
        # --- 修正2: ここで tokenizer を渡します ---
        prompts.append(build_prompt(q, tokenizer)) 
        # ----------------------------------------

    print("Running vLLM generation...")
    
    lora_request = None
    if lora_path:
        lora_request = LoRARequest("adapter", 1, lora_path)

    outputs: List[Any] = []
    # vLLM は内部でスケジューリングしてくれる
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

    if wandb_run is not None:
        log_payload = {
            "eval/em": em,
            "eval/num_correct": num_correct,
            "eval/num_total": num_total,
        }
        for reason_key, reason_count in reason_counter.items():
            log_payload[f"eval/reason/{reason_key}"] = reason_count
        wandb_run.log(log_payload)

        if wandb_log_artifacts and output_path and os.path.exists(output_path):
            import wandb

            artifact = wandb.Artifact("gsm8k_eval_outputs", type="evaluation")
            artifact.add_file(output_path)
            summary_path = output_path + ".summary.json"
            if os.path.exists(summary_path):
                artifact.add_file(summary_path)
            wandb_run.log_artifact(artifact)

    return result_summary

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Hugging Face 4-bit base model name.",
    )
    p.add_argument(
        "--lora-path",
        type=str,
        default="/workspace/model/qwen3_sft_lora_openmathinst2-10000/",
        #default="",
        help="Path to the LoRA adapter.",
    )
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="vLLM batch size (passed to max_num_seqs).",
    )
    p.add_argument("--output-path", type=str, default="/workspace/outputs/gsm8k_eval.jsonl")
    p.add_argument("--wandb-project", type=str, default=f"{WANDB_PROJECT}", help="W&B project name. If not set, wandb is disabled.")
    p.add_argument("--wandb-entity", type=str, default=f"{WANDB_ENTITY}", help="W&B entity/user.")
    p.add_argument("--wandb-run-name", type=str, default=f"{WANDB_RUNNAME}", help="Optional W&B run name.")
    p.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Set to online/offline to enable W&B logging. Default is online.",
    )
    p.add_argument(
        "--wandb-log-artifacts",
        action="store_true",
        help="Log evaluation outputs as W&B artifacts (requires --wandb-project).",
    )
    return p.parse_args()

def init_wandb(args: argparse.Namespace):
    if args.wandb_mode == "disabled" or not args.wandb_project:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed but W&B logging was requested.") from exc

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "mode": args.wandb_mode,
        "config": {
            "model_name": args.model_name,
            "lora_path": args.lora_path,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "output_path": args.output_path,
        },
    }
    # Remove None values to keep init clean
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    return wandb.init(**init_kwargs)

def main():
    args = parse_args()
    wandb_run = init_wandb(args)
    try:
        evaluate_gsm8k_with_vllm(
            model_name = args.model_name,
            lora_path = args.lora_path,
            max_samples = args.max_samples if args.max_samples > 0 else None,
            batch_size = args.batch_size,
            output_path = args.output_path,
            wandb_run = wandb_run,
            wandb_log_artifacts = args.wandb_log_artifacts,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

if __name__ == "__main__":
    main()
