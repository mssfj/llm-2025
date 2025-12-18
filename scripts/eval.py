#!/usr/bin/env python
# eval.py
"""
Unsloth 4bit Base Model + LoRA + vLLM で openai/gsm8k を評価するスクリプト。
scripts/sft-unsloth20251213.py で学習した LoRA／ベースモデルと、ベース単体を
model_preset で切り替えて評価できるようにした。


RUN examples:

  - Base only: python scripts/eval20251208.py --model-preset base
  - SFT LoRA (default paths): python scripts/eval20251208.py --model-preset sft
  - Custom combo: python scripts/eval20251208.py --model-preset custom --model-name <model> --lora-path <adapter> --output-path <file>

"""

import argparse
import json
import os
from collections import Counter
from typing import List, Dict, Any, Optional

from unsloth.chat_templates import get_chat_template

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from mymath_verify import verify_math_answer, MathVerifyConfig, MathVerifyResult

from transformers import AutoTokenizer

WANDB_PROJECT = "qwen3-4b-gsm8k-100"
WANDB_ENTITY = "mssfj-1"
WANDB_RUNNAME_SFT = "qwen3-4b-openmathinst2-structured"
WANDB_RUNNAME_BASE = "qwen3-4b-base"

DEFAULT_SFT_LORA_DIRNAME = "qwen3_sft_lora_openmathinst2-structured_1000"
DEFAULT_OUTPUT_DIR = "/workspace/outputs"

SYSTEM_PROMPT = (
    "You are given a math problem.\n"
    "First, think about the problem step by step and show your reasoning.\n"
    "Wrap all your reasoning between <think> and </think>.\n"
    "Then, output the final answer after Final Answer:.\n"
    "The final answer must be a concise expression (usually a single number)."
)

MODEL_PRESETS = {
    # Base (no LoRA)
    "base": {
        "model_name": "unsloth/Qwen3-4B-Base",
        "lora_path": None,
        "wandb_run_name": WANDB_RUNNAME_BASE,
        "output_path": f"{DEFAULT_OUTPUT_DIR}/gsm8k_eval_qwen3-4b-base.jsonl",
    },
    # SFT (LoRA) produced by scripts/sft-unsloth20251213.py
    "sft": {
        "model_name": "unsloth/Qwen3-4B-Base",
        "lora_path": f"/workspace/model/{DEFAULT_SFT_LORA_DIRNAME}/",
        "wandb_run_name": WANDB_RUNNAME_SFT,
        "output_path": f"{DEFAULT_OUTPUT_DIR}/gsm8k_eval_{WANDB_RUNNAME_SFT}.jsonl",
    },
}

def extract_gsm8k_gold_answer(answer_text: str) -> str:
    lines = [ln.strip() for ln in answer_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "####" in ln:
            after = ln.split("####", 1)[1].strip()
            return after
    return lines[-1] if lines else ""
'''
def build_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a careful mathematical problem solver."},
        {"role": "user", "content": f"Solve the following problem step by step.\nProblem:\n{question}\nOutput the answer in the format: Final Answer: <number>"},
    ]

    # Unsloth's optimized chat template for Qwen 2.5
    tokenizer = get_chat_template(
        tokenizer,
    #    chat_template = "qwen-2.5",
    #    #mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )
    # トークナイザーのテンプレートを適用
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
'''

def build_prompt(question: str, tokenizer) -> str:
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    system = (
        "You are a careful mathematical problem solver.\n"
        "You MUST follow the required output format exactly."
    )

    user = f"""Solve the following problem.

Problem:
{question}

Required output format (MUST follow exactly):
<think>
<analyze>...</analyze>
<plan>...</plan>
<verify>...</verify>
<reason>...</reason>
</think>
Final Answer: <number>

Rules:
- The output MUST start with <think> and end with the Final Answer line.
- Do not omit <think> or any of the tags.
- Do not output any extra tags like <Final Answer>.
- Put only the final numeric answer after 'Final Answer:'.
"""

    # 生成を <think> から始めるため、assistant のプレフィックスを入れる
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "<think>\n"},  # ←これが効く
    ]

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

    # ★重要: Stop Tokenの設定
    stop_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id >= 0:
        stop_ids.append(im_end_id)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048,
        stop_token_ids=stop_ids,
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
    config = MathVerifyConfig(use_exact=True, use_numeric=True, use_sympy=True, require_final_answer=True)
    num_correct = 0
    num_total = len(outputs)
    reason_counter: Counter = Counter()
    detailed_results: List[Dict[str, Any]] = []
    model_output_lengths: List[int] = []

    for i, (out, q, gold) in enumerate(zip(outputs, raw_questions, gold_answers)):
        if not out.outputs:
            pred_text = ""
        else:
            pred_text = out.outputs[0].text

        model_output_len = len(pred_text)
        model_output_lengths.append(model_output_len)

        res: MathVerifyResult = verify_math_answer(pred_text, gold, config=config)
        if res.is_correct:
            num_correct += 1
        reason_counter[res.reason] += 1

        detailed_results.append({
            "index": i, "question": q, "gold_answer": gold, "model_output": pred_text,
            "model_output_length": model_output_len,
            "extracted_pred_answer": res.pred_answer, "is_correct": res.is_correct, "reason": res.reason
        })

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{num_total} samples")

    em = num_correct / max(num_total, 1)
    total_model_output_chars = sum(model_output_lengths)
    avg_model_output_chars = total_model_output_chars / num_total if num_total > 0 else 0.0
    max_model_output_chars = max(model_output_lengths) if model_output_lengths else 0

    print(f"\n==== Evaluation Result ====")
    print(f"Base Model : {model_name}")
    print(f"LoRA Path: {lora_path}")
    print(f"EM: {em:.4f}")
    print(f"Total model_output chars: {total_model_output_chars}")
    print(f"Avg model_output chars: {avg_model_output_chars:.2f}")
    print(f"Max model_output chars: {max_model_output_chars}")

    result_summary = {
        "model_name": model_name, "lora_path": lora_path, "num_samples": num_total,
        "num_correct": num_correct, "em": em, "reason_counts": dict(reason_counter),
        "model_output_char_total": total_model_output_chars,
        "model_output_char_avg": avg_model_output_chars,
        "model_output_char_max": max_model_output_chars,
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

def resolve_model_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Choose (model_name, lora_path, output_path, wandb_run_name) from preset
    or command line overrides so we can quickly switch between base and SFT.
    """
    preset = MODEL_PRESETS.get(args.model_preset, {})

    model_name = args.model_name or preset.get("model_name")
    lora_path = args.lora_path if args.lora_path is not None else preset.get("lora_path")
    output_path = args.output_path or preset.get("output_path")
    wandb_run_name = args.wandb_run_name or preset.get("wandb_run_name")

    if not model_name:
        raise ValueError("model_name must be provided either via --model-preset or --model-name.")

    # Empty string disables LoRA
    lora_path = lora_path if lora_path else None

    if not output_path:
        preset_key = args.model_preset or "custom"
        output_path = f"{DEFAULT_OUTPUT_DIR}/gsm8k_eval_{preset_key}.jsonl"

    return {
        "model_name": model_name,
        "lora_path": lora_path,
        "output_path": output_path,
        "wandb_run_name": wandb_run_name,
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-preset",
        choices=[*MODEL_PRESETS.keys(), "custom"],
        default="sft",
        help="Switch between evaluating the base model ('base') or the SFT LoRA produced by sft-unsloth20251213.py ('sft'). "
             "Use 'custom' to rely solely on --model-name/--lora-path overrides.",
    )
    p.add_argument("--model-name", type=str, default=None, help="Override base model name (used directly when --model-preset=custom).")
    p.add_argument("--lora-path", type=str, default=None, help="Override LoRA adapter path. Leave empty to disable LoRA.")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16, help="vLLM batch size (passed to max_num_seqs).")
    p.add_argument("--output-path", type=str, default=None, help="Where to write evaluation outputs. Defaults depend on the preset.")
    p.add_argument("--wandb-project", type=str, default=f"{WANDB_PROJECT}", help="W&B project name.")
    p.add_argument("--wandb-entity", type=str, default=f"{WANDB_ENTITY}", help="W&B entity/user.")
    p.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name. Defaults depend on the preset.")
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

    resolved = resolve_model_config(args)
    args.model_name = resolved["model_name"]
    args.lora_path = resolved["lora_path"]
    args.output_path = resolved["output_path"]
    args.wandb_run_name = resolved["wandb_run_name"]

    print(f"[Eval Preset] {args.model_preset}")
    print(f" - model_name: {args.model_name}")
    print(f" - lora_path: {args.lora_path}")
    print(f" - output_path: {args.output_path}")

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
