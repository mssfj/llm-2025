#!/usr/bin/env python
# eval.py
"""
Qwen/Qwen3-8B + vLLM で openai/gsm8k を解かせ、
math_verify で正答判定して EM を出す評価スクリプト。

前提:
  - math_verify.py が同じディレクトリにある
  - `uv add vllm datasets sympy` 済み
  - GPU 上で実行すること
"""

import argparse
import json
from collections import Counter
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from vllm import LLM, SamplingParams

from math_verify import verify_math_answer, MathVerifyConfig, MathVerifyResult


def extract_gsm8k_gold_answer(answer_text: str) -> str:
    """
    GSM8K の 'answer' フィールドから最終答えだけを抜き出す。
    フォーマット例:
      "... 解説 ...\n#### 24"
    → "24"
    """
    lines = [ln.strip() for ln in answer_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "####" in ln:
            # "#### 24" のような形式を想定
            after = ln.split("####", 1)[1].strip()
            return after
    # 念のため最後の行を返すが、本来はほぼ来ない
    return lines[-1] if lines else ""


def build_prompt(question: str) -> str:
    """
    Qwen 用のプロンプト。
    最終行に "Final answer: <number>" 形式で出させる。
    math_verify 側のパターンとも整合させる。
    """
    return (
        "You are a careful mathematical problem solver.\n"
        "Solve the following problem step by step.\n"
        "Then on the last line, output only the final answer in the format:\n"
        "Final answer: <number>\n\n"
        f"Problem:\n{question}\n"
    )


def evaluate_gsm8k_with_vllm(
    model_name: str,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    # データ読み込み
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Loaded GSM8K test split: {len(ds)} samples")

    # vLLM モデル読み込み
    print(f"Loading model with vLLM: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,       # 単GPU
        max_model_len=2048,           # ★ ここが重要：KVキャッシュを小さくする
        gpu_memory_utilization=0.90,  # 必要なら 0.92〜0.95 まで上げてもよい
        # dtype="bfloat16",           # 必要なら明示（Qwen はだいたい fp16/bf16対応） 
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
    # vLLM は一括で投げられる
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)

    config = MathVerifyConfig(
        use_exact=True,
        use_numeric=True,
        use_sympy=True,
    )

    num_correct = 0
    num_total = len(outputs)
    reason_counter: Counter = Counter()
    detailed_results: List[Dict[str, Any]] = []

    for i, (out, q, gold) in enumerate(zip(outputs, raw_questions, gold_answers)):
        # vLLM の出力構造: RequestOutput -> list[RequestOutput]
        # 各 RequestOutput.outputs[0].text が生成テキスト
        if not out.outputs:
            pred_text = ""
        else:
            pred_text = out.outputs[0].text

        res: MathVerifyResult = verify_math_answer(pred_text, gold, config=config)
        if res.is_correct:
            num_correct += 1
        reason_counter[res.reason] += 1

        detailed_results.append(
            {
                "index": i,
                "question": q,
                "gold_answer": gold,
                "model_output": pred_text,
                "extracted_pred_answer": res.pred_answer,
                "is_correct": res.is_correct,
                "reason": res.reason,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{num_total} samples")

    em = num_correct / max(num_total, 1)
    print(f"\n==== Evaluation Result ====")
    print(f"Model: {model_name}")
    print(f"Samples: {num_total}")
    print(f"Exact-match EM (math-verify based): {em:.4f}")
    print("Reason breakdown:")
    for k, v in reason_counter.most_common():
        print(f"  {k}: {v}")

    result_summary = {
        "model_name": model_name,
        "num_samples": num_total,
        "num_correct": num_correct,
        "em": em,
        "reason_counts": dict(reason_counter),
    }

    if output_path is not None:
        print(f"\nSaving detailed results to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for row in detailed_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # サマリーも別ファイルに
        summary_path = output_path + ".summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result_summary, f, ensure_ascii=False, indent=2)

    return result_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Hugging Face model name to use with vLLM.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max number of GSM8K test samples to evaluate (None for all).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for vLLM generation (currently not used directly; vLLM handles batching internally).",
    )
    p.add_argument(
        "--output-path",
        type=str,
        default="/workspace/logs/gsm8k_qwen3_8b_eval.jsonl",
        help="Path to save per-sample results as JSONL.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    evaluate_gsm8k_with_vllm(
        model_name=args.model_name,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        batch_size=args.batch_size,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()

