import re
import torch
import numpy as np
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from math_verify import parse, verify
from transformers import AutoTokenizer

# --- 1. Configuration ---
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
SEED = 3407
# MODEL_NAME = "unsloth/Qwen3-4B-Base"
MODEL_NAME = "/workspace/model/qwen3_sft_lora_openmathinst2-structured_1000"

MODEL_DIR = "/workspace/model/qwen3_4b_grpo_saved_lora"
OUTPUT_DIR = "/workspace/output/"

# 思考プロセス用のタグ定義
XML_TAGS = {
    "reasoning_start": "reason",
    "reasoning_end": "Final Answer: ",
    "solution_start": "<SOLUTION>",
    "solution_end": "</SOLUTION>"
}

SYSTEM_PROMPT = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {XML_TAGS['reasoning_start']} and {XML_TAGS['reasoning_end']}.
"""

# --- 2. Model & Tokenizer Setup ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = LORA_RANK,
    gpu_memory_utilization = 0.3,
    fix_tokenizer = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = LORA_RANK * 2,
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
)

# 1. まず文字列としてテンプレートを定義（改行バックスラッシュではなく、カッコで囲む方式に変更してミスを防ぎます）
chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

# Replace with out specific template:
chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{SYSTEM_PROMPT}'")\
    .replace("'{reasoning_start}'", f"'{XML_TAGS['reasoning_start']}'")
tokenizer.chat_template = chat_template

print("Custom chat_template set.")

# --- 3. Reward Functions ---
# 正規表現のコンパイル（高速化のため外出し）
_reasoning_pattern = re.compile(
    rf"{XML_TAGS['reasoning_start']}(.*?){XML_TAGS['reasoning_end']}",
    flags=re.DOTALL
)

def _extract_completion_text(completion_obj):
    """
    vLLM + TRL の completions からプレーンテキストを取り出すヘルパー。
    - completions: List[completion] を想定
    - completion: List[{"role": ..., "content": ...}] または dict の場合を雑にケア
    """
    if isinstance(completion_obj, (list, tuple)) and len(completion_obj) > 0:
        first = completion_obj[0]
        if isinstance(first, dict) and "content" in first:
            return first["content"]
        return str(first)
    if isinstance(completion_obj, dict) and "content" in completion_obj:
        return completion_obj["content"]
    return str(completion_obj)

# 必要なら既存のものを置き換え／統一する
_reasoning_pattern = re.compile(
    r"<start_working_out>(.*?)<end_working_out>", re.DOTALL | re.IGNORECASE
)
_solution_pattern = re.compile(
    r"<SOLUTION>(.*?)</SOLUTION>", re.DOTALL | re.IGNORECASE
)

def _normalize_math_expr(s: str) -> str:
    """
    LaTeX ラッパを剥がし、前後のノイズを削る軽量正規化。
    math-verify に渡す前/フォールバック比較前に必ず通す。
    """
    if s is None:
        return ""

    s = str(s).strip()

    # よくある LaTeX ラッパを剥がす
    wrappers = [
        ("$$", "$$"),
        ("$", "$"),
        (r"\(", r"\)"),
        (r"\[", r"\]"),
        (r"\boxed{", "}"),
    ]
    changed = True
    # ネストにある程度耐えるため、剥がせる限りループ
    while changed:
        changed = False
        for left, right in wrappers:
            if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
                s = s[len(left) : -len(right)].strip()
                changed = True

    # カンマなど軽い前処理
    s = s.replace("，", ",").strip()

    return s


def _extract_sections(text: str):
    """
    completion から
    - solution 部分
    - SOLUTION の後ろにどれだけ余計なテキストがあるか
    を抽出するユーティリティ。
    """
    solution = None

    # solution 抽出
    m_s = _solution_pattern.search(text)
    if m_s:
        solution = m_s.group(1)

    # suffix 長さ
    if m_s:
        _, end = m_s.span()
        suffix_len = len(text[end:])
    else:
        # SOLUTION が無い場合は suffix を全文として扱う
        suffix_len = len(text)

    return solution, suffix_len


def reward_math_verify_improved(completions, answer=None, **kwargs):
    """
    math-verify correctness + hallucination をまとめた reward。

    - correctness: math-verify + フォールバック一致
    - hallucination: </SOLUTION> 以降の余計なテキストにペナルティ
    """
    if answer is None:
        return [0.0] * len(completions)

    # ハイパーパラメータ（あとで調整用）
    R_CORRECT = 5.0
    R_INCORRECT = -2.0

    HALLUC_PENALTY_SCALE = -0.1       # suffix_len / 100 * 係数

    rewards = []

    for comp, truth in zip(completions, answer):
        text = _extract_completion_text(comp)

        solution, suffix_len = _extract_sections(text)

        # ===== 1. correctness =====
        is_correct = False

        # gold / pred を正規化してから math-verify に渡す
        gold_str = _normalize_math_expr(truth)
        pred_str = _normalize_math_expr(solution if solution is not None else text)

        try:
            gold_parsed = parse(gold_str)
            pred_parsed = parse(pred_str)
            is_correct = bool(verify(gold_parsed, pred_parsed))
        except Exception:
            is_correct = False

        # フォールバック: プレーンテキスト一致 / 数値一致
        if not is_correct:
            if pred_str.strip() == gold_str.strip():
                is_correct = True
            else:
                try:
                    g = float(gold_str.replace(",", ""))
                    p = float(pred_str.replace(",", ""))
                    if abs(g - p) <= 1e-6:
                        is_correct = True
                except Exception:
                    pass

        correctness_reward = R_CORRECT if is_correct else R_INCORRECT

        # ===== 2. hallucination penalty (SOLUTION 後の余計な出力) =====
        # suffix_len を 100 文字単位でスケーリングしてペナルティ
        #   例: suffix_len = 250, HALLUC_PENALTY_SCALE = -0.1
        #        → -0.1 * (250/100) = -0.25
        halluc_penalty = 0.0
        if suffix_len > 0:
            halluc_penalty = HALLUC_PENALTY_SCALE * (suffix_len / 100.0)

        total_reward = correctness_reward + halluc_penalty
        rewards.append(total_reward)

    return rewards


def reward_reasoning_length(completions, **kwargs):
    """
    reasoning（<start_working_out> ... <end_working_out>）の長さを
    [L_min, L_max] に収めることを狙う長さペナルティ。

    - L < L_min  → 短すぎペナルティ
    - L > L_max  → 長すぎペナルティ
    - L_min <= L <= L_max → ペナルティ 0
    """
    rewards = []

    alpha = float(kwargs.pop("_alpha", 0.1)) if "_alpha" in kwargs else 0.1
    L_min = int(kwargs.pop("_L_min", 300)) if "_L_min" in kwargs else 300
    L_max = int(kwargs.pop("_L_max", 900)) if "_L_max" in kwargs else 900

    for comp in completions:
        text = _extract_completion_text(comp)

        m = _reasoning_pattern.search(text)
        if not m:
            # reasoning タグが欠落 → 一律軽ペナルティ
            rewards.append(-1.0 * alpha)
            continue

        reasoning_text = m.group(1)
        L = len(reasoning_text)

        if L < L_min:
            diff = L_min - L
            penalty = -alpha * (diff / 100.0)
        elif L > L_max:
            diff = L - L_max
            penalty = -alpha * (diff / 100.0)
        else:
            penalty = 0.0

        rewards.append(penalty)

    return rewards

# --- 4. Data Preparation ---
def prepare_dataset():
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    
    # プロンプト形式への変換
    ds = ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["prompt"]},
        ],
        "answer": x["solution"], # 必要に応じてハッシュ処理などを戻す
    })

    def _prompt_len(ex):
        text = tokenizer.apply_chat_template(
            ex["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt_len": len(tokenizer(text, add_special_tokens=False).input_ids)
        }

    ds = ds.map(_prompt_len)
    input_max_len = int(max(ds["prompt_len"])) if len(ds) > 0 else 0

    return ds, input_max_len

dataset, input_max_len = prepare_dataset()
print(f"Dataset prepared. Max input length: {input_max_len}")

# --- 5. Training ---
training_args = GRPOConfig(
    output_dir=f"{OUTPUT_DIR}",
    learning_rate=1e-5,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1, 
    num_generations=4, # メモリ不足なら減らす
    max_prompt_length=input_max_len + 1,
    max_completion_length=MAX_SEQ_LENGTH - (input_max_len + 1),
    max_steps=10, # テスト用に短く設定されています
    save_steps=100,
    report_to="none",
    vllm_gpu_memory_utilization=0.4, # VLLM用のメモリ確保
    vllm_sampling_params=SamplingParams(
        min_p=0.1, top_p=1.0, top_k=-1, seed=SEED,
        stop=[tokenizer.eos_token], include_stop_str_in_output=True
    ),
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_math_verify_improved, reward_reasoning_length],
	args=training_args,
    train_dataset=dataset,
)

print("Starting training...")
trainer.train()

# --- 6. Save ---
model.save_lora(f"{MODEL_DIR}")
print(f"Model saved to {MODEL_DIR}")
