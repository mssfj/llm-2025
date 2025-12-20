import re
import torch
import numpy as np
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from mymath_verify import verify_math_answer, MathVerifyConfig, extract_final_answer
from transformers import AutoTokenizer

# --- 1. Configuration ---
MAX_SEQ_LENGTH = 3500
MIN_COMPLETION_LENGTH = 512
LORA_RANK = 32
SEED = 3407
MODEL_NAME = "unsloth/Qwen3-4B-Base"
LORA_DIR = "/workspace/model/qwen3_sft_lora_openmathinst2-structured_1000"
VLLM_GPU_MEMORY_UTILIZATION = 0.6

MODEL_DIR = "/workspace/model/qwen3_4b_grpo_saved_lora"
OUTPUT_DIR = "/workspace/output/"

SYSTEM_PROMPT = ("""
You are a careful mathematical problem solver.
You MUST follow the required output format exactly.

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
- Put only the final numeric answer after 'Final Answer: '.

In <analyze>, restate the problem in your own words and identify:
- the given quantities,
- what is being asked,
- any constraints or implicit assumptions.

Do NOT perform calculations.
Do NOT outline solution steps.
Focus only on understanding and formalizing the problem.

In <plan>, describe the logical steps required to solve the problem.

- Write the steps at a high level.
- Do NOT carry out arithmetic or algebra.
- Do NOT include intermediate or final results.
- Each step should describe *what* will be done, not *the result*.

The plan must be sufficient for another solver to reproduce the solution.

In <verify>, explain why the planned solution is valid.

- Check that all given information is used exactly once.
- Confirm that no assumptions contradict the problem.
- Explain why the method guarantees a unique correct answer.

Do NOT redo the calculation.
Do NOT introduce new steps.

In <reason>, carry out the full logical reasoning and calculations.

- Follow the steps described in <plan>.
- Show intermediate calculations clearly.
- Keep the reasoning concise and linear.
- Do not add commentary unrelated to solving the problem.

Do NOT include the final answer statement here.

After </reason>, output exactly one line:

Final Answer: <number>

- No extra text.
- No units unless explicitly required.
- Use plain numerals.

""")

model_name = LORA_DIR if LORA_DIR else MODEL_NAME

# --- 2. Model & Tokenizer Setup ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = LORA_RANK,
    gpu_memory_utilization = VLLM_GPU_MEMORY_UTILIZATION,
    fix_tokenizer = False,
)

if LORA_DIR:
    print(f"Loading SFT LoRA adapter with Unsloth from: {LORA_DIR}")
else:
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = LORA_RANK * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = SEED,
    )

#Unsloth's optimized chat template for Qwen 2.5
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
    #mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
)

stop_token_ids = []
if tokenizer.eos_token_id is not None:
    stop_token_ids.append(tokenizer.eos_token_id)
im_end_token = "<|im_end|>"
if hasattr(tokenizer, "get_vocab") and im_end_token in tokenizer.get_vocab():
    im_end_id = tokenizer.convert_tokens_to_ids(im_end_token)
else:
    im_end_id = tokenizer.convert_tokens_to_ids(im_end_token)
    if tokenizer.convert_ids_to_tokens(im_end_id) != im_end_token:
        im_end_id = None
if im_end_id is not None and im_end_id not in stop_token_ids:
    stop_token_ids.append(im_end_id)

# --- 3. Reward Functions ---
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

_reasoning_pattern = re.compile(
    r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE
)

_math_verify_config = MathVerifyConfig(require_final_answer=True)

def reward_math_verify(completions, answer=None, **kwargs):
    """
    math-verify correctness reward。

    - correctness: math-verify + フォールバック一致
    """
    if answer is None:
        return [0.0] * len(completions)

    # ハイパーパラメータ（あとで調整用）
    R_CORRECT = 5.0
    R_INCORRECT = -2.0

    rewards = []

    for comp, truth in zip(completions, answer):
        text = _extract_completion_text(comp)

        # ===== 1. correctness =====
        is_correct = False

        gold_answer = extract_final_answer(str(truth))
        if not gold_answer:
            gold_answer = str(truth)
        result = verify_math_answer(text, gold_answer, config=_math_verify_config)
        is_correct = result.is_correct
        
        print(f"TEXT:{text}\n GOLD_ANSWER:{gold_answer}\n RESULT:{result}\n IS_CORRECT:{is_correct}\n")

        correctness_reward = R_CORRECT if is_correct else R_INCORRECT
        rewards.append(correctness_reward)
        
    return rewards


def reward_reasoning_length(completions, **kwargs):
    """
    reasoning（<think> ... </think>）の長さを
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

def max_length_check(input_max_len):
    max_completion_length = MAX_SEQ_LENGTH - (input_max_len + 1)
    if max_completion_length < MIN_COMPLETION_LENGTH:
        max_completion_length = MIN_COMPLETION_LENGTH
    max_prompt_length = MAX_SEQ_LENGTH - max_completion_length
    print(
        "Using lengths -> "
        f"max_prompt_length={max_prompt_length}, "
        f"max_completion_length={max_completion_length}"
    )
    return max_prompt_length

dataset, input_max_len = prepare_dataset()
print(f"Dataset prepared. Max input length: {input_max_len}")

max_prompt_length = max_length_check(input_max_len)
max_completion_length = MAX_SEQ_LENGTH - (input_max_len + 1)

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
    num_generations=2, # メモリ不足なら減らす
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=5, # テスト用に短く設定されています
    save_steps=100,
    report_to="none",
    vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, # VLLM用のメモリ確保
    vllm_sampling_params=SamplingParams(
        min_p=0.1, top_p=1.0, top_k=-1, seed=SEED,
        max_tokens=max_completion_length,
        stop_token_ids=stop_token_ids,
    ),
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_math_verify, reward_reasoning_length],
	args=training_args,
    train_dataset=dataset,
)

print("Starting training...")
trainer.train()

# --- 6. Save ---
model.save_lora(f"{MODEL_DIR}")
print(f"Model saved to {MODEL_DIR}")
