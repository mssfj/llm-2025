import re
import torch
import numpy as np
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

# --- 1. Configuration ---
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
SEED = 3407
MODEL_NAME = "Qwen3/Qwen3-4B-Base" # ※元のコードはQwen3となっていましたが、一般的には2.5かInstruct系を使います。適宜修正してください。

# 思考プロセス用のタグ定義
XML_TAGS = {
    "reasoning_start": "<start_working_out>",
    "reasoning_end": "<end_working_out>",
    "solution_start": "<SOLUTION>",
    "solution_end": "</SOLUTION>"
}

SYSTEM_PROMPT = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {XML_TAGS['reasoning_start']} and {XML_TAGS['reasoning_end']}.
Then, provide your solution between {XML_TAGS['solution_start']}{XML_TAGS['solution_end']}"""

# --- 2. Model & Tokenizer Setup ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen3/Qwen3-4B-Base", # 元コード準拠ならここを合わせる
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = LORA_RANK,
    gpu_memory_utilization = 0.66,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = LORA_RANK * 2,
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
)

# Chat Templateの適用
tokenizer.chat_template = tokenizer.chat_template \
    .replace("'{system_prompt}'", f"'{SYSTEM_PROMPT}'") \
    .replace("'{reasoning_start}'", f"'{XML_TAGS['reasoning_start']}'")

# --- 3. Reward Functions ---
# 正規表現のコンパイル（高速化のため外出し）
solution_pattern = re.compile(
    rf"{XML_TAGS['reasoning_end']}.*?{XML_TAGS['solution_start']}(.+?)(?:{XML_TAGS['solution_end']}|{re.escape(tokenizer.eos_token)})?[\s]*$",
    flags=re.MULTILINE | re.DOTALL
)
number_pattern = re.compile(
    XML_TAGS['solution_start'] + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL
)

def match_format_exactly(completions, **kwargs):
    """フォーマットが完全に一致しているか"""
    return [3.0 if solution_pattern.search(c[0]["content"]) else 0.0 for c in completions]

def match_format_approximately(completions, **kwargs):
    """タグが含まれているか（部分点）"""
    scores = []
    for c in completions:
        text = c[0]["content"]
        score = 0
        score += 0.5 if text.count(XML_TAGS['reasoning_end']) == 1 else -1.0
        score += 0.5 if text.count(XML_TAGS['solution_start']) == 1 else -1.0
        score += 0.5 if text.count(XML_TAGS['solution_end']) == 1 else -1.0
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    """正解テキストとの一致判定"""
    responses = [c[0]["content"] for c in completions]
    extracted = [m.group(1) if (m := solution_pattern.search(r)) else None for r in responses]
    
    scores = []
    for guess, truth in zip(extracted, answer):
        if guess is None:
            scores.append(-2.0)
            continue
        if guess == truth:
            scores.append(5.0)
        elif guess.strip() == truth.strip():
            scores.append(3.5)
        else:
            # 数値的な近さを判定
            try:
                ratio = float(guess) / float(truth)
                if 0.9 <= ratio <= 1.1: scores.append(2.0)
                elif 0.8 <= ratio <= 1.2: scores.append(1.5)
                else: scores.append(-2.5)
            except:
                scores.append(-4.5)
    return scores

def check_numbers(prompts, completions, answer, **kwargs):
    """数値としての正解判定"""
    responses = [c[0]["content"] for c in completions]
    extracted = [m.group(1) if (m := number_pattern.search(r)) else None for r in responses]
    
    scores = []
    for guess, truth in zip(extracted, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        try:
            val_truth = float(truth.strip())
            val_guess = float(guess.strip().replace(",", ""))
            scores.append(3.5 if val_guess == val_truth else -1.5)
        except:
            scores.append(0)
    return scores

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
    
    # 長すぎるデータをフィルタリング（90パーセンタイルでカット）
    tokenized_lengths = [len(tokenizer.apply_chat_template(p, add_generation_prompt=True)) for p in ds["prompt"]]
    max_len_cutoff = int(np.quantile(tokenized_lengths, 0.9))
    ds = ds.select([i for i, l in enumerate(tokenized_lengths) if l <= max_len_cutoff])
    
    return ds, max_len_cutoff

dataset, input_max_len = prepare_dataset()
print(f"Dataset prepared. Max input length: {input_max_len}")

# --- 5. Training ---
training_args = GRPOConfig(
    output_dir="outputs",
    learning_rate=5e-6,
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
    max_steps=100, # テスト用に短く設定されています
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
    reward_funcs=[match_format_exactly, match_format_approximately, check_answer, check_numbers],
    args=training_args,
    train_dataset=dataset,
)

print("Starting training...")
trainer.train()

# --- 6. Save ---
model.save_lora("grpo_saved_lora")
print("Model saved.")
