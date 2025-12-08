#!/usr/bin/env python
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import wandb
import random
import re

# ========= 設定 =========
MODEL_NAME = "unsloth/Qwen3-4B-Base"

DATASET_NAME = "open-r1/DAPO-Math-17k-Processed"
DATASET_SUBSET = "en"
DATASET_SPLIT = "train"

MAX_SEQ_LENGTH = 2048
WANDB_PROJECT = "math-sft-qwen3-4b-base"
WANDB_ENTITY = "mssfj-1"
WANDB_RUNNAME = "sft-qwen3-4b-dapo"
MODEL_DIR = "/workspace/model"
CHECKPOINT_DIR = "/workspace/checkpoints"
LOG_DIR = "/workspace/logs"

XML_TAGS = {
    "reasoning_start": "<start_working_out>",
    "reasoning_end": "<end_working_out>",
    "solution_start": "<SOLUTION>",
    "solution_end": "</SOLUTION>",
}

SYSTEM_PROMPT = (
    "You are given a math problem.\n"
    "First, think about the problem step by step and show your reasoning.\n"
    f"Wrap all your reasoning between {XML_TAGS['reasoning_start']} and {XML_TAGS['reasoning_end']}.\n"
    f"Then, output the final answer between {XML_TAGS['solution_start']}{XML_TAGS['solution_end']}.\n"
    "The final answer must be a concise expression (usually a single number)."
)

# ========= Model & Tokenizer (Unsloth) =========
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,        # 自動 (fp16 / bf16)
    load_in_4bit = True, # 4bit量子化
)

# ==== Chat template を自前で定義（add_generation_prompt は使わない）====
raw_chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] + eos_token }}"
    "{% elif message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)

tokenizer.chat_template = raw_chat_template
print("Custom chat_template set.")

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,  # LoRA rank（数学なら 32 くらいでいい）
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

print(f"Model loaded with Unsloth. Vocab size: {len(tokenizer)}")

# ========= データセット準備（DAPO-Math-17k） =========
raw_ds = load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)

# train / valid / test に分割
train_valid = raw_ds.train_test_split(test_size=0.1, seed=42)
valid_test = train_valid["test"].train_test_split(test_size=0.5, seed=42)

dataset_dict = DatasetDict({
    "train": train_valid["train"],
    "validation": valid_test["train"],
    "test": valid_test["test"],
})

# ========= 解答テキストから最終解を（ゆるく）抽出するヘルパ =========
def extract_final_answer(solution_text: str) -> str:
    text = str(solution_text).strip()

    # "Answer: 42" "Ans = 42" みたいなやつ
    m = re.search(r"(?:Answer|Ans|Final answer)\s*[:=]\s*([\-+]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # "#### 42" 用のパターン（GSM8K系も混ぜたくなる場合を想定）
    m = re.search(r"####\s*([\-+]?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)

    # 文末付近の数字を拾う
    nums = re.findall(r"([\-+]?\d+(?:\.\d+)?)", text)
    if nums:
        return nums[-1]

    # 何も取れなければ空（後で math-verify 側で対処）
    return ""


# ========= チャットテンプレート適用関数（数学用統一フォーマット） =========
def format_math_examples(examples):
    texts = []
    for prompt, solution in zip(examples["prompt"], examples["solution"]):
        question = str(prompt).strip()
        full_solution = str(solution).strip()
        final_answer = extract_final_answer(full_solution)

        reasoning_start = XML_TAGS["reasoning_start"]
        reasoning_end = XML_TAGS["reasoning_end"]
        sol_start = XML_TAGS["solution_start"]
        sol_end = XML_TAGS["solution_end"]

        assistant_content = (
            f"{reasoning_start}\n"
            f"{full_solution}\n"
            f"{reasoning_end}\n"
            f"{sol_start}{final_answer}{sol_end}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False,
        )
        text += tokenizer.eos_token
        texts.append(text)

    return {"text": texts}

dataset_dict = dataset_dict.map(format_math_examples, batched=True)

# ========= 推論テスト用関数 (Unsloth高速推論) =========
def generate_samples(model, tokenizer, dataset, num_samples=3):
    FastLanguageModel.for_inference(model)

    print("\n=== Generation Sample Check (Before Training) ===")
    indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    samples = dataset.select(indices)

    for sample in samples:
        question = sample["prompt"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True,  # ここで assistant 役の生成開始
        )

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 512,
                use_cache = True,
                do_sample = False,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        output_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print("=== Question ===")
        print(question)
        print("=== Model Output ===")
        print(output_text)
        print("------")

    FastLanguageModel.for_training(model)

# 学習前の動作確認
generate_samples(model, tokenizer, dataset_dict["test"])

# ========= 学習 (SFT) =========
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUNNAME)

sft_config = SFTConfig(
    output_dir = f"{CHECKPOINT_DIR}/qwen3_4b_dapo_sft",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-5,  # 8B QLoRA ならこの辺から
    num_train_epochs = 1,  # 最初は 1 epoch で様子見
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 10,
    eval_strategy = "steps",
    eval_steps = 100,
    save_strategy = "epoch",
    optim = "adamw_8bit",
    report_to = "wandb",
    seed = 3407,
)

sft_config.max_seq_length = MAX_SEQ_LENGTH
sft_config.dataset_text_field = "text"

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_dict["train"],
    eval_dataset = dataset_dict["validation"],
    args = sft_config,
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_text_field = "text",
)

print("Starting Unsloth SFT (math)...")
trainer.train()

# ========= 保存 =========
model.save_pretrained(f"{MODEL_DIR}/qwen3_4b_dapo_sft_lora")
tokenizer.save_pretrained(f"{MODEL_DIR}/qwen3_4b_dapo_sft_lora")
print("Training finished and model saved.")

