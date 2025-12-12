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

DATASET_NAME = "mssfj/openmathinstruct-2_formatted"
DATASET_SUBSET = "en"
# データ転送を抑えるため、最初から 1 万件にスライスして取得
DATASET_SPLIT = "train[:10000]"
# 上で 1 万件に絞るので追加 select は不要（None）
DATASET_MAX_SAMPLES = None
# 訓練にはその中から 1000 件だけ使用
DATASET_TRAIN_SAMPLES = 1000
DATASET_ALLOWED_CATEGORIES = {"augumented_math", "math"}

MAX_SEQ_LENGTH = 2048
WANDB_PROJECT = "qwen3-4b-sft"
WANDB_ENTITY = "mssfj-1"
WANDB_RUNNAME = "qwen3-4b-openmathinst2-sft_1000"
MODEL_DIR = "/workspace/model"
CHECKPOINT_DIR = "/workspace/checkpoints"
LOG_DIR = "/workspace/logs"

XML_TAGS = {
    "reasoning_start": "<think>",
    "reasoning_end": "</think>",
    "final_answer": "Final Answer:",
}

SYSTEM_PROMPT = (
    "You are given a math problem.\n"
    "First, think about the problem step by step and show your reasoning.\n"
    f"Wrap all your reasoning between {XML_TAGS['reasoning_start']} and {XML_TAGS['reasoning_end']}.\n"
    f"Then, output the final answer after {XML_TAGS['final_answer']}.\n"
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

# ========= データセット準備 =========
raw_ds = load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)

# 指定カテゴリのみ抽出してサンプル数を制限
raw_ds = raw_ds.shuffle(seed=42)
filtered_ds = raw_ds.filter(
    lambda example: str(example.get("category", "")).lower() in DATASET_ALLOWED_CATEGORIES
)
if DATASET_MAX_SAMPLES:
    filtered_ds = filtered_ds.select(range(min(DATASET_MAX_SAMPLES, len(filtered_ds))))

# train を 1000 件に固定し、残りを validation / test に回す（足りない場合は train を再利用）
train_sample_count = min(DATASET_TRAIN_SAMPLES, len(filtered_ds))
train_ds = filtered_ds.select(range(train_sample_count))

eval_source = filtered_ds.select(range(train_sample_count, len(filtered_ds)))
if len(eval_source) == 0:
    eval_source = train_ds

if len(eval_source) > 1:
    valid_test = eval_source.train_test_split(test_size=0.5, seed=42)
    validation_ds = valid_test["train"]
    test_ds = valid_test["test"]
else:
    validation_ds = eval_source
    test_ds = eval_source

dataset_dict = DatasetDict({
    "train": train_ds,
    "validation": validation_ds,
    "test": test_ds,
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
    for prompt, solution in zip(examples["question"], examples["answer"]):
        question = str(prompt).strip()
        # solutionを文字列に変換
        text = str(solution)

        # 正規表現で抽出
        # <think>と</think>の間にあるあらゆる文字(.*?)を抽出
        # re.DOTALL: 改行文字も . に含めるためのフラグ
        match = re.search(r'XML_TAGS["reasoning_start"](.*?)XML_TAGS["reasoning_end"]', text, re.DOTALL)

        if match:
            # タグの中身を取得し、前後の空白を除去
            full_solution = match.group(1).strip()
        else:
            # タグが見つからない場合の処理（空文字にするか、元の文字列を入れるかなど）
            full_solution = str(solution).strip()
        final_answer = extract_final_answer(full_solution)

        reasoning_start = XML_TAGS["reasoning_start"]
        reasoning_end = XML_TAGS["reasoning_end"]

        assistant_content = (
            f"{reasoning_start}\n"
            f"{full_solution}\n"
            f"{reasoning_end}\n"
            f"Final Answer:{final_answer}"
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
                max_new_tokens = 1024,
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
    output_dir = f"{CHECKPOINT_DIR}/qwen3_4b_sft_openmathinst2",
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
model.save_pretrained(f"{MODEL_DIR}/qwen3_4b_openmathins2_sft_lora")
tokenizer.save_pretrained(f"{MODEL_DIR}/qwen3_4b_openmathins2_sft_lora")
print("Training finished and model saved.")
