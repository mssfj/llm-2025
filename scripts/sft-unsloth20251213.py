#!/usr/bin/env python
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset, DatasetDict, load_dataset
from trl import SFTTrainer, SFTConfig
import wandb
import random
import re
from itertools import islice

# ========= Settings =========
MODEL_NAME = "unsloth/Qwen3-4B-Base"
LORA_NAME = "qwen3_sft_lora_openmathinst2-structured_1000"
RUN_NAME = LORA_NAME
#RUN_NAME = "default"

DATASET_NAME = "mssfj/openmathinstruct-2_structured-1000"
DATASET_SUBSET = "default"
DATASET_DOWNLOAD_SAMPLES = 1_446
DATASET_TRAIN_SAMPLES = 1_000
DATASET_TEST_SIZE = 0.2
DATASET_ALLOWED_CATEGORIES = {"augmented_math", "math"}

MAX_SEQ_LENGTH = 2048

WANDB_PROJECT = "qwen3-4b-sft-openmathinst2-structured"
WANDB_RUNNAME = "qwen3-openmathinst2-sft_structured_1000"
WANDB_ENTITY = "mssfj-1"

MODEL_DIR = "/workspace/model"
CHECKPOINT_DIR = "/workspace/checkpoints"

SYSTEM_PROMPT = (
    "You are given a math problem.\n"
    "First, think about the problem step by step and show your reasoning.\n"
    "Wrap all your reasoning between <think> and </think>.\n"
    "Then, output the final answer after Final Answer:.\n"
    "The final answer must be a concise expression (usually a single number)."
)

# ========= Model & Tokenizer =========
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
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

# Use Unsloth's optimized chat template for Qwen 2.5
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
    #mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
)

print(f"Model loaded. Vocab size: {len(tokenizer)}")

# ========= Dataset Preparation =========
def load_limited_dataset() -> Dataset:
    stream = load_dataset(
        DATASET_NAME,
        DATASET_SUBSET,
        split="train",
        streaming=True,
    )
    limited = list(islice(stream, DATASET_DOWNLOAD_SAMPLES))
    print(f"Loaded {len(limited)} samples via streaming.")
    return Dataset.from_list(limited)

raw_ds = load_limited_dataset()
#raw_ds = raw_ds.shuffle(seed=42)

filtered_ds = raw_ds.filter(
    lambda example: str(example.get("category", "")).lower() in DATASET_ALLOWED_CATEGORIES
)

train_sample_count = min(DATASET_TRAIN_SAMPLES, len(filtered_ds))
train_ds = filtered_ds.select(range(train_sample_count))
eval_source = filtered_ds.select(range(train_sample_count, len(filtered_ds)))

if len(eval_source) == 0:
    eval_source = train_ds

if len(eval_source) > 1:
    valid_test = eval_source.train_test_split(test_size=DATASET_TEST_SIZE, seed=42)
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

# ========= Helper Functions =========
def extract_final_answer(solution_text: str) -> str:
    text = str(solution_text).strip()
    m = re.search(r"(?:Answer|Ans|Final answer)\s*[:=]\s*([\-+]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r"####\s*([\-+]?\d+(?:\.\d+)?)", text)
    if m: return m.group(1)
    nums = re.findall(r"([\-+]?\d+(?:\.\d+)?)", text)
    if nums: return nums[-1]
    return ""

def format_math_examples(examples):
    texts = []
    reasoning_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    
    for prompt, solution in zip(examples["question"], examples["answer"]):
        question = str(prompt).strip()
        text = str(solution)

        match = reasoning_pattern.search(text)
        if match:
            thought = match.group(1).strip()
        else:
            thought = text.replace("<think>", "").replace("</think>", "").strip()

        final_answer = extract_final_answer(text)

        assistant_content = (
            f"<think>\n{thought}\n</think>\n"
            f"Final Answer: {final_answer}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False,
        )
        texts.append(formatted_text)

    return {"text": texts}

dataset_dict = dataset_dict.map(format_math_examples, batched=True)

# ========= Inference Test =========
def generate_samples(model, tokenizer, dataset, num_samples=3):
    FastLanguageModel.for_inference(model)
    print("\n=== Generation Sample Check (Before Training) ===")
    
    indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    samples = dataset.select(indices)

    for sample in samples:
        question = sample["question"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True,
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

generate_samples(model, tokenizer, dataset_dict["test"])

# ========= Training =========
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUNNAME)

sft_config = SFTConfig(
    output_dir = f"{CHECKPOINT_DIR}/{RUN_NAME}",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-5,
    num_train_epochs = 1,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 10,
    eval_strategy = "steps",
    eval_steps = 100,
    save_strategy = "epoch",
    optim = "adamw_8bit",
    report_to = "wandb",
    seed = 3407,
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_text_field = "text",
    overwrite_output_dir = True,  # avoid resuming old checkpoints if the directory already exists
)

# 検証用データを10件（または検証データ数の少ない方）に制限する
eval_subset = dataset_dict["validation"].select(range(min(10, len(dataset_dict["validation"]))))

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_dict["train"],
    eval_dataset = eval_subset,
    args = sft_config,
)

print("Starting Unsloth SFT...")
trainer.train()

# ========= Save =========
model.save_pretrained(f"{MODEL_DIR}/{RUN_NAME}")
tokenizer.save_pretrained(f"{MODEL_DIR}/{RUN_NAME}")
print("Training finished and model saved.")
