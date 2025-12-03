import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import wandb
import random

# ========= 設定 =========
MODEL_NAME = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2"
DATASET_URL = "https://huggingface.co/datasets/watashihakobashi/ogiri/raw/main/ogiri.tsv"
MAX_SEQ_LENGTH = 2048
WANDB_PROJECT = "llm-lecture-2025-sft"
WANDB_ENTITY = "mssfj-1"
WANDB_RUNNAME = "sft-run-unsloth"

# ========= Model & Tokenizer (Unsloth) =========
# 4bit量子化とモデル読み込みを同時に行います
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,           # Noneで自動検出 (Float16 or Bfloat16)
    load_in_4bit = True,    # 4bit量子化を有効化
)

# ========= LoRA設定 (Unsloth最適化版) =========
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,     # Unslothは0を推奨
    bias = "none",        # Unslothは"none"を推奨
    use_gradient_checkpointing = "unsloth", # メモリ節約設定 (True or "unsloth")
    random_state = 3407,
    use_rslora = False,   # ランク安定化LoRAを使う場合はTrue
    loftq_config = None,  # LoftQを使う場合は設定
)

print(f"Model loaded with Unsloth. Vocab size: {len(tokenizer)}")

# ========= データセット準備 =========
dataset = load_dataset("csv", data_files=DATASET_URL, delimiter="\t", split="train")

# データの分割
train_test = dataset.train_test_split(test_size=0.1, seed=42)
test_valid = train_test['test'].train_test_split(test_size=0.5, seed=42)

dataset_dict = DatasetDict({
    'train': train_test['train'],
    'validation': test_valid['train'],
    'test': test_valid['test']
})

# チャットテンプレート適用関数
def format_prompts(examples):
    texts = []
    for odai, boke in zip(examples["odai"], examples["boke"]):
        user_text = (
            "以下は大喜利のお題です。大喜利とはお題に対して正しい答えではなく面白い回答を出力するタスクです。"
            "お題に対する面白い回答を生成してください。\n"
            "理由は説明せず回答のみを出力してください。\n"
            f"お題: {odai}"
        )
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": boke if boke else ""},
        ]
        # Unslothのtokenizerもapply_chat_templateを持っています
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        if boke:
            text += tokenizer.eos_token
            
        texts.append(text)
    return {"text": texts}

dataset_dict = dataset_dict.map(format_prompts, batched=True)

# ========= 推論テスト用関数 (Unsloth高速推論) =========
def generate_samples(model, tokenizer, dataset, num_samples=5):
    # Unslothの高速推論モードを有効化
    FastLanguageModel.for_inference(model) 
    
    print("\n=== Generation Sample Check (Before Training) ===")
    samples = dataset.select(random.sample(range(len(dataset)), num_samples))
    
    for sample in samples:
        odai = sample["odai"]
        user_text = (
            "以下は大喜利のお題です。面白い回答を出力してください。\n"
            f"お題: {odai}"
        )
        messages = [{"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024, 
                use_cache=True,
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 入力プロンプト部分を除去してデコード
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"お題: {odai}\n回答: {output_text}\n---")
    
    # 学習モードに戻す
    FastLanguageModel.for_training(model)

# 学習前の動作確認
generate_samples(model, tokenizer, dataset_dict["test"])

# ========= 学習 (SFT) =========
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUNNAME)

# SFT設定
sft_config = SFTConfig(
    output_dir="outputs_unsloth",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    eval_strategy="steps",
    save_strategy="epoch",
    optim="adamw_8bit",
    report_to="wandb",
    seed=3407,
)

# 設定値を後付け（バージョン互換性対策）
sft_config.max_seq_length = MAX_SEQ_LENGTH
sft_config.dataset_text_field = "text"

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,           # Unslothではtokenizerも渡すのが推奨
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    args=sft_config,
    # 念のため直接引数としても渡しておく（trlのバージョン差異吸収用）
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
)

print("Starting Unsloth training...")
trainer.train()

# ========= 保存 =========
# Unsloth専用の保存メソッドを使用 (GGUF変換などもここから可能)
model.save_pretrained("lora_model_unsloth")
tokenizer.save_pretrained("lora_model_unsloth")
print("Training finished and model saved.")
