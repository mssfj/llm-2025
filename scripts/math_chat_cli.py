#!/usr/bin/env python
# filename: math_chat_cli.py

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ==== パス設定 ====
BASE_MODEL_NAME = "unsloth/Qwen3-4B-Base"  # 元のベース
#ADAPTER_PATH   = "/workspace/model/qwen3_4b_dapo_sft_lora"
ADAPTER_PATH   = "/workspace/model/grpo_saved_lora"
MAX_SEQ_LENGTH = 2048

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


def load_model():
    print(f"Loading base model: {BASE_MODEL_NAME}")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = compute_dtype,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config = bnb_config,
        device_map = "cuda",
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # ==== chat_template をこちらで上書き ====
    # system → eos、user →そのまま、assistant → eos 付き、というシンプルな構造
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
    print("Custom chat_template set (no add_generation_prompt branch).")

    return model, tokenizer


def chat_loop(model, tokenizer):
    print("=== Math Chat (type 'exit' or Ctrl+C to quit) ===")
    while True:
        try:
            user_q = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_q.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        if not user_q:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_q},
        ]

        # add_generation_prompt を自前でやる：最後に「assistantターン」を暗黙に追加
        # → chat_template には add_generation_prompt 分岐が無いので、ここで完結させる
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False,
        )
        # assistant 開始の区切りを付ける（モデルが見てきたフォーマットに揃えたいならここを調整）
        prompt = prompt + "\n"

        inputs = tokenizer(
            prompt,
            return_tensors = "pt",
            add_special_tokens = False,
        ).to(model.device)

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

        print("\n--- Raw model output ---")
        print(output_text)

        # 簡易パース
        rs, re_ = XML_TAGS["reasoning_start"], XML_TAGS["reasoning_end"]
        ss, se_ = XML_TAGS["solution_start"], XML_TAGS["solution_end"]

        reasoning = None
        solution = None

        if rs in output_text and re_ in output_text:
            try:
                reasoning = output_text.split(rs, 1)[1].split(re_, 1)[0].strip()
            except Exception:
                pass

        if ss in output_text and se_ in output_text:
            try:
                solution = output_text.split(ss, 1)[1].split(se_, 1)[0].strip()
            except Exception:
                pass

        print("\n--- Parsed ---")
        if reasoning is not None:
            print(f"[Reasoning]\n{reasoning}")
        if solution is not None:
            print(f"\n[Final Answer]\n{solution}")


if __name__ == "__main__":
    model, tokenizer = load_model()
    chat_loop(model, tokenizer)

