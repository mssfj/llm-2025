#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
from typing import Dict, Iterable, List

import requests


SYSTEM_PROMPT = """You are a mathematical reasoning editor.

Your task is NOT to solve the problem from scratch.
Assume that the solution provided is correct.

Your goal is to restructure the reasoning into a fixed, structured format
that separates analysis, planning, verification, and detailed reasoning.

You MUST strictly follow the output format.
Do not omit or rename any tags.
Do not add any extra text."""

USER_PROMPT_TEMPLATE = """We have a mathematical problem and its correct solution.

Based strictly on the contents of <question></question> and <solution></solution> below,
rewrite the reasoning into the following structured format.

Rules:
1. Assume the given solution is correct.
2. Do NOT change the final numerical or symbolic answer.
3. Do NOT introduce new solution methods.
4. Each tag must contain meaningful content (do not leave tags empty).
5. Do NOT add explanations outside the specified tags.

Output format (FOLLOW EXACTLY):

<think>
<analyze>
Summarize the given problem, its conditions, and what is being asked.
</analyze>

<plan>
Describe the high-level strategy used in the provided solution.
plan must contain problem-specific quantities (e.g., computed bag count, vertex equations, common difference), not generic steps.
plan is a blueprint that guides the reason; plan so that it does not become a summary of the answer (excluding the final value).
</plan>

<verify>
Explain why the chosen strategy is valid and why no cases are missing.
verify must be at most 2–3 sentences and must reference only the key correctness checks for this specific problem.
</verify>

Here is the input:

<question>
{question}
</question>

<solution>
{solution}
</solution>
"""

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_DATASET = "dataset/openmathinstruct-2_formatted/openmathinstruct2_formatted_10000.jsonl"
ALLOWED_CATEGORIES = {"augmented_math", "math"}
REQUIRED_MARKERS = [
    "<think>",
    "</think>",
    "<analyze>",
    "</analyze>",
    "<plan>",
    "</plan>",
    "<verify>",
    "</verify>",
    "<reason>",
    "</reason>",
    "Final Answer:",
]
SKIP_ANSWER_SUBSTRINGS = [
    "however, the original solution reported",
    "the provided solution says",
    "so we accept",
]


def read_filtered_records(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("category") in ALLOWED_CATEGORIES:
                yield record


def pick_records(path: str, count: int, sample: bool, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    if sample:
        reservoir: List[Dict[str, str]] = []
        for idx, record in enumerate(read_filtered_records(path)):
            if idx < count:
                reservoir.append(record)
                continue
            j = rng.randint(0, idx)
            if j < count:
                reservoir[j] = record
        return reservoir

    selected: List[Dict[str, str]] = []
    for record in read_filtered_records(path):
        selected.append(record)
        if len(selected) >= count:
            break
    return selected


def build_user_prompt(question: str, solution: str) -> str:
    return USER_PROMPT_TEMPLATE.format(question=question.strip(), solution=solution.strip())


def call_openrouter(api_key: str, messages: List[Dict[str, str]], model: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")
    data = resp.json()
    # OpenRouter sometimes returns 200 with an error payload; guard before reading choices.
    if "error" in data:
        raise RuntimeError(
            f"OpenRouter API error payload: {data['error'].get('message', data['error'])}"
        )
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected API response: {data}") from exc


def strip_final_answer_in_think(text: str) -> str:
    # Remove any Final Answer lines that ended up inside <think></think>.
    def _clean(match: re.Match) -> str:
        body = match.group(1)
        cleaned_lines = [
            line for line in body.splitlines() if "Final Answer:" not in line
        ]
        cleaned_body = "\n".join(cleaned_lines).rstrip()
        return f"<think>{cleaned_body}</think>"

    return re.sub(r"<think>(.*?)</think>", _clean, text, flags=re.DOTALL)


def ensure_closing_think(text: str) -> str:
    # Append </think> once after </reason> if the model forgot it.
    if "</reason>" in text and "</think>" not in text:
        return text.replace("</reason>", "</reason>\n</think>", 1)
    return text


def strip_think_blocks(text: str) -> str:
    # Remove complete <think>...</think> blocks from the original solution.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_think_content(text: str) -> str:
    # <think>と</think>の間にある文字列(group 1)を抽出する
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    
    if match:
        # 見つかった場合、タグの中身だけを返却（前後の空白はstripで削除）
        return match.group(1).strip()
    else:
        # タグが見つからない場合は空文字を返す（要件に応じてNoneでも可）
        return ""


def rename_think_to_reason(text: str) -> str:
    # <(/?)think> の意味:
    #   </?   -> "/" があってもなくてもマッチ（開始タグと終了タグの両方に対応）
    #   (...) -> "/" の有無をグループ1として記憶
    #
    # r"<\1reason>" の意味:
    #   \1    -> グループ1で記憶した "/" をここに復元（開始タグなら空、終了タグなら"/"が入る）
    return re.sub(r"<(/?)think>", r"<\1reason>", text)


def replace_reason_with_solution(text: str, solution: str) -> str:
    # Force <reason>...</reason> to contain the provided solution (without think blocks).
    cleaned_solution = strip_think_blocks(solution)
    replacement = f"<reason>\n{cleaned_solution}\n</reason>"
    if "<reason>" in text and "</reason>" in text:
        return re.sub(r"<reason>.*?</reason>", replacement, text, flags=re.DOTALL)
    suffix = "\n" if not text.endswith("\n") else ""
    return f"{text}{suffix}{replacement}"


def _strip_boxed(content: str) -> str:
    # Remove \boxed{...} and \\boxed{...}\\ while keeping inner content.
    result: List[str] = []
    i = 0
    while i < len(content):
        if content[i] == "\\":
            start = i
            slash_count = 0
            while i + slash_count < len(content) and content[i + slash_count] == "\\":
                slash_count += 1
            word_start = i + slash_count
            if word_start < len(content) and content.startswith("boxed", word_start):
                j = word_start + len("boxed")
                while j < len(content) and content[j].isspace():
                    j += 1
                if j < len(content) and content[j] == "{":
                    j += 1
                    brace_level = 1
                    inner_start = j
                    while j < len(content) and brace_level > 0:
                        if content[j] == "{":
                            brace_level += 1
                        elif content[j] == "}":
                            brace_level -= 1
                        j += 1
                    if brace_level == 0:
                        inner = content[inner_start : j - 1]
                        k = j
                        trailing_slashes = 0
                        while k < len(content) and content[k] == "\\":
                            trailing_slashes += 1
                            k += 1
                        # Skip the entire boxed sequence, optional leading/trailing backslashes are dropped.
                        result.append(inner)
                        i = k if trailing_slashes else j
                        continue
        result.append(content[i])
        i += 1
    return "".join(result)


def remove_boxed_in_reason(text: str) -> str:
    # Apply boxed stripping only inside <reason>...</reason> sections.
    def _clean(match: re.Match) -> str:
        body = match.group(1)
        cleaned = body
        # Remove boxed commands repeatedly to catch nested occurrences.
        while True:
            updated = _strip_boxed(cleaned)
            if updated == cleaned:
                break
            cleaned = updated
        return f"<reason>{cleaned}</reason>"

    return re.sub(r"<reason>(.*?)</reason>", _clean, text, flags=re.DOTALL)


def render_progress(current: int, total: int, width: int = 40) -> str:
    # Renders a simple text progress bar without external deps.
    progress = int((current / total) * width)
    bar = "#" * progress + "-" * (width - progress)
    return f"[{bar}] {current}/{total}"


def append_final_answer_line(response: str, solution: str) -> str:
    """
    Append the last line containing 'Final Answer:' from the solution to the response.
    If a Final Answer already exists in the response, replace the last occurrence to keep it single.
    """
    final_answer_line = ""
    for line in reversed(solution.splitlines()):
        if "Final Answer:" in line:
            final_answer_line = line.strip()
            break
    if not final_answer_line:
        return response

    response = response.rstrip()
    if "Final Answer:" in response:
        lines = response.splitlines()
        for idx in range(len(lines) - 1, -1, -1):
            if "Final Answer:" in lines[idx]:
                lines[idx] = final_answer_line
                return "\n".join(lines)
    return f"{response}\n{final_answer_line}"


def has_required_markers(text: str) -> bool:
    return all(text.count(marker) == 1 for marker in REQUIRED_MARKERS)


def should_skip_answer(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in SKIP_ANSWER_SUBSTRINGS):
        return True
    return bool(re.search(r"indicating that.*admit additional", lowered, flags=re.DOTALL))


def ensure_halfwidespace_after_final_answer(text: str) -> str:
    """Insert a halfwidespace right after the first 'Final Answer:' if it's missing."""
    if "Final Answer: " in text or "Final Answer:" not in text:
        return text
    return text.replace("Final Answer:", "Final Answer: ", 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate structured CoT outputs with OpenRouter on openmathinstruct-2_formatted."
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        required=True,
        help="Number of records to process.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Path to the source JSONL dataset (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenRouter model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample records uniformly at random instead of taking the first N.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --sample is set.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.stderr.write("Missing OPENROUTER_API_KEY in environment.\n")
        sys.exit(1)
    
    print("===== データセットの取得とOpenRouterへの問い合わせを開始します =====")

    records = pick_records(args.dataset, args.count, args.sample, args.seed)
    if not records:
        sys.stderr.write("No records found for the specified filters.\n")
        sys.exit(1)

    total_records = len(records)
    sys.stderr.write(f"Processing {total_records} records...\n")

    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, record in enumerate(records, start=1):
            try:
                question = record.get("question", "")
                solution = record.get("answer", "")
                category = record.get("category", "")

                user_prompt = build_user_prompt(question, solution)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                response = call_openrouter(api_key, messages, args.model)
                cleaned_solution = rename_think_to_reason(solution)
                response = f"{response}\n{cleaned_solution}"
                response = ensure_closing_think(response)
                response = remove_boxed_in_reason(response)
                if not has_required_markers(response):
                    sys.stderr.write(
                        f"\nSkipping record {idx}/{total_records}: missing required tags.\n"
                    )
                    sys.stderr.write("\r" + render_progress(idx, total_records))
                    sys.stderr.flush()
                    continue
                if should_skip_answer(response):
                    sys.stderr.write(
                        f"\nSkipping record {idx}/{total_records}: flagged wording in answer.\n"
                    )
                    sys.stderr.write("\r" + render_progress(idx, total_records))
                    sys.stderr.flush()
                    continue

                response = ensure_halfwidespace_after_final_answer(response)

                out_record = {
                    "question": question,
                    "answer": response,
                    "category": category,
                }

                out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception as exc:
                sys.stderr.write(
                    f"\nError processing record {idx}/{total_records}: {exc}\n"
                )
                continue
            sys.stderr.write("\r" + render_progress(idx, total_records))
            sys.stderr.flush()
    sys.stderr.write("\nDone.\n")


if __name__ == "__main__":
    main()
