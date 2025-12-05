cd /workspace/
source .venv/bin/activate

python - << 'PY'
import json

path = "logs/gsm8k_qwen3_8b_eval.jsonl"

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

print("total rows:", len(rows))
print("例として、最初の5件だけ表示\n")

for r in rows[20:21]:
    print("="*80)
    print(f"index: {r['index']}")
    print("Q:", r["question"])
    print("GOLD:", r["gold_answer"])
    print("PRED_EXTRACTED:", r["extracted_pred_answer"])
    print("CORRECT?:", r["is_correct"], "reason:", r["reason"])
    print("--- FULL MODEL OUTPUT ---")
    print(r["model_output"])
    print()
PY

