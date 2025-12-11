# 例: 104 5目から20行目までを表示（Pythonのスライス仕様により、終了インデックスは含まれません）
python - 1 20 << 'PY'
import json
import sys

path = "/workspace/outputs/gsm8k_eval.jsonl"

# デフォルト値
start_idx = 0
end_idx = 5

# 引数解析
# sys.argv[0] は "-" (スクリプト名)
# sys.argv[1] = 開始行
# sys.argv[2] = 終了行
try:
    if len(sys.argv) >= 3:
        start_idx = int(sys.argv[1])
        end_idx = int(sys.argv[2])
    elif len(sys.argv) == 2:
        # 引数が1つだけの場合は、0からその行までとみなす
        end_idx = int(sys.argv[1])
except ValueError:
    print("エラー: 行数は整数で指定してください。")
    sys.exit(1)

# データ読み込み
try:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません: {path}")
    sys.exit(1)

print(f"Total rows: {len(rows)}")
print(f"Displaying rows: {start_idx} to {end_idx}\n")

# 指定範囲を表示
for r in rows[start_idx:end_idx]:
    print("="*80)
    # get() を使ってキーが存在しない場合のエラーを回避
    idx = r.get("index", "N/A")
    print(f"index: {idx}")
    print("Q:", r.get("question", "") if len(r.get("question", "")) > 100 else r.get("question", ""))
    print("GOLD:", r.get("gold_answer", ""))
    print("PRED_EXTRACTED:", r.get("extracted_pred_answer", ""))
    
    is_correct = r.get("is_correct", False)
    # 正解なら緑、不正解なら赤で表示（端末が対応していれば）
    color = "\033[32m" if is_correct else "\033[31m"
    reset = "\033[0m"
    
    print(f"CORRECT?: {color}{is_correct}{reset}", "reason:", r.get("reason", ""))
    
    print("--- FULL MODEL OUTPUT (Snippet) ---")
    output = r.get("model_output", "")
    # 出力が長すぎる場合は末尾500文字だけ表示するなど調整
    print(output[-500:] if len(output) > 5000 else output)
    print()
PY
