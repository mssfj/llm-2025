# math_verify.py
"""
数学系タスク用の自動正解判定モジュール（math-verify の簡易実装）

用途：
- 評価パイプラインでの正答判定（EM / 数値 / SymPy）
- GRPO の reward 関数
- SFT / RL データのクリーニング

前提：
- gold_answer は「最終的な正解」を文字列で持つ
- pred_text は LLM の生出力（CoT込み）をそのまま渡してよい
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sympy import sympify, simplify
from sympy.core.sympify import SympifyError


# =========================
# 正規化・抽出まわり
# =========================

_FINAL_ANSWER_PATTERNS = [
    r"final answer[:：]\s*(.+)",        # Final Answer: xx
    r"最終解[:：]\s*(.+)",             # 最終解: xx
    r"答えは[:：]\s*(.+)",             # 答えは: xx
    r"答え[:：]\s*(.+)",               # 答え: xx
    r"最終的な答えは[:：]\s*(.+)",     # 最終的な答えは: xx
]


def _normalize_text(s: str) -> str:
    """空白と全角・記号の最低限の正規化"""
    s = s.strip()
    # 全角スペース → 半角
    s = s.replace("\u3000", " ")
    # 連続スペースを1個に
    s = re.sub(r"\s+", " ", s)
    return s


def extract_final_answer(raw_text: str) -> str:
    """
    CoT込みの生成テキストから「最終的な答え」っぽい部分を抜き出す。

    ルール：
      1. Final Answer / 答え / 最終解 パターンを優先的にマッチ
      2. 見つからなければ、最後の行の「数字または式」っぽい部分を返す
    """
    text = raw_text.strip()

    # 1) パターンマッチで抜き出す
    lowered = text.lower()
    for pat in _FINAL_ANSWER_PATTERNS:
        m = re.search(pat, lowered, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1)
            return _normalize_text(candidate)

    # 2) 行ごとに見て最後の「それっぽい」トークンを拾う
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    last = lines[-1]

    # 数式 or 数字だけが書かれている場合を優先
    # 例: "したがって、答えは 24 である。"
    # → 数字だけ抜く
    # 数字 or 分数 or 小数 or マイナスを含むトークンを拾う
    num_pattern = re.compile(r"[-+]?\d+(\.\d+)?(/\d+)?")
    nums = num_pattern.findall(last)
    if nums:
        # findallはタプルを返すので、searchで1個拾い直す
        m = num_pattern.search(last)
        if m:
            return _normalize_text(m.group(0))

    # 何も取れなければ行全体を返す
    return _normalize_text(last)


# =========================
# 数値パース・比較
# =========================

def _parse_number(s: str) -> Optional[float]:
    """
    文字列から数値をパースする。
    - 整数
    - 小数
    - 分数 (a/b)
    - %（パーセント記号）
    """
    s = _normalize_text(s)
    if not s:
        return None

    # %（パーセント）
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return None

    # 分数 a/b
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0])
                den = float(parts[1])
                if den == 0:
                    return None
                return num / den
            except ValueError:
                pass

    # 通常のfloat
    try:
        return float(s)
    except ValueError:
        return None


def numeric_close(a: float, b: float, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    """数値として十分近いかどうか"""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


# =========================
# SymPy を使った等価性チェック
# =========================

def sympy_equiv(pred: str, gold: str) -> bool:
    """
    SymPyで pred と gold が等価か判定する。

    例:
      pred = "(x+1)*(x-1)"
      gold = "x**2 - 1"
      → True
    """
    pred = pred.strip()
    gold = gold.strip()
    if not pred or not gold:
        return False

    try:
        ep = sympify(pred)
        eg = sympify(gold)
    except SympifyError:
        return False
    except Exception:
        return False

    try:
        diff = simplify(ep - eg)
        return diff == 0
    except Exception:
        return False


# =========================
# メイン verify ロジック
# =========================

@dataclass
class MathVerifyConfig:
    use_exact: bool = True           # 完全一致をまず見る
    use_numeric: bool = True         # 数値近似で判定
    use_sympy: bool = True           # SymPy等価性チェック
    rel_tol: float = 1e-6
    abs_tol: float = 1e-9


@dataclass
class MathVerifyResult:
    is_correct: bool
    reason: str
    pred_answer: str
    gold_answer: str


def verify_math_answer(
    pred_text: str,
    gold_answer: str,
    config: Optional[MathVerifyConfig] = None,
) -> MathVerifyResult:
    """
    math-verify のメイン関数。
    - CoT込みpred_textから最終答えを抽出
    - gold_answer と比較
    - is_correct / reason を返す
    """
    if config is None:
        config = MathVerifyConfig()

    gold = _normalize_text(gold_answer)
    pred_raw = extract_final_answer(pred_text)
    pred = _normalize_text(pred_raw)

    # 1) 完全一致
    if config.use_exact and pred == gold:
        return MathVerifyResult(
            is_correct=True,
            reason="exact_match",
            pred_answer=pred,
            gold_answer=gold,
        )

    # 2) 数値近似
    if config.use_numeric:
        gv = _parse_number(gold)
        pv = _parse_number(pred)
        if gv is not None and pv is not None:
            if numeric_close(pv, gv, rel_tol=config.rel_tol, abs_tol=config.abs_tol):
                return MathVerifyResult(
                    is_correct=True,
                    reason="numeric_close",
                    pred_answer=pred,
                    gold_answer=gold,
                )

    # 3) SymPy 等価性
    if config.use_sympy:
        if sympy_equiv(pred, gold):
            return MathVerifyResult(
                is_correct=True,
                reason="sympy_equiv",
                pred_answer=pred,
                gold_answer=gold,
            )

    # すべてダメなら不正解
    return MathVerifyResult(
        is_correct=False,
        reason="mismatch",
        pred_answer=pred,
        gold_answer=gold,
    )


# =========================
# RL 用の reward ラッパ
# =========================

def math_reward(
    pred_text: str,
    gold_answer: str,
    correct_reward: float = 1.0,
    wrong_reward: float = 0.0,
    config: Optional[MathVerifyConfig] = None,
) -> Tuple[float, MathVerifyResult]:
    """
    GRPO / RLHF 用の reward 関数。
    - 正解なら correct_reward
    - 不正解なら wrong_reward
    を返す。
    """
    result = verify_math_answer(pred_text, gold_answer, config=config)
    reward = correct_reward if result.is_correct else wrong_reward
    return reward, result

