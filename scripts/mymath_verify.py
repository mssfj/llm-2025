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
    r"final\s*answer\s*(?:[:：=]|is\b|->|⇒|=>)\s*(.+)",   # Final Answer: / Final Answer is / Final Answer -> xx
    r"final\s*ans\s*(?:[:：=]|is\b|->|⇒|=>)\s*(.+)",      # Final Ans: xx
    r"(?:the\s+)?answer\s*(?:is\b|[:：=])\s*(.+)",        # (The) answer is: xx
    r"最終解\s*(?:[:：=は]|です)\s*(.+)",                 # 最終解: xx
    r"最終答え\s*(?:[:：=は]|です)\s*(.+)",               # 最終答え: xx
    r"最終的な答えは\s*(?:[:：=])?\s*(.+)",               # 最終的な答えは: xx
    r"答えは\s*(?:[:：=])?\s*(.+)",                       # 答えは: xx
    r"答え\s*(?:[:：=は]|です)\s*(.+)",                   # 答え: xx
]


@dataclass
class ExtractedAnswer:
    answer: str
    has_final_answer: bool
    source: str


def _normalize_text(s: str) -> str:
    """空白と全角・記号の最低限の正規化"""
    s = s.strip()
    # 全角スペース → 半角
    s = s.replace("\u3000", " ")
    # 連続スペースを1個に
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_trailing_punct(s: str) -> str:
    """末尾の句読点・記号ゆらぎを吸収"""
    return re.sub(r"[，,。．、!！?？]+$", "", s).strip()


def _strip_markdown_wrappers(s: str) -> str:
    """
    Markdown/LaTeX 由来の装飾記号（**, *, `, $, \\( \\)）を外側から取り除く。
    末尾にだけ残った ** のようなケースも考慮し、前後の装飾を削る。
    """
    s = s.strip()
    # 対応する両端がある場合
    if (s.startswith("**") and s.endswith("**")) or (s.startswith("__") and s.endswith("__")):
        s = s[2:-2].strip()
    elif (s.startswith("*") and s.endswith("*")) or (s.startswith("_") and s.endswith("_")):
        s = s[1:-1].strip()
    elif s.startswith("`") and s.endswith("`"):
        s = s[1:-1].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    elif s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()

    # 末尾だけに残った装飾を削る（例: "26**"）
    s = re.sub(r"[`*_]+$", "", s).strip()
    s = re.sub(r"^[`*_]+", "", s).strip()
    return s


def _postprocess_candidate(s: str) -> str:
    """正規化＋装飾/末尾記号の削除をまとめて行う"""
    s = _normalize_text(s)
    s = _strip_markdown_wrappers(s)
    return _strip_trailing_punct(s)


_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")
_PLAIN_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?")
_NUMBER_WITH_UNIT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s+[A-Za-z%°]+")


def _extract_numeric_token(text: str) -> Optional[str]:
    """行からそれらしい数値/分数トークンを抜き出す"""
    m = _NUMERIC_TOKEN_RE.search(text)
    if m:
        return _postprocess_candidate(m.group(0))
    return None


def _looks_like_math_expression(text: str) -> bool:
    """Final Answer が数式っぽい場合に True を返す"""
    s = _postprocess_candidate(text)
    if not s:
        return False
    if _PLAIN_NUMBER_RE.fullmatch(s):
        return False
    if _NUMBER_WITH_UNIT_RE.fullmatch(s):
        return False
    if re.search(r"[A-Za-z\\]", s):
        return True
    if re.search(r"[=<>^+\-*/()\[\]{}]", s):
        return True
    if len(_NUMERIC_TOKEN_RE.findall(s)) >= 2:
        return True
    return False


def _select_final_candidate(candidate: str) -> str:
    candidate = _postprocess_candidate(candidate)
    if not candidate:
        return ""
    if _looks_like_math_expression(candidate):
        return candidate
    token = _extract_numeric_token(candidate)
    return token if token else candidate


def extract_final_answer(raw_text: str) -> str:
    return extract_final_answer_with_meta(raw_text).answer


def extract_final_answer_with_meta(raw_text: str) -> ExtractedAnswer:
    """
    CoT込みの生成テキストから「最終的な答え」っぽい部分を抜き出す。

    ルール：
      1. Final Answer / 答え / 最終解 パターンを優先的にマッチ
      2. 見つからなければ、キーワードを含む行の数値っぽい部分を拾う
      3. それも無ければ最後の行の「数字または式」っぽい部分を返す
    has_final_answer は「明示的に最終解答として提示されているか」を表す。
    """
    text = raw_text.strip()
    if not text:
        return ExtractedAnswer("", False, "empty")

    # 1) パターンマッチで抜き出す
    for pat in _FINAL_ANSWER_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            candidate = _select_final_candidate(m.group(1))
            if candidate:
                return ExtractedAnswer(candidate, True, "pattern")

    # 2) 行ごとに見て「Final Answer」「答え」などのキーワードを含む行を優先
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ExtractedAnswer("", False, "empty_lines")

    keyword_re = re.compile(r"(final answer|final ans|answer|最終解|最終答え|答え)", flags=re.IGNORECASE)
    for ln in reversed(lines):
        if keyword_re.search(ln):
            m = keyword_re.search(ln)
            remainder = ln[m.end():] if m else ln
            remainder = re.sub(
                r"^\s*(?:[:：=]|is\b|->|⇒|=>)?\s*",
                "",
                remainder,
                flags=re.IGNORECASE,
            )
            candidate = _select_final_candidate(remainder or ln)
            return ExtractedAnswer(candidate, True, "keyword_line")

    # 3) キーワードが無ければ、最後の行の「それっぽい」トークンを拾う
    last = lines[-1]
    candidate = _select_final_candidate(last)
    if candidate:
        clean_last = _normalize_text(_postprocess_candidate(last))
        bare_answer = (
            len(lines) == 1
            or clean_last == candidate
            or len(clean_last.split()) <= 3  # 数字＋簡単な単位程度
        )
        return ExtractedAnswer(candidate, bare_answer, "fallback")

    # 何も取れなければ行全体を返す
    candidate = _select_final_candidate(last)
    answer = candidate
    tokens = _postprocess_candidate(last).split()
    has_final = (
        len(lines) == 1
        or (answer and len(tokens) <= 3)  # 短い答えなら最終回答扱い
    )
    return ExtractedAnswer(answer, has_final, "fallback")


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
    require_final_answer: bool = True  # 最終回答の明示が無ければ不正解扱いにする


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
    - gold_answer と比較（require_final_answer が True の場合、最終解答が明示されていなければ不正解）
    - is_correct / reason を返す
    """
    if config is None:
        config = MathVerifyConfig()

    gold = _normalize_text(gold_answer)
    extracted = extract_final_answer_with_meta(pred_text)
    pred_raw = extracted.answer
    pred = _normalize_text(pred_raw)

    if config.require_final_answer and not extracted.has_final_answer:
        return MathVerifyResult(
            is_correct=False,
            reason="missing_final_answer",
            pred_answer=pred,
            gold_answer=gold,
        )

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
