"""Multi-step English CoT + explicit option-matching solver for Qwen3.5 0.8B."""
import hashlib
import re
from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"
VALID_LETTERS = {"A", "B", "C", "D", "E"}

# Brief English system prompt (~38 tokens), KV-cached across all problems
SYSTEM = (
    "You are an English math competition solver. Always think and write strictly in English. "
    "Break the problem into 2 to 5 short calculation steps under Steps:, write the final value "
    "after Result:, match it to the right option under Match:, and end with Answer: <A-E>."
)

# GBNF rule names use only letters (no underscores); ASCII-only text blocks Catalan drift
GRAMMAR_COT = (
    'root ::= "Steps:\\n" step ("\\n" step){1,4} "\\nResult: " res "\\nMatch: " match "\\nAnswer: " [A-E]\n'
    'step ::= "- " start body\n'
    'start ::= "Find " | "Calculate " | "Count " | "Total " | "Subtract " | "Add " | '
    '"Multiply " | "Divide " | "Compare " | "Let " | "Each " | "If " | "Since " | '
    '"First " | "Next " | "So " | "The "\n'
    'body ::= [A-Za-z0-9 ,.+*/()?:;\'=%-]{4,85}\n'
    'res ::= [A-Za-z0-9 ,.+*/()-]{1,28}\n'
    'match ::= [A-Za-z0-9 ,.+*/()-]{1,36} "option " [A-E]'
)
GRAMMAR_LETTER = "root ::= [A-E]"

_ANS_RE = re.compile(r"Answer:\s*([A-E])\b")
_RES_RE = re.compile(r"Result:\s*(.+)")


def _fallback_letter(problem):
    """Return a valid A-E fallback letter even on server timeout or error."""
    default = str(CONFIG.get("FALLBACK_ANSWER", "D")).strip().upper()
    if default in VALID_LETTERS:
        return default
    digest = hashlib.md5(str(problem.get("id", "")).encode()).digest()[0]
    return "ABCDE"[digest % 5]


def format_problem(problem, sees_image=False, include_answer_cue=False):
    lines = ["Solve step by step in English:", (problem.get("text") or "").strip()]
    options = problem.get("options") or {}
    for letter in "ABCDE":
        if letter in options:
            lines.append(f"{letter}) {options[letter]}")
    if not options:
        lines.append(
            "(The options are diagrams in the image. Pick the best letter A-E.)"
            if sees_image else "(The options are in an unseen diagram. Pick the best letter A-E.)"
        )
    if include_answer_cue:
        lines.append("Answer:")
    return "\n".join(lines)


def build_messages(problem, examples=(), image_b64=None):
    """Build English-framed chat messages (compatible with explorer/server.py)."""
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in examples:
        ex_label = str(ex.get("label") or "").strip().upper()
        ex_val = str((ex.get("options") or {}).get(ex_label, "")).strip() or ex_label
        ex_reply = (
            f"Steps:\n"
            f"- First read the given values from the problem.\n"
            f"- Calculate the target quantity as {ex_val}.\n"
            f"Result: {ex_val}\n"
            f"Match: {ex_val} is option {ex_label}\n"
            f"Answer: {ex_label}"
        )
        messages.append({"role": "user", "content": format_problem(ex)})
        messages.append({"role": "assistant", "content": ex_reply})

    text = format_problem(problem, sees_image=bool(image_b64))
    content = (
        [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
            {"type": "text", "text": text},
        ]
        if image_b64
        else text
    )
    messages.append({"role": "user", "content": content})
    return messages


def _compute_max_tokens(problem, examples, req_timeout, image_b64=None):
    """Fit ~25-30 s of generation comfortably inside the remaining time budget."""
    prompt_chars = len(SYSTEM) + len(problem.get("text") or "")
    for ex in examples:
        prompt_chars += len(ex.get("text") or "") + 120
    est_prefill_s = (prompt_chars / 40.0) + (28.0 if image_b64 else 0.0)
    avail_gen_s = max(14.0, req_timeout - est_prefill_s - 4.0)
    # ~5.5 tokens/s generation on UNO Q; cap at 175 tokens (~30 s)
    return max(110, min(175, int(avail_gen_s * 5.5)))


def generate_cot(problem, examples=(), timeout=50.0, image_b64=None, url=URL):
    """Generate 2-5 English calculation steps, Result, option Match, and Answer: <A-E>.
    Returns (raw_text, parsed_val, letter, timings).
    """
    hard_limit = float(CONFIG.get("TIME_LIMIT_SECONDS", 60))
    req_timeout = max(8.0, min(float(timeout), hard_limit - 4.0))
    messages = build_messages(problem, examples=examples, image_b64=image_b64)
    max_tokens = _compute_max_tokens(problem, examples, req_timeout, image_b64=image_b64)

    try:
        response = post(url, {
            "messages": messages,
            "grammar": GRAMMAR_COT,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": int(CONFIG["SEED"]),
            "cache_prompt": True,
        }, timeout=req_timeout)
        raw = (response["choices"][0]["message"]["content"] or "").strip()
        timings = response.get("timings", {})
    except Exception as exc:
        fb = _fallback_letter(problem)
        return f"ERROR: {exc}\nAnswer: {fb}", "", fb, {}

    m_res = _RES_RE.search(raw)
    parsed_val = m_res.group(1).strip() if m_res else ""

    m_ans = _ANS_RE.search(raw)
    letter = m_ans.group(1) if m_ans else None

    if letter not in VALID_LETTERS:
        letter, _ = answer_letter(
            problem, examples=examples, timeout=8.0, image_b64=image_b64, url=url
        )

    return raw, parsed_val, letter, timings


def answer_letter(problem, examples=(), timeout=20.0, image_b64=None, url=URL, debias=False):
    """Fast 1-token direct A-E fallback classifier."""
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in examples:
        messages.append({"role": "user", "content": format_problem(ex, include_answer_cue=True)})
        messages.append({"role": "assistant", "content": str(ex.get("label") or "").strip().upper()})

    text = format_problem(problem, sees_image=bool(image_b64), include_answer_cue=True)
    content = (
        [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
            {"type": "text", "text": text},
        ]
        if image_b64
        else text
    )
    messages.append({"role": "user", "content": content})

    try:
        response = post(url, {
            "messages": messages,
            "grammar": GRAMMAR_LETTER,
            "max_tokens": 1,
            "temperature": 0,
            "seed": int(CONFIG["SEED"]),
            "cache_prompt": True,
        }, timeout=timeout)
        letter = (response["choices"][0]["message"]["content"] or "").strip().upper()
        timings = response.get("timings", {})
    except Exception:
        return _fallback_letter(problem), {}

    if letter not in VALID_LETTERS:
        letter = _fallback_letter(problem)
    return letter, timings
