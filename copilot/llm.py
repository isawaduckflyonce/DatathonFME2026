"""Zero-shot English-prompted KV-cached CoT and calibrated classifier for Qwen3.5 0.8B."""
import hashlib
import math
import re
from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Balances blind guesses only when options are unseen diagrams (options == {})
LETTER_PRIOR_PENALTY = {"A": 0.85, "B": 0.0, "C": -0.05, "D": 0.25, "E": 0.0}

# Single English system prompt shared across all calls so llama-server keeps its KV cache warm
SYSTEM = (
    "You are an expert math competition solver. "
    "Solve the given problem directly in 2 or 3 short plain-English calculation lines "
    "(do not restate or translate the problem, no markdown, no LaTeX), "
    "then output Final: <exact value> and Answer: <A-E>."
)

# Max 3 lines * 75 chars (~60 tokens) + Final + Answer (~16 tokens) < max_tokens (110): never truncates
GRAMMAR_COT = (
    'root ::= "Work:\\n" line ("\\n" line){0,2} "\\nFinal: " val "\\nAnswer: " [A-E]\n'
    'line ::= [^#*<>\\\\$\\n]{4,75}\n'
    'val ::= [^\\n]{1,24}'
)
GRAMMAR_LETTER = "root ::= [A-E]"

_VAL_RE = re.compile(r"Final:\s*(.+?)(?:\n|Answer:|$)")
_ANS_RE = re.compile(r"Answer:\s*([A-E])")


def format_problem(problem, sees_image=False, include_answer_cue=False):
    lines = [problem["text"].strip()]
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
    """Build zero-shot English-instructed chat messages (compatible with explorer/server.py)."""
    messages = [{"role": "system", "content": SYSTEM}]

    for ex in examples:
        ex_content = format_problem(ex, include_answer_cue=False)
        ex_val = ex.get("options", {}).get(ex["label"], ex["label"])
        ex_reply = ex.get("solution") or f"Work:\nCompute from the given values.\nFinal: {ex_val}\nAnswer: {ex['label']}"
        messages.append({"role": "user", "content": ex_content})
        messages.append({"role": "assistant", "content": ex_reply})

    text = format_problem(problem, sees_image=bool(image_b64), include_answer_cue=False)
    if image_b64:
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
            {"type": "text", "text": text},
        ]
    else:
        content = text
    messages.append({"role": "user", "content": content})
    return messages


def generate_cot(problem, examples=(), timeout=50.0, image_b64=None, url=URL):
    """Run deterministic zero-shot CoT: returns (raw_text, parsed_val, letter, timings)."""
    messages = build_messages(problem, examples=examples, image_b64=image_b64)

    response = post(url, {
        "messages": messages,
        "grammar": GRAMMAR_COT,
        "max_tokens": 110,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }, timeout=timeout)

    raw = response["choices"][0]["message"]["content"].strip()
    m_val = _VAL_RE.search(raw)
    m_ans = _ANS_RE.search(raw)
    parsed_val = m_val.group(1).strip() if m_val else ""
    letter = m_ans.group(1) if m_ans else None
    return raw, parsed_val, letter, response.get("timings", {})


def answer_letter(problem, examples=(), timeout=20.0, image_b64=None, url=URL, debias=True):
    """Deterministic 1-token calibrated classifier for problems whose options are images."""
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in examples:
        messages.append({"role": "user", "content": format_problem(ex, include_answer_cue=True)})
        messages.append({"role": "assistant", "content": ex["label"]})

    text = format_problem(problem, sees_image=bool(image_b64), include_answer_cue=True)
    if image_b64:
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
            {"type": "text", "text": text},
        ]
    else:
        content = text
    messages.append({"role": "user", "content": content})

    response = post(url, {
        "messages": messages,
        "grammar": GRAMMAR_LETTER,
        "max_tokens": 1,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
        "logprobs": True,
        "top_logprobs": 5,
        "n_probs": 5,
    }, timeout=timeout)

    choice = response["choices"][0]
    letter = choice["message"]["content"].strip()

    if debias:
        scores = {}
        try:
            top = choice["logprobs"]["content"][0]["top_logprobs"]
            for entry in top:
                tok = entry["token"].strip()
                if tok in LETTER_PRIOR_PENALTY and math.isfinite(entry["logprob"]):
                    scores[tok] = entry["logprob"] - LETTER_PRIOR_PENALTY[tok]
        except (KeyError, IndexError, TypeError):
            pass

        if len(scores) >= 2:
            letter = max(scores, key=scores.get)
        else:
            digest = hashlib.md5(str(problem.get("id", "")).encode()).digest()[0]
            letter = "BCDE"[digest % 4]

    return letter, response.get("timings", {})
