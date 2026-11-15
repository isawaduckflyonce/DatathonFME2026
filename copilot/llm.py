"""Deterministic KV-cached plain-English CoT and calibrated classifier for Qwen3.5 0.8B."""
import hashlib
import math
import re
from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Empirical prior penalty to counter Qwen3.5-0.8B's heavy A/D bias when options are unseen pictures
LETTER_PRIOR_PENALTY = {"A": 0.85, "B": 0.0, "C": -0.05, "D": 0.25, "E": 0.0}

SYSTEM_COT = (
    "Solve the math competition problem in 2 to 4 concise plain-English calculation lines "
    "(no markdown, no bold, no bullet points, no LaTeX, do not restate the problem), "
    "then write Final: <exact result> and Answer: <A-E>."
)
SYSTEM = SYSTEM_COT

# Cached once in llama-server's KV cache across all problems: primes terse English equations
STATIC_EXAMPLE = [
    {
        "role": "user",
        "content": (
            "Un quadrat té perímetre 20 cm. Un triangle equilàter té el mateix costat que el quadrat. "
            "Quin és el perímetre del triangle?\n"
            "A) 10 cm\nB) 12 cm\nC) 15 cm\nD) 20 cm\nE) 25 cm"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Work:\n"
            "Square perimeter = 20, so side = 20 / 4 = 5 cm.\n"
            "Equilateral triangle has 3 sides of 5 cm, so perimeter = 3 * 5 = 15 cm.\n"
            "Final: 15 cm\n"
            "Answer: C"
        ),
    },
]

SYSTEM_DIRECT = (
    "You solve Kangaroo math competition problems. "
    "Pick the single most likely option letter A, B, C, D, or E."
)

# Blocks '#', '*', '<', '>', '\', '$' so markdown, bullets, </think>, and LaTeX cannot appear
GRAMMAR_COT = (
    'root ::= "Work:\\n" line ("\\n" line){0,4} "\\nFinal: " val "\\nAnswer: " [A-E]\n'
    'line ::= [^#*<>\\\\$\\n]{4,105}\n'
    'val ::= [^\\n]{1,28}'
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
            "(The options are diagrams in a picture. Choose the best letter A-E.)"
            if sees_image else "(The options are in the picture.)"
        )
    if include_answer_cue:
        lines.append("Answer:")
    return "\n".join(lines)


def build_messages(problem, examples=(), image_b64=None):
    """Build the CoT chat messages (compatible with explorer/server.py)."""
    messages = [{"role": "system", "content": SYSTEM_COT}] + STATIC_EXAMPLE

    for ex in examples:
        ex_content = format_problem(ex, include_answer_cue=False)
        ex_val = ex.get("options", {}).get(ex["label"], ex["label"])
        ex_reply = ex.get("solution") or f"Work:\nApply the given conditions.\nFinal: {ex_val}\nAnswer: {ex['label']}"
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
    """Run deterministic KV-cached plain-English CoT: returns (raw_text, parsed_val, letter, timings)."""
    messages = build_messages(problem, examples=examples, image_b64=image_b64)

    response = post(url, {
        "messages": messages,
        "grammar": GRAMMAR_COT,
        "max_tokens": 115,
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
    messages = [{"role": "system", "content": SYSTEM_DIRECT}]
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
