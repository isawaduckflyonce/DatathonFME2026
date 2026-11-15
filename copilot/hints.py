"""Non-AI Hint Ladder rungs (Rung 1: easier solved neighbor, Rung 2: shared topic & method)."""
import re
import urllib.error
from collections import Counter

import json
import time
import urllib.request
from copilot.config import CONFIG

# Multilingual (English + Catalan) topic signatures and Kangaroo solving heuristics
TOPIC_RULES = [
    {
        "topic": "Geometry & Spatial Reasoning",
        "keywords": {
            "triangle", "square", "rectangle", "circle", "angle", "area", "perimeter",
            "polygon", "cube", "face", "faces", "net", "fold", "folded", "symmetry",
            "symmetric", "line", "segment", "length", "side", "sides", "corner",
            "triangle", "quadrat", "rectangle", "cercle", "angle", "àrea", "perímetre",
            "cub", "cara", "cares", "plegar", "simetria", "costat", "costats",
        },
        "method": (
            "Draw or label the figure with all known lengths and angles. Look for "
            "symmetry, congruent shapes, or ways to split an irregular region into simple rectangles and triangles."
        ),
    },
    {
        "topic": "Combinatorics & Counting",
        "keywords": {
            "ways", "many", "combinations", "arrange", "arrangement", "order", "path",
            "paths", "route", "routes", "pair", "pairs", "color", "colors", "colour",
            "colours", "choose", "different", "possible", "at least", "at most",
            "maneres", "quantes", "quants", "combinacions", "camins", "parelles",
            "colors", "diferents", "possibles", "mínim", "màxim",
        },
        "method": (
            "This is a counting problem: organize the possibilities into small, non-overlapping "
            "cases (or count the total and subtract the invalid cases) so nothing is missed or counted twice."
        ),
    },
    {
        "topic": "Logic & Deduction",
        "keywords": {
            "truth", "lie", "lies", "lying", "always", "never", "statement", "clue",
            "taller", "shorter", "heavier", "lighter", "older", "younger", "race",
            "rank", "first", "last", "between", "left", "right", "neighbor",
            "veritat", "mentida", "sempre", "mai", "afirmació", "ordre", "més alt",
            "més pesat", "esquerra", "dreta",
        },
        "method": (
            "Use elimination: assume one statement or position is true, check it against every "
            "other clue, and cross out any option (A–E) that creates a contradiction."
        ),
    },
    {
        "topic": "Number Theory & Digits",
        "keywords": {
            "digit", "digits", "prime", "divisible", "remainder", "multiple", "even",
            "odd", "factor", "product", "sum", "integer", "whole", "number", "numbers",
            "xifra", "xifres", "dígit", "dígits", "primer", "divisible", "residu",
            "múltiple", "parell", "senar", "producte", "suma", "nombre", "nombres",
        },
        "method": (
            "Check parity (even vs. odd), last-digit behavior, and divisibility rules. "
            "Testing the five options (A–E) against these constraints often eliminates three or four choices immediately."
        ),
    },
    {
        "topic": "Algebra, Ratios & Working Backwards",
        "keywords": {
            "percent", "percentage", "fraction", "half", "twice", "double", "triple",
            "ratio", "age", "ages", "speed", "distance", "time", "hour", "hours",
            "minute", "minutes", "coin", "coins", "price", "cost", "total", "average",
            "percentatge", "fracció", "meitat", "doble", "triple", "edat", "velocitat",
            "distància", "temps", "hores", "minuts", "monedes", "preu", "mitjana",
        },
        "method": (
            "Name the unknown quantity with a simple variable or start from the middle option (C) "
            "and work backwards through each sentence of the problem."
        ),
    },
    {
        "topic": "Patterns, Grids & Sequences",
        "keywords": {
            "pattern", "sequence", "next", "repeat", "repeats", "step", "steps",
            "grid", "table", "row", "rows", "column", "columns", "cell", "cells",
            "patró", "seqüència", "següent", "repeteix", "pas", "passos", "quadrícula",
            "taula", "fila", "files", "columna", "columnes", "casella",
        },
        "method": (
            "Fill in the most constrained cells or write out the first 4–5 steps explicitly "
            "until you spot the repeating cycle or rule."
        ),
    },
]

DEFAULT_RULE = {
    "topic": "General Problem Solving",
    "method": (
        "Try the problem with smaller numbers first to see how it behaves, or plug the "
        "five answer choices (A–E) back into the question to see which one fits."
    ),
}

def _tokenize(text):
    return re.findall(r"[a-zà-ÿ]+", (text or "").lower())


def get_rung_1(problem, index, k=30):
    """Find the nearest solved neighbor in embeddings at a strictly lower level."""
    if index is None:
        return {"error": "Retrieval index not built."}

    try:
        hits = index.search(problem, k=k, exclude_id=problem["id"])
    except (OSError, urllib.error.URLError):
        return {"error": "Embedding server is down (port 8082)."}

    if not hits:
        return {"error": "No similar solved problems found in the index."}

    same_lang = [(s, p) for s, p in hits if p.get("language") == problem.get("language")]
    pool = same_lang or hits

    easier = [(s, p) for s, p in pool if p["level"] < problem["level"]]
    if not easier:
        return {"error": f"No easier solved problem found below level {problem['level']}."}

    score, example = easier[0]
    return {
        "rung": 1,
        "target_level": problem["level"],
        "score": round(score, 3),
        "example": example,
    }

def get_rung_2(problem, index, k=5):
    """Infer the shared topic and solving method from the problem and its top neighbors."""
    hits = []
    if index is not None:
        try:
            hits = index.search(problem, k=k, exclude_id=problem["id"])
        except (OSError, urllib.error.URLError):
            hits = []

    token_weights = Counter()
    for tok in _tokenize(problem.get("text", "")):
        token_weights[tok] += 2.0
    for _score, neighbor in hits:
        for tok in set(_tokenize(neighbor.get("text", ""))):
            token_weights[tok] += 1.0

    best_rule = DEFAULT_RULE
    best_score = 0.0
    best_matches = []

    for rule in TOPIC_RULES:
        matched = [kw for kw in rule["keywords"] if kw in token_weights]
        score = sum(token_weights[kw] for kw in matched)
        if score > best_score:
            best_score = score
            best_rule = rule
            best_matches = sorted(matched, key=lambda w: -token_weights[w])[:5]

    return {
        "rung": 2,
        "topic": best_rule["topic"],
        "method": best_rule["method"],
        "matched_keywords": best_matches,
        "neighbor_count": len(hits),
    }

LANG_NAMES = {
    "eng": "English",
    "cat": "Catalan",
    "esp": "Spanish",
    "fra": "French",
    "deu": "German",
}

LEAK_PATTERNS = [
    # Explicit option letters ("answer is B", "option (C)")
    re.compile(
        r"\b(?:answer|option|choice|letter|resposta|opció|lletra|respuesta|opción)\s*"
        r"(?:is|és|es|:|=)?\s*\(?[A-E]\)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*\(?[A-E]\)?[\)\.:\s]*$"),
    # Declarative value assignments ("must be 7", "equals 14", "= 7", "ha de ser 7")
    re.compile(
        r"(?:\b(?:must\s+be|should\s+be|will\s+be|equals|equal\s+to|is|gives|makes|"
        r"ha\s+de\s+ser|és|serà|dona|es|será)\s+|=)\s*-?\d+(?:[.,/]\d+)?\b",
        re.IGNORECASE,
    ),
]

ALLOWED_STRUCTURAL_NUMS = {"1", "2"}

def _extract_numbers(text):
    return set(re.findall(r"\b\d+(?:[.,/]\d+)?\b", text or ""))



def _chat_completion(messages, max_tokens, llm_url=None, timeout=None):
    url = llm_url or f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"
    limit = float(timeout or CONFIG["TIME_LIMIT_SECONDS"])
    payload = json.dumps({
        "messages": messages,
        "temperature": 0,
        "seed": int(CONFIG.get("SEED", 42)),
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=limit) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.monotonic() - t0
    content = (data["choices"][0]["message"]["content"] or "").strip()
    return content, elapsed


def _first_sentence(text):
    """Keep only the first sentence without splitting on decimal points like 3.5."""
    cleaned = " ".join((text or "").split())
    parts = re.split(r"(?<!\d)[.!?](?!\d)\s+", cleaned, maxsplit=1)
    sentence = parts[0].strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def contains_answer_leak(hint_text, problem):
    """Return True if the hint reveals the answer, option value, or a newly computed number."""
    if not hint_text or not hint_text.strip():
        return True

    for pat in LEAK_PATTERNS:
        if pat.search(hint_text):
            return True

    # 1. Check against known OCR option value (when available)
    label = problem.get("label")
    options = problem.get("options") or {}
    correct_val = str(options.get(label, "")).strip() if label else ""
    if correct_val:
        val_pat = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(correct_val)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        if val_pat.search(hint_text):
            return True

    # 2. Block any computed number not already present in the problem text
    #    (catches leaks even when OCR options are empty "{}")
    problem_nums = _extract_numbers(problem.get("text", ""))
    hint_nums = _extract_numbers(hint_text)
    novel_nums = hint_nums - problem_nums - ALLOWED_STRUCTURAL_NUMS
    if novel_nums:
        return True

    return False
def _format_options(problem):
    options = problem.get("options") or {}
    if not options:
        return ""
    return "\n" + "\n".join(f"{l}) {options[l]}" for l in "ABCDE" if l in options)

def get_rung_3(problem, index=None, llm_url=None):
    """Generate a one-sentence guiding question and block computed-number leaks."""
    lang = LANG_NAMES.get(problem.get("language", "eng"), "English")
    rung2 = get_rung_2(problem, index)

    # Omit A-E options so the 0.8B model focuses on method rather than solving for a choice
    system_msg = (
        f"You are a Socratic math tutor. Reply in {lang} with ONE short guiding question "
        "(max 20 words) that points out what to notice first. "
        "Do NOT calculate anything, do NOT introduce new numbers, and never state the answer."
    )
    user_msg = (
        f"Topic: {rung2['topic']}\n"
        f"Problem: {problem.get('text', '')}\n"
        "Guiding question:"
    )

    total_seconds = 0.0
    hint = ""
    leaked = True

    for attempt in range(2):
        extra = (
            " Ask a question starting with 'How', 'What', 'Which', or 'Notice' using zero new numbers."
            if attempt == 1
            else ""
        )
        msgs = [
            {"role": "system", "content": system_msg + extra},
            {"role": "user", "content": user_msg},
        ]
        raw, dt = _chat_completion(msgs, max_tokens=44, llm_url=llm_url)
        total_seconds += dt
        candidate = _first_sentence(raw)
        if not contains_answer_leak(candidate, problem):
            hint = candidate
            leaked = False
            break

    if leaked:
        hint = _first_sentence(rung2["method"])

    return {
        "rung": 3,
        "hint": hint,
        "leaked": leaked,
        "seconds": round(total_seconds, 2),
    }


def get_rung_4(problem, index=None, llm_url=None):
    """Reveal the verified answer and generate a concise step-by-step solution."""
    lang = LANG_NAMES.get(problem.get("language", "eng"), "English")
    label = problem.get("label", "")
    options = problem.get("options") or {}
    correct_val = str(options.get(label, "")).strip() if label else ""
    opts_text = _format_options(problem)

    answer_desc = f"{label}) {correct_val}" if correct_val else label
    system_msg = (
        f"You are a math tutor. Reply in {lang}. Explain clearly in 2 or 3 short sentences "
        f"why {answer_desc} is the correct answer."
    )
    user_msg = (
        f"Problem: {problem.get('text', '')}{opts_text}\n"
        f"Correct answer: {answer_desc}\n"
        "Short step-by-step solution:"
    )

    try:
        explanation, dt = _chat_completion(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=110,
            llm_url=llm_url,
        )
    except (OSError, urllib.error.URLError, KeyError, ValueError):
        rung2 = get_rung_2(problem, index)
        explanation = rung2["method"]
        dt = 0.0

    return {
        "rung": 4,
        "label": label,
        "option_value": correct_val,
        "explanation": explanation,
        "seconds": round(dt, 2),
    }