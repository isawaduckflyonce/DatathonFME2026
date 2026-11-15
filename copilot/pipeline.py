
"""Deterministic KV-cached Math Copilot: Value-First CoT + Option Verification."""
import os
import socket
import time
import urllib.error

from .config import CONFIG, ROOT
from .llm import URL, answer_letter, generate_cot
from .parser import prepare_problem
from .solver_py import match_value_to_option

TIME_LIMIT = float(CONFIG["TIME_LIMIT_SECONDS"])
FALLBACK = CONFIG["FALLBACK_ANSWER"]
MARGIN = 3.0

_DEFAULT_INDEX = None
_DEFAULT_INDEX_CHECKED = False


def _get_default_index():
    global _DEFAULT_INDEX, _DEFAULT_INDEX_CHECKED
    if not _DEFAULT_INDEX_CHECKED:
        _DEFAULT_INDEX_CHECKED = True
        db_path = os.path.join(ROOT, "data", "index.db")
        if os.path.exists(db_path):
            from .rag import Index
            _DEFAULT_INDEX = Index(db_path)
    return _DEFAULT_INDEX


def _remaining(start, time_limit):
    return time_limit - (time.perf_counter() - start) - MARGIN


def solve(problem, index=None, k=0, exclude_id=None, image_b64=None, llm_url=URL,
          time_limit=TIME_LIMIT):
    """Solve one M3Kang problem deterministically within time_limit seconds."""
    start = time.perf_counter()
    clean_prob = prepare_problem(problem)
    options = clean_prob.get("options") or {}
    has_options = len(options) >= 2

    result = {
        "answer": FALLBACK,
        "status": "ok",
        "prompt_tokens": 0,
        "retrieved": "",
        "examples": [],
        "method": "fallback",
        "debug": "",
    }

    try:
        examples = []
        active_index = index if index is not None else (_get_default_index() if k > 0 else None)
        if active_index is not None and k > 0 and _remaining(start, time_limit) > 30.0:
            hits = active_index.search(clean_prob, k=1, exclude_id=exclude_id, min_score=0.84)
            examples = [prepare_problem(p) for _score, p in hits]
            result["examples"] = hits
            result["retrieved"] = ";".join(f"{p['id']}:{s:.2f}" for s, p in hits)

        rem = _remaining(start, time_limit)
        if has_options and rem >= 14.0:
            raw_cot, parsed_val, cot_letter, timings = generate_cot(
                clean_prob, examples=examples, timeout=rem, image_b64=image_b64, url=llm_url
            )
            result["prompt_tokens"] += timings.get("prompt_n", 0)
            result["debug"] = raw_cot.replace("\n", " | ")

            # 1. High-confidence path: calculated Final value matches one of the 5 options
            val_match = match_value_to_option(parsed_val, options)
            if val_match in {"A", "B", "C", "D", "E"}:
                result["answer"] = val_match
                result["method"] = "cot_value_match"
                result["seconds"] = round(time.perf_counter() - start, 2)
                return result

            # 2. If Final missed all options, run a fast 1-token calibrated check if time allows
            rem_after = _remaining(start, time_limit)
            if rem_after >= 5.0:
                cal_letter, t2 = answer_letter(
                    clean_prob, examples=(), timeout=rem_after, image_b64=image_b64, url=llm_url, debias=True
                )
                result["prompt_tokens"] += t2.get("prompt_n", 0)
                result["answer"] = cal_letter
                result["method"] = "cot_miss_calibrated"
                result["seconds"] = round(time.perf_counter() - start, 2)
                return result

            if cot_letter in {"A", "B", "C", "D", "E"}:
                result["answer"] = cot_letter
                result["method"] = "cot_letter"
                result["seconds"] = round(time.perf_counter() - start, 2)
                return result

        rem = _remaining(start, time_limit)
        if rem >= 3.0:
            letter, timings = answer_letter(
                clean_prob, examples=(), timeout=rem, image_b64=image_b64, url=llm_url, debias=True
            )
            result["answer"] = letter
            result["prompt_tokens"] += timings.get("prompt_n", 0)
            result["method"] = "calibrated_direct"

    except (socket.timeout, TimeoutError):
        result["status"] = "timeout"
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            result["status"] = "timeout"
        else:
            result["status"] = f"error: {error.reason}"

    result["seconds"] = round(time.perf_counter() - start, 2)
    return result
