"""AST-validated Python sandbox for Program-Aided Language (PAL) math solving."""
import ast
import fractions
import io
import itertools
import math
import re
import signal
import sys

from .parser import eval_numeric

_ALLOWED_MODULES = {"math", "fractions", "itertools"}
_SAFE_BUILTINS = {
    "range": range, "len": len, "sum": sum, "min": min, "max": max,
    "abs": abs, "round": round, "int": int, "float": float, "str": str,
    "bool": bool, "list": list, "set": set, "tuple": tuple, "dict": dict,
    "sorted": sorted, "reversed": reversed, "enumerate": enumerate, "zip": zip,
    "any": any, "all": all, "map": map, "filter": filter, "pow": pow,
    "divmod": divmod, "bin": bin, "oct": oct, "hex": hex,
}


def extract_code(raw_text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", raw_text, flags=re.DOTALL)
    return (m.group(1) if m else raw_text).strip()


def validate_ast(code):
    """Ensure the generated snippet is pure math/logic with no I/O or dangerous calls."""
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_MODULES:
                    raise ValueError(f"blocked import: {alias.name}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError("blocked dunder attribute")
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in {"open", "eval", "exec", "compile", "input", "__import__"}:
                raise ValueError(f"blocked name: {node.id}")
    return tree


def run_sandboxed(code, max_steps=250_000, timeout_sec=1):
    """Execute validated math code and return the last non-empty printed line, or None."""
    try:
        tree = validate_ast(code)
    except Exception:
        return None

    stdout_buf = io.StringIO()
    safe_globals = {
        "__builtins__": {
            **_SAFE_BUILTINS,
            "print": lambda *a, **kw: print(*a, file=stdout_buf, **kw),
        },
        "math": math,
        "fractions": fractions,
        "Fraction": fractions.Fraction,
        "itertools": itertools,
        "comb": math.comb,
        "perm": math.perm,
        "gcd": math.gcd,
        "lcm": math.lcm,
        "factorial": math.factorial,
        "sqrt": math.sqrt,
    }

    steps = 0

    def _tracer(frame, event, arg):
        nonlocal steps
        if event == "line":
            steps += 1
            if steps > max_steps:
                raise RuntimeError("step limit exceeded")
        return _tracer

    has_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if has_alarm:
        def _on_alarm(signum, frame):
            raise TimeoutError("execution timeout")
        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(timeout_sec)

    old_trace = sys.gettrace()
    try:
        sys.settrace(_tracer)
        exec(compile(tree, "<pal>", "exec"), safe_globals, {})
    except Exception:
        return None
    finally:
        sys.settrace(old_trace)
        if has_alarm:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    lines = [line.strip() for line in stdout_buf.getvalue().splitlines() if line.strip()]
    return lines[-1] if lines else None


def match_value_to_option(value_str, options):
    """Return the unique option letter A..E matching value_str, or None if ambiguous/unmatched."""
    if not value_str or not options:
        return None
    cleaned = value_str.strip().rstrip(".")
    # Reject bare letter guesses so PAL cannot bypass computation by printing "B"
    if cleaned in {"A", "B", "C", "D", "E"} and set(options.values()) != {"A", "B", "C", "D", "E"}:
        return None

    # 1. Exact string match (case-insensitive, whitespace-normalized)
    norm_val = re.sub(r"\s+", "", cleaned).lower()
    exact_hits = [
        letter for letter, opt in options.items()
        if re.sub(r"\s+", "", str(opt)).lower() == norm_val
    ]
    if len(exact_hits) == 1:
        return exact_hits[0]

    # 2. Numeric / fractional / radical match
    target_num = eval_numeric(cleaned)
    if target_num is None:
        return None

    num_hits = []
    for letter, opt in options.items():
        opt_num = eval_numeric(opt)
        if opt_num is not None:
            tol = max(1e-6, 1e-5 * max(abs(target_num), abs(opt_num)))
            if abs(target_num - opt_num) <= tol:
                num_hits.append(letter)
    return num_hits[0] if len(num_hits) == 1 else None
