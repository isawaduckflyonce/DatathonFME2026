"""Clean OCR option artifacts and evaluate numeric/symbolic math expressions."""
import ast
import copy
import math
import operator
import re

_DENOM_RE = re.compile(r"^(?:\d+(?:√\d+)?|\d*√\d+)$")
_STRAY_LABELS_RE = re.compile(r"^([+-]?\d+(?:[.,]\d+)?(?:/\d+)?)\s+(?:[A-Z]\s*){2,}$")
_QNUM_RE = re.compile(r"^\s*\d+\.\s*")
_LEADING_NUM_WORDS_RE = re.compile(
    r"^([+-]?\d+(?:[.,]\d+)?(?:/\d+)?)\s+[a-zA-Zà-ÿ²³°%]+(?:\s+[a-zA-Zà-ÿ²³°%]+)*$"
)


def clean_text(text):
    return _QNUM_RE.sub("", text.strip())


def repair_options(raw_options):
    """Fix common M3Kang OCR artifacts (stacked fractions glued onto E, stray vertex labels)."""
    if not raw_options:
        return {}
    opts = {k: str(v).strip() for k, v in raw_options.items()}
    if set(opts.keys()) != {"A", "B", "C", "D", "E"}:
        return opts

    # 1. Strip stray diagram labels appended to E (e.g. "20 G H K L M" -> "20")
    m = _STRAY_LABELS_RE.match(opts["E"])
    if m and all(re.match(r"^[+-]?\d+(?:[.,]\d+)?(?:/\d+)?$", opts[k]) for k in "ABCD"):
        opts["E"] = m.group(1)
        return opts

    # 2. Recover stacked fractions where denominators landed at the end of E
    tokens_e = opts["E"].split()
    if len(tokens_e) >= 6:
        tail5 = tokens_e[-5:]
        if all(_DENOM_RE.match(t) for t in tail5) and not any("/" in opts[k] for k in "ABCD"):
            prefix_e = " ".join(tokens_e[:-5]).strip()
            if prefix_e:
                nums = [opts["A"], opts["B"], opts["C"], opts["D"], prefix_e]
                for letter, num, den in zip("ABCDE", nums, tail5):
                    opts[letter] = f"({num})/{den}" if ("+" in num or "-" in num[1:]) else f"{num}/{den}"
                return opts

    # 3. Case where 4 of the 5 options are fractions and 1 is a small integer
    if len(tokens_e) >= 5:
        tail4 = tokens_e[-4:]
        if all(_DENOM_RE.match(t) for t in tail4) and not any("/" in opts[k] for k in "ABCD"):
            int_keys = [k for k in "ABCD" if re.match(r"^\d+$", opts[k]) and int(opts[k]) <= 12]
            other_keys = [k for k in "ABCD" if k not in int_keys]
            if len(int_keys) == 1 and all(re.match(r"^\d+$", opts[k]) and int(opts[k]) > 12 for k in other_keys):
                prefix_e = " ".join(tokens_e[:-4]).strip()
                if prefix_e and re.match(r"^\d+$", prefix_e):
                    frac_keys = other_keys + ["E"]
                    nums = [opts[k] for k in other_keys] + [prefix_e]
                    for letter, num, den in zip(frac_keys, nums, tail4):
                        opts[letter] = f"{num}/{den}"
                    return opts

    return opts


def prepare_problem(problem):
    """Return a shallow-copied problem with cleaned text and repaired options."""
    p = copy.copy(problem)
    p["text"] = clean_text(problem.get("text", ""))
    p["options"] = repair_options(problem.get("options") or {})
    return p


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def eval_numeric(expr):
    """Safely evaluate a numeric/radical/pi option string into a float, or return None."""
    if expr is None:
        return None
    s = str(expr).strip()
    if not s or len(s) > 48:
        return None

    # Strip leading "B) " or "x = " if the model wrote an assignment in Final:
    s = re.sub(r"^[A-E]\)\s*", "", s)
    if "=" in s:
        s = s.split("=")[-1].strip()

    # Extract leading number when followed by units/words (e.g. "216 cm.", "110 puntes", "45°")
    s = s.rstrip(".")
    m_words = _LEADING_NUM_WORDS_RE.match(s)
    if m_words:
        s = m_words.group(1)

    s = s.replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*").replace("÷", "/")
    s = s.replace(",", ".")
    s = re.sub(r"^([+-]?\d+)\s+(\d+/\d+)$", r"(\1+\2)", s)
    s = re.sub(r"(\d)\s*√\s*(\d+)", r"\1*sqrt(\2)", s)
    s = re.sub(r"√\s*(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"(\d)\s*π", r"\1*pi", s)
    s = s.replace("π", "pi")
    s = re.sub(r"(\d)\s*\(", r"\1*(", s)
    s = s.replace("^", "**")

    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return None

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            if isinstance(node.op, ast.Pow):
                exp = _eval(node.right)
                if abs(exp) > 20:
                    raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
            if node.func.id == "sqrt":
                val = _eval(node.args[0])
                if val < 0:
                    raise ValueError("negative sqrt")
                return math.sqrt(val)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        raise ValueError("unsupported syntax")

    try:
        val = _eval(tree)
        return val if math.isfinite(val) else None
    except Exception:
        return None
