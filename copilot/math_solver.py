"""Safe AST math expression parser, equation solver, and A-E option matcher."""
import ast
from fractions import Fraction
import math
import operator
import re

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    "abs": abs,
    "sqrt": math.sqrt,
    "round": round,
    "int": int,
    "floor": math.floor,
    "ceil": math.ceil,
}

_TARGET_NAMES = ("ans", "answer", "result", "res", "total", "final", "out", "x")
_VAR_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_STMT_RE = re.compile(r"^\s*([^=\n]+?)\s*=\s*([^=\n]+?)\s*$")
_NUM_TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def _normalize_expr(text):
    """Normalize unicode operators, decimal commas, units, and implicit multiplication."""
    s = text.strip()
    s = (
        s.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("^", "**")
    )
    # Decimal commas like 3,5 -> 3.5
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    # Strip trailing unit words after a number or closing paren (e.g., "8 kg", "176 cm2")
    s = re.sub(r"(?<=[\d\)])\s*[a-zA-Z°²³]+(?:\d+)?\s*$", "", s)
    # Implicit multiplication: 2x -> 2*x, 2(x+1) -> 2*(x+1), (a)(b) -> (a)*(b)
    s = re.sub(r"(\d)\s*([a-zA-Z\(])", r"\1*\2", s)
    s = re.sub(r"\)\s*(\d|[a-zA-Z\(])", r")*\1", s)
    return s.strip()


def _extract_vars(node):
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
    return names


def _eval_ast(node, env):
    """Safely evaluate an AST node using exact Fractions where possible."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, env)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, int):
            return Fraction(node.value, 1)
        return Fraction(str(node.value))

    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise KeyError(node.id)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_ast(node.operand, env))

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, env)
        right = _eval_ast(node.right, env)
        if type(node.op) in _BIN_OPS:
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ZeroDivisionError
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node.op, ast.Pow):
            if abs(float(right)) > 100:
                raise ValueError("Exponent too large")
            if isinstance(left, Fraction) and isinstance(right, Fraction) and right.denominator == 1:
                return left ** int(right.numerator)
            return Fraction(str(float(left) ** float(right)))

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _SAFE_FUNCS.get(node.func.id)
        if fn and len(node.args) == 1:
            val = float(_eval_ast(node.args[0], env))
            return Fraction(str(fn(val)))

    raise ValueError("Unsupported AST construct")


def _try_linear_system(equations, known):
    """Solve linear equations in unknown variables using exact Fraction Gaussian elimination."""
    unknowns = sorted({v for _, _, vs in equations for v in vs if v not in known})
    if not unknowns or len(unknowns) > 8:
        return {}

    n = len(unknowns)
    rows = []

    for lhs_node, rhs_node, _ in equations:
        def f(vals):
            env = dict(known)
            env.update(zip(unknowns, vals))
            return _eval_ast(lhs_node, env) - _eval_ast(rhs_node, env)

        try:
            c0 = f([Fraction(0)] * n)
            coeffs = []
            for i in range(n):
                unit = [Fraction(0)] * n
                unit[i] = Fraction(1)
                coeffs.append(f(unit) - c0)
            # Linearity check at (1, 1, ..., 1)
            if f([Fraction(1)] * n) != c0 + sum(coeffs):
                continue
            rows.append(coeffs + [-c0])
        except Exception:
            continue

    if not rows:
        return {}

    # Gauss-Jordan elimination to Reduced Row Echelon Form
    m = len(rows)
    pivot_row = 0
    for col in range(n):
        sel = None
        for r in range(pivot_row, m):
            if rows[r][col] != 0:
                sel = r
                break
        if sel is None:
            continue
        rows[pivot_row], rows[sel] = rows[sel], rows[pivot_row]
        piv = rows[pivot_row][col]
        rows[pivot_row] = [x / piv for x in rows[pivot_row]]
        for r in range(m):
            if r != pivot_row and rows[r][col] != 0:
                factor = rows[r][col]
                rows[r] = [rows[r][c] - factor * rows[pivot_row][c] for c in range(n + 1)]
        pivot_row += 1
        if pivot_row >= m:
            break

    solved = {}
    for r in range(m):
        non_zero = [c for c in range(n) if rows[r][c] != 0]
        if len(non_zero) == 1:
            col = non_zero[0]
            solved[unknowns[col]] = rows[r][n]
    return solved


def _try_single_var_search(equations, known):
    """Fallback numeric check for a single non-linear unknown (e.g. x * x = 16)."""
    unknowns = sorted({v for _, _, vs in equations for v in vs if v not in known})
    if len(unknowns) != 1:
        return {}
    var = unknowns[0]
    candidates = [Fraction(k, 2) for k in range(-400, 401)]
    hits = []
    for cand in candidates:
        env = dict(known)
        env[var] = cand
        ok = True
        for lhs_node, rhs_node, _ in equations:
            try:
                if abs(float(_eval_ast(lhs_node, env) - _eval_ast(rhs_node, env))) > 1e-6:
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            hits.append(cand)
    non_neg = [h for h in hits if h >= 0]
    if len(non_neg) == 1:
        return {var: non_neg[0]}
    if len(hits) == 1:
        return {var: hits[0]}
    return {}


def solve_statements(raw_text):
    """Parse equations/assignments and return (resolved_value, target_var, env_dict)."""
    parsed_stmts = []
    last_lhs_var = None

    for line in (raw_text or "").splitlines():
        line = line.strip()
        if not line or line.upper().startswith("ANSWER"):
            continue
        m = _STMT_RE.match(line)
        if not m:
            continue
        lhs_raw, rhs_raw = m.group(1), m.group(2)
        lhs_norm = _normalize_expr(lhs_raw)
        rhs_norm = _normalize_expr(rhs_raw)
        try:
            lhs_ast = ast.parse(lhs_norm, mode="eval")
            rhs_ast = ast.parse(rhs_norm, mode="eval")
        except SyntaxError:
            continue

        lhs_var = lhs_norm if _VAR_RE.match(lhs_norm) else None
        if lhs_var:
            last_lhs_var = lhs_var
        all_vars = _extract_vars(lhs_ast) | _extract_vars(rhs_ast)
        parsed_stmts.append((lhs_var, lhs_ast, rhs_ast, all_vars))

    if not parsed_stmts:
        return None, None, {}

    known = {}
    remaining = []

    # Pass 1: sequential evaluation (handles direct assignments & self-updates like x = x + 2)
    for lhs_var, lhs_ast, rhs_ast, all_vars in parsed_stmts:
        rhs_vars = _extract_vars(rhs_ast)
        if lhs_var and rhs_vars.issubset(known.keys()):
            try:
                known[lhs_var] = _eval_ast(rhs_ast, known)
                continue
            except Exception:
                pass
        remaining.append((lhs_var, lhs_ast, rhs_ast, all_vars))

    # Pass 2: iterative fixed-point & simultaneous equation solving
    for _ in range(6):
        if not remaining:
            break
        progress = False
        next_rem = []
        for lhs_var, lhs_ast, rhs_ast, all_vars in remaining:
            rhs_vars = _extract_vars(rhs_ast)
            if lhs_var and lhs_var not in known and rhs_vars.issubset(known.keys()):
                try:
                    known[lhs_var] = _eval_ast(rhs_ast, known)
                    progress = True
                    continue
                except Exception:
                    pass
            next_rem.append((lhs_var, lhs_ast, rhs_ast, all_vars))
        remaining = next_rem

        if remaining:
            eq_list = [(l, r, vs) for _, l, r, vs in remaining]
            lin_solved = _try_linear_system(eq_list, known)
            if not lin_solved:
                lin_solved = _try_single_var_search(eq_list, known)
            if lin_solved:
                known.update(lin_solved)
                progress = True
                remaining = [
                    item for item in remaining
                    if not (item[0] and item[0] in known)
                ]
        if not progress:
            break

    if not known:
        return None, None, {}

    target_var = None
    for name in _TARGET_NAMES:
        if name in known:
            target_var = name
            break
    if target_var is None and last_lhs_var in known:
        target_var = last_lhs_var
    if target_var is None:
        target_var = list(known.keys())[-1]

    return known[target_var], target_var, known


def format_value(val):
    """Format a Fraction cleanly as an integer, decimal, or fraction string."""
    if val is None:
        return ""
    if isinstance(val, Fraction):
        if val.denominator == 1:
            return str(val.numerator)
        as_float = float(val)
        if abs(as_float - round(as_float, 6)) < 1e-9:
            return f"{as_float:g}"
        return f"{val.numerator}/{val.denominator}"
    return str(val)


def _parse_option_numbers(opt_text):
    """Extract candidate numeric values from an option string (ignoring unit digits like cm2)."""
    s = str(opt_text or "").strip().replace(",", ".")
    # Remove trailing unit exponents like cm2, m3, cm²
    s_clean = re.sub(r"([a-zA-Z°]+)\d+\b", r"\1", s)
    tokens = _NUM_TOKEN_RE.findall(s_clean)
    vals = []
    for tok in tokens:
        try:
            if "/" in tok:
                num, den = tok.split("/", 1)
                if float(den) != 0:
                    vals.append(Fraction(num) / Fraction(den))
            else:
                vals.append(Fraction(tok))
        except Exception:
            continue
    return vals


def match_option(value, options):
    """Return the unique matching option letter ('A'-'E') for `value`, or None."""
    if value is None or not options:
        return None

    target = float(value)
    parsed_opts = {
        letter: _parse_option_numbers(options[letter])
        for letter in "ABCDE"
        if letter in options
    }

    # 1. Match against primary (first) numeric value of each option
    primary_hits = [
        letter for letter, vals in parsed_opts.items()
        if vals and abs(float(vals[0]) - target) <= 1e-6
    ]
    if len(primary_hits) == 1:
        return primary_hits[0]

    # 2. Fallback: match against any token (handles OCR noise like "1 17" -> 17)
    any_hits = [
        letter for letter, vals in parsed_opts.items()
        if any(abs(float(v) - target) <= 1e-6 for v in vals)
    ]
    if len(any_hits) == 1:
        return any_hits[0]

    return None


def solve_and_match(raw_text, options):
    """Solve expressions in `raw_text` and match against `options`.
    Returns (matched_letter_or_None, formatted_value_str).
    """
    val, _, _ = solve_statements(raw_text)
    if val is None:
        return None, ""
    return match_option(val, options), format_value(val)
