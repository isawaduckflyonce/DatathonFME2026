"""Hybrid retrieval index in SQLite: dense embeddings + math/lexical overlap + similarity gating."""
import json
import re
import sqlite3
from array import array

from .embed import embed
from .llm import format_problem

SCHEMA = """
CREATE TABLE IF NOT EXISTS problems (
    key TEXT PRIMARY KEY,   -- id|language
    problem TEXT NOT NULL,  -- the full problem as JSON
    vec BLOB NOT NULL       -- float32 vector
)"""

_TOKEN_RE = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)


def _math_tokens(text):
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 3 or t.isdigit()}


def build(db_path, problems, batch=16):
    db = sqlite3.connect(db_path)
    db.execute(SCHEMA)
    for i in range(0, len(problems), batch):
        chunk = problems[i:i + batch]
        vectors = embed([format_problem(p, include_answer_cue=False) for p in chunk])
        db.executemany("INSERT OR REPLACE INTO problems VALUES (?, ?, ?)", [
            (f"{p['id']}|{p['language']}", json.dumps(p, ensure_ascii=False), array("f", v).tobytes())
            for p, v in zip(chunk, vectors)])
        db.commit()
        yield min(i + batch, len(problems))
    db.close()


class Index:
    """Loads vectors + precomputed math tokens into memory for sub-millisecond hybrid search."""

    def __init__(self, db_path):
        db = sqlite3.connect(db_path)
        self.rows = []
        for key, problem_json, blob in db.execute("SELECT key, problem, vec FROM problems"):
            vec = array("f")
            vec.frombytes(blob)
            prob = json.loads(problem_json)
            tokens = _math_tokens(prob.get("text", ""))
            self.rows.append((key, prob, vec, tokens))
        db.close()

    def search(self, problem, k=1, exclude_id=None, min_score=0.76):
        """Return up to k high-confidence (score, problem) pairs."""
        if k <= 0 or not self.rows:
            return []
        q_text = format_problem(problem, include_answer_cue=False)
        query = embed([q_text], query=True)[0]
        q_tokens = _math_tokens(problem.get("text", ""))
        q_level = problem.get("level")
        q_text_only = problem.get("text_only")

        scored = []
        for _key, stored, vec, s_tokens in self.rows:
            if exclude_id and stored["id"] == exclude_id:
                continue
            if q_text_only is not None and stored.get("text_only") != q_text_only:
                continue

            cos = sum(a * b for a, b in zip(query, vec))
            union = len(q_tokens | s_tokens)
            jacc = (len(q_tokens & s_tokens) / union) if union else 0.0
            lvl_bonus = 0.04 if (q_level is not None and stored.get("level") == q_level) else 0.0
            score = 0.78 * cos + 0.22 * jacc + lvl_bonus

            if score >= min_score:
                scored.append((score, stored))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]
