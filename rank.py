"""Rank short texts with TF-IDF cosine. Jaccard is the count-free baseline."""

from __future__ import annotations

import math
import re

_TOKEN = re.compile(r"[0-9A-Za-z\u0590-\u05FF]+")

DOCS = (
    "alpha alpha alpha",
    "alpha beta gamma",
    "החתול ישן על השטיח",
    "הכלב רץ בחצר",
)
QUERY = "alpha alpha beta"
HE_QUERY = "החתול ישן"


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(text or "")]


def term_freq(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens)
    return {token: count / total for token, count in counts.items()}


def idf_map(docs: list[list[str]]) -> dict[str, float]:
    df: dict[str, int] = {}
    for doc in docs:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1
    n_docs = len(docs)
    return {token: math.log((1 + n_docs) / (1 + count)) + 1.0 for token, count in df.items()}


def tfidf(tokens: list[str], weights: dict[str, float]) -> dict[str, float]:
    return {token: freq * weights.get(token, 0.0) for token, freq in term_freq(tokens).items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right[token] for token in left if token in right)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def char_trigrams(text: str) -> set[str]:
    grams: set[str] = set()
    for token in tokenize(text):
        if len(token) < 3:
            if token:
                grams.add(token)
            continue
        grams.update(token[index : index + 3] for index in range(len(token) - 2))
    return grams


def overlap(left: set[str], right: set[str]) -> float:
    """Share of the shorter trigram set that also sits in the longer one."""
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def dice(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rank(query: str, docs: list[str]) -> list[dict[str, float | int]]:
    tokenized = [tokenize(doc) for doc in docs]
    query_tokens = tokenize(query)
    weights = idf_map(tokenized)
    query_vec = tfidf(query_tokens, weights)
    rows: list[dict[str, float | int]] = []
    for index, tokens in enumerate(tokenized):
        rows.append(
            {
                "index": index,
                "cosine": cosine(query_vec, tfidf(tokens, weights)),
                "jaccard": jaccard(query_tokens, tokens),
                "dice": dice(char_trigrams(query), char_trigrams(docs[index])),
                "overlap": overlap(char_trigrams(query), char_trigrams(docs[index])),
            }
        )
    rows.sort(key=lambda row: (-float(row["cosine"]), int(row["index"])))
    return rows


def format_report(query: str, docs: list[str]) -> str:
    lines = [f"query={query}"]
    for row in rank(query, docs):
        lines.append(
            f"{int(row['index'])}  cosine={float(row['cosine']):.3f}  "
            f"jaccard={float(row['jaccard']):.3f}  dice={float(row['dice']):.3f}  "
            f"overlap={float(row['overlap']):.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    print(format_report(QUERY, list(DOCS[:2])))
    print(format_report(HE_QUERY, list(DOCS[2:])))
    print(format_report("חתול", list(DOCS[2:])))


if __name__ == "__main__":
    main()
