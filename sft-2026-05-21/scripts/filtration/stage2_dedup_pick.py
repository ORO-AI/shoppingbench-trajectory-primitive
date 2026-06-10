"""Stage 2 — Dedup by user-query text + pick 1.5–2 trajectories per problem.

Input: format_valid.jsonl from Stage 1.
Output: stage2_picked.jsonl + stage2_rejected.jsonl + stats.

Within-problem rank: weighted sum of four structural signals.
  think_depth        — mean <think> chars per assistant turn, clipped to [50, 800]   weight 0.30
  search_refinement  — count of distinct find_product query strings                  weight 0.30
  verification       — 1 if view_product_information called at least once            weight 0.25
  step_shape_health  — assistant-turn count, peaks in [3, 12], linear falloff        weight 0.15

Tool-call shape signature: ordered string of tool-name initials per assistant turn.
  F = find_product
  V = view_product_information
  R = recommend_product
  T = terminate
  Multiple tools in the same assistant turn are concatenated in order.

Pick policy per problem:
  - 1 candidate  → take it
  - >=2          → take highest-ranked, then next-highest with different shape
                   if all share one shape, take top 2 from that shape

Usage:
  python -m scripts.filtration.stage2_dedup_pick \
    --input data/filtration/format_valid.jsonl \
    --out-picked data/filtration/stage2_picked.jsonl \
    --out-rejected data/filtration/stage2_rejected.jsonl \
    --max-per-problem 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# ----- weights (tunable) -----
W_THINK_DEPTH = 0.30
W_SEARCH_REFINEMENT = 0.30
W_VERIFICATION = 0.25
W_STEP_SHAPE = 0.15

# ----- bounds -----
THINK_DEPTH_CLAMP = (50, 800)
STEP_SHAPE_PEAK = (3, 12)


# Tool-name → letter for shape signature
TOOL_LETTER = {
    "find_product": "F",
    "view_product_information": "V",
    "recommend_product": "R",
    "terminate": "T",
}


def _iter_assistant_messages(messages):
    return [m for m in messages if m.get("role") == "assistant"]


def think_depth_score(messages) -> float:
    """Mean chars of assistant.content across assistant turns, clipped + scaled."""
    asst = _iter_assistant_messages(messages)
    if not asst:
        return 0.0
    lengths = []
    for m in asst:
        c = m.get("content") or ""
        if isinstance(c, str):
            lengths.append(len(c.strip()))
    if not lengths:
        return 0.0
    mean = sum(lengths) / len(lengths)
    lo, hi = THINK_DEPTH_CLAMP
    if mean <= lo:
        return 0.0
    if mean >= hi:
        return 1.0
    return (mean - lo) / (hi - lo)


def search_refinement_score(messages) -> float:
    """Number of distinct find_product query strings → log-scaled (0,1)."""
    queries = set()
    for m in _iter_assistant_messages(messages):
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            if fn != "find_product":
                continue
            args = (tc.get("function") or {}).get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            q = args.get("q") or args.get("query") or ""
            if isinstance(q, str) and q.strip():
                queries.add(q.strip().lower())
    n = len(queries)
    if n <= 0:
        return 0.0
    if n == 1:
        return 0.3   # at least one search happened
    if n == 2:
        return 0.7
    return 1.0       # 3+ distinct queries = full credit


def verification_score(messages) -> float:
    """1 if view_product_information called at least once, else 0."""
    for m in _iter_assistant_messages(messages):
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            if fn == "view_product_information":
                return 1.0
    return 0.0


def step_shape_health_score(messages) -> float:
    """Peaks at [3, 12] assistant turns; linear falloff outside."""
    n = len(_iter_assistant_messages(messages))
    lo, hi = STEP_SHAPE_PEAK
    if lo <= n <= hi:
        return 1.0
    if n < lo:
        return max(0.0, n / lo)
    # falloff above hi
    fade = max(0.0, 1.0 - (n - hi) / hi)
    return fade


def quality_score(messages) -> dict:
    """Return {score, parts} where parts has each component."""
    parts = {
        "think_depth": think_depth_score(messages),
        "search_refinement": search_refinement_score(messages),
        "verification": verification_score(messages),
        "step_shape_health": step_shape_health_score(messages),
    }
    score = (
        W_THINK_DEPTH * parts["think_depth"]
        + W_SEARCH_REFINEMENT * parts["search_refinement"]
        + W_VERIFICATION * parts["verification"]
        + W_STEP_SHAPE * parts["step_shape_health"]
    )
    return {"score": round(score, 4), "parts": {k: round(v, 4) for k, v in parts.items()}}


def shape_signature(messages) -> str:
    """Ordered string of tool-name letters, one block per assistant turn,
    separated by '|' so turns are distinguishable. E.g., 'F|FV|R|T'."""
    blocks = []
    for m in _iter_assistant_messages(messages):
        block = ""
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            block += TOOL_LETTER.get(fn, "?")
        if block:
            blocks.append(block)
        else:
            blocks.append(".")  # content-only turn
    return "|".join(blocks)


def coarse_shape(messages) -> str:
    """Set-of-tool-letters used (order-free). Two trajectories share coarse
    shape iff they used the same tools regardless of count or order. Used as
    the diversity-tiebreak signature so {F,V,R} differs from {F,R}."""
    letters = set()
    for m in _iter_assistant_messages(messages):
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            l = TOOL_LETTER.get(fn)
            if l:
                letters.add(l)
    return "".join(sorted(letters))


def group_key(row) -> str:
    """User-query text from the second message."""
    msgs = row.get("messages", [])
    if len(msgs) < 2:
        return ""
    return (msgs[1].get("content") or "").strip()


def annotate_row(row) -> dict:
    """Add _quality, _shape, _coarse_shape, _problem_query to row."""
    q = quality_score(row["messages"])
    row["_quality"] = q
    row["_shape"] = shape_signature(row["messages"])
    row["_coarse_shape"] = coarse_shape(row["messages"])
    row["_problem_query"] = group_key(row)
    return row


def pick_for_problem(candidates: list[dict], max_per: int) -> list[dict]:
    """Pick up to max_per from candidates.

    Rank by quality desc. Take #1. For #2, prefer a different coarse shape
    than #1. Iterate to max_per (default 2)."""
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda r: -r["_quality"]["score"])
    picked: list[dict] = [ranked[0]]
    if max_per <= 1:
        return picked
    used_shapes = {ranked[0]["_coarse_shape"]}
    for r in ranked[1:]:
        if r["_coarse_shape"] not in used_shapes:
            picked.append(r)
            used_shapes.add(r["_coarse_shape"])
            if len(picked) >= max_per:
                return picked
    # No different shape available; backfill with top-ranked remaining.
    if len(picked) < max_per:
        for r in ranked[1:]:
            if r not in picked:
                picked.append(r)
                if len(picked) >= max_per:
                    break
    return picked


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", type=Path,
                   default=Path("data/filtration/format_valid.jsonl"))
    p.add_argument("--out-picked", type=Path,
                   default=Path("data/filtration/stage2_picked.jsonl"))
    p.add_argument("--out-rejected", type=Path,
                   default=Path("data/filtration/stage2_rejected.jsonl"))
    p.add_argument("--max-per-problem", type=int, default=2)
    args = p.parse_args()

    t0 = time.time()
    by_query: dict[str, list[dict]] = defaultdict(list)
    n_in = 0
    n_no_query = 0

    print("loading + annotating...", file=sys.stderr)
    with open(args.input) as fin:
        for line in fin:
            n_in += 1
            row = json.loads(line)
            row = annotate_row(row)
            key = row["_problem_query"]
            if not key:
                n_no_query += 1
                continue
            by_query[key].append(row)

    print(f"  {n_in} rows, {len(by_query)} distinct problems, "
          f"{n_no_query} dropped (no query) in {time.time()-t0:.1f}s",
          file=sys.stderr)

    n_picked = 0
    n_rejected = 0
    shape_counts: dict[str, int] = {}
    per_problem_picks: list[int] = []

    with open(args.out_picked, "w") as fout, \
         open(args.out_rejected, "w") as frej:
        for key, rows in by_query.items():
            picks = pick_for_problem(rows, args.max_per_problem)
            picked_set = {id(r) for r in picks}
            per_problem_picks.append(len(picks))
            for r in rows:
                if id(r) in picked_set:
                    n_picked += 1
                    shape_counts[r["_coarse_shape"]] = (
                        shape_counts.get(r["_coarse_shape"], 0) + 1
                    )
                    fout.write(json.dumps(r) + "\n")
                else:
                    n_rejected += 1
                    r["_reject_reason"] = "stage2_not_picked_per_problem_cap"
                    frej.write(json.dumps(r) + "\n")

    summary = {
        "input_rows": n_in,
        "rows_without_query": n_no_query,
        "distinct_problems": len(by_query),
        "picked_rows": n_picked,
        "rejected_rows": n_rejected,
        "rows_per_problem_mean": (
            round(sum(per_problem_picks) / max(len(per_problem_picks), 1), 2)
        ),
        "rows_per_problem_min": min(per_problem_picks) if per_problem_picks else 0,
        "rows_per_problem_max": max(per_problem_picks) if per_problem_picks else 0,
        "coarse_shape_distribution": dict(sorted(shape_counts.items(),
                                                  key=lambda x: -x[1])),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    summary_path = args.out_picked.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
