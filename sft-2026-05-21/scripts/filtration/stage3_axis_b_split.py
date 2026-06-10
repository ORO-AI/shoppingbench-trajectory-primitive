"""Stage 3 — Axis-B split.

Reads stage2_picked.jsonl. Splits into:
  - Corpus B (sft_corpus_B.jsonl): all Stage 2 picks, no further filtering.
  - Corpus A (sft_corpus_A.jsonl): Stage 2 picks minus any row that fires ANY
    of the five axis-B signals.

Axis-B signals (any one firing = reject from Corpus A):
  1. canned_search          : single distinct find_product query AND >=5 assistant turns
  2. blind_multi_pick       : no view_product_information call AND recommend has >=2 IDs
  3. narration_bloat        : mean think length per turn in top 10% of input corpus
  4. query_parroting        : first find_product query byte-identical to problem_query
  5. structural_clone       : exact shape_signature appears in >5% of input corpus

Signals 3 and 5 need corpus-wide statistics. Computed in one prepass.

Held-out problem set computed AFTER Corpus B is finalized:
  holdout = set(750 ShoppingBench problem queries) - set(unique queries in Corpus B)
Held-out is loaded from a JSONL file if --holdout-source is given; otherwise
skipped (status report will note that held-out wasn't computed and the user
must run stage3 again with --holdout-source).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path


# ----- signal predicates -----

def _iter_assistant(messages):
    return [m for m in messages if m.get("role") == "assistant"]


def _find_queries(messages) -> list[str]:
    """All find_product query strings in order, lowercased + stripped."""
    out: list[str] = []
    for m in _iter_assistant(messages):
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
                out.append(q.strip().lower())
    return out


def _view_called(messages) -> bool:
    for m in _iter_assistant(messages):
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            if fn == "view_product_information":
                return True
    return False


def _recommend_id_count(messages) -> int:
    for m in _iter_assistant(messages):
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            if fn != "recommend_product":
                continue
            args = (tc.get("function") or {}).get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return 0
            if "product_ids" in args:
                raw = args["product_ids"]
                if isinstance(raw, list):
                    return len([x for x in raw if str(x).strip()])
                if isinstance(raw, str):
                    return len([p for p in raw.split(",") if p.strip()])
            if "product_id" in args:
                return 1
    return 0


def _mean_think_length(messages) -> float:
    lengths = []
    for m in _iter_assistant(messages):
        c = m.get("content") or ""
        if isinstance(c, str):
            lengths.append(len(c.strip()))
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def signal_canned_search(messages) -> bool:
    qs = _find_queries(messages)
    n_asst = len(_iter_assistant(messages))
    distinct = len(set(qs))
    return distinct <= 1 and n_asst >= 5


def signal_blind_multi_pick(messages) -> bool:
    return (not _view_called(messages)) and _recommend_id_count(messages) >= 2


def signal_narration_bloat(messages, think_p90: float) -> bool:
    if think_p90 <= 0:
        return False
    return _mean_think_length(messages) >= think_p90


def signal_query_parroting(messages) -> bool:
    qs = _find_queries(messages)
    if not qs:
        return False
    user_query = (messages[1].get("content") or "").strip().lower() if len(messages) >= 2 else ""
    return user_query and qs[0] == user_query


def signal_structural_clone(shape_sig: str, frequent_shapes: set[str]) -> bool:
    return shape_sig in frequent_shapes


# ----- corpus-wide statistics -----

def compute_corpus_stats(rows: list[dict]) -> dict:
    """Compute p90 of mean-think-length + shape-signature frequencies on the
    INPUT corpus (Stage 2 picks). These thresholds are then used for Signal 3
    and Signal 5."""
    think_means = []
    shape_counts: Counter[str] = Counter()
    for r in rows:
        think_means.append(_mean_think_length(r["messages"]))
        shape_counts[r.get("_shape", "")] += 1
    think_means.sort()
    if think_means:
        idx = int(0.90 * len(think_means))
        idx = min(idx, len(think_means) - 1)
        think_p90 = think_means[idx]
    else:
        think_p90 = 0.0
    n = len(rows)
    threshold = max(1, int(0.05 * n))
    frequent_shapes = {sig for sig, c in shape_counts.items() if c > threshold}
    return {
        "think_p90": think_p90,
        "n_rows": n,
        "frequent_shape_threshold_count": threshold,
        "frequent_shapes": sorted(frequent_shapes),
        "shape_count_top10": dict(shape_counts.most_common(10)),
    }


# ----- row-level axis-B check -----

def axis_b_reasons(row: dict, think_p90: float, frequent_shapes: set[str]) -> list[str]:
    msgs = row["messages"]
    reasons = []
    if signal_canned_search(msgs):
        reasons.append("canned_search")
    if signal_blind_multi_pick(msgs):
        reasons.append("blind_multi_pick")
    if signal_narration_bloat(msgs, think_p90):
        reasons.append("narration_bloat")
    if signal_query_parroting(msgs):
        reasons.append("query_parroting")
    shape = row.get("_shape", "")
    if signal_structural_clone(shape, frequent_shapes):
        reasons.append("structural_clone")
    return reasons


# ----- held-out -----

def load_shoppingbench_problem_queries(path: Path | None) -> set[str]:
    """Loads ShoppingBench problem queries from a JSONL file. Each row must
    have a `problem_query` or `query` field. Returns a set of normalized
    lowercased queries.

    Optional. If not provided, held-out is not computed.
    """
    if path is None or not path.exists():
        return set()
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            q = row.get("problem_query") or row.get("query") or ""
            if isinstance(q, str) and q.strip():
                out.add(q.strip().lower())
    return out


# ----- driver -----

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", type=Path,
                   default=Path("data/filtration/stage2_picked.jsonl"))
    p.add_argument("--out-corpus-b", type=Path,
                   default=Path("data/filtration/sft_corpus_B.jsonl"))
    p.add_argument("--out-corpus-a", type=Path,
                   default=Path("data/filtration/sft_corpus_A.jsonl"))
    p.add_argument("--out-axis-b-rejects", type=Path,
                   default=Path("data/filtration/axis_b_rejected.jsonl"))
    p.add_argument("--holdout-source", type=Path, default=None,
                   help="JSONL of ShoppingBench problems with `problem_query` field")
    p.add_argument("--out-holdout", type=Path,
                   default=Path("data/filtration/holdout_problems.json"))
    args = p.parse_args()

    t0 = time.time()
    rows = []
    with open(args.input) as fin:
        for line in fin:
            rows.append(json.loads(line))
    print(f"loaded {len(rows)} rows in {time.time()-t0:.1f}s", file=sys.stderr)

    stats = compute_corpus_stats(rows)
    print("corpus stats:", json.dumps(stats, indent=2), file=sys.stderr)

    frequent_shapes = set(stats["frequent_shapes"])
    think_p90 = stats["think_p90"]

    reasons_count: Counter[str] = Counter()
    n_a = 0
    n_b = 0
    seen_queries_b: set[str] = set()

    with open(args.out_corpus_b, "w") as fb, \
         open(args.out_corpus_a, "w") as fa, \
         open(args.out_axis_b_rejects, "w") as fr:
        for r in rows:
            # Corpus B always gets it.
            fb.write(json.dumps(r) + "\n")
            n_b += 1
            q = r.get("_problem_query", "")
            if q:
                seen_queries_b.add(q.strip().lower())

            reasons = axis_b_reasons(r, think_p90, frequent_shapes)
            if reasons:
                for x in reasons:
                    reasons_count[x] += 1
                r_out = dict(r)
                r_out["_axis_b_reasons"] = reasons
                fr.write(json.dumps(r_out) + "\n")
            else:
                fa.write(json.dumps(r) + "\n")
                n_a += 1

    # Held-out
    holdout_payload: dict = {}
    if args.holdout_source is not None:
        sb = load_shoppingbench_problem_queries(args.holdout_source)
        if sb:
            holdout = sorted(sb - seen_queries_b)
            holdout_payload = {
                "source": str(args.holdout_source),
                "shoppingbench_size": len(sb),
                "seen_in_corpus_b": len(seen_queries_b),
                "holdout_count": len(holdout),
                "holdout_queries": holdout,
            }
            args.out_holdout.write_text(json.dumps(holdout_payload, indent=2))

    summary = {
        "input_rows": len(rows),
        "corpus_b_rows": n_b,
        "corpus_a_rows": n_a,
        "axis_b_rejected": n_b - n_a,
        "axis_b_reasons": dict(reasons_count.most_common()),
        "think_p90_chars": round(think_p90, 1),
        "frequent_shape_threshold_count": stats["frequent_shape_threshold_count"],
        "frequent_shapes_count": len(stats["frequent_shapes"]),
        "shape_count_top10": stats["shape_count_top10"],
        "distinct_queries_in_b": len(seen_queries_b),
        "holdout": (
            {k: v for k, v in holdout_payload.items() if k != "holdout_queries"}
            if holdout_payload else "not_computed_no_source"
        ),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    summary_path = args.out_corpus_b.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
