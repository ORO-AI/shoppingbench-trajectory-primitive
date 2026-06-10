"""Leak-cluster fingerprinting adapted from Trac's freeze_sft_eval_splits.py.

A leak cluster groups rows that share enough surface features (problem_id,
recommended product_ids, query-title fingerprint) that training on one and
evaluating on another would let the model memorize rather than generalize.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Digit-only: avoids extracting tokens like "none", "null", or "see notes" as product IDs.
_PRODUCT_ID_RE = re.compile(r"\d+")


def title_fingerprint(query: str) -> str:
    tokens = _TOKEN_RE.findall((query or "").lower())
    return " ".join(tokens[:12])


def extract_recommended_product_ids(steps: list[dict[str, Any]]) -> list[str]:
    pids: list[str] = []
    for step in steps or []:
        for call in step.get("tool_calls") or []:
            if (call or {}).get("name") != "recommend_product":
                continue
            params = call.get("parameters", {}) or {}
            raw = params.get("product_ids") or params.get("product_id") or ""
            if isinstance(raw, list):
                pids.extend(str(x) for x in raw)
            else:
                pids.extend(_PRODUCT_ID_RE.findall(str(raw)))
    return sorted(set(pids))


def build_cluster_id(row: dict[str, Any]) -> str:
    """Return a priority-OR fingerprint for the row.

    Priority order: product_ids > problem_id > title_fingerprint. The highest-
    priority non-empty signal wins; lower-priority signals are NOT mixed in.

    This means two rows can belong to the same true leak cluster yet receive
    different fingerprints, e.g., rows sharing only title_fingerprint won't
    collide if either has a problem_id. Use `cluster_dimensions` + union-find
    when full multi-dimension leak detection is needed (see Task 3).
    """
    pids = extract_recommended_product_ids(row.get("steps", []))
    if pids:
        material = "pids|" + ",".join(pids)
    elif row.get("problem_id"):
        material = "pid|" + str(row["problem_id"])
    else:
        material = "title|" + title_fingerprint(row.get("problem_query", ""))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def cluster_dimensions(row: dict[str, Any]) -> dict[str, Any]:
    """Return the three independent leak-cluster dimensions for a row.

    Task 3 uses these to detect leakage across rows that share ANY dimension,
    which is stricter than the single-hash build_cluster_id can express.
    """
    return {
        "problem_id": str(row.get("problem_id", "")),
        "product_ids": frozenset(extract_recommended_product_ids(row.get("steps", []))),
        "title_fingerprint": title_fingerprint(row.get("problem_query", "")),
    }


def split_by_holdout(
    rows: list[dict[str, Any]],
    *,
    holdout_fraction: float = 0.15,
    seed: int = 17,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into (train, eval, dropped_due_to_leak).

    Selects holdout_fraction of distinct problem_ids as the eval seed, then expands
    each seed to its full leak cluster. Any row sharing a cluster with an eval seed
    goes to dropped (if its problem_id isn't itself in eval) or eval.

    Rows missing `problem_id` are treated as cluster-blank and routed to train unless
    they share a cluster fingerprint with an eval seed.
    """
    rng = random.Random(seed)
    by_pid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pid[row.get("problem_id", "")].append(row)
    distinct_pids = sorted(by_pid.keys())
    rng.shuffle(distinct_pids)
    n_eval_pids = max(1, int(len(distinct_pids) * holdout_fraction))
    eval_pids = set(distinct_pids[:n_eval_pids])

    cluster_ids: list[str] = [build_cluster_id(row) for row in rows]
    clusters_with_eval_pid: set[str] = {
        cluster_ids[i] for i, row in enumerate(rows) if row.get("problem_id", "") in eval_pids
    }

    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        cid = cluster_ids[i]
        pid = row.get("problem_id", "")
        if pid in eval_pids:
            eval_rows.append(row)
        elif cid in clusters_with_eval_pid:
            dropped.append(row)
        else:
            train.append(row)
    return train, eval_rows, dropped
