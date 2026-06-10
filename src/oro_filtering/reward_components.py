"""Granular reward shaping per Jarrod 2026-05-19.

Decomposes binary trajectory success into four 0.25-weighted components so that
partial-credit trajectories carry training signal instead of being dropped.
"""
from __future__ import annotations

from typing import Any

from oro_filtering.leak_cluster import extract_recommended_product_ids

ALL_COMPONENTS = (
    "tool_call_correct",
    "retrieval_correct",
    "constraint_satisfied",
    "final_recommendation_correct",
)

VALID_TOOLS = {"find_product", "view_product_information", "recommend_product", "terminate"}


def score_components(
    row: dict[str, Any],
    *,
    ground_truth_pids: list[str],
) -> dict[str, float]:
    steps = row.get("steps") or []
    tool_calls = [call for step in steps for call in (step.get("tool_calls") or [])]

    tool_call_correct = 0.25 if any(
        (call or {}).get("name") in VALID_TOOLS for call in tool_calls
    ) else 0.0

    retrieved_pids: set[str] = set()
    for call in tool_calls:
        if (call or {}).get("name") != "find_product":
            continue
        for hit in (call.get("result") or []):
            pid = (hit or {}).get("product_id")
            if pid is not None:
                retrieved_pids.add(str(pid))
    retrieval_correct = 0.25 if any(str(pid) in retrieved_pids for pid in ground_truth_pids) else 0.0

    recommended = extract_recommended_product_ids(steps)
    final_recommendation_correct = (
        0.25 if recommended and set(recommended) == set(map(str, ground_truth_pids)) else 0.0
    )

    score = float(row.get("per_problem_score") or 0.0)
    constraint_satisfied = round(0.25 * score, 4)

    return {
        "tool_call_correct": tool_call_correct,
        "retrieval_correct": retrieval_correct,
        "constraint_satisfied": constraint_satisfied,
        "final_recommendation_correct": final_recommendation_correct,
    }
