"""Lightweight trace-quality re-check on top of Seth's pre-filter."""
from __future__ import annotations

import json
from typing import Any


def passes_quality_gate(
    row: dict[str, Any],
    *,
    allow_partial: bool = True,
    min_partial_score: float = 0.3,
) -> tuple[bool, list[str]]:
    """Permissive trace-quality gate. Drops rows that are obviously garbage.

    Returns (ok, reasons). When allow_partial is True (default), rows with
    per_problem_score >= min_partial_score are accepted — this implements
    Jarrod's 2026-05-19 guidance that partial-success trajectories carry
    training signal as long as they clear ~30% of the 4-component reward.
    When allow_partial is False, only per_problem_score >= 1.0 rows pass.

    Reason strings: abnormal_termination, no_assistant_tool_calls,
    tool_call_arguments_not_json, per_problem_score_not_numeric, below_pass_threshold.
    """
    reasons: list[str] = []
    if row.get("abnormal_termination"):
        reasons.append("abnormal_termination")
    steps = row.get("steps") or []
    tool_calls = [call for step in steps for call in (step.get("tool_calls") or [])]
    if not tool_calls:
        reasons.append("no_assistant_tool_calls")
    for call in tool_calls:
        args = (call or {}).get("parameters")
        if isinstance(args, str):
            try:
                json.loads(args)
            except (json.JSONDecodeError, TypeError):
                reasons.append("tool_call_arguments_not_json")
                break
    score = row.get("per_problem_score")
    if score is not None:
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            reasons.append("per_problem_score_not_numeric")
        else:
            threshold = min_partial_score if allow_partial else 1.0
            if score_f < threshold:
                reasons.append("below_pass_threshold")
    return len(reasons) == 0, reasons
