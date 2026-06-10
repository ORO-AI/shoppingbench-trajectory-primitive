"""Tests for Stage 2 dedup + pick."""
from __future__ import annotations

import pytest

from scripts.filtration.stage2_dedup_pick import (
    think_depth_score,
    search_refinement_score,
    verification_score,
    step_shape_health_score,
    quality_score,
    shape_signature,
    coarse_shape,
    group_key,
    annotate_row,
    pick_for_problem,
)


def make_messages(assistant_turns):
    """Build a messages list with system+user then provided assistant/tool blocks.

    assistant_turns is a list of dicts each like:
      {"content": "<think text>", "tool_calls": [{"name": "find_product", "q": "..."}]}
    """
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Find me red shoes"},
    ]
    for i, turn in enumerate(assistant_turns):
        tcs = []
        for j, tc in enumerate(turn.get("tool_calls", [])):
            tc_id = f"call_{i}_{j}"
            tcs.append({
                "id": tc_id, "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc.get("arguments", {}),
                },
            })
        asst_msg = {"role": "assistant", "content": turn.get("content", "")}
        if tcs:
            asst_msg["tool_calls"] = tcs
        msgs.append(asst_msg)
        for tc in tcs:
            msgs.append({
                "role": "tool",
                "name": tc["function"]["name"],
                "tool_call_id": tc["id"],
                "content": "{}",
            })
    return msgs


# ----- think_depth -----

def test_think_depth_zero_on_empty_think():
    m = make_messages([
        {"content": "", "tool_calls": [{"name": "recommend_product",
                                          "arguments": {"product_ids": "1"}}]}
    ])
    assert think_depth_score(m) == 0.0


def test_think_depth_clamps_high_to_one():
    m = make_messages([
        {"content": "x" * 2000, "tool_calls": [{"name": "recommend_product",
                                                  "arguments": {"product_ids": "1"}}]}
    ])
    assert think_depth_score(m) == 1.0


def test_think_depth_linear_in_band():
    m = make_messages([
        {"content": "x" * 425,  # midpoint of [50, 800]
         "tool_calls": [{"name": "recommend_product",
                          "arguments": {"product_ids": "1"}}]}
    ])
    score = think_depth_score(m)
    assert 0.4 < score < 0.6


# ----- search_refinement -----

def test_search_refinement_zero_no_search():
    m = make_messages([
        {"content": "go", "tool_calls": [{"name": "recommend_product",
                                            "arguments": {"product_ids": "1"}}]}
    ])
    assert search_refinement_score(m) == 0.0


def test_search_refinement_distinct_queries():
    m = make_messages([
        {"content": "a", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "shoes"}}]},
        {"content": "b", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "red shoes"}}]},
        {"content": "c", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "running"}}]},
        {"content": "d", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    assert search_refinement_score(m) == 1.0  # 3 distinct


def test_search_refinement_dedups_identical_queries():
    m = make_messages([
        {"content": "a", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "shoes"}}]},
        {"content": "b", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "shoes"}}]},
        {"content": "c", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    # Two identical queries collapse to 1 distinct → score 0.3
    assert search_refinement_score(m) == 0.3


# ----- verification -----

def test_verification_one_if_view_called():
    m = make_messages([
        {"content": "a", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "x"}}]},
        {"content": "b", "tool_calls": [{"name": "view_product_information",
                                           "arguments": {"product_id": "1"}}]},
        {"content": "c", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    assert verification_score(m) == 1.0


def test_verification_zero_otherwise():
    m = make_messages([
        {"content": "a", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "x"}}]},
        {"content": "b", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    assert verification_score(m) == 0.0


# ----- step_shape_health -----

def test_step_shape_health_peaks_in_band():
    for n in [3, 5, 8, 12]:
        m = make_messages(
            [{"content": "x", "tool_calls": [{"name": "find_product",
                                                "arguments": {"q": "x"}}]}] * (n - 1)
            + [{"content": "x", "tool_calls": [{"name": "recommend_product",
                                                  "arguments": {"product_ids": "1"}}]}]
        )
        assert step_shape_health_score(m) == 1.0, f"n={n}"


def test_step_shape_health_falls_off_above_band():
    m = make_messages(
        [{"content": "x", "tool_calls": [{"name": "find_product",
                                            "arguments": {"q": f"q{i}"}}]} for i in range(23)]
        + [{"content": "x", "tool_calls": [{"name": "recommend_product",
                                              "arguments": {"product_ids": "1"}}]}]
    )
    # 24 turns: linear falloff. 12 above peak hi=12 → score = 1 - (24-12)/12 = 0.0
    score = step_shape_health_score(m)
    assert score == 0.0


def test_step_shape_health_falls_off_below_band():
    m = make_messages(
        [{"content": "x", "tool_calls": [{"name": "recommend_product",
                                            "arguments": {"product_ids": "1"}}]}]
    )
    # 1 turn vs lo=3 → 1/3
    score = step_shape_health_score(m)
    assert abs(score - 1.0 / 3) < 0.01


# ----- shape signature -----

def test_shape_signature_orders_blocks():
    m = make_messages([
        {"content": "a", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "x"}}]},
        {"content": "b", "tool_calls": [
            {"name": "view_product_information", "arguments": {"product_id": "1"}},
            {"name": "find_product", "arguments": {"q": "y"}},
        ]},
        {"content": "c", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    assert shape_signature(m) == "F|VF|R"


def test_coarse_shape_is_orderless_set():
    m1 = make_messages([
        {"content": "a", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "x"}}]},
        {"content": "b", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    m2 = make_messages([
        {"content": "a", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
        {"content": "b", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "x"}}]},
    ])
    assert coarse_shape(m1) == coarse_shape(m2) == "FR"


# ----- annotate -----

def test_annotate_attaches_keys():
    m = make_messages([
        {"content": "a", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "x"}}]},
        {"content": "b", "tool_calls": [{"name": "view_product_information",
                                           "arguments": {"product_id": "1"}}]},
        {"content": "c", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    row = annotate_row({"messages": m})
    assert "_quality" in row
    assert "_shape" in row
    assert "_coarse_shape" in row
    assert "_problem_query" in row
    assert row["_quality"]["score"] > 0
    assert row["_coarse_shape"] == "FRV"


# ----- pick policy -----

def make_candidate(score, coarse, label=""):
    return {
        "_quality": {"score": score},
        "_coarse_shape": coarse,
        "_problem_query": "q",
        "_label": label,
    }


def test_pick_one_when_single_candidate():
    cs = [make_candidate(0.5, "FR", "A")]
    out = pick_for_problem(cs, max_per=2)
    assert len(out) == 1
    assert out[0]["_label"] == "A"


def test_pick_prefers_different_shape():
    cs = [
        make_candidate(0.9, "FR", "A"),     # top
        make_candidate(0.8, "FR", "B"),     # same shape as A
        make_candidate(0.7, "FVR", "C"),    # different shape, lower
    ]
    out = pick_for_problem(cs, max_per=2)
    labels = [r["_label"] for r in out]
    assert labels == ["A", "C"]  # diversity over score


def test_pick_falls_back_to_top_when_all_share_shape():
    cs = [
        make_candidate(0.9, "FR", "A"),
        make_candidate(0.8, "FR", "B"),
        make_candidate(0.7, "FR", "C"),
    ]
    out = pick_for_problem(cs, max_per=2)
    labels = [r["_label"] for r in out]
    assert labels == ["A", "B"]


def test_pick_handles_three():
    cs = [
        make_candidate(0.9, "FR", "A"),
        make_candidate(0.85, "FVR", "B"),
        make_candidate(0.5, "F", "C"),
    ]
    out = pick_for_problem(cs, max_per=3)
    labels = [r["_label"] for r in out]
    # A picks first. B has different shape. C has yet another. All three picked.
    assert set(labels) == {"A", "B", "C"}


def test_quality_score_components_present():
    m = make_messages([
        {"content": "x" * 200,
         "tool_calls": [{"name": "find_product", "arguments": {"q": "a"}}]},
        {"content": "y" * 200,
         "tool_calls": [{"name": "view_product_information",
                          "arguments": {"product_id": "1"}}]},
        {"content": "z" * 200,
         "tool_calls": [{"name": "recommend_product",
                          "arguments": {"product_ids": "1"}}]},
    ])
    q = quality_score(m)
    assert set(q["parts"].keys()) == {
        "think_depth", "search_refinement", "verification", "step_shape_health"
    }
    assert 0 <= q["score"] <= 1
