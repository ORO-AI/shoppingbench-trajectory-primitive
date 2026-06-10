"""Tests for Stage 3 axis-B split."""
from __future__ import annotations

import pytest

from scripts.filtration.stage3_axis_b_split import (
    signal_canned_search,
    signal_blind_multi_pick,
    signal_narration_bloat,
    signal_query_parroting,
    signal_structural_clone,
    axis_b_reasons,
    compute_corpus_stats,
)
from tests.filtration.test_stage2 import make_messages


# ----- canned_search -----

def test_canned_search_fires_on_one_query_many_turns():
    m = make_messages(
        [{"content": "x", "tool_calls": [{"name": "find_product",
                                            "arguments": {"q": "shoes"}}]}] * 4
        + [{"content": "y", "tool_calls": [{"name": "recommend_product",
                                              "arguments": {"product_ids": "1"}}]}]
    )
    # 5 assistant turns, all find_product with same query → fires
    assert signal_canned_search(m) is True


def test_canned_search_passes_with_query_refinement():
    m = make_messages([
        {"content": "x", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "shoes"}}]},
        {"content": "y", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "red shoes"}}]},
        {"content": "z", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "red running shoes"}}]},
        {"content": "w", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "trail running"}}]},
        {"content": "v", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    assert signal_canned_search(m) is False


def test_canned_search_passes_with_few_turns():
    m = make_messages([
        {"content": "x", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "shoes"}}]},
        {"content": "y", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    # 1 query but only 2 turns → doesn't fire
    assert signal_canned_search(m) is False


# ----- blind_multi_pick -----

def test_blind_multi_pick_fires():
    m = make_messages([
        {"content": "x", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "x"}}]},
        {"content": "y", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1,2,3"}}]},
    ])
    assert signal_blind_multi_pick(m) is True


def test_blind_multi_pick_passes_with_view():
    m = make_messages([
        {"content": "x", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "x"}}]},
        {"content": "y", "tool_calls": [{"name": "view_product_information",
                                           "arguments": {"product_id": "1"}}]},
        {"content": "z", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1,2,3"}}]},
    ])
    assert signal_blind_multi_pick(m) is False


def test_blind_multi_pick_passes_on_single_pick():
    m = make_messages([
        {"content": "x", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "x"}}]},
        {"content": "y", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    assert signal_blind_multi_pick(m) is False


# ----- narration_bloat -----

def test_narration_bloat_fires_above_p90():
    m = make_messages([
        {"content": "x" * 900, "tool_calls": [{"name": "find_product",
                                                 "arguments": {"q": "x"}}]},
        {"content": "y" * 900, "tool_calls": [{"name": "recommend_product",
                                                 "arguments": {"product_ids": "1"}}]},
    ])
    # think p90 of 500 → mean 900 fires
    assert signal_narration_bloat(m, think_p90=500.0) is True


def test_narration_bloat_passes_below_p90():
    m = make_messages([
        {"content": "x" * 100, "tool_calls": [{"name": "recommend_product",
                                                 "arguments": {"product_ids": "1"}}]},
    ])
    assert signal_narration_bloat(m, think_p90=500.0) is False


def test_narration_bloat_safe_when_p90_zero():
    m = make_messages([
        {"content": "x" * 5000, "tool_calls": [{"name": "recommend_product",
                                                  "arguments": {"product_ids": "1"}}]},
    ])
    assert signal_narration_bloat(m, think_p90=0.0) is False


# ----- query_parroting -----

def test_query_parroting_fires_when_byte_identical():
    # The user query in make_messages is "Find me red shoes".
    m = make_messages([
        {"content": "thinking", "tool_calls": [{"name": "find_product",
                                                  "arguments": {"q": "Find me red shoes"}}]},
        {"content": "y", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    assert signal_query_parroting(m) is True


def test_query_parroting_passes_when_query_differs():
    m = make_messages([
        {"content": "thinking", "tool_calls": [{"name": "find_product",
                                                  "arguments": {"q": "red running shoes"}}]},
        {"content": "y", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    assert signal_query_parroting(m) is False


def test_query_parroting_passes_when_no_find():
    m = make_messages([
        {"content": "y", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    assert signal_query_parroting(m) is False


# ----- structural_clone -----

def test_structural_clone_fires_on_frequent_shape():
    frequent = {"F|R", "F|V|R"}
    assert signal_structural_clone("F|R", frequent) is True
    assert signal_structural_clone("F|V|F|V|R", frequent) is False


# ----- corpus stats -----

def test_compute_corpus_stats_p90():
    rows = []
    for length in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        m = make_messages([
            {"content": "x" * length,
             "tool_calls": [{"name": "recommend_product",
                              "arguments": {"product_ids": "1"}}]}
        ])
        rows.append({"messages": m, "_shape": f"shape_{length}"})
    stats = compute_corpus_stats(rows)
    # p90 of [100..1000] → index int(0.9*10)=9 → value 1000
    assert stats["think_p90"] == 1000


def test_compute_corpus_stats_frequent_shapes():
    rows = []
    # 100 rows, 80 with shape "F|R" → 80% > 5% threshold → frequent
    for _ in range(80):
        m = make_messages([
            {"content": "x", "tool_calls": [{"name": "find_product",
                                               "arguments": {"q": "x"}}]},
            {"content": "y", "tool_calls": [{"name": "recommend_product",
                                               "arguments": {"product_ids": "1"}}]},
        ])
        rows.append({"messages": m, "_shape": "F|R"})
    for i in range(20):
        m = make_messages([
            {"content": "x", "tool_calls": [{"name": "find_product",
                                               "arguments": {"q": f"q{i}"}}]},
            {"content": "y", "tool_calls": [{"name": "view_product_information",
                                               "arguments": {"product_id": "1"}}]},
            {"content": "z", "tool_calls": [{"name": "recommend_product",
                                               "arguments": {"product_ids": "1"}}]},
        ])
        rows.append({"messages": m, "_shape": f"F|V|R_{i}"})
    stats = compute_corpus_stats(rows)
    assert "F|R" in stats["frequent_shapes"]
    # Each unique F|V|R_i shape appears 1 time, threshold is 5 (5% of 100), so none are frequent
    for i in range(20):
        assert f"F|V|R_{i}" not in stats["frequent_shapes"]


# ----- axis_b_reasons orchestrator -----

def test_axis_b_reasons_returns_all_firing_signals():
    # Build a row that hits canned_search + blind_multi_pick.
    m = make_messages(
        [{"content": "x", "tool_calls": [{"name": "find_product",
                                            "arguments": {"q": "shoes"}}]}] * 4
        + [{"content": "y", "tool_calls": [{"name": "recommend_product",
                                              "arguments": {"product_ids": "1,2"}}]}]
    )
    row = {"messages": m, "_shape": "F|F|F|F|R", "_problem_query": "find me red shoes"}
    reasons = axis_b_reasons(row, think_p90=10000, frequent_shapes=set())
    assert "canned_search" in reasons
    assert "blind_multi_pick" in reasons


def test_axis_b_reasons_clean_on_good_row():
    m = make_messages([
        {"content": "x", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "red running shoes"}}]},
        {"content": "y", "tool_calls": [{"name": "find_product",
                                           "arguments": {"q": "trail runners"}}]},
        {"content": "z", "tool_calls": [{"name": "view_product_information",
                                           "arguments": {"product_id": "1"}}]},
        {"content": "w", "tool_calls": [{"name": "recommend_product",
                                           "arguments": {"product_ids": "1"}}]},
    ])
    row = {"messages": m, "_shape": "F|F|V|R", "_problem_query": "find me red shoes"}
    reasons = axis_b_reasons(row, think_p90=10000, frequent_shapes=set())
    assert reasons == []
