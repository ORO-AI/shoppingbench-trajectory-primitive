"""Unit tests for Stage 1 format-validity gate.

Run from repo root:
  .venv/bin/python -m pytest tests/filtration/test_stage1.py -v
"""
from __future__ import annotations

import json
import pytest

from scripts.filtration.stage1_format_valid import (
    check_tool_call_pairing,
    check_recommend_count,
    check_recommend_args,
    audit_role_alternation,
    check_row,
)


# ----- golden good row -----

def make_good_row():
    return {
        "messages": [
            {"role": "system", "content": "You are Teacher..."},
            {"role": "user", "content": "Find me red running shoes under $100"},
            {
                "role": "assistant",
                "content": "I'll search for red running shoes.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "find_product", "arguments": {"q": "red running shoes"}},
                }],
            },
            {
                "role": "tool", "name": "find_product", "tool_call_id": "call_1",
                "content": '[{"product_id": "12345", "title": "Red Shoe"}]',
            },
            {
                "role": "assistant",
                "content": "Found a match. Let me verify.",
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "view_product_information",
                                 "arguments": {"product_id": "12345"}},
                }],
            },
            {
                "role": "tool", "name": "view_product_information", "tool_call_id": "call_2",
                "content": '{"price": 95}',
            },
            {
                "role": "assistant",
                "content": "Recommending now.",
                "tool_calls": [{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "recommend_product",
                                 "arguments": {"product_ids": "12345"}},
                }],
            },
            {
                "role": "tool", "name": "recommend_product", "tool_call_id": "call_3",
                "content": '{"status": "received"}',
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "terminate", "arguments": {}},
                }],
            },
            {
                "role": "tool", "name": "terminate", "tool_call_id": "call_4",
                "content": '{"status": "terminated"}',
            },
        ],
        "task_name": "trac_shoppingbench_sft",
        "tools": [],
        "trac_export_version": "trac-nemo-sft-openai-v0.1-oro-converted",
    }


# ----- pair check tests -----

def test_pairing_passes_on_good():
    assert check_tool_call_pairing(make_good_row()["messages"]) is None


def test_pairing_rejects_orphan_assistant_call():
    r = make_good_row()
    # Add an extra tool_call to an assistant but no matching tool response.
    r["messages"][2]["tool_calls"].append({
        "id": "call_orphan",
        "type": "function",
        "function": {"name": "find_product", "arguments": {"q": "x"}},
    })
    assert check_tool_call_pairing(r["messages"]).startswith("tool_call_mismatch")


def test_pairing_rejects_orphan_tool_msg():
    r = make_good_row()
    r["messages"].insert(2, {
        "role": "tool", "name": "find_product", "tool_call_id": "call_nonexistent",
        "content": '{"x": 1}',
    })
    # tool response BEFORE the matching assistant tool_call is also an orphan
    assert check_tool_call_pairing(r["messages"]).startswith("tool_call_mismatch")


def test_pairing_rejects_missing_id_on_assistant():
    r = make_good_row()
    r["messages"][2]["tool_calls"][0].pop("id")
    assert check_tool_call_pairing(r["messages"]).startswith("tool_call_mismatch")


# ----- recommend_count tests -----

def test_recommend_count_passes_on_one():
    assert check_recommend_count(make_good_row()["messages"]) is None


def test_recommend_count_rejects_zero():
    r = make_good_row()
    # Replace the recommend_product call with a find_product call.
    r["messages"][6]["tool_calls"][0]["function"]["name"] = "find_product"
    assert check_recommend_count(r["messages"]) == "recommend_count:0"


def test_recommend_count_rejects_two():
    r = make_good_row()
    # Insert a second recommend_product before the terminate.
    r["messages"].insert(8, {
        "role": "assistant",
        "content": "Updating",
        "tool_calls": [{
            "id": "call_dup",
            "type": "function",
            "function": {"name": "recommend_product",
                         "arguments": {"product_ids": "67890"}},
        }],
    })
    r["messages"].insert(9, {
        "role": "tool", "name": "recommend_product", "tool_call_id": "call_dup",
        "content": '{"status": "received"}',
    })
    assert check_recommend_count(r["messages"]) == "recommend_count:2"


# ----- recommend_args tests -----

def test_recommend_args_passes_on_csv_string():
    assert check_recommend_args(make_good_row()["messages"]) is None


def test_recommend_args_passes_on_list():
    r = make_good_row()
    r["messages"][6]["tool_calls"][0]["function"]["arguments"]["product_ids"] = ["12345", "67890"]
    assert check_recommend_args(r["messages"]) is None


def test_recommend_args_passes_on_product_id_single():
    r = make_good_row()
    r["messages"][6]["tool_calls"][0]["function"]["arguments"] = {"product_id": "12345"}
    assert check_recommend_args(r["messages"]) is None


def test_recommend_args_rejects_empty():
    r = make_good_row()
    r["messages"][6]["tool_calls"][0]["function"]["arguments"]["product_ids"] = ""
    assert check_recommend_args(r["messages"]).startswith("recommend_args_invalid")


def test_recommend_args_rejects_non_digit():
    r = make_good_row()
    r["messages"][6]["tool_calls"][0]["function"]["arguments"]["product_ids"] = "not_an_id"
    assert check_recommend_args(r["messages"]).startswith("recommend_args_invalid:non_digit")


def test_recommend_args_rejects_no_product_field():
    r = make_good_row()
    r["messages"][6]["tool_calls"][0]["function"]["arguments"] = {"other": "x"}
    assert check_recommend_args(r["messages"]).startswith("recommend_args_invalid:no_product_field")


def test_recommend_args_accepts_json_string_args():
    """Some upstream sources serialize arguments as JSON-encoded strings."""
    r = make_good_row()
    r["messages"][6]["tool_calls"][0]["function"]["arguments"] = json.dumps({"product_ids": "12345"})
    assert check_recommend_args(r["messages"]) is None


# ----- audit role_alternation tests -----

def test_audit_passes_on_good():
    assert audit_role_alternation(make_good_row()["messages"]) is None


def test_audit_flags_missing_system():
    r = make_good_row()
    r["messages"][0]["role"] = "user"
    assert audit_role_alternation(r["messages"]).startswith("role_alternation:expected_system")


def test_audit_flags_tool_without_assistant():
    r = make_good_row()
    # Insert a stray tool message right after user.
    r["messages"].insert(2, {
        "role": "tool", "name": "x", "tool_call_id": "call_x", "content": "{}",
    })
    assert audit_role_alternation(r["messages"]).startswith("role_alternation")


# ----- check_row orchestrator -----

def test_check_row_passes_on_good_without_tokenizer():
    reject, audit = check_row(make_good_row(), counter=None, max_length=14336)
    assert reject is None
    assert audit is None


def test_check_row_returns_first_failing_check():
    r = make_good_row()
    # Break two things at once: pairing and recommend.
    r["messages"][6]["tool_calls"][0]["id"] = "call_99"  # break pairing
    r["messages"][6]["tool_calls"][0]["function"]["name"] = "find_product"  # remove recommend
    reject, _ = check_row(r, counter=None, max_length=14336)
    assert reject is not None
    # Pairing check runs first.
    assert reject.startswith("tool_call_mismatch")
