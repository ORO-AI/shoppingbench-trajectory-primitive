import pytest

from oro_filtering.student_gap import (
    GapResult,
    TOOL_SCHEMAS,
    run_gap_batch,
)


def test_compute_gap_dataclass():
    result = GapResult(
        row_id="r1",
        teacher_score=0.8,
        student_score=0.4,
        teacher_components={
            "tool_call_correct": 0.25,
            "retrieval_correct": 0.25,
            "constraint_satisfied": 0.2,
            "final_recommendation_correct": 0.1,
        },
        student_components={
            "tool_call_correct": 0.25,
            "retrieval_correct": 0.0,
            "constraint_satisfied": 0.15,
            "final_recommendation_correct": 0.0,
        },
    )
    assert result.gap == 0.4
    assert result.passes_filter(min_gap=0.05)
    assert not result.passes_filter(min_gap=0.5)


def test_compute_gap_via_property():
    r = GapResult(
        row_id="r1",
        teacher_score=0.9,
        student_score=0.7,
        teacher_components={},
        student_components={},
    )
    assert r.gap == 0.2


def test_run_gap_batch_with_stub_rollout():
    def stub_rollout(arg_tuples):
        return [
            {
                "steps": [
                    {
                        "tool_calls": [
                            {
                                "name": "find_product",
                                "parameters": {"q": "x"},
                                "result": [{"product_id": "111"}],
                            }
                        ],
                        "think": "",
                        "content": "",
                        "response": None,
                    },
                    {
                        "tool_calls": [
                            {
                                "name": "recommend_product",
                                "parameters": {"product_ids": "111"},
                            }
                        ],
                        "think": "",
                        "content": "",
                        "response": None,
                    },
                ],
                "n_turns": 2,
            }
            for _ in arg_tuples
        ]

    rows = [
        {
            "run_id": "r1",
            "problem_id": "p1",
            "problem_query": "find a red sneaker",
            "steps": [
                {
                    "tool_calls": [
                        {
                            "name": "find_product",
                            "parameters": {"q": "sneaker"},
                            "result": [{"product_id": "111"}],
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "name": "recommend_product",
                            "parameters": {"product_ids": "111"},
                        }
                    ]
                },
            ],
            "per_problem_score": 1.0,
        }
    ]
    results = run_gap_batch(rows, student_rollout=stub_rollout)
    assert len(results) == 1
    assert results[0].row_id == "r1"
    assert results[0].teacher_score == 1.0
    assert results[0].student_score == 1.0  # stub matches teacher exactly
    assert results[0].gap == 0.0
    assert not results[0].passes_filter(min_gap=0.05)


def test_run_gap_batch_skips_rows_without_ground_truth():
    def stub_rollout(arg_tuples):
        # Skipped rows shouldn't trigger any rollout calls.
        return [{"steps": [], "n_turns": 0} for _ in arg_tuples]

    rows = [
        {
            "run_id": "r1",
            "problem_id": "p1",
            "problem_query": "x",
            "steps": [],
            "per_problem_score": 0.0,
        }
    ]
    results = run_gap_batch(rows, student_rollout=stub_rollout)
    assert len(results) == 1
    assert results[0].gap == 0.0
    assert not results[0].passes_filter()


def test_run_gap_batch_real_gap_when_student_fails():
    def stub_rollout(arg_tuples):
        return [
            {
                "steps": [
                    {
                        "tool_calls": [
                            {
                                "name": "recommend_product",
                                "parameters": {"product_ids": "999"},
                            }
                        ]
                    }
                ],
                "n_turns": 1,
            }
            for _ in arg_tuples
        ]

    rows = [
        {
            "run_id": "r1",
            "problem_id": "p1",
            "problem_query": "find x",
            "steps": [
                {
                    "tool_calls": [
                        {
                            "name": "find_product",
                            "parameters": {"q": "x"},
                            "result": [{"product_id": "111"}],
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "name": "recommend_product",
                            "parameters": {"product_ids": "111"},
                        }
                    ]
                },
            ],
            "per_problem_score": 1.0,
        }
    ]
    results = run_gap_batch(rows, student_rollout=stub_rollout)
    # Teacher = 1.0 (all 4 components). Student lands tool_call_correct only
    # (0.25 of 3-component max 0.75), rescaled by 4/3 → ~0.333. Gap ~0.667.
    assert results[0].gap > 0.6
    assert results[0].passes_filter(min_gap=0.05)


def test_run_gap_batch_handles_multiple_tool_calls_per_turn():
    """Qwen3 chains tool_calls in one turn; parser must capture all of them."""

    def stub_rollout(arg_tuples):
        # Student emits find_product + view_product_information +
        # recommend_product all in one assistant turn.
        return [
            {
                "steps": [
                    {
                        "tool_calls": [
                            {
                                "name": "find_product",
                                "parameters": {"q": "x"},
                                "result": [{"product_id": "111"}],
                            },
                            {
                                "name": "view_product_information",
                                "parameters": {"product_id": "111"},
                                "result": {},
                            },
                            {
                                "name": "recommend_product",
                                "parameters": {"product_ids": "111"},
                            },
                        ],
                        "think": "",
                        "content": "",
                        "response": None,
                    }
                ],
                "n_turns": 1,
            }
            for _ in arg_tuples
        ]

    rows = [
        {
            "run_id": "r1",
            "problem_id": "p1",
            "problem_query": "find x",
            "steps": [
                {
                    "tool_calls": [
                        {
                            "name": "find_product",
                            "parameters": {"q": "x"},
                            "result": [{"product_id": "111"}],
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "name": "recommend_product",
                            "parameters": {"product_ids": "111"},
                        }
                    ]
                },
            ],
            "per_problem_score": 1.0,
        }
    ]
    results = run_gap_batch(rows, student_rollout=stub_rollout)
    # Student matched: tool_call_correct (0.25) + retrieval_correct (0.25)
    # + final_recommendation_correct (0.25) = 0.75 * (4/3) = 1.0 normalized.
    assert results[0].student_score == 1.0
    assert results[0].gap == 0.0


@pytest.mark.modal
def test_smoke_qwen_rollout_three_rows():
    """Manually-run smoke test. Skipped by default; requires `MODAL_PROFILE=oro-ai modal run`."""
    from oro_filtering.student_gap import Qwen3StudentServer

    server = Qwen3StudentServer()
    out = server.rollout.remote(
        system_prompt="Be brief.",
        user_prompt="Find me a red sneaker under $50.",
        tool_schemas=TOOL_SCHEMAS,
        max_turns=3,
    )
    assert "steps" in out
    assert isinstance(out["n_turns"], int)
