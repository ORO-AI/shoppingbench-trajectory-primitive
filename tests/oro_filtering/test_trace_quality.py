from oro_filtering.trace_quality import passes_quality_gate


def test_accepts_clean_row():
    row = {
        "problem_query": "find sneakers",
        "steps": [
            {"tool_calls": [{"name": "find_product", "parameters": {"q": "sneakers"}}]},
            {"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": "111"}}]},
        ],
        "abnormal_termination": False,
        "n_steps": 2,
    }
    ok, reasons = passes_quality_gate(row)
    assert ok, reasons


def test_rejects_no_tool_calls():
    row = {"problem_query": "x", "steps": [{"think": "hmm"}], "abnormal_termination": False, "n_steps": 1}
    ok, reasons = passes_quality_gate(row)
    assert not ok
    assert "no_assistant_tool_calls" in reasons


def test_rejects_abnormal_termination():
    row = {"problem_query": "x", "steps": [], "abnormal_termination": True, "n_steps": 0}
    ok, reasons = passes_quality_gate(row)
    assert not ok
    assert "abnormal_termination" in reasons


def test_partial_success_allowed_when_flag_set():
    row = {
        "problem_query": "x",
        "steps": [{"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": "111"}}]}],
        "abnormal_termination": False,
        "n_steps": 1,
        "per_problem_score": 0.4,
    }
    ok, _ = passes_quality_gate(row, allow_partial=True, min_partial_score=0.3)
    assert ok
    ok, reasons = passes_quality_gate(row, allow_partial=False)
    assert not ok
    assert "below_pass_threshold" in reasons


def test_rejects_unparseable_string_arguments():
    row = {
        "problem_query": "x",
        "steps": [{"tool_calls": [{"name": "find_product", "parameters": "{not json"}]}],
        "abnormal_termination": False, "n_steps": 1,
    }
    ok, reasons = passes_quality_gate(row)
    assert not ok
    assert "tool_call_arguments_not_json" in reasons


def test_rejects_non_numeric_score():
    row = {
        "problem_query": "x",
        "steps": [{"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": "111"}}]}],
        "abnormal_termination": False, "n_steps": 1,
        "per_problem_score": "not a number",
    }
    ok, reasons = passes_quality_gate(row)
    assert not ok
    assert "per_problem_score_not_numeric" in reasons
