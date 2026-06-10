from oro_filtering.reward_components import score_components, ALL_COMPONENTS


def _row(steps, score=1.0):
    return {"steps": steps, "per_problem_score": score, "problem_query": "find product"}


def test_perfect_trajectory_scores_one():
    steps = [
        {"tool_calls": [{"name": "find_product", "parameters": {"q": "sneakers"}, "result": [{"product_id": "111"}]}]},
        {"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": "111"}}]},
    ]
    scores = score_components(_row(steps), ground_truth_pids=["111"])
    assert scores["tool_call_correct"] == 0.25
    assert scores["retrieval_correct"] == 0.25
    assert scores["constraint_satisfied"] == 0.25
    assert scores["final_recommendation_correct"] == 0.25
    assert sum(scores[c] for c in ALL_COMPONENTS) == 1.0


def test_no_recommend_loses_final_component():
    steps = [{"tool_calls": [{"name": "find_product", "parameters": {"q": "x"}, "result": [{"product_id": "111"}]}]}]
    scores = score_components(_row(steps, score=0.0), ground_truth_pids=["111"])
    assert scores["final_recommendation_correct"] == 0.0
    assert scores["tool_call_correct"] == 0.25  # used a real tool


def test_wrong_recommendation_loses_final_and_retrieval():
    steps = [
        {"tool_calls": [{"name": "find_product", "parameters": {"q": "x"}, "result": [{"product_id": "999"}]}]},
        {"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": "999"}}]},
    ]
    scores = score_components(_row(steps, score=0.0), ground_truth_pids=["111"])
    assert scores["final_recommendation_correct"] == 0.0
    assert scores["retrieval_correct"] == 0.0  # 111 not in any tool result
