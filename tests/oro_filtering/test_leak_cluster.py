from oro_filtering.leak_cluster import build_cluster_id, extract_recommended_product_ids, title_fingerprint


def test_same_problem_id_same_cluster():
    row_a = {"problem_id": "p1", "problem_query": "Find sneakers", "steps": []}
    row_b = {"problem_id": "p1", "problem_query": "Locate sneakers", "steps": []}
    assert build_cluster_id(row_a) == build_cluster_id(row_b)


def test_same_product_id_same_cluster():
    rec = lambda pids: [{"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": pids}}]}]
    row_a = {"problem_id": "p1", "problem_query": "X", "steps": rec("123,456")}
    row_b = {"problem_id": "p2", "problem_query": "Y", "steps": rec("123,456")}
    assert build_cluster_id(row_a) == build_cluster_id(row_b)


def test_extract_pids_from_list_or_string():
    steps_list = [{"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": ["123", "456"]}}]}]
    steps_str = [{"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": "123,456"}}]}]
    assert extract_recommended_product_ids(steps_list) == ["123", "456"]
    assert extract_recommended_product_ids(steps_str) == ["123", "456"]


def test_title_fingerprint_is_first_12_tokens_lowercased():
    q = "Find ME the BEST shoes for running with maximum cushion and at least 4 stars for under fifty dollars"
    assert title_fingerprint(q).split() == ["find", "me", "the", "best", "shoes", "for", "running", "with", "maximum", "cushion", "and", "at"]


def test_different_query_no_shared_product_different_cluster():
    rec = lambda pids: [{"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": pids}}]}]
    row_a = {"problem_id": "p1", "problem_query": "alpha bravo charlie", "steps": rec("111")}
    row_b = {"problem_id": "p2", "problem_query": "delta echo foxtrot", "steps": rec("222")}
    assert build_cluster_id(row_a) != build_cluster_id(row_b)


def test_cluster_dimensions_exposes_three_signals():
    from oro_filtering.leak_cluster import cluster_dimensions
    rec = lambda pids: [{"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": pids}}]}]
    row = {"problem_id": "p1", "problem_query": "Find ME some RED sneakers please", "steps": rec("123")}
    dims = cluster_dimensions(row)
    assert dims["problem_id"] == "p1"
    assert dims["product_ids"] == frozenset({"123"})
    assert dims["title_fingerprint"].startswith("find me some red sneakers please")


def test_product_id_priority_over_problem_id():
    rec = lambda pids: [{"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": pids}}]}]
    row_a = {"problem_id": "different_a", "problem_query": "qa", "steps": rec("999")}
    row_b = {"problem_id": "different_b", "problem_query": "qb", "steps": rec("999")}
    assert build_cluster_id(row_a) == build_cluster_id(row_b)  # product_ids wins despite different problem_ids


def test_holdout_split_proportions():
    from oro_filtering.leak_cluster import split_by_holdout
    rows = [
        {"problem_id": f"p{i}", "problem_query": f"q{i}", "steps": []}
        for i in range(100)
    ]
    train, eval_, dropped = split_by_holdout(rows, holdout_fraction=0.15, seed=42)
    assert len(train) + len(eval_) + len(dropped) == 100
    assert 12 <= len(eval_) <= 18  # roughly 15%
    # No problem_id appears in both train and eval
    assert {r["problem_id"] for r in train} & {r["problem_id"] for r in eval_} == set()


def test_holdout_cluster_expansion_drops_mutation():
    from oro_filtering.leak_cluster import split_by_holdout
    rec = lambda pids: [{"tool_calls": [{"name": "recommend_product", "parameters": {"product_ids": pids}}]}]
    rows = [
        {"problem_id": "eval_seed", "problem_query": "anchor query", "steps": rec("999")},
        {"problem_id": "mutation", "problem_query": "anchor query rephrased", "steps": rec("999")},
        {"problem_id": "unrelated", "problem_query": "totally different", "steps": rec("111")},
    ]
    # Find a seed that puts eval_seed into eval (deterministic search, not RNG-dependent).
    for candidate_seed in range(10):
        train, eval_, dropped = split_by_holdout(rows, holdout_fraction=0.34, seed=candidate_seed)
        eval_pids = {r["problem_id"] for r in eval_}
        if "eval_seed" in eval_pids:
            break
    else:
        raise AssertionError("could not find a seed that selects eval_seed within 10 tries")
    # Now the assertions ALWAYS run.
    train_pids = {r["problem_id"] for r in train}
    dropped_pids = {r["problem_id"] for r in dropped}
    assert "mutation" not in train_pids
    assert "mutation" in dropped_pids or "mutation" in eval_pids
