from oro_filtering.stratified_sampler import stratified_sample, BUCKET_WEIGHTS


def _r(bucket, gap, i):
    return {"problem_id": f"p{i}", "_bucket": bucket, "_gap": gap}


def test_sample_hits_target_size():
    rows = [_r("sku_variant_product_inspection", 0.5, i) for i in range(200)]
    out = stratified_sample(rows, target=50)
    assert len(out) == 50


def test_sample_respects_bucket_weights():
    rows = []
    for b in BUCKET_WEIGHTS:
        rows += [_r(b, 0.5, f"{b}_{i}") for i in range(50)]
    out = stratified_sample(rows, target=200)
    counts = {b: sum(1 for r in out if r["_bucket"] == b) for b in BUCKET_WEIGHTS}
    # each bucket within +/-2 of its quota
    for b, w in BUCKET_WEIGHTS.items():
        expected = round(200 * w / sum(BUCKET_WEIGHTS.values()))
        assert abs(counts[b] - expected) <= 2, (b, counts[b], expected)


def test_sample_prefers_largest_gap_within_bucket():
    rows = [_r("sku_variant_product_inspection", gap, i) for i, gap in enumerate([0.1, 0.2, 0.9, 0.05])]
    out = stratified_sample(rows, target=2)
    assert sorted(r["_gap"] for r in out) == [0.2, 0.9]
