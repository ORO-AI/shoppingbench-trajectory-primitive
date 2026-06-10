from oro_filtering.bucket_tag import classify_row, BUCKETS, classify_batch


class _FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return self.response


def test_classify_row_returns_valid_bucket():
    client = _FakeClient("voucher_math_auditable_provenance")
    bucket = classify_row({"problem_query": "give me a 5% off voucher math"}, client=client)
    assert bucket in BUCKETS
    assert bucket == "voucher_math_auditable_provenance"


def test_classify_row_falls_back_on_invalid_output():
    client = _FakeClient("nonsense_bucket")
    bucket = classify_row({"problem_query": "foo"}, client=client)
    assert bucket in BUCKETS


def test_classify_batch_calls_once_per_row():
    client = _FakeClient("sku_variant_product_inspection")
    rows = [{"problem_query": f"q{i}"} for i in range(5)]
    out = classify_batch(rows, client=client)
    assert client.calls == 5
    assert len(out) == 5
