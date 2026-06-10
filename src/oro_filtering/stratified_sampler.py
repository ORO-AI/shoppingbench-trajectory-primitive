"""Bucket-stratified gap-preferring sampler.

Picks `target` rows distributed across Trac's 7 capability buckets according to
their weights; within each bucket, prefers rows with the largest student-teacher gap.
"""
from __future__ import annotations
from typing import Any

BUCKET_WEIGHTS = {
    "sku_variant_product_inspection": 62,
    "same_shop_full_constraint_reasoning": 55,
    "knowledge_bridge_literal_sourced": 40,
    "voucher_math_auditable_provenance": 35,
    "recommendation_provenance_multi_id": 30,
    "robust_search_exact_constraint_reformulation": 20,
    "benchmark_artifact_resilience_probe": 8,
}


def stratified_sample(rows: list[dict[str, Any]], *, target: int) -> list[dict[str, Any]]:
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_bucket.setdefault(row["_bucket"], []).append(row)
    total_weight = sum(BUCKET_WEIGHTS.values())
    out: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for bucket, weight in BUCKET_WEIGHTS.items():
        quota = round(target * weight / total_weight)
        pool = sorted(by_bucket.get(bucket, []), key=lambda r: -float(r.get("_gap", 0.0)))
        out.extend(pool[:quota])
        leftovers.extend(pool[quota:])
    if len(out) < target:
        leftovers.sort(key=lambda r: -float(r.get("_gap", 0.0)))
        out.extend(leftovers[: target - len(out)])
    return out[:target]
