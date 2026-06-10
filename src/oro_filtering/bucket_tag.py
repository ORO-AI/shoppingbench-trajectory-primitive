"""Bucket classification of trajectory rows into Trac's 7 capability buckets."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

BUCKETS = (
    "sku_variant_product_inspection",
    "same_shop_full_constraint_reasoning",
    "knowledge_bridge_literal_sourced",
    "voucher_math_auditable_provenance",
    "recommendation_provenance_multi_id",
    "robust_search_exact_constraint_reformulation",
    "benchmark_artifact_resilience_probe",
)

_FALLBACK = "recommendation_provenance_multi_id"

_SYSTEM = (
    "You classify a shopping query into ONE of these capability buckets. "
    "Respond with the bucket name only, no other text.\n\n"
    "Buckets:\n"
    "- sku_variant_product_inspection: single product retrieval + verifying one attribute/SKU/variant\n"
    "- same_shop_full_constraint_reasoning: multi-item same-shop request with per-item constraints\n"
    "- knowledge_bridge_literal_sourced: knowledge lookup then grounded product verification\n"
    "- voucher_math_auditable_provenance: voucher/budget threshold with discount math\n"
    "- recommendation_provenance_multi_id: comparing multiple candidate IDs, recommend grounded matches\n"
    "- robust_search_exact_constraint_reformulation: query reformulation + exact constraint check\n"
    "- benchmark_artifact_resilience_probe: reward-hacking-resistant probe"
)


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


def classify_row(row: dict[str, Any], *, client: ChatClient) -> str:
    query = row.get("problem_query", "")
    raw = client.complete(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": query[:1200]},
        ]
    ).strip()
    return raw if raw in BUCKETS else _FALLBACK


def classify_batch(
    rows: list[dict[str, Any]], *, client: ChatClient, max_workers: int = 16
) -> list[str]:
    """Classify ``rows`` concurrently.

    The default 16-way concurrency turns a ~25 min serial pass (~500 ms per
    OpenAI call x 3K rows) into roughly one minute. ``client`` must be
    thread-safe; ``openai.OpenAI`` is per OpenAI's docs. ``executor.map``
    preserves input order, so the returned list aligns with ``rows``.
    """
    if not rows:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda row: classify_row(row, client=client), rows))
