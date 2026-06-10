"""Tiered reward for Dr.GRPO.

Tiers (matching the iteration plan in notes.md):
  0.00 — rollout never called recommend_product
  0.05 — recommend_product called, but product_id not in catalog (hallucination)
  0.10 + 0.80 * rule_score — real catalog ID, partial credit on attributes
  1.00 — exact ground-truth product_id match

rule_score is the price/service/sku/attributes component, lifted from the
reference single-shot GRPO script (compute_rule_score). Title similarity is
skipped — too slow for online reward.
"""

from __future__ import annotations

from typing import Any

from .rollout_harness import SEARCH_URL, RolloutResult, extract_recommended_ids


# ---------- rule_score (price / service / sku / attributes) ----------


def compute_rule_score(product: dict, reward: dict) -> float:
    if product is None:
        return 0.0
    if product.get("product_id") == reward.get("product_id"):
        return 1.0

    total_count = 0
    hit_count = 0

    if "price" in reward:
        price = product.get("price")
        if price is not None:
            for price_range in reward["price"]:
                if not isinstance(price_range, dict):
                    continue
                for mode, bounds in price_range.items():
                    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                        continue
                    lo, hi = bounds
                    total_count += 1
                    if mode == "less than" and hi is not None and price <= hi:
                        hit_count += 1
                    elif mode == "greater than" and lo is not None and price >= lo:
                        hit_count += 1
                    elif mode == "between" and lo is not None and hi is not None and lo <= price <= hi:
                        hit_count += 1

    if "service" in reward:
        product_services = product.get("service", []) or []
        for serv in reward["service"]:
            total_count += 1
            if serv in product_services:
                hit_count += 1

    sku_flattens: list[set] = [set()]
    if product.get("sku_options"):
        for option in product["sku_options"].values():
            if isinstance(option, dict):
                sku_flattens.append({(k, v) for k, v in option.items()})
    attr_flatten: set = set()
    if product.get("attributes"):
        for k, vs in product["attributes"].items():
            if isinstance(vs, list):
                for v in vs:
                    attr_flatten.add((k, v))

    max_total = 0
    max_hit = 0
    for sku_flatten in sku_flattens:
        cur_total = 0
        cur_hit = 0
        if "sku_options" in reward:
            for option in reward["sku_options"]:
                if isinstance(option, dict):
                    for k, v in option.items():
                        cur_total += 1
                        if (k, v) in sku_flatten or (k, v) in attr_flatten:
                            cur_hit += 1
        if "attributes" in reward:
            for attr in reward["attributes"]:
                if isinstance(attr, dict):
                    for k, vs in attr.items():
                        if isinstance(vs, list):
                            for v in vs:
                                cur_total += 1
                                if (k, v) in sku_flatten or (k, v) in attr_flatten:
                                    cur_hit += 1
        if cur_total > max_total:
            max_total = cur_total
            max_hit = cur_hit
    total_count += max_total
    hit_count += max_hit

    if total_count == 0:
        return 0.0
    return hit_count / total_count


# ---------- catalog lookup ----------


def _fetch_product(session, pid: str) -> dict | None:
    try:
        r = session.get(
            f"{SEARCH_URL}/get_product_raw",
            params={"product_ids": pid},
            timeout=10,
        )
        if not r.ok:
            return None
        body = r.json()
        if not body:
            return None
        return body[0] if isinstance(body, list) else body
    except Exception:
        return None


# ---------- main entry ----------


def _normalize_reward_dict(problem: dict) -> dict:
    """Pull the per-task scoring dict out of a problem record.

    Shop / voucher problems store full_reward as a list of sub-rewards; for v1
    we score against the first one (matches reference behavior).
    """
    full_reward = problem.get("full_reward") or problem.get("reward") or {}
    if isinstance(full_reward, list):
        full_reward = full_reward[0] if full_reward else {}
    return full_reward or {}


def _outcome_reward(
    rollout: RolloutResult,
    problem: dict,
    *,
    session=None,
    continuous: bool = False,
) -> dict[str, Any]:
    """Tiered reward. Returns dict so trainer can log component breakdown.

    Keys:
      reward       — final float in [0, 1]
      tier         — one of "none" | "hallucinated" | "partial" | "exact"
      rec_ids      — extracted product_ids
      gt_match     — exact ground-truth match
      catalog_hit  — product_id resolved to a real catalog entry
      rule_score   — partial-match component (0 if no catalog hit)
    """
    import requests

    if session is None:
        session = requests.Session()

    rec_ids = extract_recommended_ids(rollout.messages)
    gt_ids = problem.get("ground_truth_product_ids") or []
    reward_dict = _normalize_reward_dict(problem)

    out: dict[str, Any] = {
        "reward": 0.0,
        "tier": "none",
        "rec_ids": rec_ids,
        "gt_match": False,
        "catalog_hit": False,
        "rule_score": 0.0,
    }

    if not rec_ids:
        return out

    pid = rec_ids[0]

    if pid in set(gt_ids):
        out.update(reward=1.0, tier="exact", gt_match=True, catalog_hit=True, rule_score=1.0)
        return out

    product = _fetch_product(session, pid)
    if product is None:
        # Hallucinated catalog ID — small shaping signal so the policy at least
        # learns to emit recommend_product, then learns to ground it.
        out.update(reward=0.05, tier="hallucinated")
        return out

    out["catalog_hit"] = True
    rs = compute_rule_score(product, reward_dict)
    out["rule_score"] = rs
    if continuous:
        # Continuous reward = raw rule_score. Finer-grain reward → more advantage variance
        # when multiple rollouts all "succeed" but differ in quality.
        out["reward"] = rs
    else:
        out["reward"] = 0.10 + 0.80 * rs
    out["tier"] = "partial"
    return out


# ---------------------------------------------------------------------------
# Per-step tool-matching reward (ToolRL-style, ORO-1266 v19).
#
# Dense shaping reward that scores a rollout's tool-call sequence against the
# problem's reference success trajectories (rejection-sampled v3 training
# rollouts). Gives gradient WITHIN a group even when every rollout fails the
# outcome — the signal-starvation case that flatlined v17/v18. Validated
# standalone: right-search-wrong-product ~0.74, junk ~0.13, empty 0.0.
# ---------------------------------------------------------------------------
import re as _re

_TC_RE = _re.compile(r"<tool_call>\s*(\[.*?\])\s*</tool_call>", flags=_re.DOTALL)


def extract_toolseq(messages: list[dict]) -> list[tuple]:
    """Ordered [(tool_name, args_dict)] from assistant turns. Handles BOTH
    structured `tool_calls` (reference trajectories) and Seth's `<tool_call>`
    text blocks in content (policy rollouts)."""
    import json as _json
    seq: list[tuple] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls")
        if tcs:
            for tc in tcs:
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except Exception:
                        args = {"_raw": args}
                seq.append((fn.get("name"), args or {}))
        else:
            for mt in _TC_RE.finditer(m.get("content", "") or ""):
                try:
                    calls = _json.loads(mt.group(1))
                except Exception:
                    continue
                for c in calls if isinstance(calls, list) else []:
                    if isinstance(c, dict) and "name" in c:
                        seq.append((c.get("name"), c.get("arguments") or c.get("parameters") or {}))
    return seq


def _tok(s):
    return set(_re.findall(r"[a-z0-9]+", str(s).lower()))


def _jacc(a, b):
    A, B = _tok(a), _tok(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _pids(args):
    return set(str(args.get("product_ids") or args.get("product_id") or "").replace(",", " ").split())


def _argsim(name, ra, rb):
    if name == "find_product":
        return _jacc(ra.get("q") or ra.get("query", ""), rb.get("q") or rb.get("query", ""))
    if name == "view_product_information":
        a, b = _pids(ra), _pids(rb)
        return (len(a & b) / len(a | b)) if (a or b) else 1.0
    if name == "recommend_product":
        a, b = _pids(ra), _pids(rb)
        return (len(a & b) / len(b)) if b else 0.0
    return 1.0


def _align_score(roll, ref):
    """Greedy in-order tool-name match; for each matched pair add arg similarity.
    0.5 name-coverage + 0.5 arg-quality. Robust to query-string divergence."""
    if not ref:
        return 0.0
    j = 0
    matched = 0
    argsum = 0.0
    for rn, ra in roll:
        while j < len(ref) and ref[j][0] != rn:
            j += 1
        if j < len(ref):
            matched += 1
            argsum += _argsim(rn, ra, ref[j][1])
            j += 1
    name_cov = matched / max(len(roll), len(ref))
    arg_q = (argsum / matched) if matched else 0.0
    return 0.5 * name_cov + 0.5 * arg_q


def tool_match_reward(rollout_seq, ref_seqs) -> float:
    """Best match over the problem's reference success trajectories. [0, 1]."""
    return max((_align_score(rollout_seq, r) for r in ref_seqs), default=0.0)


def compute_reward(
    rollout: RolloutResult,
    problem: dict,
    *,
    session=None,
    continuous: bool = False,
    tool_weight: float = 0.0,
) -> dict[str, Any]:
    """Outcome reward, optionally blended with the dense per-step tool-match
    reward when ``tool_weight > 0`` and the problem carries ``_ref_toolseqs``.
    blended = (1 - w) * outcome + w * tool_match. Wraps the unchanged outcome
    logic so eval (tool_weight=0) still measures true success."""
    out = _outcome_reward(rollout, problem, session=session, continuous=continuous)
    refs = problem.get("_ref_toolseqs")
    if tool_weight > 0.0 and refs:
        tm = tool_match_reward(extract_toolseq(rollout.messages), refs)
        out["tool_match"] = tm
        out["reward"] = (1.0 - tool_weight) * out["reward"] + tool_weight * tm
    return out
