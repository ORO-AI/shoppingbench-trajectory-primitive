"""Build KTO preference dataset from Seth's Sonnet teacher rollouts (ORO-1307 Phase B).

Trajectory-level KTO: each row is one full Sonnet trajectory.
  - prompt    = system prompt + user query
  - completion = full assistant turn sequence (think + tool_call + response across all steps)
  - label      = trajectory success (score_ok=True for ~210 wins, False for ~288 failures)

Why trajectory-level (vs step-wise): KTO is more robust to mode collapse than DPO,
so we don't need step-wise risk mitigation. Trajectory-level gives cleaner signal
per row (success/failure is a property of the whole trajectory, not individual steps).

Input:
  - opd_sonnet_500.jsonl                (498 rollouts, paper XML format)
  - scored_opd_sonnet_500_final.json    (per-rollout score_ok via rule==1.0 + bucket rules)

Output:
  - kto_dataset_opd_renderers_v1.jsonl  (TRL KTOTrainer format: prompt, completion, label, bucket, pid)

Usage:
    python3 sft-2026-05-21/scripts/50_build_kto_dataset.py \
        --teacher-jsonl sft-2026-05-21/phase3b_teacher_data/opd_sonnet_500.jsonl \
        --teacher-scored sft-2026-05-21/phase3b_teacher_data/scored_opd_sonnet_500_final.json \
        --out sft-2026-05-21/phase3c_kto/kto_dataset_opd_renderers_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# Match opd_renderers' training distribution: standard Qwen3 chat template.
# Seth's training corpus used Sonnet's paper-XML format wrapped in chat messages.
# The system prompt is the ORO fork's BASE_SYSTEM_PROMPT (loaded by ShoppingBench
# at agent_main()). For KTO we use a faithful approximation since we don't have
# the fork's per-bucket strategy prompts; the prompt+completion still match
# the format opd_renderers expects to GENERATE.
SYSTEM_PROMPT = (
    "You are a helpful multi-turn dialogue assistant capable of leveraging tool "
    "calls to solve user tasks and provide structured chat responses. "
    "Use <think>...</think> for reasoning and "
    "<tool_call>[{\"name\": ..., \"parameters\": {...}}]</tool_call> for tool calls. "
    "Required finalization: call recommend_product exactly once with product_ids as a "
    "comma-separated string, then call terminate."
)


def is_success(category: str, score: dict) -> bool:
    """Match the production-strict definition Seth used."""
    if not score:
        return False
    if (score.get("rule") or 0) < 1.0:
        return False
    cat = (category or "").lower()
    if cat == "shop":
        return (score.get("shop") or 0) >= 1.0
    if cat == "voucher":
        return (score.get("budget") or 0) >= 1.0
    return True


def render_step_content(step: dict) -> str:
    """Render one trajectory step into its paper-XML completion content.

    Each step in opd_sonnet_500.jsonl has `completion.content` which already
    contains `<think>...</think>` + `<tool_call>[...]</tool_call>` markers.
    Tool responses (the `result` field inside tool_call) are appended after
    each tool_call so the model sees them in-line during generation.
    """
    comp = step.get("completion", {})
    content = comp.get("content", "") or ""
    # Append tool_call results inline as <tool_response>...</tool_response>
    msg = comp.get("message", {})
    if isinstance(msg, dict):
        tool_calls = msg.get("tool_call") or []
        for tc in tool_calls:
            if isinstance(tc, dict) and "result" in tc:
                result = tc["result"]
                # Compact JSON to save tokens; truncate huge responses
                result_str = json.dumps(result, default=str)[:4000]
                content += f"\n<tool_response>{result_str}</tool_response>"
    return content


def render_completion(trajectory: dict) -> str:
    """Render full trajectory as a single completion string.

    Concatenates all steps' contents (think + tool_call + tool_response). The
    final assistant turn ends with recommend_product + terminate per the protocol.
    """
    parts: list[str] = []
    for step in trajectory.get("steps", []):
        parts.append(render_step_content(step))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-jsonl", required=True, type=Path)
    ap.add_argument("--teacher-scored", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # Load all trajectories
    trajectories: dict[str, dict] = {}
    with open(args.teacher_jsonl) as f:
        for line in f:
            r = json.loads(line)
            trajectories[r["pid"]] = r
    print(f"Loaded {len(trajectories)} Sonnet trajectories", file=sys.stderr)

    # Load scores. The scored file has two variants (claude-code-sonnet and
    # openrouter-sonnet); rollouts within each are a LIST of dicts indexed by
    # query string. Build a query-keyed lookup, preferring openrouter when both
    # variants ran the same query.
    scored = json.load(open(args.teacher_scored))
    query_to_score: dict[str, dict] = {}
    for variant_name, payload in scored.items():
        for rollout_score in payload["rollouts"]:
            q = rollout_score.get("query")
            if q is None:
                continue
            if q not in query_to_score or variant_name == "openrouter-sonnet":
                query_to_score[q] = rollout_score
    print(f"Loaded scores for {len(query_to_score)} unique queries", file=sys.stderr)

    # Build KTO rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    label_counts: Counter = Counter()
    bucket_label_counts: Counter = Counter()
    written = 0
    skipped_no_score = 0
    skipped_empty_traj = 0

    with open(args.out, "w") as fout:
        for pid, traj in trajectories.items():
            if traj.get("error"):
                skipped_empty_traj += 1
                continue
            if not traj.get("steps"):
                skipped_empty_traj += 1
                continue
            score_record = query_to_score.get(traj["query"])
            if score_record is None:
                skipped_no_score += 1
                continue
            score = score_record.get("score") or {}
            label = is_success(traj["category"], score)

            user_query = traj["query"]
            completion = render_completion(traj)
            if not completion.strip():
                skipped_empty_traj += 1
                continue

            row = {
                "pid": pid,
                "bucket": traj["category"],
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                "completion": [
                    {"role": "assistant", "content": completion},
                ],
                "label": bool(label),
            }
            fout.write(json.dumps(row) + "\n")
            written += 1
            label_counts[label] += 1
            bucket_label_counts[(traj["category"], label)] += 1

    print(f"\nWrote {written} KTO rows to {args.out}")
    print(f"  desirable (label=True): {label_counts[True]}")
    print(f"  undesirable (label=False): {label_counts[False]}")
    print(f"  ratio desirable: {label_counts[True] / max(written, 1):.1%}")
    print(f"\nPer bucket:")
    for bucket in ("product", "shop", "voucher"):
        n_pos = bucket_label_counts.get((bucket, True), 0)
        n_neg = bucket_label_counts.get((bucket, False), 0)
        total = n_pos + n_neg
        print(f"  {bucket:>8}: {n_pos:>3} desirable / {total:>3} total "
              f"({100*n_pos/max(total,1):.1f}% pos)")
    print(f"\nSkipped: {skipped_no_score} no-score, {skipped_empty_traj} empty/errored")


if __name__ == "__main__":
    main()
