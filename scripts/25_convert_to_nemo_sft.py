#!/usr/bin/env python3
"""Convert race-corpus trajectories to Trac NeMo SFT OpenAI format.

Input: sft_corpus_unanim_50.jsonl (our per-(av, pid, race) format with steps[])
Output: nemo_sft_openai.jsonl (Shardul's Trac format — OpenAI ChatML messages)

Defaults:
  - Strip <think> reasoning (matches Shardul; avoids template mimicry)
  - Skip render metadata (no PrimeIntellect renderers dependency)
  - Coerce product_ids to comma-separated string
  - Add explicit terminate call at end
  - Use Shardul's exact system prompt + tool schemas

Flags:
  --keep-think        preserve <think> content as assistant content
  --no-terminate      skip the final terminate call
  --add-render        emit placeholder render metadata (still no token mask)
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

# Shardul's exact system prompt
SYSTEM_PROMPT = (
    "You are Teacher, a shopping assistant in a synthetic data gate. "
    "Complete the user task using only the available tools and observed tool results. "
    "Do not answer with prose instead of tools. "
    "Search the catalog with find_product. "
    "When the request says verify, inspect, check, compare, SKU, variant, option, "
    "attribute, ingredient, material, description, or product information, call "
    "view_product_information before recommending. "
    "Required finalization: once you have enough evidence, call recommend_product "
    "exactly once with product_ids as a comma-separated string, then call terminate. "
    "For product tasks recommend one ID; for shop, voucher, and multi-item tasks "
    "recommend comma-separated IDs in the requested order. "
    "Do not mention private rewards, benchmark answers, split metadata, source lineage, "
    "or evaluation metadata. Prefer careful verification over speed. "
    "If candidate titles are ambiguous, inspect product information."
)

# Shardul's exact 4-tool schema
TOOLS = [
    {
        "description": "Search the ShoppingBench product catalog.",
        "name": "find_product",
        "parameters": {
            "additionalProperties": True,
            "properties": {
                "page": {"type": "integer"},
                "price": {"type": "string"},
                "q": {"type": "string"},
                "query": {"type": "string"},
                "service": {"type": "string"},
                "shop_id": {"type": "string"},
                "sort": {"type": "string"},
            },
            "type": "object",
        },
        "type": "function",
    },
    {
        "description": "Inspect product details for one or more product IDs.",
        "name": "view_product_information",
        "parameters": {
            "additionalProperties": True,
            "properties": {
                "product_id": {"type": "string"},
                "product_ids": {"items": {"type": "string"}, "type": "array"},
            },
            "type": "object",
        },
        "type": "function",
    },
    {
        "description": "Submit final ShoppingBench product recommendation IDs.",
        "name": "recommend_product",
        "parameters": {
            "additionalProperties": True,
            "properties": {
                "product_id": {"type": "string"},
                "product_ids": {"items": {"type": "string"}, "type": "array"},
            },
            "type": "object",
        },
        "type": "function",
    },
    {
        "description": "End the shopping episode after final recommendation.",
        "name": "terminate",
        "parameters": {"additionalProperties": True, "properties": {}, "type": "object"},
        "type": "function",
    },
]


# Some agents emit compact field names in tool_result entries; remap to Shardul's full schema
COMPACT_TO_FULL = {"pid": "product_id", "p": "price", "s": "shop_id", "t": "title"}


def expand_compact_result(result):
    """Recursively rename compact field names (pid→product_id, p→price, s→shop_id, t→title)."""
    if isinstance(result, list):
        return [expand_compact_result(x) for x in result]
    if isinstance(result, dict):
        out = {}
        for k, v in result.items():
            out_k = COMPACT_TO_FULL.get(k, k)
            out[out_k] = expand_compact_result(v) if isinstance(v, (list, dict)) else v
        return out
    return result


def normalize_pids(raw) -> str:
    """Normalize product_ids to comma-separated string (per Shardul's system prompt)."""
    if isinstance(raw, list):
        return ",".join(str(x).strip() for x in raw if str(x).strip())
    if isinstance(raw, (int, float)):
        return str(int(raw))
    if isinstance(raw, str):
        return ",".join(p.strip() for p in raw.split(",") if p.strip())
    return ""


def step_to_messages(step: dict, call_idx: int, keep_think: bool) -> tuple[list[dict], int]:
    """Convert one of our `steps[]` entries to ≥0 Trac OpenAI messages.

    Returns (messages, next_call_idx). Each tool_call becomes:
      - assistant w/ tool_calls (content = think if keep_think else "")
      - tool w/ result keyed by tool_call_id
    """
    msgs: list[dict] = []
    think = step.get("think") or ""
    if not isinstance(think, str):
        think = json.dumps(think)
    response = step.get("response")
    tcs = step.get("tool_calls") or []

    if tcs:
        assistant_msg = {
            "role": "assistant",
            "content": (think.strip() if keep_think else ""),
            "tool_calls": [],
        }
        tool_msgs = []
        for tc in tcs:
            name = (tc or {}).get("name")
            if not name:
                continue
            params = (tc or {}).get("parameters") or {}
            # Coerce recommend_product / view_product_information pid args
            if name in ("recommend_product",):
                if "product_ids" in params:
                    params = {**params, "product_ids": normalize_pids(params["product_ids"])}
                elif "product_id" in params:
                    params = {**params, "product_id": str(params["product_id"])}
            elif name == "view_product_information":
                if "product_ids" in params:
                    val = params["product_ids"]
                    if isinstance(val, str):
                        params = {**params, "product_ids": [p.strip() for p in val.split(",") if p.strip()]}
                elif "product_id" in params:
                    params = {**params, "product_id": str(params["product_id"])}
            call_id = f"call_trac_{call_idx}"
            assistant_msg["tool_calls"].append({
                "function": {"name": name, "arguments": params},
                "id": call_id,
                "type": "function",
            })
            # tool result — expand compact keys then serialize JSON
            result = (tc or {}).get("result")
            if isinstance(result, (dict, list)):
                result = expand_compact_result(result)
                result_str = json.dumps(result)
            else:
                result_str = str(result) if result is not None else ""
            tool_msgs.append({
                "role": "tool",
                "content": result_str + "\n",
                "name": name,
                "tool_call_id": call_id,
            })
            call_idx += 1
        msgs.append(assistant_msg)
        msgs.extend(tool_msgs)
    elif keep_think and think.strip():
        # Pure thinking step with no tool calls — only emit if keeping think
        msgs.append({"role": "assistant", "content": think.strip()})

    if response and isinstance(response, str) and response.strip():
        msgs.append({"role": "assistant", "content": response.strip()})

    return msgs, call_idx


def trajectory_to_nemo(traj: dict, keep_think: bool, add_terminate: bool, add_render: bool) -> dict | None:
    """Convert one of our trajectory records to a Trac NeMo SFT JSONL row.

    Returns None if trajectory is malformed (no user query OR no recommend_product).
    """
    query = traj.get("problem_query") or ""
    if not query.strip():
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query.strip()},
    ]

    call_idx = 0
    had_recommend = False
    had_terminate = False
    for step in traj.get("steps") or []:
        new_msgs, call_idx = step_to_messages(step, call_idx, keep_think)
        messages.extend(new_msgs)
        for m in new_msgs:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}).get("name")
                if fn == "recommend_product":
                    had_recommend = True
                elif fn == "terminate":
                    had_terminate = True

    if not had_recommend:
        return None  # malformed: no recommend

    if add_terminate and not had_terminate:
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {"name": "terminate", "arguments": {}},
                "id": f"call_trac_{call_idx}",
                "type": "function",
            }],
        })
        messages.append({
            "role": "tool",
            "content": '{"status": "terminated"}\n',
            "name": "terminate",
            "tool_call_id": f"call_trac_{call_idx}",
        })

    out = {
        "messages": messages,
        "task_name": "trac_shoppingbench_sft",
        "tools": TOOLS,
        "trac_export_version": "trac-nemo-sft-openai-v0.1-oro-converted",
    }
    if add_render:
        out["render"] = {
            "renderer_backend": "renderers",
            "renderer_name": "model_native",
            "renderer_version": "PENDING",
            "note": "Loss-mask hashes NOT computed. Re-run through PrimeIntellect renderers before training.",
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", type=Path,
                   default=Path("data/training/sft_corpus_unanim_50.jsonl"))
    p.add_argument("--output", "-o", type=Path,
                   default=Path("data/training/nemo_sft_openai_oro.jsonl"))
    p.add_argument("--keep-think", action="store_true",
                   help="preserve <think> content as assistant content (default: strip)")
    p.add_argument("--no-terminate", action="store_true",
                   help="don't append explicit terminate call (default: add)")
    p.add_argument("--add-render", action="store_true",
                   help="emit placeholder render metadata block (default: omit)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap on number of input rows (smoke test)")
    args = p.parse_args()

    n_in = 0
    n_out = 0
    n_dropped_no_query = 0
    n_dropped_no_recommend = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            n_in += 1
            if args.limit and n_in > args.limit:
                break
            traj = json.loads(line)
            if not (traj.get("problem_query") or "").strip():
                n_dropped_no_query += 1
                continue
            row = trajectory_to_nemo(
                traj,
                keep_think=args.keep_think,
                add_terminate=not args.no_terminate,
                add_render=args.add_render,
            )
            if row is None:
                n_dropped_no_recommend += 1
                continue
            fout.write(json.dumps(row) + "\n")
            n_out += 1

    print(f"input:                 {n_in}", file=sys.stderr)
    print(f"output:                {n_out}", file=sys.stderr)
    print(f"dropped (no query):    {n_dropped_no_query}", file=sys.stderr)
    print(f"dropped (no recommend): {n_dropped_no_recommend}", file=sys.stderr)
    print(f"write rate: {n_out / max(n_in, 1) * 100:.1f}%", file=sys.stderr)
    print(f"saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
