"""Stage 1 — Format-validity gate.

Five hard rejects + one audit-only check, applied to each row of the input
NeMo SFT JSONL. Pure structural checks; no Trac render-block hash dependency.

Reject reasons (returned by `check_row`):
  - tool_call_mismatch        : an assistant.tool_calls[i].id has no matching tool message
  - recommend_count           : `recommend_product` not called exactly once
  - recommend_args_invalid    : `recommend_product` args don't parse to comma-sep digit IDs
  - token_count_exceeded      : tokens > max_length

Audit-only (logged but not rejected):
  - role_alternation          : pattern deviates from system/user/(assistant/tool*)*/(optional final assistant)

Note: the `with_think` converter variant emits legitimate content-only assistant
turns mid-trajectory (pure thinking steps and per-step finalize responses). These
are NOT format errors — Stage 3's `narration_bloat` axis-B signal handles the
pathological-narration case at corpus-statistics level.

Usage:
  python -m scripts.filtration.stage1_format_valid \
    --input data/filtration/nemo_sft_unanim_50_with_think.jsonl \
    --out-valid data/filtration/format_valid.jsonl \
    --out-rejects data/filtration/format_rejects.jsonl \
    --max-length 14336
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Qwen3-4B-Instruct-2507 is what the training config uses; tokenizer is downloaded
# from HF on first use (~5 MB).
TOKENIZER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MAX_LENGTH = 14336

DIGIT_ID_RE = re.compile(r"^\d{4,}$")


# ----- per-row checks -----

def check_tool_call_pairing(messages: list[dict]) -> str | None:
    """Every assistant.tool_calls[i].id must have exactly one matching subsequent
    tool message with `tool_call_id == id`. No orphans either way."""
    expected_ids: list[str] = []
    seen_tool_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = tc.get("id")
                if not tc_id:
                    return "tool_call_mismatch:assistant_tool_call_missing_id"
                expected_ids.append(tc_id)
        elif msg.get("role") == "tool":
            tcid = msg.get("tool_call_id")
            if not tcid:
                return "tool_call_mismatch:tool_msg_missing_tool_call_id"
            if tcid in seen_tool_ids:
                return f"tool_call_mismatch:duplicate_tool_response:{tcid}"
            seen_tool_ids.add(tcid)
    expected_set = set(expected_ids)
    if expected_set != seen_tool_ids:
        missing = expected_set - seen_tool_ids
        extra = seen_tool_ids - expected_set
        return f"tool_call_mismatch:missing={sorted(missing)[:3]}_extra={sorted(extra)[:3]}"
    return None


def _iter_tool_calls(messages: list[dict]) -> Iterable[tuple[int, dict]]:
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                yield idx, tc


def check_recommend_count(messages: list[dict]) -> str | None:
    """Exactly one `recommend_product` tool_call across all assistant turns."""
    count = sum(
        1 for _, tc in _iter_tool_calls(messages)
        if (tc.get("function") or {}).get("name") == "recommend_product"
    )
    if count != 1:
        return f"recommend_count:{count}"
    return None


def check_recommend_args(messages: list[dict]) -> str | None:
    """recommend_product args must contain a non-empty list of digit-IDs.

    The converter writes `product_ids` as a comma-sep string for shop/voucher/multi
    tasks, or `product_id` for single-product tasks.
    """
    for _, tc in _iter_tool_calls(messages):
        fn = (tc.get("function") or {}).get("name")
        if fn != "recommend_product":
            continue
        args = (tc.get("function") or {}).get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return "recommend_args_invalid:args_not_json"
        ids: list[str] = []
        if "product_ids" in args:
            raw = args["product_ids"]
            if isinstance(raw, list):
                ids = [str(x).strip() for x in raw if str(x).strip()]
            elif isinstance(raw, str):
                ids = [p.strip() for p in raw.split(",") if p.strip()]
            else:
                return "recommend_args_invalid:product_ids_wrong_type"
        elif "product_id" in args:
            v = args["product_id"]
            if isinstance(v, (int, float)):
                v = str(int(v))
            if not isinstance(v, str):
                return "recommend_args_invalid:product_id_wrong_type"
            ids = [v.strip()] if v.strip() else []
        else:
            return "recommend_args_invalid:no_product_field"
        if not ids:
            return "recommend_args_invalid:empty_ids"
        for pid in ids:
            if not DIGIT_ID_RE.match(pid):
                return f"recommend_args_invalid:non_digit_id:{pid[:20]}"
        return None
    return "recommend_args_invalid:no_recommend_call"


def check_mid_traj_content_only(messages: list[dict]) -> str | None:
    """An assistant turn that's content-only (no tool_calls) is only allowed as
    the very last assistant message (the final response). Anywhere else, reject."""
    # Find indices of all assistant messages.
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not assistant_indices:
        return None  # caught by recommend_count
    last_assistant_idx = assistant_indices[-1]
    for idx in assistant_indices:
        msg = messages[idx]
        has_tool_calls = bool(msg.get("tool_calls"))
        if not has_tool_calls and idx != last_assistant_idx:
            return f"mid_traj_content_only:position={idx}"
    return None


def audit_role_alternation(messages: list[dict]) -> str | None:
    """Audit-only. Expected pattern: system, user, then (assistant + tool*)*
    optionally followed by a final content-only assistant.
    """
    if len(messages) < 3:
        return "role_alternation:too_few_messages"
    if messages[0].get("role") != "system":
        return f"role_alternation:expected_system_at_0_got_{messages[0].get('role')}"
    if messages[1].get("role") != "user":
        return f"role_alternation:expected_user_at_1_got_{messages[1].get('role')}"
    # From position 2 onward, every message must be assistant or tool, and tool
    # messages must follow an assistant-with-tool_calls.
    expecting = "assistant"  # next role we expect
    for i, msg in enumerate(messages[2:], start=2):
        role = msg.get("role")
        if role not in ("assistant", "tool"):
            return f"role_alternation:unexpected_role_{role}_at_{i}"
        if role == "tool" and expecting != "tool":
            return f"role_alternation:tool_without_assistant_at_{i}"
        if role == "assistant":
            # If previous message was a tool (or the start), this is fine.
            tcs = msg.get("tool_calls") or []
            expecting = "tool" if tcs else "assistant"
    return None


# ----- token-count check (separate because it needs the tokenizer) -----

class TokenCounter:
    """Wraps HF apply_chat_template for max-length enforcement."""

    def __init__(self, model_name: str = TOKENIZER_MODEL):
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {model_name}", file=sys.stderr)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def count(self, messages: list[dict], tools: list[dict] | None) -> int:
        ids = self.tok.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=False,
            tokenize=True,
        )
        return len(ids)


def check_token_count(messages: list[dict], tools: list[dict] | None,
                       counter: TokenCounter, max_length: int) -> str | None:
    n = counter.count(messages, tools)
    if n > max_length:
        return f"token_count_exceeded:{n}"
    return None


# ----- row-level orchestrator -----

ROW_CHECKS = [
    ("tool_call_mismatch", lambda r: check_tool_call_pairing(r["messages"])),
    ("recommend_count", lambda r: check_recommend_count(r["messages"])),
    ("recommend_args_invalid", lambda r: check_recommend_args(r["messages"])),
]


def check_row(row: dict, counter: TokenCounter | None,
              max_length: int) -> tuple[str | None, str | None]:
    """Returns (reject_reason, audit_warning). reject_reason is None if pass.
    """
    for _name, fn in ROW_CHECKS:
        reason = fn(row)
        if reason is not None:
            audit = audit_role_alternation(row["messages"])
            return reason, audit

    if counter is not None:
        tc_reason = check_token_count(
            row["messages"], row.get("tools"), counter, max_length
        )
        if tc_reason is not None:
            audit = audit_role_alternation(row["messages"])
            return tc_reason, audit

    audit = audit_role_alternation(row["messages"])
    return None, audit


# ----- driver -----

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", type=Path,
                   default=Path("data/filtration/nemo_sft_unanim_50_with_think.jsonl"))
    p.add_argument("--out-valid", type=Path,
                   default=Path("data/filtration/format_valid.jsonl"))
    p.add_argument("--out-rejects", type=Path,
                   default=Path("data/filtration/format_rejects.jsonl"))
    p.add_argument("--out-audit", type=Path,
                   default=Path("data/filtration/format_audit.jsonl"))
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--limit", type=int, default=None,
                   help="cap rows for smoke testing")
    p.add_argument("--no-token-check", action="store_true",
                   help="skip token-count check (faster, for debugging only)")
    args = p.parse_args()

    counter = None
    if not args.no_token_check:
        counter = TokenCounter()

    n_in = 0
    n_valid = 0
    rejected_reasons: dict[str, int] = {}
    audit_warnings: dict[str, int] = {}

    t0 = time.time()
    with open(args.input) as fin, \
         open(args.out_valid, "w") as fvalid, \
         open(args.out_rejects, "w") as freject, \
         open(args.out_audit, "w") as faudit:
        for line in fin:
            n_in += 1
            if args.limit and n_in > args.limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                rejected_reasons["json_decode_error"] = (
                    rejected_reasons.get("json_decode_error", 0) + 1
                )
                continue
            reject, audit = check_row(row, counter, args.max_length)
            if audit is not None:
                key = audit.split(":")[0]
                audit_warnings[key] = audit_warnings.get(key, 0) + 1
                faudit.write(
                    json.dumps({"row_index": n_in - 1, "audit": audit}) + "\n"
                )
            if reject is not None:
                key = reject.split(":")[0]
                rejected_reasons[key] = rejected_reasons.get(key, 0) + 1
                row["_reject_reason"] = reject
                row["_row_index"] = n_in - 1
                freject.write(json.dumps(row) + "\n")
            else:
                fvalid.write(line)  # write original line as-is
                n_valid += 1

            if n_in % 1000 == 0:
                elapsed = time.time() - t0
                print(
                    f"  processed {n_in} rows in {elapsed:.1f}s "
                    f"({n_in/elapsed:.0f}/s), valid={n_valid}",
                    file=sys.stderr,
                )

    summary = {
        "input_rows": n_in,
        "valid_rows": n_valid,
        "reject_rows": n_in - n_valid,
        "valid_pct": round(100 * n_valid / max(n_in, 1), 2),
        "rejected_reasons": dict(sorted(rejected_reasons.items(),
                                         key=lambda x: -x[1])),
        "audit_warnings": dict(sorted(audit_warnings.items(),
                                       key=lambda x: -x[1])),
        "max_length": args.max_length,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    summary_path = args.out_valid.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
