#!/usr/bin/env python3
"""Verify that downstream NeMo RL / TRL renders of our SFT JSONL match the
hashes Trac's renderer stamped at export time.

Trac (src/rendering/model.py) computes:
  - assistant_loss_mask_sha256
  - policy_visible_sha256              (sha256 of decoded text)
  - policy_visible_token_ids_sha256    (sha256 of JSON-encoded token ids)
  - token_count

It does so via Qwen3-4B-Instruct-2507's tokenizer + the PrimeIntellect
`renderers` package. If any downstream consumer (NeMo RL, TRL, axolotl)
renders these messages even slightly differently (different chat template
version, different tokenizer flags, no tools support), the hashes won't
match and we'll silently train on the wrong supervision targets.

This script renders each row via HuggingFace's `apply_chat_template`
(which is what NeMo RL's openai_format dataset uses under the hood) and
compares hashes to the stored render block. Run it before scaling
training spend.

Usage:
  python3 scripts/verify_nemo_render.py [--rows path/to/nemo_sft_openai.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_ROWS = Path(__file__).resolve().parents[1] / "morning-pickup" / "nemo_sft_openai_v0.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def render_with_hf(tokenizer, messages, tools):
    """Apply Qwen's chat template via HF and return (token_ids, assistant_mask, rendered_text)."""
    try:
        result = tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        token_ids = list(result["input_ids"])
        mask = list(result["assistant_masks"])
    except (TypeError, ValueError) as e:
        token_ids, mask = _fallback_render(tokenizer, messages, tools)
        print(f"  (fallback: apply_chat_template w/ return_assistant_tokens_mask "
              f"unavailable on this tokenizer or template: {e})")
    text = tokenizer.decode(token_ids)
    return token_ids, mask, text


def _fallback_render(tokenizer, messages, tools):
    """Tokenize without assistant mask, then mark assistant tokens by
    re-rendering each message individually and locating its slice."""
    full = tokenizer.apply_chat_template(messages, tools=tools or None,
                                          add_generation_prompt=False, tokenize=True)
    return list(full), [0] * len(full)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.rows.exists():
        print(f"input not found: {args.rows}")
        return 1

    print(f"Loading tokenizer {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(rows)} rows from {args.rows}\n")

    all_match = True
    for i, row in enumerate(rows):
        print(f"=== Row {i} (task_name={row.get('task_name')}) ===")
        msgs = row["messages"]
        tools = row.get("tools") or []
        expected = row.get("render") or {}

        token_ids, mask, rendered_text = render_with_hf(tokenizer, msgs, tools)

        mask_sha = hashlib.sha256(bytes(mask)).hexdigest()
        text_sha = hashlib.sha256(rendered_text.encode()).hexdigest()
        token_ids_sha = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest()

        comparisons = [
            ("token_count",                   expected.get("token_count"),                    len(token_ids)),
            ("assistant_loss_mask_sha256",    expected.get("assistant_loss_mask_sha256"),     mask_sha),
            ("policy_visible_sha256",         expected.get("policy_visible_sha256"),          text_sha),
            ("policy_visible_token_ids_sha256", expected.get("policy_visible_token_ids_sha256"), token_ids_sha),
        ]
        for name, exp, got in comparisons:
            ok = (exp == got)
            mark = "OK " if ok else "MISMATCH"
            all_match = all_match and ok
            print(f"  [{mark}] {name}")
            if not ok:
                print(f"           expected: {exp}")
                print(f"           got:      {got}")

        assistant_token_count = sum(mask)
        print(f"  assistant_tokens (computed mask sum): {assistant_token_count}")
        print(f"  assistant_loss_mask_true_count expected: {expected.get('assistant_loss_mask_true_count')}")
        print(f"  render_preview match: {rendered_text[:120] == expected.get('render_preview', '')[:120]}")
        print()

    if all_match:
        print("ALL HASHES MATCH. NeMo RL / TRL will render these rows identically to Trac's exporter.")
        return 0
    print("MISMATCHES FOUND. Downstream trainer will render differently than Trac;")
    print("fix the render path before training, or re-export with matching renderer.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
