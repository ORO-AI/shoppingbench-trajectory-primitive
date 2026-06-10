# Fast vLLM eval pipeline (HF generate → vLLM server)

Replaces in-process HuggingFace `model.generate()` eval (~3 hr per 75-set) with a
vLLM-served pipeline that scores post-trained Qwen3-4B variants in **~24 min**
(≈7.5×), validated to **match the HF numbers** (Linear ORO-1319).

## Why

The HF eval (`52_eval_kto_7tool_harness.py`) generates one sequence at a time,
re-encoding the full message history every turn (quadratic in trajectory
length). 75 problems × multi-turn × per-turn re-encode ≈ 3 hr. That throttled how
many checkpoints we could compare before Paris.

## Architecture

```
vLLM OpenAI server (H200)                 threaded client (CPU)
  Qwen/Qwen3-4B + swappable LoRA   <--HTTP--   N workers each run agent_main
  /v1/completions, continuous batching        one problem; each turn POSTs
                                               /v1/completions (chat-template
                                               parity with the HF eval)
```

The server's continuous batching coalesces the concurrent per-turn requests —
that's the speedup. (Same engine + mechanism as Seth's drgrpo v17 rollout
server, `40_serve_opd_renderers_vllm.py`.)

## Scripts (`sft-2026-05-21/scripts/`)

| Script | Role |
|---|---|
| `55_compose_adapters_vllm.py` | Cat-composes `opd_renderers ⊕ KTO_vN` into one rank-32 LoRA + a **weight-parity gate** (asserts composed-on-base == sequential HF merge). |
| `56_serve_eval_vllm.py` | vLLM OpenAI server on `Qwen/Qwen3-4B` + swappable adapters (`opd`, `opd_kto_v3/v4`, `drgrpo_step5/10/15/20`). Pins `vllm==0.9.2`, `transformers==4.53.1`. |
| `57_eval_vllm_client.py` | Threaded client. `apply_chat_template` parity with `52`, retry-with-backoff, resume-on-good-rows. Generic `::run --adapter X --out-subdir Y`. |
| `53_score_rollouts.py::score_file` | (additive entrypoint) score any `rollouts.jsonl` pulled from the volume. |

## Run

```bash
# 1. one-time: compose the KTO adapters (+ parity gate)
modal run sft-2026-05-21/scripts/55_compose_adapters_vllm.py::compose_v4
modal run sft-2026-05-21/scripts/55_compose_adapters_vllm.py::compose_v3

# 2. deploy the server
modal deploy sft-2026-05-21/scripts/56_serve_eval_vllm.py

# 3. eval any adapter (full 75-set)
modal run --detach sft-2026-05-21/scripts/57_eval_vllm_client.py::run \
  --adapter opd_kto_v4 --out-subdir eval_kto_v4_vllm

# 4. score
modal volume get oro-distill-data eval_kto_v4_vllm/rollouts.jsonl /tmp/r.jsonl
modal run sft-2026-05-21/scripts/53_score_rollouts.py::score_file \
  --path /tmp/r.jsonl --out-subdir eval_kto_v4_vllm
```

## Hard-won learnings (read before extending)

1. **vLLM 0.9.2 throws `HFValidationError` on a merged *local* base path.** Do not
   serve a merged base. Instead cat-compose `opd ⊕ KTO` into one rank-32 adapter
   (PEFT `add_weighted_adapter(combination_type="cat", weights=[1,1])`) and serve
   it on plain `Qwen/Qwen3-4B`. Verified weight-equivalent to the sequential
   merge (max_abs_diff = 1 bf16 ULP). `55` runs this parity gate.
2. **Retry inside the inference call.** Under concurrency a transient vLLM
   timeout, if returned as empty content, makes the agent treat the turn as dead
   and terminate early → deflated score (HF never returns empty). `57` retries
   with backoff so a timeout never leaks an empty turn.
3. **`--max-model-len 32768`, not 16384.** Long multi-item shop/voucher histories
   overflow 16384 → HTTP 400 → silent auto-recommend fallback on ~20% of problems.
4. **`max_lora_rank 32`** (composed adapters are rank 32; opd/KTO/drgrpo are 16).

## Caveat: Dr.GRPO needs Seth's harness, not this client

drgrpo policies emit Seth's `<tool_call>[...]</tool_call>` format parsed by
`src/drgrpo/rollout_harness.py`, **not** the ShoppingBench 7-tool surface this
client drives. Running a drgrpo policy through this client 0-steps every problem
(format mismatch). Eval drgrpo via Seth's `rollout_harness` pointed at the same
vLLM server (serve the `drgrpo_stepN` adapter alone on base — it's
self-contained, do not stack `opd` underneath).

## Verified parity (ORO-1319)

| checkpoint | vLLM | HF | 
|---|---|---|
| opd_renderers | 42.7% (32/75) | 41.3% (31/75) — within 1 problem |
| KTO v3 | 42.7% (32/75) | 42.7% (32/75) — exact |
