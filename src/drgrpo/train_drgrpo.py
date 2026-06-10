"""Dr.GRPO trainer for ShoppingBench multi-turn shopping agent.

Single Modal app. Hand-rolled, not TRL — TRL's GRPOTrainer assumes single-shot
generate(), our task is multi-turn (model -> tool -> model -> ...).

Per-step algorithm (see notes.md "Architecture"):
  1. Sample B problems
  2. K rollouts per problem via the multi-turn harness
  3. Tiered reward per rollout
  4. Dr.GRPO advantage: a_i = r_i - group_mean (NO length norm, NO std norm)
  5. (iteration 5) positive-only advantage filter
  6. Per-token loss on assistant tokens only: -a * log_prob(token)
  7. (iteration 5) + beta * KL(policy || ref) via dual-adapter forward
  8. Backprop, step, save adapter every N steps

Iteration 2 adds the decode-and-assert mask validator (see
`validate_assistant_mask` below).

Iteration 3 adds model.eval() / model.train() discipline with active-dropout
assertions (see `_count_active_dropout_modules` and the eval/train guards
around the generate/backward boundary).

Iteration 4 adds behavioral metrics logged every step from step 0 (see
`_aggregate_step_metrics`) and persists them as JSONL inside `out_dir`.

Iteration 5 adds (a) positive-only advantage filter (see
`_filter_positive_advantage_examples`) to neutralise shared-prefix gradient
suppression in multi-turn rollouts, and (b) a KL anchor at beta=0.1 computed
via dual-adapter forwards (k3 estimator, see `_kl_k3_per_token`).

Iteration 7 adds problem-set hygiene: stable query-hash exclusion against
shardul_holdout.json + problem_suite_v{1,2,3}.json (the 75-set / 90-set
scoring suites) so we never train on a problem we evaluate on, plus a
deterministic 10% in-distribution eval split (`_split_train_eval`). See
`_load_problems_and_split`.

Iteration 9 wires the reserved 10% eval slice into a periodic in-distribution
eval pass (`_collect_eval_groups` + `eval_every` knob). The training
`mean_reward` rises by construction (optimizer gradient-ascends on it); only
held-out eval mean_reward tells us the policy is generalizing rather than
overfitting the sampled training problems.

Iteration 10 adds length-bucketed batch sampling
(`_length_bucketed_grouped`) so each step's batch is homogeneous in
expected trajectory length. Prevents padding from dominating compute and
the long tail from being clipped against short prompts in the same batch.

Iteration 11 adds 75/90-set scoring (`_load_scoring_suite`, `score_adapter`)
so an arbitrary adapter checkpoint can be evaluated against the same
problems used to score SFT baselines (shardul_holdout / problem_suite_v*).
This is what lets us measure the RL lift vs SFT — the objective.

Iteration 12 enables gradient checkpointing (`_setup_gradient_checkpointing`)
on the PEFT-wrapped model. At the practice-run constraint of 20K max_seq
+ Qwen3-4B vocab ~152K, the fp32 log_softmax tensor + softmax saved for
backward exceeds 12 GB per forward — two of those (policy + ref) plus
B*K rollout activations OOMs an 80 GB H100 before training starts.
Includes the PEFT-specific `enable_input_require_grads()` dance — without
it, LoRA params silently receive zero gradient and the run looks healthy
in loss but learns nothing.

Still deferred:
  - full 20-step train run (iteration 13+)

Entrypoints:
  modal run src/train_drgrpo.py::smoke    # 3 steps, B=1, K=2 — must pass first
  modal run src/train_drgrpo.py::train    # 20 steps, B=2, K=4
  modal run src/train_drgrpo.py::score    # score one adapter on 75/90-set
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import modal


app = modal.App("oro-drgrpo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "peft>=0.14.0",
        "accelerate>=1.2.0",
        "hf_transfer>=0.1.6",
        "requests>=2.31.0",
        "wandb>=0.18.0",  # ORO-1310 unified paris-distillation dashboard
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("src")
)

ckpt_volume = modal.Volume.from_name("oro-sft-checkpoints", create_if_missing=True)
trac_volume = modal.Volume.from_name("oro-trac-data", create_if_missing=True)
distill_volume = modal.Volume.from_name("oro-distill-data", create_if_missing=True)  # per-step tool-match references (v19)


BACKBONE = "Qwen/Qwen3-4B"
POLICY_ADAPTER_PATH = "/ckpt/sft_opd_sonnet_rebal_from_rejection_v2/final"

# Fallback problems if no race_bank mount found — enough to let smoke() actually
# run without depending on volume contents on first invocation.
_FALLBACK_PROBLEMS: list[dict] = [
    {
        "query": "I need a wireless bluetooth keyboard under 200000 rupiah with backlit keys.",
        "ground_truth_product_ids": [],
        "full_reward": {"price": [{"less than": [None, 200000]}]},
        "category": "product",
    },
    {
        "query": "Find me a stainless steel water bottle that keeps drinks cold for 24 hours.",
        "ground_truth_product_ids": [],
        "full_reward": {},
        "category": "product",
    },
]


# ----------------------------------------------------------------------
# Problem loading + split (iteration 7)
#
# BUG MITIGATED: training on the same problems we evaluate on (75-set,
# 90-set, shardul_holdout) would silently inflate scores. Race_bank.json
# is a superset of those eval suites — the production scorer's "hard" sets
# live inside problem_suite_v{1,2,3}.json and shardul_holdout.json. We
# exclude any problem whose normalized-query hash appears in those files.
#
# Also reserves a deterministic 10% in-distribution eval split (seeded
# shuffle, fixed seed) so we can spot generalization drift across training
# steps without touching the 75/90-set scoring suites.
# ----------------------------------------------------------------------


_HOLDOUT_CANDIDATE_PATHS = [
    # Modal /trac mount layouts (when oro-trac-data carries the suites)
    "/trac/shardul_holdout.json",
    "/trac/suites/shardul_holdout.json",
    "/trac/data/suites/shardul_holdout.json",
    "/trac/problem_suite_v1.json",
    "/trac/problem_suite_v2.json",
    "/trac/problem_suite_v3.json",
    "/trac/suites/problem_suite_v1.json",
    "/trac/suites/problem_suite_v2.json",
    "/trac/suites/problem_suite_v3.json",
    "/trac/data/suites/problem_suite_v1.json",
    "/trac/data/suites/problem_suite_v2.json",
    "/trac/data/suites/problem_suite_v3.json",
]

_RACE_BANK_CANDIDATES = [
    "/trac/race_bank.json",
    "/trac/data/suites/race_bank.json",
    "/data/race_bank.json",
]


def _problem_query_hash(problem: dict) -> str:
    """Stable hash key for a problem.

    Normalisation: lowercase, collapse whitespace runs. Picks `query` if
    present else `prompt` (some suites use either). 16 hex chars of sha256
    is plenty — 2^64 collision space against a few-thousand-problem corpus.
    """
    q = problem.get("query") or problem.get("prompt") or ""
    norm = " ".join(str(q).lower().split())
    if not norm:
        return ""
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _extract_problems_from_json(data) -> list[dict]:
    """Accept either a list[problem] or a dict-with-'problems' top level."""
    if isinstance(data, dict):
        if isinstance(data.get("problems"), list):
            return [d for d in data["problems"] if isinstance(d, dict)]
        # Otherwise treat dict values as problems if they look like dicts.
        return [d for d in data.values() if isinstance(d, dict) and (d.get("query") or d.get("prompt"))]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict) and (d.get("query") or d.get("prompt"))]
    return []


def _load_holdout_hashes(paths: list[str] | None = None) -> tuple[set[str], list[str]]:
    """Read every holdout suite file we can find and return (hashes, found_paths).

    Missing files are silently skipped — different mount layouts expose
    different subsets — but the returned `found_paths` lets the caller log
    exactly which files contributed so a missing mount fails LOUD via a
    visible "no holdout files found" warning rather than silent leakage.
    """
    if paths is None:
        paths = _HOLDOUT_CANDIDATE_PATHS
    hashes: set[str] = set()
    found: list[str] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p) as fh:
                data = json.load(fh)
        except Exception as e:  # malformed file should not silently skip
            print(f"[holdout] WARN: failed to parse {p}: {e}")
            continue
        problems = _extract_problems_from_json(data)
        added = 0
        for prob in problems:
            h = _problem_query_hash(prob)
            if h and h not in hashes:
                hashes.add(h)
                added += 1
        if added:
            found.append(p)
            print(f"[holdout] loaded {added} from {p}")
    return hashes, found


def _split_train_eval(
    problems: list[dict],
    *,
    eval_frac: float = 0.1,
    seed: int = 1234,
) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle then partition.

    Splits AFTER holdout exclusion so the eval slice is real in-distribution
    novelty for the policy — not just the 75/90-set re-tested under a
    different name.
    """
    if not problems:
        return [], []
    rng = random.Random(seed)
    indexed = list(enumerate(problems))
    # Sort by hash so input order doesn't affect the split (race_bank.json
    # ordering can drift). Then shuffle with seeded RNG.
    indexed.sort(key=lambda ip: _problem_query_hash(ip[1]) or f"_idx{ip[0]:08d}")
    rng.shuffle(indexed)
    n_eval = max(1, int(len(indexed) * eval_frac)) if 0 < eval_frac < 1 else 0
    eval_items = [p for _, p in indexed[:n_eval]]
    train_items = [p for _, p in indexed[n_eval:]]
    return train_items, eval_items


def _load_problems_and_split(
    *,
    max_problems: int | None = None,
    eval_frac: float = 0.1,
    seed: int = 1234,
    race_bank_paths: list[str] | None = None,
    holdout_paths: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """End-to-end: load race_bank → drop holdouts → train/eval split.

    Returns (train, eval). Both lists exclude any problem whose normalized
    query hash matches anything in the holdout suites.
    """
    candidates = race_bank_paths if race_bank_paths is not None else _RACE_BANK_CANDIDATES
    raw_problems: list[dict] = []
    src_path: str | None = None
    for p in candidates:
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            data = json.load(fh)
        raw_problems = _extract_problems_from_json(data)
        if raw_problems:
            src_path = p
            print(f"[problems] loaded {len(raw_problems)} from {p}")
            break
    if not raw_problems:
        print("[problems] WARN: race_bank not found on any candidate path; using embedded fallback")
        raw_problems = list(_FALLBACK_PROBLEMS)

    holdout_hashes, found = _load_holdout_hashes(holdout_paths)
    if not found:
        print(
            "[holdout] WARN: no holdout suite files found on disk — "
            "training and 75/90-set eval problems will OVERLAP. "
            "Mount oro-trac-data or set --holdout-paths to fix."
        )
    pre = len(raw_problems)
    filtered = [p for p in raw_problems if _problem_query_hash(p) not in holdout_hashes]
    print(f"[problems] holdout exclusion: {pre} → {len(filtered)} (dropped {pre - len(filtered)})")

    if max_problems is not None:
        filtered = filtered[:max_problems]

    train, eval_set = _split_train_eval(filtered, eval_frac=eval_frac, seed=seed)
    print(
        f"[problems] split (seed={seed} eval_frac={eval_frac}): "
        f"train={len(train)} eval={len(eval_set)} (src={src_path})"
    )
    return train, eval_set


# ----------------------------------------------------------------------
# Mask construction
# ----------------------------------------------------------------------


_ASSISTANT_OPEN = "<|im_start|>assistant\n"
_TURN_END = "<|im_end|>"


def _tokenize_with_assistant_mask(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Render `messages` and return (input_ids, assistant_mask).

    Strategy: render to STRING, re-tokenize with return_offsets_mapping=True,
    then locate each `<|im_start|>assistant\\n ... <|im_end|>` region in the
    string and flip the mask for tokens overlapping that char range.

    Why not prefix-diff: Qwen3's chat template STRIPS `<think>...</think>`
    from every assistant turn except the most-recent one. The prior approach
    (incremental re-render of `messages[:i+1]` + `len()` arithmetic) produced
    spans that overshot into the next user message for non-final assistant
    turns. Surfaced by iteration-2 validator on the first test case.
    """
    full_str: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(full_str, return_offsets_mapping=True, add_special_tokens=False)
    ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = list(enc["offset_mapping"])
    mask = [0] * len(ids)

    pos = 0
    while True:
        s = full_str.find(_ASSISTANT_OPEN, pos)
        if s == -1:
            break
        content_start = s + len(_ASSISTANT_OPEN)
        e = full_str.find(_TURN_END, content_start)
        if e == -1:
            # No closing — assistant turn never finished. Mask through end-of-string
            # so the model still learns to terminate even if the rollout truncated.
            content_end = len(full_str)
            pos = len(full_str)
        else:
            # Include the <|im_end|> token so the policy is trained to stop.
            content_end = e + len(_TURN_END)
            pos = content_end
        for tok_i, (a, b) in enumerate(offsets):
            if b <= content_start:
                continue
            if a >= content_end:
                break
            mask[tok_i] = 1
    return ids, mask


# ----------------------------------------------------------------------
# Mask validator (iteration 2)
#
# BUG MITIGATED: assistant-token masks built by prefix-diff can silently
# include system-prompt or tool-result tokens if the chat template adds extra
# boilerplate at message boundaries. Wrong masks train on the wrong tokens
# while loss curves look healthy. Mitigation: decode-and-assert — for each
# contiguous mask==1 span, the decoded text must contain the corresponding
# assistant message content, and tool-result content from neighbouring
# `user` turns must NOT appear in any mask==1 span.
# ----------------------------------------------------------------------


class MaskValidationError(AssertionError):
    """Raised when the assistant mask fails decode-and-assert."""


def _mask_spans(mask: list[int]) -> list[tuple[int, int]]:
    """Return [(start, end), ...] for each maximal contiguous run of mask==1."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i] == 1:
            j = i
            while j < n and mask[j] == 1:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def _norm_ws(s: str) -> str:
    """Whitespace-collapse for substring comparison; defends against
    tokenizer-introduced whitespace drift between encode/decode round-trips."""
    return " ".join(s.split())


def _strip_ws(s: str) -> str:
    """Remove ALL whitespace. Chat templates often re-format `</think>X` as
    `</think>\\n\\nX`; the validator only needs to see the same characters in
    the same order, not identical whitespace."""
    return "".join(s.split())


def validate_assistant_mask(
    tokenizer,
    messages: list[dict],
    input_ids: list[int],
    mask: list[int],
    *,
    leak_probe_chars: int = 40,
) -> None:
    """Decode-and-assert. Raises MaskValidationError if the mask is wrong.

    Checks:
      A. Span count == number of assistant messages.
      B. For each (span, assistant message) pair, the assistant content
         (whitespace-normalised) appears inside the decoded span.
      C. For each non-assistant message with substantial content (system/user/
         tool), the first `leak_probe_chars` chars do NOT appear in any
         mask==1 span — catches off-by-one mask shifts.
    """
    if len(input_ids) != len(mask):
        raise MaskValidationError(
            f"input_ids/mask length mismatch: {len(input_ids)} vs {len(mask)}"
        )

    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    spans = _mask_spans(mask)

    # (A) one span per assistant message
    if len(spans) != len(assistant_msgs):
        raise MaskValidationError(
            f"mask span count {len(spans)} != assistant message count "
            f"{len(assistant_msgs)}; spans={spans[:5]}{'...' if len(spans) > 5 else ''}"
        )

    # (B) each span's decoded text contains the corresponding assistant content.
    #
    # Qwen3's chat template drops <think>...</think> from every assistant turn
    # except the most-recent one. So we compare against the post-</think>
    # portion of the recorded content for all non-final assistant turns; for
    # the final turn the full content survives.
    # Qwen3 preserves <think>...</think> only on the final assistant turn AND
    # only when that turn is the last message in the convo (loop.last in the
    # chat template). All other assistant turns get their think block stripped.
    final_keeps_think = (
        bool(messages)
        and messages[-1].get("role") == "assistant"
        and assistant_msgs
        and messages[-1] is assistant_msgs[-1]
    )
    span_decoded_stripped: list[str] = []
    last_idx = len(assistant_msgs) - 1
    for span_idx, (span, msg) in enumerate(zip(spans, assistant_msgs)):
        start, end = span
        decoded = tokenizer.decode(input_ids[start:end], skip_special_tokens=True)
        decoded_stripped = _strip_ws(decoded)
        span_decoded_stripped.append(decoded_stripped)
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        keeps_think = final_keeps_think and span_idx == last_idx
        if not keeps_think and "</think>" in content:
            content = content.split("</think>", 1)[1].lstrip("\n")
        if not content.strip():
            continue
        content_stripped = _strip_ws(content)
        if content_stripped not in decoded_stripped:
            raise MaskValidationError(
                "assistant content not found inside its mask==1 span.\n"
                f"  span_idx={span_idx} span [{start}:{end}]\n"
                f"  decoded[:200]={decoded[:200]!r}\n"
                f"  expected[:200]={content[:200]!r}"
            )

    # (C) no non-assistant message content leaks into any mask==1 span.
    full_masked_stripped = "".join(span_decoded_stripped)
    for msg in messages:
        if msg.get("role") == "assistant":
            continue
        body = (msg.get("content") or "").strip()
        if len(body) < leak_probe_chars:
            continue
        probe = _strip_ws(body[:leak_probe_chars])
        if not probe:
            continue
        # The system prompt's tool list is benign and may be paraphrased by the
        # model inside <think>, so only flag when the probe text appears
        # verbatim AND wasn't also present in the assistant content (i.e., the
        # mask actually grabbed non-assistant tokens, not just a quoted echo).
        if probe in full_masked_stripped:
            quoted_by_assistant = any(
                probe in _strip_ws(m.get("content") or "")
                for m in assistant_msgs
            )
            if quoted_by_assistant:
                continue
            raise MaskValidationError(
                f"non-assistant content (role={msg.get('role')!r}) leaked into "
                f"mask==1 span. probe={probe[:120]!r}"
            )


# ----------------------------------------------------------------------
# eval()/train() discipline (iteration 3)
#
# BUG MITIGATED: PeftModel.from_pretrained(is_trainable=True) returns the
# model in TRAIN mode. The LoRA adapter has lora_dropout=0.05; dropout
# during autoregressive generate() corrupts late-turn output by compounding
# stochasticity across turns. Loss curves stay healthy while behavior
# degrades. Mitigation: model.eval() before EVERY generate() call (enforced
# inside `run_episode._generate` as well), model.train() before backward,
# and a runtime assertion that no Dropout/LoRA-dropout module is in training
# mode at rollout time.
# ----------------------------------------------------------------------


def _iter_dropout_modules(model):
    """Yield (name, module) pairs for every dropout-like submodule.

    Catches both plain nn.Dropout and PEFT's LoRA `lora_dropout` (which is
    nn.Dropout when p>0, nn.Identity when p==0). Identity counts only when
    explicitly nested under `lora_dropout` so the structural presence of LoRA
    is observable even on adapters with p=0.
    """
    import torch.nn as nn

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Dropout):
            yield name, mod
            continue
        if "lora_dropout" in name and isinstance(mod, nn.Identity):
            yield name, mod


def _count_active_dropout_modules(model) -> int:
    """How many dropout-like modules currently have .training == True.

    `nn.Dropout` only actually drops samples when `self.training` is True
    AND `p > 0`. We want both conditions to enforce discipline; the count
    returned here is the dangerous one (training-mode dropouts with p>0).
    """
    import torch.nn as nn

    n = 0
    for _name, mod in _iter_dropout_modules(model):
        p = getattr(mod, "p", 0.0)
        if isinstance(mod, nn.Dropout) and mod.training and p and p > 0:
            n += 1
    return n


def _assert_no_active_dropout(model, where: str) -> None:
    """Fail loud if any dropout module is active. Called before generate()."""
    n = _count_active_dropout_modules(model)
    if n > 0:
        # Surface the first few offenders for debuggability.
        import torch.nn as nn

        offenders = []
        for name, mod in _iter_dropout_modules(model):
            p = getattr(mod, "p", 0.0)
            if isinstance(mod, nn.Dropout) and mod.training and p and p > 0:
                offenders.append((name, p))
                if len(offenders) >= 5:
                    break
        raise RuntimeError(
            f"[{where}] {n} dropout modules active during generation. "
            f"First offenders: {offenders}. "
            f"Call model.eval() before any generate() call."
        )


# ----------------------------------------------------------------------
# Gradient checkpointing (iteration 12)
#
# BUG MITIGATED: at the practice-run constraint of 20K-24K max_seq_length,
# Qwen3-4B's vocab (~152K) means log_softmax(logits.float()) materializes a
# 20K * 152K * 4 byte = ~12 GB fp32 tensor per forward, plus the softmax
# saved by autograd for backward. Two of these per training example (policy
# + ref) plus B*K rollout activations OOMs an 80 GB H100 before training
# can start. Gradient checkpointing trades ~30% compute for the activation
# memory savings that make the 20K case fit.
#
# PEFT subtlety: the base model is frozen, so gradients normally don't flow
# through the input embeddings to reach LoRA-injected layers. Calling
# `enable_input_require_grads()` flips that on by registering a forward
# hook on the embeddings that re-enables grad on the output. Without it,
# `gradient_checkpointing_enable()` silently produces zero gradients on
# the LoRA parameters — same symptoms as a wrong mask.
#
# use_reentrant=False is the modern API; reentrant=True still works but
# is incompatible with input-require-grads in some PyTorch versions and
# is being deprecated by HuggingFace.
# ----------------------------------------------------------------------


def _setup_gradient_checkpointing(model) -> bool:
    """Enable gradient checkpointing on a PEFT-wrapped model.

    Returns True iff both calls succeeded. Returns False (and prints) when
    the model doesn't expose the relevant hooks — defensive so a future
    refactor that swaps the model class doesn't silently lose the memory
    savings on the next run.

    Always: `enable_input_require_grads()` BEFORE `gradient_checkpointing_enable()`
    — reverse order means the hook is registered after gradient checkpointing
    has already cached the (no-grad) embedding output, and LoRA params get
    zero gradient.
    """
    enable_grads = getattr(model, "enable_input_require_grads", None)
    enable_ckpt = getattr(model, "gradient_checkpointing_enable", None)
    if enable_grads is None or enable_ckpt is None:
        print(
            "[gc] WARN: model lacks enable_input_require_grads or "
            "gradient_checkpointing_enable; skipping (LoRA grads may OOM at "
            "long context)."
        )
        return False
    enable_grads()
    try:
        enable_ckpt(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        # Older transformers versions don't accept the kwargs dict.
        enable_ckpt()
    print("[gc] gradient checkpointing enabled (use_reentrant=False)")
    return True


# ----------------------------------------------------------------------
# Behavioral metrics (iteration 4)
#
# BUG MITIGATED: loss and KL alone do NOT surface gradient-direction bugs.
# Wrong assistant masks, sign-flipped advantages, or a degenerating policy
# all leave loss curves looking healthy while behavior collapses. Mitigation:
# log avg_turns / recommend_emit_rate / catalog_hit_rate / gt_match_rate
# every step from step 0 so direction-of-travel is visible immediately, and
# persist as JSONL inside `out_dir` so post-hoc plotting doesn't need parsing
# of free-form logs.
# ----------------------------------------------------------------------


def _aggregate_step_metrics(groups: list[list]) -> dict:
    """Reduce one step's rollouts + rewards into a flat metrics dict.

    `groups` is the per-step structure built in the trainer:
    list[ list[ (RolloutResult, reward_dict) ] ] (outer = per-problem, inner =
    K rollouts). Output keys are stable so a downstream JSONL parser can index
    by name.
    """
    flat = [(rollout, rwd) for grp in groups for (rollout, rwd) in grp]
    n = len(flat)
    tier_counts = {"none": 0, "hallucinated": 0, "partial": 0, "exact": 0}
    if n == 0:
        return {"n_rollouts": 0, "tier_counts": tier_counts}

    n_term = sum(1 for r, _ in flat if r.terminated)
    n_trunc = sum(1 for r, _ in flat if r.truncated)
    n_err = sum(1 for r, _ in flat if r.error)
    turns = [r.n_steps for r, _ in flat]

    n_emit = sum(1 for _, rw in flat if rw.get("tier", "none") != "none")
    n_cat = sum(1 for _, rw in flat if rw.get("catalog_hit"))
    n_gt = sum(1 for _, rw in flat if rw.get("gt_match"))

    for _, rw in flat:
        t = rw.get("tier", "none")
        tier_counts[t] = tier_counts.get(t, 0) + 1

    rewards = [rw.get("reward", 0.0) for _, rw in flat]
    rule_scores = [rw.get("rule_score", 0.0) for _, rw in flat]

    # Within-group reward variance: when this is 0 across all groups, Dr.GRPO
    # advantages collapse to 0 → no learning signal that step. Track so we can
    # spot dead-batch steps without re-deriving from per-rollout logs.
    nonzero_var_groups = 0
    for grp in groups:
        rs = [rw.get("reward", 0.0) for _, rw in grp]
        if len(rs) < 2:
            continue
        mu = sum(rs) / len(rs)
        var = sum((r - mu) ** 2 for r in rs) / len(rs)
        if var > 1e-12:
            nonzero_var_groups += 1

    return {
        "n_rollouts": n,
        "n_groups": len(groups),
        "n_groups_with_signal": nonzero_var_groups,
        "avg_turns": sum(turns) / n,
        "max_turns": max(turns),
        "min_turns": min(turns),
        "terminated_rate": n_term / n,
        "truncated_rate": n_trunc / n,
        "error_rate": n_err / n,
        "recommend_emit_rate": n_emit / n,
        "catalog_hit_rate": n_cat / n,
        "gt_match_rate": n_gt / n,
        "mean_reward": sum(rewards) / n,
        "mean_rule_score": sum(rule_scores) / n,
        "tier_counts": tier_counts,
    }


def _format_step_metrics(m: dict) -> str:
    """One-line human-readable summary for the per-step print."""
    if m.get("n_rollouts", 0) == 0:
        return "no rollouts"
    tc = m["tier_counts"]
    return (
        f"n={m['n_rollouts']} "
        f"turns(avg={m['avg_turns']:.1f}/max={m['max_turns']}) "
        f"emit={m['recommend_emit_rate']:.2f} "
        f"cat_hit={m['catalog_hit_rate']:.2f} "
        f"gt={m['gt_match_rate']:.2f} "
        f"rule={m['mean_rule_score']:.2f} "
        f"reward={m['mean_reward']:.3f} "
        f"trunc={m['truncated_rate']:.2f} "
        f"err={m['error_rate']:.2f} "
        f"signal_groups={m['n_groups_with_signal']}/{m['n_groups']} "
        f"tiers[n/h/p/e]={tc['none']}/{tc['hallucinated']}/{tc['partial']}/{tc['exact']}"
    )


# ----------------------------------------------------------------------
# In-distribution eval pass (iteration 9)
#
# BUG MITIGATED: training mean_reward rises by construction (optimizer
# gradient-ascends on it). Without a held-out eval the trainer cannot tell
# overfitting-to-sampled-problems from real generalization. Iteration 7
# reserved a deterministic 10% slice; this iteration runs the rollout harness
# on that slice every `eval_every` steps, reuses `_aggregate_step_metrics` for
# reduction, and persists results as `eval_metrics.jsonl` alongside the
# step_metrics.jsonl train trace.
#
# The helper is pure-python (rollout_fn + reward_fn are injected) so eval
# arithmetic can be tested without GPU/Modal.
# ----------------------------------------------------------------------


def _collect_eval_groups(problems, rollout_fn, reward_fn, *, K_eval):
    """Run K_eval rollouts per problem and pair each with its reward.

    Output shape matches the per-step `groups` structure built in the trainer
    (`list[list[(rollout, reward_dict)]]`) so `_aggregate_step_metrics` can
    reduce eval and train rollouts identically — fewer downstream parsing
    branches, identical signal definitions across the two JSONLs.
    """
    groups: list[list] = []
    for problem in problems:
        per_problem: list = []
        for _ in range(K_eval):
            rollout = rollout_fn(problem)
            rwd = reward_fn(rollout, problem)
            per_problem.append((rollout, rwd))
        groups.append(per_problem)
    return groups


# ----------------------------------------------------------------------
# Positive-only advantage filter + KL k3 estimator (iteration 5)
#
# BUG MITIGATED (filter): in multi-turn rollouts the K rollouts from one
# problem share an opening prefix. Per-token gradients on those shared tokens
# get pulled up by the single +adv rollout and pulled down by the N-1
# negative-adv rollouts → net suppression of exactly the prefix tokens you
# want to keep. Simplest correct fix: drop any rollout with adv <= 0. Smarter
# token-level credit assignment is future work; positive-only is the v1
# safety net.
#
# BUG MITIGATED (KL): without a KL anchor against the SFT init the policy
# drifts off-manifold within a handful of steps (degenerate outputs, repeated
# tool-calls, dead chain-of-thought). beta=0.1 is the floor — the k3
# estimator yields ~1e-3 to 2e-2 per token, so beta=0.04 (RLHF default) is
# practically inert. Beta is computed via dual-adapter: we load the same SFT
# adapter twice (policy=trainable, ref=frozen) and switch active adapter
# during the loss forward.
# ----------------------------------------------------------------------


def _drgrpo_advantages(rewards: list[float]) -> list[float]:
    """Dr.GRPO: advantage = reward - group_mean. No std norm, no length norm."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def _filter_positive_advantage_examples(examples: list[tuple]) -> tuple[list[tuple], int, int]:
    """Drop (rollout, adv) entries whose advantage is <= 0.

    Returns (kept, n_kept, n_dropped). Strict > 0 — exactly-zero advantages
    contribute no gradient anyway but DO add cost to the KL forward, so we
    skip them too.
    """
    kept = [(r, a) for (r, a) in examples if a > 0]
    return kept, len(kept), len(examples) - len(kept)


def _kl_k3_per_token(logp_policy, logp_ref):
    """Schulman k3 unbiased estimator for KL(policy || ref) per token.

    k3 = exp(log_ratio) - log_ratio - 1, where log_ratio = logp_ref - logp_policy.
    Always >= 0. Lower variance than k1, unbiased unlike k2.
    Inputs are token log-probs of the SAME tokens under the two distributions.
    """
    log_ratio = logp_ref - logp_policy
    return log_ratio.exp() - log_ratio - 1.0


def _grouped(problems: list[dict], B: int, step: int) -> list[dict]:
    """Naive round-robin sampler. Retained for tests / fallback only.

    Length-bucketed sampling (`_length_bucketed_grouped` below) is now the
    default for the trainer because mixing 500-token and 35K-token problems
    in the same batch wastes compute on padding and truncates the long tail.
    """
    if not problems:
        return []
    n = len(problems)
    start = (step * B) % n
    if start + B <= n:
        return problems[start : start + B]
    # wrap
    return (problems[start:] + problems[: (start + B) % n])[:B]


# ----------------------------------------------------------------------
# Length-bucketed batch sampling (iteration 10)
#
# BUG MITIGATED (#5 in notes.md): ShoppingBench rollouts range 500-35K
# tokens. Naive batching (a) wastes compute on padding because every short
# rollout in the batch gets padded to the longest one, and (b) pushes the
# long tail past `max_seq_length` truncation when one long rollout sits
# next to short ones. Sorting by expected trajectory length and cycling
# through equal-size buckets keeps each step's batch homogeneous.
#
# Determinism: same (problems, B, step) -> same batch. Bucket index
# advances by `step % n_buckets`; within-bucket offset advances by
# `step // n_buckets`. So consecutive steps visit different buckets,
# but after `n_buckets` steps we return to the first bucket with a
# fresh within-bucket offset — covering every problem in expectation.
# ----------------------------------------------------------------------


def _estimate_problem_length(problem: dict) -> int:
    """Cheap proxy for expected trajectory token length.

    Larger value => longer expected rollout. We don't tokenize the prompt
    here — this is called for every problem on every step, and the goal is
    relative ordering for bucketing, not absolute accuracy.

    Heuristic contributions (additive):
      - voucher tasks tend to run longer than product (multi-shop search loop)
      - longer query strings correlate with more spec constraints to satisfy
      - more ground-truth product ids ~= multi-target voucher problems
    """
    base = len(str(problem.get("query") or problem.get("prompt") or ""))
    cat = str(problem.get("category") or "").lower()
    if "voucher" in cat:
        base += 400
    elif "shop" in cat:
        base += 200
    gts = problem.get("ground_truth_product_ids") or []
    if isinstance(gts, list):
        base += 50 * len(gts)
    return base


def _length_bucketed_grouped(
    problems: list[dict],
    B: int,
    step: int,
    *,
    n_buckets: int = 3,
) -> list[dict]:
    """Length-bucketed sampler. Replaces `_grouped` as the trainer default.

    Sort the corpus by `_estimate_problem_length` (stable tiebreak on
    `_problem_query_hash` so identical-length problems don't reshuffle
    across runs), partition into `n_buckets` contiguous slices, cycle
    through buckets by step. Within a bucket, round-robin by `step // n_buckets`.

    Returns at most B problems. Falls back to returning everything when
    `len(problems) <= B` (a B-of-1 corpus has nothing to bucket).
    """
    if not problems:
        return []
    n = len(problems)
    if n <= B:
        return list(problems)
    # Clamp n_buckets so every bucket can supply at least one batch.
    n_buckets = max(1, min(n_buckets, n // max(B, 1)))
    sorted_problems = sorted(
        problems,
        key=lambda p: (_estimate_problem_length(p), _problem_query_hash(p)),
    )
    bucket_size = n // n_buckets
    bucket_idx = step % n_buckets
    start = bucket_idx * bucket_size
    end = n if bucket_idx == n_buckets - 1 else (bucket_idx + 1) * bucket_size
    bucket = sorted_problems[start:end]
    bs = len(bucket)
    if bs == 0:
        return []
    cycle = step // n_buckets
    offset = (cycle * B) % bs
    if offset + B <= bs:
        return bucket[offset : offset + B]
    # wrap within bucket
    return (bucket[offset:] + bucket[: (offset + B) % bs])[:B]


# ----------------------------------------------------------------------
# 75/90-set scoring suite loader (iteration 11)
#
# BUG MITIGATED: without an apples-to-apples scoring entrypoint, RL lift
# vs the SFT baseline cannot be measured. The SFT baseline numbers
# (40.0% / 33.3%) come from running the policy against the 75-set
# (shardul_holdout.json) and the 90-set (union of problem_suite_v1/2/3
# qualifying problems). This loader picks the right files for each suite
# name and dedupes by query hash so the 90-set count matches the published
# baseline. If no candidate file exists on disk, we WARN LOUD and return
# empty — silent fallback to an embedded toy set would falsely report a
# successful scoring run.
# ----------------------------------------------------------------------


_SUITE_75 = "75"
_SUITE_90 = "90"
_VALID_SUITES = (_SUITE_75, _SUITE_90)

_SUITE_PATH_CANDIDATES = {
    _SUITE_75: [
        # The 75-set lives in shardul_holdout.json.
        "/trac/shardul_holdout.json",
        "/trac/suites/shardul_holdout.json",
        "/trac/data/suites/shardul_holdout.json",
    ],
    _SUITE_90: [
        # The 90-set is the union of qualifying problems across v1/v2/v3.
        "/trac/problem_suite_v1.json",
        "/trac/problem_suite_v2.json",
        "/trac/problem_suite_v3.json",
        "/trac/suites/problem_suite_v1.json",
        "/trac/suites/problem_suite_v2.json",
        "/trac/suites/problem_suite_v3.json",
        "/trac/data/suites/problem_suite_v1.json",
        "/trac/data/suites/problem_suite_v2.json",
        "/trac/data/suites/problem_suite_v3.json",
    ],
}


def _load_scoring_suite(
    suite: str,
    *,
    paths: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Load a named scoring suite. Returns (problems, found_paths).

    Dedupes by `_problem_query_hash` so the 90-set (union of v1/v2/v3) does
    not double-count problems shared across files. Loud WARN when no
    candidate file is present — silent fallback would let a misconfigured
    mount produce a "successful" scoring run on an embedded toy set.
    """
    if suite not in _VALID_SUITES:
        raise ValueError(
            f"unknown suite {suite!r}; valid: {_VALID_SUITES}"
        )
    candidates = paths if paths is not None else _SUITE_PATH_CANDIDATES[suite]
    seen_hashes: set[str] = set()
    out: list[dict] = []
    found: list[str] = []
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            with open(p) as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"[score] WARN: failed to parse {p}: {e}")
            continue
        problems = _extract_problems_from_json(data)
        added = 0
        for prob in problems:
            h = _problem_query_hash(prob)
            if not h:
                continue
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            out.append(prob)
            added += 1
        if added:
            found.append(p)
            print(f"[score] suite={suite} loaded {added} from {p} (total {len(out)})")
    if not found:
        print(
            f"[score] WARN: suite={suite} found no candidate files on disk. "
            f"Returning EMPTY problem list. Mount oro-trac-data or pass --paths."
        )
    return out, found


def _aggregate_scoring_metrics(groups: list[list]) -> dict:
    """Score-set summary. Adds a `success_rate` alias for gt_match_rate so the
    headline number maps directly to the published SFT baseline metric (40.0%
    on the 75-set, 33.3% on the 90-set). Everything else reuses the train
    metrics shape so downstream tooling can parse one schema.
    """
    m = _aggregate_step_metrics(groups)
    # "production-strict success" in the notes is the exact ground-truth
    # match rate — partial-credit tier doesn't count toward the headline.
    m["success_rate"] = m.get("gt_match_rate", 0.0)
    return m


# ----------------------------------------------------------------------
# Main remote function
# ----------------------------------------------------------------------


@app.function(
    image=image,
    gpu="H200",  # torch 2.6 missing Blackwell sm_100 kernels (v15 tried B200 -> "no kernel image")
    timeout=12 * 60 * 60,
    volumes={"/ckpt": ckpt_volume, "/trac": trac_volume, "/distill": distill_volume},
    secrets=[modal.Secret.from_name("oro-wandb-api-key")],  # ORO-1310; re-added for oro-ai run (ORO-1239)
)
def run_training(
    *,
    max_steps: int = 20,
    B: int = 2,
    K: int = 4,
    lr: float = 1e-7,
    beta: float = 0.1,  # iteration 5: KL anchor strength (k3 against frozen ref adapter)
    max_rollout_steps: int = 25,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    save_every: int = 2,
    eval_every: int = 5,
    K_eval: int = 2,
    use_gradient_checkpointing: bool = True,  # iter 12: 20K-seq OOM mitigation
    out_dir: str = "/ckpt/drgrpo_v1",
    smoke: bool = False,
    use_positive_only_filter: bool = True,
    continuous_reward: bool = False,
    keep_top_n: int = 2,
    policy_adapter: str = POLICY_ADAPTER_PATH,
    use_batched_rollouts: bool = False,
    use_ratio_clip: bool = False,
    ratio_clip_eps: float = 0.2,
    vllm_rollout_url: str = "",
    vllm_rollout_model: str = "opd_renderers",
    vllm_rollout_workers: int = 24,
    tool_weight: float = 0.0,  # v19: blend dense per-step tool-match reward into training reward
    ref_traj_path: str = "/distill/rejection_sample_train_v3_trajectories.jsonl",
):
    import requests
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The harness/reward modules ship with the image via add_local_python_source.
    from src.drgrpo.rollout_harness import run_episode, run_episode_batch, run_batch_via_vllm
    from src.drgrpo.reward import compute_reward

    if smoke:
        max_steps = 3
        B = 1
        K = 2
        max_rollout_steps = 15  # opd_rebal typically needs 5-10 turns to reach recommend_product
        max_new_tokens = 512
        out_dir = "/ckpt/drgrpo_smoke"
        save_every = max_steps  # only at end
        # Iteration 9: smoke eval = run once after the final step on a single
        # problem so the eval code path is exercised without ballooning runtime.
        eval_every = max_steps
        K_eval = 1

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== Dr.GRPO {'SMOKE' if smoke else 'TRAIN'} ===")
    print(f"  backbone:           {BACKBONE}")
    print(f"  policy adapter:     {policy_adapter}")
    print(f"  ref adapter:        {policy_adapter} (frozen, dual-adapter)")
    print(f"  max_steps={max_steps}  B={B}  K={K}  lr={lr}  beta={beta}")
    print(f"  max_rollout_steps={max_rollout_steps}  max_new_tokens={max_new_tokens}")
    print(f"  eval_every={eval_every}  K_eval={K_eval}")
    print(f"  out_dir={out_dir}")

    # ORO-1310 wandb wiring: project paris-distillation. Run name = out_dir basename.
    import os as _os
    _wandb_run = None
    if _os.environ.get("WANDB_API_KEY") and not smoke:
        try:
            import wandb
            _wandb_run = wandb.init(
                project="paris-distillation",
                name=_os.path.basename(out_dir.rstrip("/")) or "drgrpo",
                config={"max_steps": max_steps, "B": B, "K": K, "lr": lr, "beta": beta,
                        "max_rollout_steps": max_rollout_steps, "temperature": temperature,
                        "policy_adapter": policy_adapter,
                        "use_positive_only_filter": use_positive_only_filter,
                        "continuous_reward": continuous_reward, "keep_top_n": keep_top_n,
                        "use_batched_rollouts": use_batched_rollouts},
                reinit=True,
            )
            print(f"  wandb: {_wandb_run.url if _wandb_run else 'init failed'}")
        except Exception as e:
            print(f"  wandb init failed: {e}")
            _wandb_run = None

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BACKBONE, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # Iteration 5: dual-adapter — same SFT weights loaded twice. "policy" is
    # trainable, "ref" is frozen and used only inside torch.no_grad to compute
    # the KL anchor. set_adapter("policy") between forwards so generation +
    # backward always run on the trainable copy.
    model = PeftModel.from_pretrained(
        base, policy_adapter, adapter_name="policy", is_trainable=True
    )
    model.load_adapter(policy_adapter, adapter_name="ref", is_trainable=False)
    model.set_adapter("policy")
    # Sanity-check: ref params must NOT require grad. If they do, the KL
    # forward would silently accumulate gradients into the "reference" weights
    # and destroy the anchor.
    n_ref_trainable = sum(
        p.numel() for n, p in model.named_parameters() if "ref" in n and p.requires_grad
    )
    if n_ref_trainable != 0:
        raise RuntimeError(
            f"ref adapter has {n_ref_trainable:,} trainable params; expected 0"
        )
    # Iteration 12: gradient checkpointing — without it, 20K-seq forward
    # OOMs an H100 on the fp32 log_softmax + softmax-for-backward. PEFT
    # requires enable_input_require_grads() before gradient_checkpointing_enable()
    # or LoRA params silently get zero gradient.
    if use_gradient_checkpointing:
        _setup_gradient_checkpointing(model)
    # Iteration 3: model is in train mode after PeftModel.from_pretrained when
    # is_trainable=True. Force eval before the first generate() and report the
    # observable dropout footprint so misconfigured adapters fail loud here
    # rather than silently corrupting rollouts.
    n_dropout_total = sum(1 for _ in _iter_dropout_modules(model))
    model.train()
    active_in_train = _count_active_dropout_modules(model)
    model.eval()
    active_in_eval = _count_active_dropout_modules(model)
    print(
        f"  dropout modules:    total={n_dropout_total} "
        f"active_in_train={active_in_train} active_in_eval={active_in_eval}"
    )
    if active_in_eval != 0:
        raise RuntimeError(
            f"model.eval() did not disable dropout (active={active_in_eval}); "
            "discipline assertion would never fire"
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  trainable params:   {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    session = requests.Session()

    # Iteration 7: exclude 75/90-set + shardul_holdout problems from training
    # via stable query-hash, reserve 10% deterministic in-distribution eval.
    train_problems, eval_problems = _load_problems_and_split(
        max_problems=None if not smoke else 16,
        eval_frac=0.1,
        seed=1234,
    )
    problems = train_problems
    if not problems:
        raise RuntimeError("no training problems after holdout exclusion")
    print(f"[problems] using {len(problems)} train (eval reserved: {len(eval_problems)})")

    # v19: attach per-step tool-match references (rejection-sampled success
    # trajectories on the training set) to each training problem, keyed by
    # normalized query. Eval problems are left unreferenced so eval reward
    # stays pure-outcome.
    if tool_weight > 0.0:
        import json as _json
        from src.drgrpo.reward import extract_toolseq as _ets

        def _ref_seq_from_row(r):
            # opd_sonnet teacher format: steps[].completion.message.tool_call[{name, parameters}]
            if r.get("steps"):
                seq = []
                for s in r["steps"]:
                    m = (s.get("completion") or {}).get("message") or {}
                    for c in (m.get("tool_call") or []):
                        seq.append((c.get("name"), c.get("parameters") or {}))
                return seq
            # rejection-sample format: OpenAI messages (structured tool_calls or <tool_call> text)
            return _ets(r.get("messages", []))

        def _row_ok(r):
            # rejection refs gate on score_ok; opd_sonnet teacher trajs are valid unless errored
            return bool(r.get("score_ok")) if "score_ok" in r else (not r.get("error"))

        ref_map: dict[str, list] = {}
        n_ref = 0
        with open(ref_traj_path) as _rf:
            for _line in _rf:
                if not _line.strip():
                    continue
                _r = _json.loads(_line)
                if not _row_ok(_r):
                    continue
                _seq = _ref_seq_from_row(_r)
                if _seq:
                    ref_map.setdefault(_r["query"].strip().lower(), []).append(_seq)
                    n_ref += 1
        hit = 0
        for _p in problems:
            _refs = ref_map.get(_p["query"].strip().lower())
            if _refs:
                _p["_ref_toolseqs"] = _refs
                hit += 1
        print(f"[v19] tool_weight={tool_weight} refs={n_ref} success-trajs over "
              f"{len(ref_map)} problems; matched {hit}/{len(problems)} train problems")
        if hit == 0:
            raise RuntimeError("v19: 0 training problems matched a reference trajectory — check ref_traj_path / query keys")

    # Iteration 4: per-step behavioral metrics persisted as JSONL for offline
    # plotting. Append mode so a resumed run accumulates rather than truncates.
    metrics_path = Path(out_dir) / "step_metrics.jsonl"
    metrics_fh = open(metrics_path, "a")
    # Iteration 9: separate JSONL for held-out eval so a downstream plotter can
    # join on `step` without filtering a mixed train/eval stream.
    eval_metrics_path = Path(out_dir) / "eval_metrics.jsonl"
    eval_metrics_fh = open(eval_metrics_path, "a")

    def _run_eval(step_label: int) -> None:
        """Run a no-grad eval pass on the reserved slice and persist metrics.

        `step_label` is the train step that just completed (0 == pre-training
        baseline). Caller decides cadence.
        """
        if not eval_problems or K_eval <= 0:
            return
        model.set_adapter("policy")
        model.eval()
        _assert_no_active_dropout(model, where=f"eval s{step_label}")

        def _rollout_fn(problem):
            return run_episode(
                model,
                tokenizer,
                problem,
                session=session,
                max_steps=max_rollout_steps,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        def _reward_fn(rollout, problem):
            return compute_reward(rollout, problem, session=session, continuous=continuous_reward)

        eval_groups = _collect_eval_groups(
            eval_problems, _rollout_fn, _reward_fn, K_eval=K_eval
        )
        em = _aggregate_step_metrics(eval_groups)
        em["step"] = step_label
        em["K_eval"] = K_eval
        em["n_eval_problems"] = len(eval_problems)
        print(f"   [eval @ step={step_label}] {_format_step_metrics(em)}")
        eval_metrics_fh.write(json.dumps(em) + "\n")
        eval_metrics_fh.flush()

    # Iteration 9: step-0 baseline so the first real training step has a
    # before/after comparison without waiting `eval_every` more steps.
    # Skipped on train (58 problems × K_eval × 25 turns = hours before step 1
    # even starts). Smoke still runs it via reduced eval subset.
    if smoke:
        _run_eval(step_label=0)

    for step in range(max_steps):
        # Iteration 10: length-bucketed sampling. Keeps each step's batch
        # homogeneous in expected trajectory length so padding doesn't
        # dominate compute and the long tail isn't clipped.
        batch = _length_bucketed_grouped(problems, B, step)
        print(f"\n--- step {step + 1}/{max_steps}  (batch={len(batch)})")

        # Iteration 5: paranoia — guarantee POLICY adapter is active before
        # generation. The loss loop swaps to "ref" and back, but a future
        # change/refactor that returns early could leave "ref" hot.
        model.set_adapter("policy")

        # 1+2. Rollout K times per problem
        groups: list[list] = []  # list[ list[ (rollout, reward_info) ] ]
        if vllm_rollout_url:
            # v17: fire all (B × K) rollouts in parallel against an external vllm.
            # Vllm continuous batching + thread-pool concurrency = ~5-10× speedup
            # vs in-process model.generate. The in-process policy is updated each
            # step; rollouts are off-policy (vllm = frozen behavior policy =
            # opd_renderers base). PPO ratio_clip handles the importance-sampling
            # correction implicitly via the same logp(policy)/logp(ref) ratio
            # already computed for the KL anchor (vllm policy == ref adapter).
            import time as _t
            t0 = _t.time()
            batch_results = run_batch_via_vllm(
                list(batch),
                K=K,
                vllm_url=vllm_rollout_url,
                model_name=vllm_rollout_model,
                max_workers=vllm_rollout_workers,
                max_steps=max_rollout_steps,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            print(f"   [vllm rollouts] {len(batch)} probs × {K} K = {len(batch)*K} rollouts in {_t.time()-t0:.1f}s")
            for p_idx, (problem, rollouts_for_problem) in enumerate(zip(batch, batch_results)):
                rollouts_for_p: list = []
                for k, rollout in enumerate(rollouts_for_problem):
                    rwd = compute_reward(rollout, problem, session=session, continuous=continuous_reward, tool_weight=tool_weight)
                    print(
                        f"   [p{p_idx} k{k}] tier={rwd['tier']:11s} "
                        f"reward={rwd['reward']:.3f} n_steps={rollout.n_steps} "
                        f"term={rollout.terminated} rec={rwd['rec_ids'][:1]}"
                    )
                    rollouts_for_p.append((rollout, rwd))
                groups.append(rollouts_for_p)
        else:
          for p_idx, problem in enumerate(batch):
            model.eval()
            _assert_no_active_dropout(model, where=f"rollout p{p_idx}")
            rollouts_for_p: list = []
            if use_batched_rollouts:
                # K parallel rollouts for this problem via batched generate.
                batch_rollouts = run_episode_batch(
                    model,
                    tokenizer,
                    problem,
                    K=K,
                    session=session,
                    max_steps=max_rollout_steps,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                for k, rollout in enumerate(batch_rollouts):
                    rwd = compute_reward(rollout, problem, session=session, continuous=continuous_reward, tool_weight=tool_weight)
                    print(
                        f"   [p{p_idx} k{k}] tier={rwd['tier']:11s} "
                        f"reward={rwd['reward']:.3f} n_steps={rollout.n_steps} "
                        f"term={rollout.terminated} rec={rwd['rec_ids'][:1]}"
                    )
                    rollouts_for_p.append((rollout, rwd))
            else:
                for k in range(K):
                    # Iteration 3: eval mode + decode-time assertion. Belt:
                    # `model.eval()` here. Suspenders: `_generate` in the harness
                    # also forces eval. The assertion catches either being skipped.
                    model.eval()
                    _assert_no_active_dropout(model, where=f"rollout p{p_idx} k{k}")
                    rollout = run_episode(
                        model,
                        tokenizer,
                        problem,
                        session=session,
                        max_steps=max_rollout_steps,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    rwd = compute_reward(rollout, problem, session=session, continuous=continuous_reward, tool_weight=tool_weight)
                    print(
                        f"   [p{p_idx} k{k}] tier={rwd['tier']:11s} "
                        f"reward={rwd['reward']:.3f} n_steps={rollout.n_steps} "
                        f"term={rollout.terminated} rec={rwd['rec_ids'][:1]}"
                    )
                    rollouts_for_p.append((rollout, rwd))
            groups.append(rollouts_for_p)

        # 3+4. Dr.GRPO advantages, per group
        raw_examples: list = []  # (rollout, advantage) BEFORE filter
        for rollouts_for_p in groups:
            rewards = [r["reward"] for _, r in rollouts_for_p]
            advs = _drgrpo_advantages(rewards)
            for (rollout, _rwd), adv in zip(rollouts_for_p, advs):
                raw_examples.append((rollout, adv))

        # 5. Iteration 5: positive-only advantage filter. Drops shared-prefix
        # gradient suppression from multi-turn rollouts. When every group has
        # zero within-group variance (group_mean == every reward), nothing
        # survives → skip the optimizer step rather than learning nothing.
        if use_positive_only_filter:
            flat_examples, n_kept, n_dropped = _filter_positive_advantage_examples(raw_examples)
        else:
            # Variant: keep ALL non-zero-advantage examples (signed). Higher gradient noise
            # but won't drop everything when all rollouts converge to same tier.
            flat_examples = [e for e in raw_examples if abs(e[1]) > 1e-8]
            n_kept = len(flat_examples)
            n_dropped = len(raw_examples) - n_kept
        # OOM fix: cap kept examples to top-2 by |advantage|. With Dr.GRPO's no-length-norm
        # loss and 25-turn rollouts, 5+ examples × 12K tokens × policy+ref forwards OOMs
        # an 80GB H100 and Modal crash-loops. Top-2 keeps the highest-signal examples.
        if len(flat_examples) > keep_top_n:
            flat_examples = sorted(flat_examples, key=lambda e: abs(e[1]), reverse=True)[:keep_top_n]
            n_dropped += n_kept - len(flat_examples)
            n_kept = len(flat_examples)
        print(
            f"   filter: kept={n_kept} dropped={n_dropped} "
            f"(of {len(raw_examples)} rollouts, top-{keep_top_n} by |adv|)"
        )

        # 6. Per-token loss.
        # Iteration 3: re-enable train mode so lora_dropout regularizes the
        # forward used for log-prob computation; verify it actually flipped on
        # so a silently-Identity dropout doesn't make future tuning of
        # lora_dropout a no-op.
        model.train()
        if n_dropout_total > 0 and _count_active_dropout_modules(model) == 0 and active_in_train > 0:
            raise RuntimeError(
                "model.train() failed to re-activate dropout; eval/train discipline broken"
            )
        optimizer.zero_grad(set_to_none=True)

        total_loss = torch.tensor(0.0, device=model.device, dtype=torch.float32)
        total_pg_loss = torch.tensor(0.0, device=model.device, dtype=torch.float32)
        total_kl_loss = torch.tensor(0.0, device=model.device, dtype=torch.float32)
        total_tokens = 0
        kl_per_token_max = 0.0
        kl_per_token_sum = 0.0

        for rollout, adv in flat_examples:
            input_ids_list, mask_list = _tokenize_with_assistant_mask(tokenizer, rollout.messages)
            # iteration 2: fail loud if the mask covers the wrong tokens.
            # Wrong mask = silently training on system/tool tokens.
            validate_assistant_mask(tokenizer, rollout.messages, input_ids_list, mask_list)
            if sum(mask_list) == 0:
                continue
            input_ids = torch.tensor([input_ids_list], device=model.device, dtype=torch.long)
            mask = torch.tensor([mask_list], device=model.device, dtype=torch.long)

            # Policy forward (graph-tracked).
            model.set_adapter("policy")
            out = model(input_ids=input_ids)
            logits = out.logits[:, :-1, :]            # predict next-token
            targets = input_ids[:, 1:]
            tgt_mask = mask[:, 1:].float()            # assistant tokens at predicted positions

            log_probs = torch.log_softmax(logits.float(), dim=-1)
            tok_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            # Iteration 5: reference forward under no_grad with the frozen
            # adapter swapped in. Switch back to "policy" before backward so
            # any downstream forward (e.g. accidental generate) hits the
            # trainable copy. Same `targets` so token alignment is exact.
            with torch.no_grad():
                model.set_adapter("ref")
                ref_out = model(input_ids=input_ids)
                ref_logits = ref_out.logits[:, :-1, :]
                ref_log_probs = torch.log_softmax(ref_logits.float(), dim=-1)
                ref_tok_logp = ref_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            model.set_adapter("policy")

            # k3 KL per token. Mask first so padding/non-assistant positions
            # contribute zero — important because exp(log_ratio) explodes if
            # the two distributions disagree wildly on non-assistant tokens
            # (system prompt, tool results), which we don't care about.
            kl_per_token = _kl_k3_per_token(tok_logp, ref_tok_logp) * tgt_mask

            # Dr.GRPO: NO length normalization → sum, not mean — both terms.
            if use_ratio_clip:
                # PPO-style ratio clip vs the frozen reference policy.
                # ratio = exp(logp_policy - logp_ref) per token.
                # We clip the SURROGATE so positive-advantage steps can't push
                # the ratio past 1+eps and negative-advantage past 1-eps.
                # Note: this measures drift from SFT base (ref), not from a
                # frozen step-N snapshot. Functionally similar for low-drift regimes.
                log_ratio = tok_logp - ref_tok_logp  # [B, T-1]
                ratio = log_ratio.exp()
                clipped_ratio = ratio.clamp(1.0 - ratio_clip_eps, 1.0 + ratio_clip_eps)
                # Surrogate per token: adv (scalar here) × ratio. Take min of clipped/unclipped.
                unclipped = adv * ratio
                clipped = adv * clipped_ratio
                # min of (unclipped, clipped) — for negative adv it's max-then-negate.
                surrogate = torch.minimum(unclipped, clipped) if adv >= 0 else torch.maximum(unclipped, clipped)
                pg_loss = -((surrogate * tgt_mask).sum())
            else:
                pg_loss = -(adv * (tok_logp * tgt_mask).sum())
            kl_loss = beta * kl_per_token.sum()
            sample_loss = pg_loss + kl_loss

            sample_loss.backward()
            total_loss = total_loss + sample_loss.detach()
            total_pg_loss = total_pg_loss + pg_loss.detach()
            total_kl_loss = total_kl_loss + kl_loss.detach()
            total_tokens += int(tgt_mask.sum().item())
            kl_per_token_max = max(kl_per_token_max, float(kl_per_token.detach().max()))
            kl_per_token_sum += float(kl_per_token.detach().sum())

        if total_tokens == 0:
            print("   [warn] no assistant tokens in batch; skipping optimizer step")
        else:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

        # Iteration 3: leave the model in eval() between steps so any code
        # path that calls generate() outside the rollout loop (debug prints,
        # validation hooks, etc.) cannot accidentally run with dropout on.
        model.eval()

        # Iteration 4: behavioral metrics every step. Loss/KL alone don't
        # surface mask bugs or sign-flipped advantages; these do.
        # Iteration 5: also surface KL stats so anchor-drift is observable.
        metrics = _aggregate_step_metrics(groups)
        metrics["step"] = step + 1
        metrics["loss"] = float(total_loss)
        metrics["pg_loss"] = float(total_pg_loss)
        metrics["kl_loss"] = float(total_kl_loss)
        metrics["beta"] = beta
        metrics["kl_per_token_mean"] = (kl_per_token_sum / total_tokens) if total_tokens else 0.0
        metrics["kl_per_token_max"] = kl_per_token_max
        metrics["n_examples_kept"] = n_kept
        metrics["n_examples_dropped"] = n_dropped
        metrics["train_tokens"] = total_tokens
        print(
            f"   step={step+1} loss={metrics['loss']:.4f} "
            f"pg={metrics['pg_loss']:.4f} kl={metrics['kl_loss']:.4f} "
            f"kl/tok(mean={metrics['kl_per_token_mean']:.4f},"
            f"max={metrics['kl_per_token_max']:.4f}) "
            f"tokens={total_tokens} kept={n_kept}/{len(raw_examples)}"
        )
        print(f"   metrics: {_format_step_metrics(metrics)}")
        metrics_fh.write(json.dumps(metrics) + "\n")
        metrics_fh.flush()
        if _wandb_run is not None:
            try:
                _wandb_run.log({k: v for k, v in metrics.items()
                                if isinstance(v, (int, float, bool))},
                               step=step + 1)
            except Exception as _e:
                print(f"   [wandb log err] {_e}")

        if (step + 1) % save_every == 0 or (step + 1) == max_steps:
            save_path = f"{out_dir}/step_{step+1:04d}"
            print(f"   [save] adapter -> {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            ckpt_volume.commit()

        # Iteration 9: eval after each cadence boundary AND after the final
        # step so the run always closes with an apples-to-apples last-step
        # eval row.
        if (step + 1) % eval_every == 0 or (step + 1) == max_steps:
            _run_eval(step_label=step + 1)

    metrics_fh.close()
    eval_metrics_fh.close()
    print(f"\n=== done === metrics: {metrics_path} eval: {eval_metrics_path}")
    return {
        "max_steps": max_steps,
        "out_dir": out_dir,
        "metrics_path": str(metrics_path),
        "eval_metrics_path": str(eval_metrics_path),
    }


# ----------------------------------------------------------------------
# Scoring remote (iteration 11)
# ----------------------------------------------------------------------


@app.function(
    image=image,
    gpu="H100",
    timeout=8 * 60 * 60,
    volumes={"/ckpt": ckpt_volume, "/trac": trac_volume},
)
def score_adapter(
    *,
    suite: str = "75",
    adapter_path: str = POLICY_ADAPTER_PATH,
    K_eval: int = 1,
    max_rollout_steps: int = 25,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    out_dir: str = "/ckpt/drgrpo_score",
    max_problems: int | None = None,
):
    """Score a single adapter against a scoring suite (75 or 90).

    Mirrors run_training's eval pass but drives the entire chosen suite (not
    the reserved 10% in-distribution slice) and writes one JSONL row with
    the headline `success_rate`. Defaults to temperature=0 so the
    reported number is the deterministic policy success rate — matches the
    published SFT baseline (40.0% / 33.3%).
    """
    import requests
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.drgrpo.rollout_harness import run_episode
    from src.drgrpo.reward import compute_reward

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"=== Dr.GRPO SCORE === suite={suite} adapter={adapter_path}")
    print(f"  K_eval={K_eval} max_rollout_steps={max_rollout_steps} temp={temperature}")

    problems, found = _load_scoring_suite(suite)
    if not problems:
        raise RuntimeError(
            f"no problems loaded for suite={suite!r}; mount oro-trac-data"
        )
    if max_problems is not None:
        problems = problems[:max_problems]
    print(f"  problems={len(problems)} found={found}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BACKBONE, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # Single-adapter load for scoring — no KL needed, no trainable params.
    model = PeftModel.from_pretrained(
        base, adapter_path, adapter_name="policy", is_trainable=False
    )
    model.set_adapter("policy")
    model.eval()
    _assert_no_active_dropout(model, where="score")

    session = requests.Session()

    def _rollout_fn(problem):
        return run_episode(
            model,
            tokenizer,
            problem,
            session=session,
            max_steps=max_rollout_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def _reward_fn(rollout, problem):
        # score_adapter doesn't accept continuous_reward — default to False (binary)
        return compute_reward(rollout, problem, session=session, continuous=False)

    groups = _collect_eval_groups(problems, _rollout_fn, _reward_fn, K_eval=K_eval)
    metrics = _aggregate_scoring_metrics(groups)
    metrics["suite"] = suite
    metrics["adapter_path"] = adapter_path
    metrics["K_eval"] = K_eval
    metrics["temperature"] = temperature
    metrics["n_problems"] = len(problems)

    score_path = Path(out_dir) / f"score_{suite}.jsonl"
    with open(score_path, "a") as fh:
        fh.write(json.dumps(metrics) + "\n")
    ckpt_volume.commit()

    print(f"\n=== suite={suite} success_rate={metrics['success_rate']:.3f} ===")
    print(f"   {_format_step_metrics(metrics)}")
    print(f"   wrote -> {score_path}")
    return metrics


# ----------------------------------------------------------------------
# Entrypoints
# ----------------------------------------------------------------------


@app.local_entrypoint()
def smoke():
    """3-step smoke test. Must pass before train() runs."""
    out = run_training.remote(smoke=True)
    print(f"smoke done: {out}")


@app.local_entrypoint()
def smoke_v14():
    """v14 wallclock smoke: full v13 config (B=3 K=6) but max_steps=3 with batched generate.
    Compare step time to sequential v13 (~17min/step expected) — batched should hit ~6min/step.
    If wallclock holds + no OOM, cancel v13 and re-fire as v14 full (60 steps)."""
    out = run_training.remote(
        smoke=False,
        max_steps=3,
        B=3,
        K=6,
        lr=1e-6,
        beta=0.1,
        max_rollout_steps=25,
        max_new_tokens=1024,
        temperature=0.7,
        save_every=3,
        eval_every=3,
        K_eval=1,
        use_positive_only_filter=False,
        keep_top_n=4,
        use_batched_rollouts=True,
        policy_adapter="/ckpt/sft_opd_renderers_from_rejection_v2/final",
        out_dir="/ckpt/drgrpo_v14_smoke",
    )
    print(f"v14 smoke done: {out}")


@app.local_entrypoint()
def train_v17():
    """v17: vllm-backed rollouts. Same hyperparams as v16 (K=6, B=3, ratio_clip,
    positive-only filter, keep_top=2) but rollouts via external vllm server
    instead of in-process model.generate. Vllm continuous batching + 18-way
    thread pool → ~5-10× faster per step. Vllm policy = frozen opd_renderers
    = ref adapter, so ratio_clip naturally handles off-policy IS correction.
    Requires vllm serving 'opd_renderers' at the given URL."""
    call = run_training.spawn(
        smoke=False,
        max_steps=60,
        B=3,
        K=6,
        lr=1e-6,
        beta=0.1,
        max_rollout_steps=15,
        max_new_tokens=1024,
        temperature=0.7,
        save_every=5,
        eval_every=10,
        K_eval=2,
        use_positive_only_filter=True,
        keep_top_n=6,
        use_ratio_clip=True,
        ratio_clip_eps=0.2,
        policy_adapter="/ckpt/sft_opd_renderers_from_rejection_v2/final",
        out_dir="/ckpt/drgrpo_v17_vllm_rollouts",
        vllm_rollout_url="https://oro-ai--oro-vllm-opdrenderers-h200-b-serve.modal.run",
        vllm_rollout_model="opd_renderers",
        vllm_rollout_workers=24,
    )
    print(f"v17 spawned: call_id={call.object_id}")


def _v18_kwargs(max_steps: int, out_dir: str) -> dict:
    """v18 stability fix off v17. step_metrics on v17 showed reward thrashing
    (0.05-0.78, no trend) + KL spikes to 17-21 with n_groups_with_signal mostly
    3/3 — i.e. signal present but optimization unstable. Fix: bigger B (variance),
    lower lr (smaller steps), higher beta (tighter anchor). Everything else = v17."""
    return dict(
        smoke=False, max_steps=max_steps,
        B=8,        # v17: 3 — cut gradient variance
        K=6,
        lr=5e-7,    # v17: 1e-6 — smaller steps, keep policy near frozen rollout policy
        beta=0.2,   # v17: 0.1 — tighten KL anchor (kill the 17-21 spikes)
        max_rollout_steps=15, max_new_tokens=1024, temperature=0.7,
        save_every=5, eval_every=10, K_eval=2,
        use_positive_only_filter=True, keep_top_n=6,
        use_ratio_clip=True, ratio_clip_eps=0.2,
        policy_adapter="/ckpt/sft_opd_renderers_from_rejection_v2/final",
        out_dir=out_dir,
        vllm_rollout_url="https://oro-ai--oro-vllm-opdrenderers-h200-b-serve.modal.run",
        vllm_rollout_model="opd_renderers",
        vllm_rollout_workers=24,
    )


@app.local_entrypoint()
def smoke_v18():
    """4 steps at the real B=8 to validate stability (bounded KL, non-collapsing
    reward) before committing ~4hr to the full run."""
    call = run_training.spawn(**{**_v18_kwargs(max_steps=4, out_dir="/ckpt/drgrpo_v18_smoke"),
                                 "save_every": 4, "eval_every": 4})
    print(f"v18 smoke spawned: call_id={call.object_id}")


@app.local_entrypoint()
def train_v18():
    """v18 full: stability-fixed Dr.GRPO, 30 steps."""
    call = run_training.spawn(**_v18_kwargs(max_steps=30, out_dir="/ckpt/drgrpo_v18_stable"))
    print(f"v18 spawned: call_id={call.object_id}")


def _v19_kwargs(max_steps: int, out_dir: str) -> dict:
    """v19 = v18 stability hyperparams + dense per-step tool-match reward
    (the paper's ToolRL-style signal our outcome-only RL was missing).
    Refs = rejection-sampled v3 training success trajectories."""
    return {**_v18_kwargs(max_steps=max_steps, out_dir=out_dir),
            "tool_weight": 0.5,
            "max_rollout_steps": 25,  # match eval (was 15); 15 truncated hard multi-item problems before recommend
            "ref_traj_path": "/distill/opd_sonnet_500.jsonl",  # Sonnet teacher targets (Seth): richer, better coverage, sep 0.59
            # dedicated 32k rollout server on oro-ai (-a); -b left for Seth's concurrent rule_reward runs (ORO-1239)
            "vllm_rollout_url": "https://oro-ai--oro-vllm-opdrenderers-h200-a-serve.modal.run"}


@app.local_entrypoint()
def smoke_v19():
    """4 steps at B=8 with tool_weight=0.5 — validate the dense reward gives
    in-group variance + an early climb before committing the full run."""
    call = run_training.spawn(**{**_v19_kwargs(max_steps=4, out_dir="/ckpt/drgrpo_v19_smoke"),
                                 "save_every": 4, "eval_every": 4})
    print(f"v19 smoke spawned: call_id={call.object_id}")


@app.local_entrypoint()
def train_v19():
    """v19 full: dense per-step tool-match reward, 30 steps."""
    call = run_training.spawn(**_v19_kwargs(max_steps=30, out_dir="/ckpt/drgrpo_v19_toolreward"))
    print(f"v19 spawned: call_id={call.object_id}")


@app.local_entrypoint()
def smoke_v17():
    """v17 smoke: same vllm setup, 3 steps. Confirm step time + reward signal."""
    call = run_training.spawn(
        smoke=False,
        max_steps=3,
        B=3,
        K=6,
        lr=1e-6,
        beta=0.1,
        max_rollout_steps=25,
        max_new_tokens=1024,
        temperature=0.7,
        save_every=3,
        eval_every=3,
        K_eval=1,
        use_positive_only_filter=True,
        keep_top_n=2,
        use_ratio_clip=True,
        ratio_clip_eps=0.2,
        policy_adapter="/ckpt/sft_opd_renderers_from_rejection_v2/final",
        out_dir="/ckpt/drgrpo_v17_smoke",
        vllm_rollout_url="https://oro-ai--oro-vllm-opdrenderers-h200-b-serve.modal.run",
        vllm_rollout_model="opd_renderers",
        vllm_rollout_workers=24,
    )
    print(f"v17 smoke spawned: call_id={call.object_id}")


@app.local_entrypoint()
def train_v16():
    """v16: Dr.GRPO + PPO-style ratio clip (eps=0.2) on opd_renderers base.
    v15 oscillated because no ratio clip → per-token gradient magnitude unbounded.
    v16 adds clip; gradient still positive-dominated by filter, but bounded.
    H200, spawn+detach."""
    call = run_training.spawn(
        smoke=False,
        max_steps=60,
        B=3,
        K=6,
        lr=1e-6,
        beta=0.1,
        max_rollout_steps=25,
        max_new_tokens=1024,
        temperature=0.7,
        save_every=5,
        eval_every=10,
        K_eval=2,
        use_positive_only_filter=True,
        keep_top_n=2,
        use_ratio_clip=True,
        ratio_clip_eps=0.2,
        policy_adapter="/ckpt/sft_opd_renderers_from_rejection_v2/final",
        out_dir="/ckpt/drgrpo_v16_ratioclip",
    )
    print(f"v16 spawned: call_id={call.object_id}")


@app.local_entrypoint()
def train_v15():
    """v15: filter back ON + keep_top_n=2 on H200. Uses .spawn() so local entrypoint
    exits immediately + the function continues regardless of bash/Modal client state.
    Previous v15 attempts cancelled at ~45min mark because .remote() blocked the
    entrypoint and any local-client SIGTERM propagated as cancel to Modal."""
    call = run_training.spawn(
        smoke=False,
        max_steps=60,
        B=3,
        K=6,
        lr=1e-6,
        beta=0.1,
        max_rollout_steps=25,
        max_new_tokens=1024,
        temperature=0.7,
        save_every=5,
        eval_every=10,
        K_eval=2,
        use_positive_only_filter=True,   # back on — controls Dr.GRPO loss magnitude
        keep_top_n=2,                    # back to v11; v13 top-4 + filter off blew up
        policy_adapter="/ckpt/sft_opd_renderers_from_rejection_v2/final",
        out_dir="/ckpt/drgrpo_v15_filter_on_h200",
    )
    print(f"v15 spawned: call_id={call.object_id}")


@app.local_entrypoint()
def train_v14():
    """v14: 60 steps with batched generate. Same params as v13 except use_batched_rollouts=True.
    ETA ~6-9hr if batched gives 2-3x rollout speedup."""
    out = run_training.remote(
        smoke=False,
        max_steps=60,
        B=3,
        K=6,
        lr=1e-6,
        beta=0.1,
        max_rollout_steps=25,
        max_new_tokens=1024,
        temperature=0.7,
        save_every=5,
        eval_every=10,
        K_eval=2,
        use_positive_only_filter=False,
        keep_top_n=4,
        use_batched_rollouts=True,
        policy_adapter="/ckpt/sft_opd_renderers_from_rejection_v2/final",
        out_dir="/ckpt/drgrpo_v14_opdrenderers_lr1e6_batched",
    )
    print(f"v14 done: {out}")


@app.local_entrypoint()
def smoke_batched():
    """3-step minimal-config run with batched generate (K parallel rollouts/problem).
    Bypasses smoke=True (which clamps K=2) so we exercise K=4 + temp=0.7 actual divergence.
    Out dir distinct from smoke() so checkpoints don't clobber."""
    out = run_training.remote(
        smoke=False,
        max_steps=3,
        B=1,
        K=4,
        lr=1e-6,
        max_rollout_steps=15,
        max_new_tokens=512,
        temperature=0.7,
        save_every=3,
        eval_every=3,
        K_eval=1,
        use_positive_only_filter=False,
        keep_top_n=4,
        use_batched_rollouts=True,
        policy_adapter="/ckpt/sft_opd_renderers_from_rejection_v2/final",
        out_dir="/ckpt/drgrpo_smoke_batched",
    )
    print(f"smoke_batched done: {out}")


@app.local_entrypoint()
def train():
    """20-step real training run."""
    out = run_training.remote(smoke=False, max_steps=20)
    print(f"train done: {out}")


@app.local_entrypoint()
def train_v13():
    """v13: 60 steps, lr 1e-6, K=6 B=3, no positive-only filter,
    stacked on opd_renderers, top-4 OOM cap, dispatch_tool str-params fix in harness.
    ETA ~17hr on H100."""
    out = run_training.remote(
        smoke=False,
        max_steps=60,
        B=3,
        K=6,
        lr=1e-6,
        beta=0.1,
        max_rollout_steps=25,
        max_new_tokens=1024,
        temperature=0.7,
        save_every=5,
        eval_every=10,
        K_eval=2,
        use_positive_only_filter=False,
        keep_top_n=4,
        policy_adapter="/ckpt/sft_opd_renderers_from_rejection_v2/final",
        out_dir="/ckpt/drgrpo_v13_opdrenderers_lr1e6",
    )
    print(f"v13 done: {out}")


@app.local_entrypoint()
def score(
    suite: str = "75",
    adapter_path: str = POLICY_ADAPTER_PATH,
    k_eval: int = 1,  # lowercase — Modal CLI lowercases kwargs, would otherwise mismatch
    temperature: float = 0.0,
    max_problems: int = 0,
):
    """Score an adapter against a scoring suite (75 or 90).

    Defaults score the SFT baseline (POLICY_ADAPTER_PATH) on the 75-set
    at temperature=0 so the number is reproducible.
    """
    mp = None if max_problems <= 0 else int(max_problems)
    out = score_adapter.remote(
        suite=suite,
        adapter_path=adapter_path,
        K_eval=k_eval,  # translate back to uppercase for the inner function
        temperature=temperature,
        max_problems=mp,
    )
    print(f"score done: suite={suite} success_rate={out['success_rate']:.3f}")
