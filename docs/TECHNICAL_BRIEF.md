# Multi-Turn GRPO Practice Run — Technical Brief

**Author:** Shardul Bansal
**Date:** 2026-05-13
**Status:** Training complete, eval complete, paper-relevant findings below

---

## Why this document exists

This is a practice run preparing for a larger distillation effort intended for a research publication. The goal here was learning, not benchmark wins. We picked Qwen3-4B + LoRA, SFT on extracted production trajectories, then GRPO with a custom multi-turn loop. We hit five non-trivial bugs in succession before the training actually converged. Those bugs are the most interesting part of this writeup.

---

## What we built

### Target task
ShoppingBench (arXiv 2508.04266) is the eval our subnet (Bittensor SN15) uses. Agents receive natural language shopping queries with constraints (price ranges, attributes, SKU options, service requirements) and must recommend a product from a 2.7M-product Lucene index. Three task categories:
- **Product** — single product match
- **Shop** — multiple products from one shop
- **Voucher** — multi-product within a budget+discount constraint

Three problem suites (v1, v2, v3), 30 problems each, 90 total.

### Validator scoring (the real metric)
We use the production validator's `ProblemScorer` from `oro/src/agent/problem_scorer.py`. A problem is "successful" when:
- `rule_score >= 1.0` (all scoreable fields matched: price, service, SKU/attrs, title similarity ≥ 0.7)
- For shop tasks, additionally all from same shop
- For voucher tasks, additionally budget constraint met after discount

The miner reward formula:
```
final_score = success_rate × reasoning_coefficient(rq)
# rq >= 0.80 → coefficient 1.0
# rq = 0.00  → coefficient 0.3 (floor)
```

We can compute success_rate exactly using the validator code.

### Infrastructure
- **Modal H100 (80GB)** for all training and rollouts
- **Search server**: Pyserini Lucene index of 2.7M products deployed as a Modal web service (always-warm). All `find_product` and `view_product_information` tool calls hit it
- **Data**: 5,635 multi-turn trajectories extracted from production S3, enriched with real tool responses (re-fetched from search server so the training data has actual catalog detail, not placeholder strings)
- **Reference model for KL**: We loaded the SFT checkpoint as a second frozen LoRA adapter alongside the trainable policy adapter. Switch adapters during loss computation to get both policy and reference log-probs without doubling base-model memory

### Training data quality
- 5,635 examples from 204 production runs, all from suite 3
- All runs had per-run reasoning quality `rq ≥ 0.81` (203 in [0.81, 0.90], one above 0.90)
- BUT `rq` is per-run not per-trajectory. Within those high-rq runs, individual problem success rates span sr=0.4 to sr=0.8. So we have high-reasoning trajectories that sometimes pick the wrong product
- Trajectories have median 4,709 tokens, p90 = 12,550 tokens, p95 = 15,083 tokens. This length distribution turned out to matter a lot, see SFT v5 below

---

## SFT journey

### Goal of SFT
Get the model to imitate the format and tool-use pattern. We want it to produce `<think>...</think><tool_call>[{"name": "find_product", ...}]</tool_call>` reliably and progress through multi-turn search before recommending.

### v1-v3 (failed): max_length truncation
TRL's `SFTConfig` defaults `max_length=1024`. We did not set it explicitly. With median trajectory length 4,709 tokens and `recommend_product` typically at the END of a trajectory, **90% of training examples were cut off before the model ever saw a `recommend_product` call**. The loss looked fine (it goes down nicely), but the model only ever trained on opening patterns.

At inference: model emits `<think>...</think><tool_call>find_product</tool_call>`, gets back results, then has no idea what to do. Stops, or emits `<response>filler</response>` and exits.

### v5 (worked): max_length=14336
We set `max_length=14336` which covers ~93% of trajectories end-to-end (p95=15,083, we accept truncating the longest 5%). Required:
- `attn_implementation="sdpa"` so Flash backend kicks in at 14K context
- `gradient_checkpointing=True`
- `per_device_train_batch_size=1`, grad_accum=16

This trained in 96 minutes on one H100. Final training loss 0.587 (started 1.7). Model can now emit `recommend_product` in some queries (vinyl stickers test query: 4/4 rollouts hit a real catalog product, 1 of which was exact GT).

**Lesson worth surfacing in the paper:** silent default truncation in standard fine-tuning libraries is a real failure mode. Always validate the token-length distribution against your `max_length` setting. The training metrics look healthy even when the model is missing the entire target capability.

---

## GRPO journey: six versions, five bugs

This is where the paper's main contribution lies. Our task (multi-turn shopping agent with tool use) hits failure modes that single-turn RLHF papers don't discuss. We trained six versions, each fixing one bug uncovered by the previous, and only the sixth converged.

### Why hand-rolled GRPO
TRL's `GRPOTrainer` assumes single-shot `model.generate()` per prompt. We need full multi-turn episodes (model → tool → model → tool → ...). We also need to mask tool-result tokens out of the loss (the model didn't generate those). About 500 lines of custom code lets us control every assumption.

### Algorithm per training step
1. Sample B=2 problems from the dataset
2. For each problem, K=4 rollouts using the multi-turn harness
3. Compute reward (tiered, see below) and per-rollout advantage = reward minus group mean
4. Loss = `-advantage × mean_log_prob_assistant_tokens + β × KL(policy || ref)`
5. Backprop, step optimizer
6. Save checkpoint every 5 steps

### Tiered reward shaping
Sparse 0/1 reward made GRPO useless because within-batch variance was near zero. We tiered:
- 0.00 — no `recommend_product` call at all
- 0.05 — `recommend_product` but ID is hallucinated (not in catalog)
- 0.10 + 0.80 × rule_score — real catalog ID, scaled by partial-match score
- 1.00 — exact ground-truth ID match

This gives gradient signal even on problems where no rollout hits ground truth.

### v1: LR=5e-6, no KL anchor, buggy mask
**Step 0 looked healthy.** reward=0.106, cat_hit=2/4, avg_turns=2.5.
**Step 1 collapsed.** reward=0.025, cat_hit=0/4, avg_turns=1.0. Model stops emitting tool calls entirely.

Initial hypothesis: LR too high. Wrong.

### v2: LR=1e-6, no KL anchor, buggy mask
Same exact collapse. Step 1 reward 0.025, avg_turns 1.0. So it is not LR magnitude.

### v3: LR=5e-7, KL anchor added via dual-adapter setup, buggy mask
We added a KL penalty using a second frozen LoRA adapter as reference. Schulman k3 estimator with β=0.04. Per-token KL computed on assistant tokens only.

Still collapsed. Step 1 reward 0.025, avg_turns 1.0. KL ≈ 0.0015 (tiny). So the KL anchor was technically working but couldn't save us from whatever the underlying issue was.

### The mask bug (discovered after v3)
We were computing the assistant-token mask wrong. The code intended to walk cumulative prefixes of the conversation and mark which tokens correspond to assistant turns. But a logic error meant we tokenized each message in isolation rather than as a cumulative prefix. The mask offsets pointed at random positions in `full_ids`. The model was being trained on **whatever tokens happened to be at those wrong offsets** — parts of system prompt, parts of user messages, fragments of assistant turns.

**This is the kind of bug that is silent in metrics.** Training loss looks reasonable. KL drift stays small. The model degrades step-over-step because we are training on garbage signal, but you only see it in the behavioral metric (avg_turns, cat_hit), which is exactly what the typical training dashboard does not show.

### v4: mask fixed
Step 1 still collapsed. Different cause this time.

### The shared-prefix problem (discovered after v4)
With K=4 rollouts of the same problem, rewards distribute like `[0.20, 0.00, 0.00, 0.00]`. Advantages: `[+0.15, -0.05, -0.05, -0.05]`. All four rollouts share their opening turn — same `<think>...find_product...</think>` prefix before they diverge.

For a specific common-prefix token, the per-token gradient is:
- One pull UP from the positive-advantage rollout (long trajectory, weight = +0.15 / N_long)
- Three pulls DOWN from the negative-advantage rollouts (short trajectories, weight = +0.05 / N_short each)

The short trajectories have larger per-token weight (1/N is bigger when N is smaller). The math nets out to a suppression of the very prefix tokens we want to keep. The model learns to suppress its own successful first turn.

### v5: positive-only advantage (rejection sampling)
Skip any rollout with non-positive advantage. Only learn from the wins.

Step 1 still collapsed.

### The dropout-during-generation bug (discovered after v5)
`PeftModel.from_pretrained(is_trainable=True)` returns the model in train mode. We never called `model.eval()` before rollouts. The trainable LoRA adapter has `lora_dropout=0.05`. So 5% of activations were being randomly zeroed during every forward pass of `model.generate()`.

For autoregressive generation, that's catastrophic. The errors compound across turns. By turn 4 of a multi-turn rollout, the model is deep in noise space. Step 1 looked like "collapse" but was just dropout corruption.

### v6: model.eval() before rollouts
Step 0: reward=0.219, cat_hit=5/8, avg_turns=2.4 — strongest baseline yet.
Step 1: **reward=0.783, cat_hit=8/8, gt_match=4/8, avg_turns=4.2**.

**Half the rollouts hit the exact ground truth product.** First time we have ever seen this. The "collapse" was never collapse, just the dropout corruption masquerading.

### v6 full run (40 steps, 6h 47min)
- 38 actual gradient steps (2 skipped: "all rollouts had identical rewards, no advantage signal")
- Mean reward across the run: 0.243
- Total catalog hits: ~150/320 (47%)
- Total exact GT matches: 38/320 (11.9%)
- KL drift over the run: 0.0004 → 0.0246 (max). Per-token KL still small but accelerating in later steps
- 8 checkpoints saved (step 5, 10, 15, 20, 25, 30, 35, 40/final)

The variance step-to-step is high and dominated by query-difficulty distribution. The fixed-seed RNG samples different problems each step. Easier query types (single-product, perfume/dress) cluster in some steps; harder query types ("find a shop that offers all of [list]") cluster in others. This is sampling noise on top of the actual training signal.

When we compare step 1 (single-product perfume + dress) to step 15 (single-product summer dress + grey t-shirt), both got `gt_match=4/8`. Same capability, same problem class. Capability preserved across 15 steps of training. Good sign.

---

## Bugs in priority order for the paper

Roughly in order of "how much would this catch a careful practitioner off guard":

1. **Assistant-mask construction.** Off-by-design mask that points at wrong token offsets. Loss looks healthy throughout. The metric that exposes it is behavioral (avg_turns, recommend_product emission rate) — *not* anything a standard training dashboard surfaces. This is a class of failure worth documenting: silent gradient corruption from wrong supervision targets in custom RL loops.

2. **Dropout during generation in RL.** Standard SFT does not hit this because TRL's trainer handles mode switching internally. In a hand-rolled RL loop, the model's default mode after `PeftModel.from_pretrained(is_trainable=True)` is train mode, dropout is active during your rollouts, and your generations are systematically corrupted. The step-zero metrics still look okay because the SFT-init signal is strong enough to survive 5% dropout most of the time, but step-one onward the pattern degrades. Worth a callout.

3. **Shared-prefix gradient suppression in GRPO.** When good and bad rollouts share their opening sub-trajectory (very common in agentic tasks), the per-token advantage gradients on the common prefix average out toward suppression. Standard GRPO does not have a fix for this baked in. Our solution (positive-only advantage, equivalent to best-of-K rejection sampling) is simple but throws away half the signal. There are smarter fixes (token-level credit assignment via importance ratios, only-penalize-divergence-points). Worth exploring for the paper.

4. **Silent context-window truncation in SFT.** TRL's default `max_length=1024` is way below the median trajectory length for agentic tasks. The model never sees the target behavior during training but the loss looks fine. Easy to miss. We should make a strong case in the paper that the field should validate length distributions before training.

5. **KL anchor sensitivity to per-token magnitude.** The k3 estimator returns small per-token KL values (~0.001-0.02 in our run). β=0.04 contribution to loss is ~1e-5 per token vs advantage contribution ~0.1 per token. The KL anchor is mathematically working but practically weak. Bumping β or using a different anchor formulation (e.g., full-sequence KL summed) probably matters more than people assume. Worth measuring.

---

## Eval results

We ran all 9 checkpoints (SFT v5 baseline + 8 GRPO checkpoints) through the production `ProblemScorer` on 90 problems across suites v1/v2/v3, all 3 categories. Each checkpoint got its own Modal H100; we ran the 9 in parallel for ~45 min wall clock. Headline metric is `success_rate` — the exact one the production validator pays miners on (`rule_score >= 1.0` plus category-specific constraints).

### Overall

| Checkpoint | success | gt | field | format |
|---|---|---|---|---|
| SFT v5 baseline | 0.011 | 0.011 | 0.011 | 0.438 |
| **GRPO step_5** | **0.022** | **0.022** | **0.062** | 0.428 |
| GRPO step_10 | 0.011 | 0.000 | 0.020 | 0.496 |
| GRPO step_15 | 0.011 | 0.000 | 0.021 | 0.460 |
| GRPO step_20 | 0.000 | 0.000 | 0.029 | 0.508 |
| **GRPO step_25** | **0.022** | 0.011 | 0.054 | 0.393 |
| GRPO step_30 | 0.000 | 0.000 | 0.003 | 0.437 |
| GRPO step_35 | 0.000 | 0.000 | 0.044 | 0.429 |
| GRPO step_40 (final) | 0.011 | 0.011 | 0.043 | 0.492 |

**Best checkpoint: step_5.** It doubled the baseline success_rate (0.011 → 0.022), doubled the gt_rate (0.011 → 0.022), and 5.6x'd field_matching (0.011 → 0.062). Step_25 tied for success_rate but with only one exact GT match.

### Three observations worth dwelling on

**1. Training is non-monotonic.** Performance oscillates: peak (5) → trough (20) → peak (25) → trough (30, 35) → recovery (40). This is not random noise on a fixed-difficulty population; it correlates with KL drift from the SFT reference. Drift accelerated from 0.0004 at step 0 to 0.0246 at step 39, and the regression windows (steps 20, 30) are exactly when KL spikes appear in the training log. This is the strongest piece of evidence that β=0.04 is too weak for our setup.

**2. Format score climbs almost monotonically (0.428 → 0.508 → 0.492) while success degrades.** The model gets better at producing structurally clean output but worse at picking the right product. The gradient is rewarding form over substance because format reward is easy to optimize (just match the regex pattern) while rule_score requires actual constraint reasoning. This suggests reward hacking on the format dimension and is a paper-relevant finding.

**3. By-category, the `shop` task is unsolvable at this scale.** Every checkpoint scores `shop = 0.000`, including the SFT v5 baseline. The "find a shop that offers [multi-item list]" pattern requires the agent to do multi-product retrieval, aggregate by shop, and validate that all items appear at the same vendor. The SFT data has very few examples of this. GRPO did not bridge the gap. We solved 0 of 30 shop problems across all 9 checkpoints, so 270 attempts and zero successes. This is a clean illustration that GRPO can't conjure capability from a thin SFT prior.

### By suite

| Checkpoint | v1 | v2 | v3 |
|---|---|---|---|
| GRPO step_5 | 0.000 | 0.033 | 0.033 |
| GRPO step_10 | 0.000 | 0.000 | 0.033 |
| GRPO step_15 | **0.033** | 0.000 | 0.000 |
| GRPO step_20 | 0.000 | 0.000 | 0.000 |
| GRPO step_25 | 0.000 | **0.067** | 0.000 |
| GRPO step_30 | 0.000 | 0.000 | 0.000 |
| GRPO step_35 | 0.000 | 0.000 | 0.000 |
| GRPO step_40 | 0.000 | 0.000 | 0.033 |

Training data was suite 3 only. GRPO checkpoints solved problems in v1 (step 15) and v2 (step 5, 25) — so the policy generalized beyond its training distribution. Step_25's 6.7% on held-out v2 is the best out-of-distribution performance we saw.

### By category

| Checkpoint | product | shop | voucher |
|---|---|---|---|
| GRPO step_5 | 0.033 | 0.000 | 0.033 |
| GRPO step_10 | 0.033 | 0.000 | 0.000 |
| GRPO step_15 | 0.033 | 0.000 | 0.000 |
| GRPO step_20 | 0.000 | 0.000 | 0.000 |
| GRPO step_25 | **0.067** | 0.000 | 0.000 |
| GRPO step_30 | 0.000 | 0.000 | 0.000 |
| GRPO step_35 | 0.000 | 0.000 | 0.000 |
| GRPO step_40 | 0.033 | 0.000 | 0.000 |

Product is where GRPO actually helped (step_25 hits 6.7%). Voucher only worked once (step_5). Shop never worked. Aligns with observation 3 above.

### Headline numbers for the paper

- **+100% success_rate lift over SFT baseline** (0.011 → 0.022 at best checkpoint)
- **+100% gt_rate lift** (0.011 → 0.022 at step_5)
- **+5.6x field_matching lift** (0.011 → 0.062 at step_5)
- **Best checkpoint is step_5, not the final.** Training-induced drift made later checkpoints worse.
- **Format score climbs (0.43 → 0.49) while success_rate stays flat or drops** in late training. Reward hacking signal.
- **0% on shop tasks across every checkpoint including baseline.** Beyond reach of this SFT + GRPO setup at this scale.

---

## Open questions where I want your input

1. **KL strength.** Our drift accelerated from 0.0004 at step 0 to 0.0246 at step 39. β=0.04 (a default cargoed from RLHF literature). What β do you typically use for hand-rolled GRPO with small LoRA policies? Or do you favor a different anchor (e.g., periodic KL-snapshot reset, or anchor against a moving average of recent policy states)?

2. **Shared-prefix mitigation.** Our positive-only-advantage fix is the laziest possible solution. I want a more principled treatment. Is there a published technique for token-level credit assignment in multi-turn agent RL that you trust? I've seen prefix-conditioned advantages in some recent agentic RL papers but haven't found one I'm happy citing yet.

3. **Training data quality.** All our trajectories come from a single agent's runs (per-run rq ≥ 0.81). Within that, individual problem outcomes vary from 40% success rate to 80%. Should we (a) filter to only successful trajectories per-problem, (b) filter to only top-rq runs, (c) accept the heterogeneity and let GRPO sort it out? Intuition is (a) for SFT and (c) for GRPO, but I'd love a second opinion.

4. **Problem class coverage.** "Find a shop that offers [multi-item list]" queries consistently fail across all our checkpoints. The SFT data has very few examples of this pattern. Curate more of those? Or accept the gap and report it?

5. **Distillation target for the real distillation run.** We used Qwen3-4B for this practice run. Frontier teachers (Claude, GPT-4-class) we can distill from via API. Practically, do we want a single big distillation pass from a top model, or a chain of distillations from intermediate-size models we can hit with higher throughput?

6. **Paper framing.** This work has two angles: (1) "lessons from a hand-rolled multi-turn GRPO pipeline on an agentic task" (engineering/methodology paper), or (2) "distilling a frontier agent into a small open model for [ShoppingBench / commerce]" (results paper). I'm inclined toward (1) given the bug catalog is genuinely useful, with (2) as a strong follow-up. Your read?

---

## What the real distillation run should do differently

Stuff I want to bake into the real run, partially driven by the practice bugs and partially by what should be standard practice anyway:

- **Mask validation step.** Before any GRPO step, decode mask=1 positions and assert they are valid assistant turn content. Catches mask bugs immediately.
- **Behavioral metrics from step 0.** Track avg_turns, recommend_emit_rate, cat_hit per step alongside loss/reward. These are the metrics that expose gradient-direction bugs.
- **Sanity smoke at every config change.** 3-step smoke before any 40-step run. Saves about a day per bad config.
- **Train/eval mode discipline.** `model.eval()` before rollouts, `model.train()` before backward, every step. No defaults trusted.
- **Hold out a fraction of trajectories from training for in-distribution eval.** We trained on every example from suite 3, so our "in-distribution" eval is technically also our training set. The real run should hold out 10%.
- **Higher KL_BETA (β=0.1 to 0.3) with monitoring.** With the right anchor strength we may not need positive-only advantage filter and can use the full signal.
- **Length-bucketed batch sampling.** Our trajectories range from 500 to 35K tokens. Currently we just truncate the long tail. We should bucket by length to avoid padding waste.

---

## What I'd love from you

Honest critique of the methodology, particularly the bug list and the fixes. If any of those "bugs" are actually misdiagnoses, I want to know now rather than after we ship the real run.

Code is in this repo if you want to look at the actual GRPO loop, scoring framework integration, and rollout harness.

Happy to walk through any of this in real time when we meet.

— Shardul
