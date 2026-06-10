# Empirical evidence: top miners produce axis-B (sub-task) data

**Date:** 2026-05-13
**Context:** Independent validation of the framing in ORO-1143 (training-corpus skew). After our practice run hit a 2/90 ceiling, we asked whether the source corpus could even produce an agentic distillation target.

## What we did

Pulled the **5 currently top-performing agents** on the live subnet (last 7 days, ranked by best score) along with their agent code and 1-3 sample trajectories per agent. Code + trajectories live in `data/recent_tops/`.

| Agent | Best score (last 7d) | Last completed |
|---|---|---|
| `06a05103` | 0.70 | 2026-05-14 00:06 |
| `06a04ffb` | 0.667 | 2026-05-13 22:54 |
| `069fdfe2` | 0.667 | 2026-05-13 21:39 |
| `06a04e01` | 0.633 | 2026-05-13 22:31 |
| `06a04cac` | 0.633 | 2026-05-13 22:27 |

These five agents represent the working leaderboard ceiling on ShoppingBench right now.

## Heuristic scan

For each agent's `agent.py`, we counted: lines, presence of XML tag parsing, count of "classify"/"judge"/"score" terms.

| Agent | Lines | XML tag parsing | "classify"/"judge" terms |
|---|---|---|---|
| Default reference agent (axis-A baseline) | 843 | yes | few |
| `069fdfe2` | 2,447 | **0** | **55** |
| `06a04cac` | 5,347 | **0** | **71** |
| `06a04e01` | 2,195 | **0** | **35** |
| `06a04ffb` | 2,641 | **0** | **47** |
| `06a05103` | 3,341 | **0** | **46** |

All five top agents are 2.5-6x the size of the default ReAct template, contain zero XML tool-call parsing, and use heavily classification-oriented language. This matches the axis-B pattern Seth describes in ORO-1143.

## The smoking-gun trace

Top agent `06a05103` (0.70 success), step 0 of trajectory `trajectory_06a05103_069dd141.json`:

**Query:** "Looking for a blue button flex replacement for Oppo A5S"

**Proxy call 0 — LLM #1 (DeepSeek-V3.1) returns structured JSON:**
```json
{
  "reasoning": "Extracted product type 'button flex replacement', brand 'Oppo', model 'A5S', and color 'blue' as explicit attributes.",
  "products": [{
    "keywords": "button flex replacement Oppo A5S blue",
    "constraints": {"color": "blue", "brand": "oppo", "model": "a5s"},
    "hypothetical_title": "Original Blu..."
  }]
}
```
This is **structured field extraction** — a sub-task. The LLM was prompted to parse the query into JSON. The output is JSON, not a tool call. The LLM never says "search for X."

**Proxy call 1 — LLM #2 (DeepSeek-V3.1) returns prose narration:**
> "I'm analyzing the user's request for a blue button flex replacement specifically for Oppo A5S. I will search using the exact keywords 'button flex oppo a5s blue' to find compatible parts. Since there are no price or service constraints mentioned, I'll focus on matching the product type and color."

This is **narration generation** — another sub-task. Pretty prose meant to make the trajectory look agentic when read. It does not actually decide anything; the keywords were already locked in by Call 0.

**Proxy call 2 — Direct GET from Python:**
```
GET /search/find_product  q=button+flex+oppo+a5s+blue
```
Python takes `keywords` from Call 0 and dispatches the search. The LLM was never asked "should I search now?"

## Implication

**Agent behavior lives in the Python orchestrator, not the LLM output.** The LLM is reduced to:
1. Structured field extraction (Call 0)
2. Narration text generation (Call 1)
3. Probably re-ranking and shop aggregation later in the trajectory (need to inspect deeper)

A model fine-tuned to imitate these LLM responses learns to parse queries and generate narration. It does NOT learn:
- When to search vs. view vs. recommend
- How to choose query keywords (Python decides)
- When to switch keywords on empty results
- When to terminate

This is exactly why our practice run capped at 2/90: we trained on supervision that doesn't contain agentic decisions.

## What this means for the distillation paper

Three live framings (from ORO-1143 and follow-ups):

**Path A — accept and ship a pipeline of specialists.** Train 3 small models: intent classifier, match scorer, narrator. Plug into a Python orchestrator (the open-source default agent or our own). Honest about what the data is. Doable in weeks.

**Path B — synthesize agentic trajectories from the existing axis-B corpus.** ORO-1145 (completed 2026-05-13) showed that a Claude-based synthesizer hits **94.4% faithfulness** on this task — the synthesizer doesn't hallucinate. ORO-1148 plans the next test: 5K trajectories × 3 data shapes (raw `<think>` blocks, full synth, hybrid) → LoRA on Qwen2.5-7B → eval on 30 held-out problems vs 4 baselines including GPT-4-class.

**Path C — fix the supply via incentive.** Tune the leaderboard scoring to reward LLM-driven dispatch (ORO-1143 Option 4: `agentic = llm_tool_calls / total_tool_calls`). Slow (months) but produces a clean corpus going forward.

**Bonus path — match the ShoppingBench paper directly.** Generate fresh GPT-4.1 trajectories on the 2,410-instruction training set, rejection-sample to absolute success, SFT Qwen3-4B, then ToolRL-style RL. This is what the paper did to hit 48.7% ASR with the same base model we used.

## Important new finding inside ORO-1148

> "Update 2026-05-13: while inspecting trajectories side-by-side we discovered miner trajectories already contain coherent first-person `<think>` reasoning blocks (produced by an internal narrator LLM, e.g. Race 29 winner's `_MODEL_NARR_*` constants). This is 'free' axis-A-shaped data that doesn't need synthesis. The distillation eval should test whether we even need expensive synthesis at all, or whether raw `<think>` blocks train just as well."

We see this directly in our `06a05103` evidence above (Call 1 is the narrator). So the question isn't "do we have axis-A data," it's "**is the narrator output good enough as supervision, or do we need to clean / regenerate it**." ORO-1148 designs the bake-off to answer this.

## Related Linear tickets

- **ORO-1143** (parent) — Training-corpus skew framing (filed by Seth, 2026-05-13).
- **ORO-1145** (Done) — Path B confabulation spot-check, 94.4% faithfulness.
- **ORO-1148** (Backlog) — Path B sanity check #1: hold-one-out distillation eval. The next experiment.
- **ORO-1147** (Done) — 10% of trajectory artifacts are empty on FAIL runs (data quality issue, fixed).
- **ORO-627** — Native `tool_calls` preservation + 2KB response truncation in proxy logs. We worked around this in our practice run via re-fetching from the search server, but it explains some of the data-quality issues.

## Why this matters for the paper

The five-bug catalog from the practice run is the engineering contribution. The axis-B finding documented here is the **data-shape contribution** — it says clearly: distilling agent behavior from this corpus, as-is, has a structural ceiling that no amount of training-loop tuning will fix. That's a generalizable result for any team trying to distill agents from miner-style production data.

The paper's strongest framing is probably: "**Lessons from a hand-rolled multi-turn GRPO pipeline + an empirical ceiling on what agentic distillation can extract from axis-B production trajectories. Three paths forward.**" That's both the methodology contribution and the framing contribution, and the practice run's numbers (2/90) become evidence for the ceiling rather than a failure to apologize for.
