# ShoppingBench Trajectory Primitive

**Paper:** [arXiv:2606.10064](https://arxiv.org/abs/2606.10064)

Code companion for the paper **Bittensor Agent Arenas as a Trajectory Primitive: Distilling a Shopping Agent from ShoppingBench Subnet Traces** (Bansal, Schilbe, Barnes).

This repository contains the working code behind the paper: the structural-quality **filter** that turns the raw ORO Subnet 15 (SN15) trajectory firehose into a trainable corpus, plus the post-training (SFT, reward-weighted rejection sampling, KTO, Dr. GRPO) and evaluation pipelines used to distill Qwen3-4B from 18.0% to 42.7% ASR on a leak-cluster-guarded, production-strict held-out partition.

It is a curated extraction from an internal research repo: only the canonical, working code referenced in the paper is included. Trial scripts, debug/smoke runs, and superseded iterations were intentionally left out.

## Released artifacts

| Artifact | Where |
|----------|-------|
| Collection (all artifacts) | https://huggingface.co/collections/oro-ai/shoppingbench-sn15-trajectory-primitive-6a270a77dc67269551a378e6 |
| Raw 18K traces | [oro-ai/sn15-shoppingbench-traces-18k](https://huggingface.co/datasets/oro-ai/sn15-shoppingbench-traces-18k) |
| Filtered 15K + eval holdout | [oro-ai/sn15-shoppingbench-sft-15k](https://huggingface.co/datasets/oro-ai/sn15-shoppingbench-sft-15k) |
| SFT model | [oro-ai/qwen3-4b-shoppingbench-sft](https://huggingface.co/oro-ai/qwen3-4b-shoppingbench-sft) |
| Rejection-sampled model | [oro-ai/qwen3-4b-shoppingbench-rejection](https://huggingface.co/oro-ai/qwen3-4b-shoppingbench-rejection) |
| SFT + KTO model | [oro-ai/qwen3-4b-shoppingbench-kto](https://huggingface.co/oro-ai/qwen3-4b-shoppingbench-kto) |

## The agentic trajectories filter

> "We release the filter as code along with the trainable corpus it produces."

The filter selects **agentic** trajectories (the language model emits the tool calls) and rejects **sub-task** trajectories (the model only classifies or narrates over a deterministic search loop), then guarantees an anti-memorised held-out split.

- `src/oro_filtering/` — the filter library:
  - `leak_cluster.py` — leak-cluster guard (groups paraphrase / attribute-reorder / retailer-substitute problem variants; partitions into train / eval / never-touch pools and verifies empty train-eval intersection)
  - `trace_quality.py` — structural-quality signals (think-and-tool-call depth, query reformulation count, verification-step presence, step-shape regularity)
  - `reward_components.py` — reasoning-quality coefficient gate
  - `bucket_tag.py` — intent bucketing (product / shop / voucher)
  - `stratified_sampler.py` — stratified held-out construction
  - `student_gap.py` — student-teacher gap signals
- `sft-2026-05-21/scripts/filtration/` — the four-stage pipeline:
  - `stage1_format_valid.py` — five format-validity hard rejects (tool-call/response pairing, single `recommend_product`, parseable recommendation args, `max_length=14336` budget, no think-then-stop termination)
  - `stage2_dedup_pick.py` — dedup and rank, 1-2 trajectories per problem
  - `stage3_axis_b_split.py` — Axis-A (agentic) vs Axis-B (sub-task) split
- `scripts/30_build_oro_sft_corpus.py` — end-to-end corpus builder that orchestrates the above
- `scripts/25_convert_to_nemo_sft.py`, `scripts/verify_nemo_render.py` — NeMo SFT (OpenAI tool-call) conversion and render verification
- Tests: `tests/oro_filtering/`, `sft-2026-05-21/tests/filtration/`

## Post-training pipeline (paper stage to code)

| Stage | Code | Key recipe |
|-------|------|-----------|
| SFT | `sft-2026-05-21/scripts/27_train_sft_oro_corpus_B.py`, `33_train_sft_oro_corpus_B_v2.py` | 1 epoch LoRA, lr sweep {1e-5, 3e-5, 1e-4} by held-out loss, `max_length=14336`, SDPA, gradient checkpointing, PrimeIntellect `renderers==0.1.7` for the assistant loss mask (bare HF `apply_chat_template` mis-masks Qwen3) |
| Reward-weighted rejection sampling | `sft-2026-05-21/scripts/43_v3_student_rollout_training_500.py`, `48_v3_student_rollout_training_500_extended_tools.py` | k=8 rollouts/problem, temp 0.9, top_p 0.95, max_turns 8, keep score=1.0 rollouts, 50/50 concat with SFT corpus |
| KTO preference refinement | `sft-2026-05-21/scripts/50_build_kto_dataset.py`, `51_train_kto_opd_renderers.py`, `52_eval_kto_7tool_harness.py` | TRL, beta 0.02, lr 5e-6, 1 epoch, attention-only LoRA r=16, ctx 16384, early-stop if `recommend_product` rate < 0.9 |
| Dr. GRPO (turn-level RL) | `src/drgrpo/` (`train_drgrpo.py`, `rollout_harness.py`, `reward.py`, `_shoppingbench_prompts.py`) | v19: B=8, lr 5e-7, beta 0.2, blended outcome + per-step tool-match reward (`tool_weight=0.5`) graded against the Sonnet 4.6 teacher, `max_rollout_steps=25`, vLLM 0.9.2, `max_model_len=32768` |

## Evaluation

- `sft-2026-05-21/scripts/55_compose_adapters_vllm.py`, `56_serve_eval_vllm.py`, `57_eval_vllm_client.py` — the vLLM eval pipeline (compose adapters with a weight-parity gate, serve, batched client). See `docs/vllm-eval-pipeline.md`.
- `sft-2026-05-21/scripts/41_openrouter_teacher_eval.py`, `42_openrouter_paper_harness_eval.py` — teacher (Sonnet 4.6) and published-paper-harness baselines.

Protocol: 75-problem leak-cluster-guarded held-out set, stratified by intent bucket and frozen before training, scored production-strict through the ORO 7-tool ShoppingBench harness. ASR = fraction where the final recommendation satisfies all four rule axes (price, service, SKU, attribute).

Headline numbers (75-set, production-strict, temp 0): base 18.0%, this work (SFT family) 42.7%, synthetic-data SFT-only baseline 43.6%, pass@1 34.8% / pass@8 53.3%, Sonnet 4.6 teacher 64.0%, published SFT+GRPO bar 48.7%.

## Setup

```bash
uv sync          # or: pip install -e .
uv run pytest tests/ sft-2026-05-21/tests/   # run the filter test suites
```

Training and eval scripts run on [Modal](https://modal.com) (see `modal_app.py`). They expect Modal named secrets (`oro-openrouter-api-key`, `oro-openai-api-key`, `oro-wandb-api-key`) and a local ShoppingBench checkout (set `SHOPPINGBENCH_LOCAL`). The corpus splits and trained checkpoints are on Hugging Face (table above), not in git.

## Notes on layout

File paths preserve the original working layout (including the dated `sft-2026-05-21/` directory) so imports and entrypoints run unchanged. A few training scripts carry `# adapted from ...` provenance comments that reference superseded scripts not included here; these are comments only, not runtime dependencies.

## Related repositories

- [ORO-AI/ShoppingBench](https://github.com/ORO-AI/ShoppingBench) — ORO's 7-tool fork of the benchmark SN15 evaluates against
- [jbarnes850/Trac](https://github.com/jbarnes850/Trac) — the Trac annotation framework (converter front-end used in production)

## License

Apache-2.0. See `LICENSE`.

## Citation

```bibtex
@misc{bansal2026trajectoryprimitive,
  title  = {Bittensor Agent Arenas as a Trajectory Primitive: Distilling a Shopping Agent from ShoppingBench Subnet Traces},
  author = {Bansal, Shardul and Schilbe, Seth and Barnes, Jarrod},
  year   = {2026},
  eprint = {2606.10064},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```
