#!/usr/bin/env python3
"""Build the ORO SFT corpus from Seth's sft_corpus_unanim_50.jsonl.

Stages: leak-cluster eval holdout -> trace-quality re-check -> Qwen3-4B
student-teacher gap filter -> 4-component reward annotation -> NeMo SFT export.

Stage E (bucket-tag + stratified sampling) is intentionally skipped — every
Stage C survivor is emitted as a final training row. The bucket_tag and
stratified_sampler modules remain in src/oro_filtering/ for possible later use.

See docs/plans/2026-05-20-oro-sft-corpus.md for design.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from oro_filtering.leak_cluster import split_by_holdout
from oro_filtering.trace_quality import passes_quality_gate
from oro_filtering.reward_components import score_components, ALL_COMPONENTS


# Load Seth's NeMo converter as a module (its filename starts with a digit, so plain
# `import` won't work). Treat scripts/25_convert_to_nemo_sft.py as read-only.
def _load_nemo_converter():
    converter_path = ROOT / "scripts" / "25_convert_to_nemo_sft.py"
    spec = importlib.util.spec_from_file_location("oro_nemo_converter", converter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load NeMo converter from {converter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_nemo_converter = _load_nemo_converter()
trajectory_to_nemo = _nemo_converter.trajectory_to_nemo

# Fields we attach to each row for downstream eval/debug. Kept in a parallel
# metadata file so the clean NeMo train file matches Trac's export contract.
# `_bucket` is intentionally omitted now that Stage E is skipped.
_METADATA_FIELDS = ("_gap", "_reward_components")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _extract_ground_truth_pids(row: dict[str, Any]) -> list[str]:
    """Pull recommended PIDs from teacher trajectory; these are our ground truth."""
    from oro_filtering.leak_cluster import extract_recommended_product_ids
    return extract_recommended_product_ids(row.get("steps", []))


def stage_a_eval_holdout(
    rows: list[dict[str, Any]],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return split_by_holdout(rows, holdout_fraction=holdout_fraction, seed=seed)


def stage_b_trace_quality(
    rows: list[dict[str, Any]],
    *,
    allow_partial: bool = True,
    min_partial_score: float = 0.3,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    survivors: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for row in rows:
        ok, reasons = passes_quality_gate(row, allow_partial=allow_partial, min_partial_score=min_partial_score)
        if ok:
            survivors.append(row)
        else:
            for reason in reasons:
                rejection_counts[reason] += 1
    return survivors, dict(rejection_counts)


def stage_c_student_gap(
    rows: list[dict[str, Any]],
    *,
    min_gap: float,
    max_student_score: float,
    use_stub: bool,
) -> list[dict[str, Any]]:
    """Run Qwen3-4B student rollout (or stub in --probe-only mode), keep rows with gap >= min_gap."""
    from oro_filtering.student_gap import run_gap_batch

    if use_stub:
        # Probe mode: skip real Modal and assume gap=0 for all rows (so we test the wiring, not the gap).
        for row in rows:
            row["_gap"] = 0.0
            row["_teacher_score"] = 1.0
            row["_student_score"] = 1.0
        return rows

    # Real path: Modal Qwen3-4B rollout.
    results = run_gap_batch(rows)
    survivors: list[dict[str, Any]] = []
    # Build the row lookup with an explicit duplicate guard. Silent dict
    # comprehension overwrites would drop one row's results onto a stale row.
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("run_id") or row.get("problem_id", "")
        if key in by_id:
            print(
                f"warning: duplicate run_id/problem_id={key!r}, "
                "second occurrence overwrites first",
                flush=True,
            )
        by_id[key] = row
    for result in results:
        if result.passes_filter(min_gap=min_gap, max_student_score=max_student_score):
            row = by_id.get(result.row_id)
            if row is not None:
                row["_gap"] = result.gap
                row["_teacher_score"] = result.teacher_score
                row["_student_score"] = result.student_score
                row["_teacher_components"] = result.teacher_components
                row["_student_components"] = result.student_components
                survivors.append(row)
    return survivors


def stage_d_score_components(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute 4-component reward for each surviving row (uses teacher trajectory)."""
    for row in rows:
        ground_truth_pids = _extract_ground_truth_pids(row)
        row["_reward_components"] = score_components(row, ground_truth_pids=ground_truth_pids)
    return rows


def _convert_rows_to_nemo(
    rows: list[dict[str, Any]],
    *,
    keep_think: bool = False,
    add_terminate: bool = True,
    add_render: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Run rows through Seth's trajectory_to_nemo.

    Returns (clean_nemo_rows, metadata_rows, dropped_count). The two output lists are
    aligned by index. Malformed rows that the converter returns None for are dropped
    from both outputs and counted in `dropped_count`.
    """
    clean: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        nemo_row = trajectory_to_nemo(
            row,
            keep_think=keep_think,
            add_terminate=add_terminate,
            add_render=add_render,
        )
        if nemo_row is None:
            dropped += 1
            continue
        clean.append(nemo_row)
        metadata.append({
            "problem_id": row.get("problem_id"),
            **{k: row.get(k) for k in _METADATA_FIELDS},
        })
    return clean, metadata, dropped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/training/sft_corpus_unanim_50.jsonl")
    parser.add_argument("--out-train", type=Path, default=ROOT / "data/training/oro_sft_15k_v0.jsonl")
    parser.add_argument("--out-train-metadata", type=Path,
                        default=ROOT / "data/training/oro_sft_200_metadata_v0.jsonl")
    parser.add_argument("--out-eval", type=Path, default=ROOT / "data/training/oro_sft_eval_v0.jsonl")
    parser.add_argument("--out-eval-metadata", type=Path,
                        default=ROOT / "data/training/oro_sft_eval_metadata_v0.jsonl")
    parser.add_argument("--out-audit", type=Path, default=ROOT / "data/training/leak_cluster_audit_v0.json")
    parser.add_argument("--out-funnel", type=Path, default=ROOT / "data/training/oro_sft_funnel_v0.json")
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--max-student-score", type=float, default=0.90)
    parser.add_argument("--probe-only", action="store_true",
                        help="Run stages A+B+C(stub)+D+F locally; skip Modal student rollout")
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Limit input rows (for fast probe). Default: use all.")
    args = parser.parse_args()

    print(f"loading rows from {args.input}", flush=True)
    rows = _load_rows(args.input)
    if args.sample_limit is not None:
        rows = rows[:args.sample_limit]
    print(f"loaded {len(rows)} rows", flush=True)

    # Stage A: eval holdout + leak-cluster expansion
    train, eval_, dropped = stage_a_eval_holdout(rows, holdout_fraction=args.holdout_fraction, seed=args.seed)
    print(f"stage A: train={len(train)} eval={len(eval_)} dropped={len(dropped)}", flush=True)

    # Stage B: trace-quality re-check on train side
    survivors, b_rejections = stage_b_trace_quality(train)
    print(f"stage B: survivors={len(survivors)} rejections={b_rejections}", flush=True)

    # Stage C: student-teacher gap filter
    after_gap = stage_c_student_gap(
        survivors,
        min_gap=args.min_gap,
        max_student_score=args.max_student_score,
        use_stub=args.probe_only,
    )
    print(f"stage C: survivors after gap filter={len(after_gap)}", flush=True)

    # Stage D: reward components (annotation only, no filtering). Last filter-y stage.
    annotated = stage_d_score_components(after_gap)
    print(f"stage D: annotated={len(annotated)}", flush=True)

    # Stage F: convert through Seth's NeMo converter. Train and eval both go through
    # the same converter so both files are in the same format. We emit two parallel
    # files: clean NeMo (messages/tools/task_name/trac_export_version only) and a
    # metadata sidecar (gap/reward_components/problem_id) for downstream eval/debug
    # joins. The clean file matches Trac's FORBIDDEN_EXPORT_KEYS contract.
    train_clean, train_metadata, train_dropped = _convert_rows_to_nemo(annotated)
    eval_clean, eval_metadata, eval_dropped = _convert_rows_to_nemo(eval_)
    print(
        f"stage F: train_out={len(train_clean)} train_dropped={train_dropped} "
        f"eval_out={len(eval_clean)} eval_dropped={eval_dropped}",
        flush=True,
    )

    _write_jsonl(args.out_train, train_clean)
    _write_jsonl(args.out_train_metadata, train_metadata)
    _write_jsonl(args.out_eval, eval_clean)
    _write_jsonl(args.out_eval_metadata, eval_metadata)

    # Audit + funnel reports
    audit = {
        "split_version": "oro-sft-corpus-v0",
        "input_rows": len(rows),
        "holdout_fraction": args.holdout_fraction,
        "seed": args.seed,
        "train_count_after_stage_a": len(train),
        "eval_count": len(eval_),
        "dropped_count": len(dropped),
        "train_eval_problem_id_intersection": sorted(
            {r["problem_id"] for r in train_metadata}
            & {r["problem_id"] for r in eval_metadata}
        ),
    }
    funnel = {
        "input_rows": len(rows),
        "stage_a_train": len(train),
        "stage_a_eval": len(eval_),
        "stage_a_dropped": len(dropped),
        "stage_b_survivors": len(survivors),
        "stage_b_rejections": b_rejections,
        "stage_c_survivors": len(after_gap),
        "stage_d_annotated": len(annotated),
        "stage_f_train_out": len(train_clean),
        "stage_f_train_dropped": train_dropped,
        "stage_f_eval_out": len(eval_clean),
        "stage_f_eval_dropped": eval_dropped,
        "final_train_rows": len(train_clean),
        "probe_only": args.probe_only,
    }
    _write_json(args.out_audit, audit)
    _write_json(args.out_funnel, funnel)
    print(f"wrote {args.out_train} ({len(train_clean)} rows)", flush=True)
    print(f"wrote {args.out_train_metadata} ({len(train_metadata)} rows)", flush=True)
    print(f"wrote {args.out_eval} ({len(eval_clean)} rows)", flush=True)
    print(f"wrote {args.out_eval_metadata} ({len(eval_metadata)} rows)", flush=True)
    print(f"wrote {args.out_audit}", flush=True)
    print(f"wrote {args.out_funnel}", flush=True)
    return 0


def _make_openai_client(args: argparse.Namespace) -> "Any":
    """Build the real GPT-4o-mini-backed ChatClient.

    Currently unused — Stage E (bucket-tag + stratified sample) is skipped, so no
    OpenAI call is made by `main`. Kept in case the bucket-tag pipeline is reinstated;
    the matching `ChatClient` protocol lives in `oro_filtering.bucket_tag`.
    """
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY required for bucket tagging")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai package required. Run `uv add openai` first.") from exc

    class _OpenAIClient:
        def __init__(self, api_key: str) -> None:
            self._client = OpenAI(api_key=api_key)

        def complete(self, messages: list[dict[str, str]]) -> str:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=64,
            )
            return resp.choices[0].message.content or ""

    return _OpenAIClient(api_key)


if __name__ == "__main__":
    raise SystemExit(main())
