"""Modal orchestration for the Trac trace-to-curriculum pipeline.

Survives laptop sleep. All work happens on Modal compute; status is persisted to a Modal volume.

Stages:
  1. Convert raw_trajectories.jsonl → Trac OpenAI tool_calls format (uses SFT-cache, no search server)
  2. Freeze SFT/eval splits
  3. Run annotation (5 agents per trace)
  4. Validate artifacts (warnings only)
  5. Autodata probe (endpoint health)
  6. Autodata 3-row gate
  7. Autodata production batch (target_accepted)
  8. Export NeMo RL SFT JSONL
  9. Persist artifacts back to /data volume

Setup (one-time, run from laptop):
  modal profile activate oro-ai
  modal secret create oro-openai-api-key OPENAI_API_KEY=$OPENAI_API_KEY
  modal secret create oro-openrouter-api-key OPENROUTER_API_KEY=$OPENROUTER_API_KEY
  modal volume create oro-trac-data
  modal volume put oro-trac-data <local-paths> /

Run:
  modal run modal_app.py::main --limit 200 --target-accepted 50 --max-attempts 300
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# Local layout we'll package into the Modal image.
LOCAL_REPO = Path(__file__).resolve().parent
LOCAL_TRAC = LOCAL_REPO / "trac"
LOCAL_CONVERTER = (
    LOCAL_REPO.parent
    / ".work"
    / "oro-sn-internal-distill"
    / "distill-practice"
    / "scripts"
    / "24_raw_to_trac.py"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "git")
    .pip_install("uv==0.5.6", "requests")
    .add_local_dir(
        str(LOCAL_TRAC),
        remote_path="/workspace/trac",
        copy=True,
        ignore=[".venv", "__pycache__", "artifacts", "*.pyc", ".pytest_cache", ".ruff_cache", ".mypy_cache"],
    )
    .add_local_file(str(LOCAL_CONVERTER), remote_path="/workspace/scripts/24_raw_to_trac.py", copy=True)
    .run_commands(
        [
            "cd /workspace/trac && uv sync --no-dev",
            (
                "cd /workspace/trac && uv run python -c "
                "\"from transformers import AutoTokenizer; "
                "AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')\""
            ),
        ]
    )
)

data_volume = modal.Volume.from_name("oro-trac-data", create_if_missing=True)

app = modal.App("oro-trac-pipeline")

BASE_ENV = {
    # Route the annotator through OpenRouter to Claude Sonnet 4.6.
    # Our OpenAI account hit `insufficient_quota`; OpenRouter exposes the
    # Responses API for non-OpenAI models, so the OpenAI Agents SDK works
    # transparently when OPENAI_BASE_URL + OPENAI_API_KEY point there.
    "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
    # Annotator path so far:
    #   gpt-chat-latest ($400/run, works)
    #   -> qwen3.5-plus (Alibaba json quirk, fails)
    #   -> gpt-5-mini (Azure store=true quirk, fails)
    #   -> gpt-4.1-mini (hallucinates trace IDs, fails Trac strict validation)
    #   -> gpt-4o-mini (appends prose after JSON, breaks Agents SDK parser)
    #   -> mistral-medium-3-5 (got 925/5873 batches before OpenRouter 429 upstream)
    #   -> claude-haiku-4.5 (OpenRouter routed via Amazon Bedrock; Bedrock rejects
    #      JSON schemas with min/max on numbers, which Trac uses everywhere; failed
    #      on first call with 400)
    #   -> gpt-chat-latest (fallback; only known-good annotator; ~$60-80 at this
    #      volume since we're on 845 batches, not 5,873)
    "ANNOTATOR_MODEL": "openai/gpt-chat-latest",
    "STUDENT_OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
    # Reverted from google/gemma-3-4b-it after that produced 54% runtime
    # exceptions (24/44 attempts) due to Gemma's tool-use incompatibility
    # with Trac's surface. Qwen3-8B is bigger than our SFT target but has
    # proven tool-use. We compensate for "student too good" via the
    # multi-constraint task-design bias + max-per-problem 15 breadth.
    "STUDENT_MODEL": "qwen/qwen3-8b",
    "TEACHER_OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
    "TEACHER_MODEL": "qwen/qwen3.6-plus",
    "TASK_DESIGNER_OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
    "TASK_DESIGNER_MODEL": "qwen/qwen3.6-flash",
    "TRACE_CURRICULUM_RENDERER_MODEL_ID": "Qwen/Qwen3-4B-Instruct-2507",
    "TRACE_CURRICULUM_RENDERER": "auto",
    "PYTHONUNBUFFERED": "1",
    # Allow success=False traces through freeze; supervisor decides usability.
    "ORO_ALLOW_FAILED_TRACES": "1",
    # Tracing tries to authenticate against api.openai.com directly with our
    # OpenRouter key. Non-fatal but spammy and triggers a 401 on every call.
    "OPENAI_AGENTS_DISABLE_TRACING": "1",
}


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("oro-openai-api-key"),
        modal.Secret.from_name("oro-openrouter-api-key"),
    ],
    volumes={"/data": data_volume},
    timeout=60 * 60 * 8,
    cpu=4.0,
    memory=16384,
)
def run_full_pipeline(
    limit: int = 200,
    target_accepted: int = 50,
    max_attempts: int = 300,
    batch_size: int = 1,
    concurrency: int = 4,
    stop_after: str = "export",
    max_per_problem: int = 0,
    use_combined: bool = True,
) -> dict:
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    trac_dir = Path("/workspace/trac")
    workspace = Path("/workspace")

    # Build the distill-practice layout the converter expects:
    #   <distill_root>/data/{raw_trajectories,problems_v123,sft_train_multi_turn_v2}.jsonl
    #   <distill_root>/scripts/24_raw_to_trac.py
    distill_root = workspace / "distill-practice"
    distill_data = distill_root / "data"
    distill_scripts = distill_root / "scripts"
    distill_data.mkdir(parents=True, exist_ok=True)
    distill_scripts.mkdir(parents=True, exist_ok=True)

    # raw_trajectories_combined.jsonl is the 60K-trajectory combined file (existing
    # 5,635 + 55,001 race-only HQ pulled 2026-05-18). If present on the volume,
    # the converter is pointed at it via --source-jsonl; otherwise falls back to
    # raw_trajectories.jsonl.
    required = ["raw_trajectories.jsonl", "problems_v123.jsonl",
                "problems_combined.jsonl", "sft_train_multi_turn_v2.jsonl"]
    optional = ["raw_trajectories_combined.jsonl"]
    for name in required + optional:
        src = Path("/data") / name
        dst = distill_data / name
        if not src.exists():
            if name in required:
                raise FileNotFoundError(f"missing input: {src}")
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    # Source-of-truth selection: combined file gives breadth (race-phase pulls)
    # but race-phase trajectories aren't in sft_train_multi_turn_v2.jsonl, so
    # they fall through to HTTP fetch against the search server. That's slow
    # (~10h for 1900 batches) and produced the 2026-05-19 wedge. use_combined=False
    # forces the original SFT-cached corpus (5,635 trajectories, fast).
    if use_combined and (distill_data / "raw_trajectories_combined.jsonl").exists():
        converter_source = distill_data / "raw_trajectories_combined.jsonl"
    else:
        converter_source = distill_data / "raw_trajectories.jsonl"

    converter_src = Path("/workspace/scripts/24_raw_to_trac.py")
    converter_dst = distill_scripts / "24_raw_to_trac.py"
    if not converter_src.exists():
        raise FileNotFoundError(f"missing converter: {converter_src}")
    if converter_dst.exists() or converter_dst.is_symlink():
        converter_dst.unlink()
    shutil.copy(converter_src, converter_dst)

    env = os.environ.copy()
    env.update(BASE_ENV)
    env["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]  # route annotator via OpenRouter
    env["STUDENT_OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    env["TEACHER_OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    env["TASK_DESIGNER_OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    env["PYTHONPATH"] = str(trac_dir / "src")
    env["ORO_TRAC_ROOT"] = str(trac_dir)

    def run(cmd: list[str], cwd: Path, check: bool = True, extra_env: dict | None = None) -> int:
        local_env = env.copy()
        if extra_env:
            local_env.update(extra_env)
        print(f"\n--- $ {' '.join(cmd)}  (cwd={cwd}) ---", flush=True)
        proc = subprocess.run(cmd, cwd=cwd, env=local_env)
        if check and proc.returncode != 0:
            raise RuntimeError(f"command failed: {cmd!r} (rc={proc.returncode})")
        return proc.returncode

    print(f"\n=== Stage 1: convert (source: {converter_source.name}) ===")
    convert_cmd = [
        "python3", str(converter_dst),
        "--limit", str(limit),
        "--source-jsonl", str(converter_source),
    ]
    if max_per_problem > 0:
        convert_cmd.extend(["--max-per-problem", str(max_per_problem)])
    run(convert_cmd, cwd=workspace / "distill-practice")
    data_volume.commit()
    if stop_after == "convert":
        return _persist_and_return(trac_dir, "stopped_after_convert")

    print("\n=== Stage 2: freeze ===")
    run(["uv", "run", "python", "scripts/freeze_sft_eval_splits.py"], cwd=trac_dir)
    if stop_after == "freeze":
        return _persist_and_return(trac_dir, "stopped_after_freeze")

    print("\n=== Stage 3: annotate ===")
    run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_annotation.py",
            "--limit",
            str(limit),
            "--batch-size",
            str(batch_size),
            "--concurrency",
            str(concurrency),
            "--resume-partial",
        ],
        cwd=trac_dir,
    )
    data_volume.commit()
    if stop_after == "annotate":
        return _persist_and_return(trac_dir, "stopped_after_annotate")

    print("\n=== Stage 4: validate (warnings only) ===")
    run(["uv", "run", "python", "scripts/validate_artifacts.py"], cwd=trac_dir, check=False)

    print("\n=== Stage 5: autodata probe ===")
    probe_rc = run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_autodata.py",
            "--target-accepted",
            "3",
            "--max-attempts",
            "24",
            "--probe-only",
        ],
        cwd=trac_dir,
        check=False,
    )
    if probe_rc != 0:
        print(f"probe failed (rc={probe_rc}), stopping before autodata generation.")
        return _persist_and_return(trac_dir, "probe_failed")
    if stop_after == "probe":
        return _persist_and_return(trac_dir, "stopped_after_probe")

    print("\n=== Stage 6: autodata 3-row gate (best-effort) ===")
    gate_rc = run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_autodata.py",
            "--target-accepted",
            "3",
            "--max-attempts",
            "24",
            "--resume",
        ],
        cwd=trac_dir,
        check=False,
    )
    if gate_rc != 0:
        print(f"[gate] WARNING gate didn't hit 3 accepted (rc={gate_rc}); "
              "checkpoints persist, production batch will continue from there")
    if stop_after == "gate":
        return _persist_and_return(trac_dir, "stopped_after_gate")

    print(f"\n=== Stage 7: autodata production batch (target={target_accepted}, max={max_attempts}) ===")
    batch_rc = run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_autodata.py",
            "--target-accepted",
            str(target_accepted),
            "--max-attempts",
            str(max_attempts),
            "--resume",
        ],
        cwd=trac_dir,
        check=False,
    )
    if batch_rc != 0:
        print(f"[batch] WARNING production batch didn't hit target (rc={batch_rc}); "
              "exporting whatever accepted rows landed in checkpoint")

    print("\n=== Stage 8: export SFT (best-effort) ===")
    run(["uv", "run", "python", "scripts/export_sft.py"], cwd=trac_dir, check=False)

    return _persist_and_return(trac_dir, "complete")


def _persist_and_return(trac_dir, status: str) -> dict:
    import shutil
    from pathlib import Path

    out_root = Path("/data/output-artifacts")
    if out_root.exists():
        shutil.rmtree(out_root)
    shutil.copytree(trac_dir / "artifacts", out_root)
    print(f"\nPersisted artifacts to {out_root}. status={status}")
    data_volume.commit()
    return {"status": status}


@app.local_entrypoint()
def main(
    limit: int = 200,
    target_accepted: int = 50,
    max_attempts: int = 300,
    batch_size: int = 1,
    concurrency: int = 4,
    stop_after: str = "export",
    max_per_problem: int = 0,
    use_combined: bool = True,
):
    # .spawn() returns a FunctionCall handle that keeps the remote function
    # running on Modal even after this local process exits. Required for true
    # detached runs -- with .remote() + --detach, Modal cancels the input
    # roughly 10 minutes after the local caller disconnects (we hit this).
    call = run_full_pipeline.spawn(
        limit=limit,
        target_accepted=target_accepted,
        max_attempts=max_attempts,
        batch_size=batch_size,
        concurrency=concurrency,
        stop_after=stop_after,
        max_per_problem=max_per_problem,
        use_combined=use_combined,
    )
    print(f"FunctionCall: {call.object_id}")
    print(f"Track at https://modal.com/apps/oro-ai/main/")
