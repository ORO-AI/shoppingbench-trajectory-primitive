"""vLLM eval server: Qwen3-4B + swappable eval adapters, OpenAI-compatible.

Clones Seth's PROVEN config from scripts/40_serve_opd_renderers_vllm.py
(plain `Qwen/Qwen3-4B` base + local LoRA paths via --lora-modules, pinned
vllm==0.9.2 + transformers==4.53.1, H200). The ONLY change is the adapter set:

    opd          = opd_renderers alone (rank 16)        — SFT baseline
    opd_kto_v3   = composed opd ⊕ KTO_v3 (rank 32)       — built by script 55
    opd_kto_v4   = composed opd ⊕ KTO_v4 (rank 32)       — built by script 55
    drgrpo       = Dr.GRPO policy (rank 16, self-contained, loaded ALONE)

`--max-lora-rank 32` covers the rank-32 composed adapters. The eval client
(script 57) selects the adapter per request via the OpenAI `model=` field.

Why serve (not offline LLM): the OpenAI server's continuous batching coalesces
the concurrent /v1/completions requests from the threaded client — that is what
gives the 6×+ speedup. Offline `LLM.generate()` called from threads does NOT
batch and is not thread-safe (the bug in script 54).

Deploy:
    modal deploy sft-2026-05-21/scripts/56_serve_eval_vllm.py
URL:
    https://oro-ai--oro-vllm-eval-serve.modal.run
"""
from __future__ import annotations

import modal

BACKBONE = "Qwen/Qwen3-4B"

# name -> adapter dir on the oro-sft-checkpoints volume. Entries whose dir is
# absent at boot are skipped (so this serves whatever has been built so far).
ADAPTERS = {
    "opd": "/ckpt/sft_opd_renderers_from_rejection_v2/final",
    "opd_kto_v3": "/ckpt/composed/opd_kto_v3/composed",
    "opd_kto_v4": "/ckpt/composed/opd_kto_v4/composed",
    # Dr.GRPO policies are self-contained (init from opd + GRPO updates in one
    # LoRA) → loaded ALONE on base. step_5/10/15/20 give the SFT→RL curve.
    "drgrpo_step5": "/ckpt/drgrpo_v17_vllm_rollouts/step_0005/policy",
    "drgrpo_step10": "/ckpt/drgrpo_v17_vllm_rollouts/step_0010/policy",
    "drgrpo_step15": "/ckpt/drgrpo_v17_vllm_rollouts/step_0015/policy",
    "drgrpo_step20": "/ckpt/drgrpo_v17_vllm_rollouts/step_0020/policy",
}

app = modal.App("oro-vllm-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("vllm==0.9.2", "transformers==4.53.1", "peft>=0.14.0")
)

ckpt_volume = modal.Volume.from_name("oro-sft-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H200",
    timeout=4 * 60 * 60,
    volumes={"/ckpt": ckpt_volume},
    scaledown_window=10 * 60,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve():
    import os
    import subprocess

    available = {k: v for k, v in ADAPTERS.items() if os.path.isdir(v)}
    print(f"available adapters: {list(available)}", flush=True)

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", BACKBONE,                 # HF repo — Seth's proven path, no local-base landmine
        "--dtype", "bfloat16",
        "--host", "0.0.0.0",
        "--port", "8000",
        # 32768 (Qwen3-4B native) so long multi-item shop/voucher histories
        # (8+ tool results, dozens of products) don't overflow context and 400.
        # At 16384 the tail problems errored and fell back to auto-recommend,
        # which both wasted retries and diverged from the HF eval (no hard cap).
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.90",
        "--served-model-name", "base",
    ]
    if available:
        cmd += [
            "--enable-lora",
            "--max-lora-rank", "32",          # composed adapters are rank 32
            "--max-loras", str(len(available)),
            "--lora-modules", *[f"{k}={v}" for k, v in available.items()],
        ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd)
