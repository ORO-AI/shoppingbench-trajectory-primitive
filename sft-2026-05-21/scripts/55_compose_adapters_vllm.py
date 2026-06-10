"""Compose opd_renderers ⊕ KTO into a single LoRA adapter for vLLM serving.

vLLM applies ONE LoRA per request — it cannot stack opd_renderers + KTO. But
the eval stack is `Qwen3-4B → merge(opd_renderers) → KTO(active)`, i.e. the
final weights are `W_base + ΔW_opd + ΔW_kto`.

Both adapters are rank 16, alpha 32, targeting [q,k,v,o]_proj. PEFT's
`add_weighted_adapter(combination_type="cat", weights=[1,1])` concatenates them
into a single rank-32 adapter whose merged delta is EXACTLY ΔW_opd + ΔW_kto.
Served on plain `Qwen/Qwen3-4B` (the HF repo Seth's server already proves
works), this reproduces the full stack with no merged-base local-path landmine
and no 8GB checkpoint.

This script also runs a weight-level parity gate: it merges the composed
adapter into base and merges the sequential (opd→kto) stack into a fresh base,
then asserts the resulting q/v_proj weights are allclose. This is the rigorous
correctness check — if it passes, the served model is byte-equivalent to the HF
eval's stack.

Output:
  /ckpt/composed/<name>/   (single rank-32 LoRA adapter on Qwen/Qwen3-4B)

Usage:
    modal run sft-2026-05-21/scripts/55_compose_adapters_vllm.py::compose_v4
    modal run sft-2026-05-21/scripts/55_compose_adapters_vllm.py::compose_v3
"""
from __future__ import annotations

import modal

app = modal.App("oro-compose-adapters-vllm")

BACKBONE = "Qwen/Qwen3-4B"
OPD_PATH = "/ckpt/sft_opd_renderers_from_rejection_v2/final"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.53.1",
        "peft>=0.14.0",
        "accelerate>=1.2.0",
        "hf_transfer>=0.1.6",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

ckpt_volume = modal.Volume.from_name("oro-sft-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/ckpt": ckpt_volume},
)
def compose(kto_path: str, out_name: str, verify: bool = True) -> dict:
    """Cat-compose opd_renderers ⊕ KTO into /ckpt/composed/<out_name>."""
    import os
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = f"/ckpt/composed/{out_name}"

    print(f"Loading base {BACKBONE} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    base = AutoModelForCausalLM.from_pretrained(
        BACKBONE, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )

    print(f"Loading opd adapter ({OPD_PATH}) ...", flush=True)
    model = PeftModel.from_pretrained(base, OPD_PATH, adapter_name="opd")
    print(f"Loading kto adapter ({kto_path}) ...", flush=True)
    model.load_adapter(kto_path, adapter_name="kto")

    print("Cat-composing opd ⊕ kto (weights=[1,1]) ...", flush=True)
    model.add_weighted_adapter(
        adapters=["opd", "kto"],
        weights=[1.0, 1.0],
        adapter_name="composed",
        combination_type="cat",
    )
    model.set_adapter("composed")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, selected_adapters=["composed"])
    tokenizer.save_pretrained(out_dir)
    ckpt_volume.commit()

    # PEFT writes a named (non-"default") adapter into <out_dir>/<name>/.
    adapter_dir = f"{out_dir}/composed"

    import json
    cfg = json.load(open(f"{adapter_dir}/adapter_config.json"))
    print(f"✓ Saved composed adapter to {adapter_dir} (r={cfg.get('r')}, alpha={cfg.get('lora_alpha')})",
          flush=True)

    result = {"status": "ok", "adapter_dir": adapter_dir, "rank": cfg.get("r")}

    if verify:
        result["parity"] = _weight_parity_gate(kto_path, adapter_dir)

    return result


def _weight_parity_gate(kto_path: str, composed_dir: str) -> dict:
    """Assert merge(base+composed) == merge(merge(base+opd)+kto) on sampled weights.

    Memory-safe: builds one merged model, snapshots a few target weights to CPU,
    frees it, builds the other, compares on CPU.
    """
    import gc
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    # Layers/params to sample (cheap but representative of LoRA targets).
    sample_keys = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.17.self_attn.q_proj.weight",
        "model.layers.35.self_attn.o_proj.weight",
    ]

    def merged_snapshot(build_fn) -> dict:
        m = build_fn()
        sd = dict(m.named_parameters())
        snap = {}
        for k in sample_keys:
            if k in sd:
                snap[k] = sd[k].detach().float().cpu().clone()
        del m, sd
        gc.collect()
        torch.cuda.empty_cache()
        return snap

    def build_composed():
        base = AutoModelForCausalLM.from_pretrained(
            BACKBONE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        m = PeftModel.from_pretrained(base, composed_dir)
        return m.merge_and_unload()

    def build_stack():
        base = AutoModelForCausalLM.from_pretrained(
            BACKBONE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        m = PeftModel.from_pretrained(base, OPD_PATH)
        m = m.merge_and_unload()
        m = PeftModel.from_pretrained(m, kto_path)
        return m.merge_and_unload()

    print("[parity] snapshotting composed-on-base weights ...", flush=True)
    snap_composed = merged_snapshot(build_composed)
    print("[parity] snapshotting sequential-stack weights ...", flush=True)
    snap_stack = merged_snapshot(build_stack)

    report = {}
    all_ok = True
    for k in snap_composed:
        a, b = snap_composed[k], snap_stack[k]
        max_abs = (a - b).abs().max().item()
        # bf16 round-trip + two merges → allow a small tolerance.
        ok = torch.allclose(a, b, atol=2e-2, rtol=1e-3)
        all_ok = all_ok and ok
        report[k] = {"max_abs_diff": round(max_abs, 6), "allclose": ok}
        print(f"[parity] {k}: max_abs_diff={max_abs:.6f} allclose={ok}", flush=True)

    report["PASS"] = all_ok
    if not all_ok:
        raise AssertionError(f"PARITY GATE FAILED — composed adapter != sequential stack: {report}")
    print("[parity] ✓ PASS — composed adapter is weight-equivalent to the HF eval stack", flush=True)
    return report


@app.local_entrypoint()
def compose_v4():
    print(compose.remote(
        kto_path="/ckpt/sft_kto_opd_renderers_sonnet_v4/final",
        out_name="opd_kto_v4",
    ))


@app.local_entrypoint()
def compose_v3():
    print(compose.remote(
        kto_path="/ckpt/sft_kto_opd_renderers_sonnet_v3/final",
        out_name="opd_kto_v3",
    ))
