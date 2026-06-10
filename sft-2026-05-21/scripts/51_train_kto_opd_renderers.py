"""KTO training on top of opd_renderers — Phase 3c v2.

LoRA stack: Qwen3-4B → opd_renderers (merged) → fresh KTO LoRA.

IMPORTANT — v1 fix: opd_renderers was trained directly from base Qwen3-4B,
NOT stacked on rejection_v2 (despite the path name "from_rejection_v2").
Stacking rejection_v2 underneath corrupts weights at merge time. Validated
2026-05-28 via 4-way bisect: base alone coherent, rejection_v2 alone coherent,
opd_renderers alone coherent, rejection_v2+opd_renderers garbled.

Dataset (~480 rows): Sonnet teacher trajectories from opd_sonnet_500, labeled
desirable (score_ok=True, ~207 wins) or undesirable (score_ok=False, ~274 fails).

Modal volume `oro-sft-checkpoints` (same workspace Seth used):
  Direct adapter:      sft_opd_renderers_from_rejection_v2/final
Output:                sft_kto_opd_renderers_sonnet_v2/final

Usage:
    modal run --detach sft-2026-05-21/scripts/51_train_kto_opd_renderers.py::upload
    modal run --detach sft-2026-05-21/scripts/51_train_kto_opd_renderers.py::train
"""
from __future__ import annotations

import modal

app = modal.App("oro-kto-opd-renderers")

BACKBONE = "Qwen/Qwen3-4B"
OPD_RENDERERS_PATH = "/ckpt/sft_opd_renderers_from_rejection_v2/final"
KTO_OUT_DIR = "/ckpt/sft_kto_opd_renderers_sonnet_v4"
DATASET_PATH = "/data/kto_dataset_opd_renderers_v1.jsonl"

# KTO hyperparameters — v4: tighter beta + 2 epochs vs v3
BETA = 0.01
DESIRABLE_WEIGHT = 1.4   # TRL recommended [1.22, 1.63] for 207/274 ratio
UNDESIRABLE_WEIGHT = 1.0
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 5e-6
NUM_EPOCHS = 2
PER_DEVICE_BATCH = 2  # KTO requires actual batch > 1 for KL term
GRAD_ACCUM = 4         # effective batch 8 stays the same
MAX_LENGTH = 16384  # ~90% of completions fit; truncate_end preserves recommend_product
MAX_PROMPT_LENGTH = 1000
WARMUP_RATIO = 0.05

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",  # Qwen3 architecture support
        "peft>=0.14.0",
        "trl==0.19.1",  # latest 0.19.x — has max_prompt_length/truncation_mode AND Qwen3-compatible
        "accelerate>=1.2.0",
        "datasets>=3.0.0",
        "hf_transfer>=0.1.6",
        "bitsandbytes>=0.43.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

data_volume = modal.Volume.from_name("oro-distill-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("oro-sft-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=5 * 60,
)
def upload_dataset(jsonl_bytes: bytes):
    import pathlib
    pathlib.Path(DATASET_PATH).write_bytes(jsonl_bytes)
    data_volume.commit()
    print(f"Uploaded {len(jsonl_bytes)} bytes to {DATASET_PATH}")


@app.function(
    image=image,
    gpu="H100",
    timeout=10 * 60 * 60,
    volumes={"/data": data_volume, "/ckpt": ckpt_volume},
)
def train(smoke: bool = False, resume: bool = False):
    import json
    import os
    import time

    import torch
    from datasets import Dataset
    from peft import PeftModel, LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import KTOTrainer, KTOConfig

    out_dir = KTO_OUT_DIR if not smoke else KTO_OUT_DIR + "_smoke"
    max_steps = -1 if not smoke else 10
    sample_size = None if not smoke else 20
    run_name = "kto-opd-renderers-sonnet" + ("-smoke" if smoke else "")

    print(f"=== {run_name} ===")
    print(f"  Base: {BACKBONE}")
    print(f"  Stack (merged): {OPD_RENDERERS_PATH}")
    print(f"  New LoRA: r={LORA_R} α={LORA_ALPHA}")
    print(f"  KTO: beta={BETA} lr={LEARNING_RATE} epochs={NUM_EPOCHS}")
    print(f"  max_length={MAX_LENGTH} max_prompt_length={MAX_PROMPT_LENGTH}")

    # Load dataset
    rows = []
    with open(DATASET_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    if sample_size:
        rows = rows[:sample_size]
    print(f"\nDataset: {len(rows)} rows ({sum(r['label'] for r in rows)} desirable)")

    # Tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model: stack rejection_v2 + opd_renderers and merge into base
    print(f"Loading {BACKBONE}...")
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    print(f"Loading + merging opd_renderers adapter from {OPD_RENDERERS_PATH}...")
    model = PeftModel.from_pretrained(model, OPD_RENDERERS_PATH)
    model = model.merge_and_unload()
    model.config.use_cache = False
    # TRL 0.19's KTOTrainer expects model.warnings_issued (removed in transformers 4.51+)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    print(f"Model loaded, parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Attach fresh KTO LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["k_proj", "q_proj", "o_proj", "v_proj"],  # match Seth's attn-only
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Render dataset to KTO format: TRL KTOTrainer accepts conversational format
    # (lists of messages) and applies the chat template internally.
    ds_dict = {
        "prompt": [r["prompt"] for r in rows],
        "completion": [r["completion"] for r in rows],
        "label": [r["label"] for r in rows],
    }
    ds = Dataset.from_dict(ds_dict)

    args = KTOConfig(
        output_dir=out_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_steps=max_steps,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=3,
        report_to="none",
        run_name=run_name,
        remove_unused_columns=False,
        beta=BETA,
        desirable_weight=DESIRABLE_WEIGHT,
        undesirable_weight=UNDESIRABLE_WEIGHT,
        truncation_mode="keep_end",  # preserve recommend_product at end of trajectory
    )

    trainer = KTOTrainer(
        model=model,
        ref_model=None,  # PEFT model → reference auto-derived by disabling adapter
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Resume from latest checkpoint if requested + present
    resume_target = None
    if resume and os.path.isdir(out_dir):
        ckpts = sorted(
            [d for d in os.listdir(out_dir) if d.startswith("checkpoint-")],
            key=lambda d: int(d.split("-")[1]),
        )
        if ckpts:
            resume_target = os.path.join(out_dir, ckpts[-1])
            print(f"Resuming from {resume_target}")

    print("Starting KTO training...")
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_target)
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.0f}s")
    print(f"Final loss: {train_result.training_loss:.4f}")

    final_dir = f"{out_dir}/final"
    trainer.save_model(final_dir)
    ckpt_volume.commit()
    print(f"Saved final adapter to {final_dir}")

    return {
        "status": "ok",
        "out_dir": final_dir,
        "train_loss": float(train_result.training_loss),
        "elapsed_sec": elapsed,
        "n_examples": len(rows),
    }


@app.local_entrypoint()
def upload():
    import pathlib
    p = pathlib.Path("sft-2026-05-21/phase3c_kto/kto_dataset_opd_renderers_v1.jsonl")
    print(f"Uploading {p} ({p.stat().st_size / 1024:.1f} KB)...")
    upload_dataset.remote(p.read_bytes())


@app.local_entrypoint()
def smoke():
    train.remote(smoke=True)


@app.local_entrypoint()
def full():
    print("Running full KTO train (~481 rows, ~6 hr)...")
    train.remote(smoke=False)


@app.local_entrypoint()
def resume_full():
    print("Resuming KTO train from latest checkpoint...")
    train.remote(smoke=False, resume=True)
