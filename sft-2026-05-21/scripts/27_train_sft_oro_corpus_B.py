"""SFT training on the 1k-filtered Corpus B (no axis-B filter).

Differences from 18_train_sft_v5.py:
  - Base model: Qwen3-4B-Instruct-2507 (was Qwen3-4B)
  - Data: data/filtration/sft_corpus_B.jsonl (1052 rows)
  - Tokenization: PrimeIntellect `renderers` package (Qwen3Renderer) — fixes
    ORO-1215, gives correct loss mask on assistant tokens only. Bare HF
    apply_chat_template doesn't work for Qwen3 mask computation.

Usage on Modal:
  modal run --detach scripts/27_train_sft_oro_corpus_B.py::smoke
  modal run --detach scripts/27_train_sft_oro_corpus_B.py::full

LR is fixed at 3e-5 (middle of the sweep range, no time for full sweep tonight).
"""
from __future__ import annotations

import modal

app = modal.App("oro-sft-corpus-b")

# Image: torch + transformers + trl + peft + renderers (PrimeIntellect)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "trl>=0.13.0",
        "peft>=0.14.0",
        "accelerate>=1.2.0",
        "datasets>=3.2.0",
        "bitsandbytes>=0.45.0",
        "hf_transfer>=0.1.6",
        "renderers==0.1.7",  # PrimeIntellect renderers (ORO-1215 fix)
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Reuse same volume as v5 for consistency.
data_volume = modal.Volume.from_name("oro-distill-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("oro-distill-checkpoints", create_if_missing=True)


# Embedded literal constant — matches design doc.
BACKBONE = "Qwen/Qwen3-4B-Instruct-2507"
MAX_LENGTH = 14336
DATA_PATH = "/data/sft_corpus_B.jsonl"   # uploaded by upload_corpus.py


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=10 * 60,
)
def upload_corpus(local_jsonl_bytes: bytes):
    """Upload Corpus B from local laptop into the Modal volume."""
    import pathlib
    p = pathlib.Path(DATA_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(local_jsonl_bytes)
    data_volume.commit()
    print(f"Wrote {len(local_jsonl_bytes)} bytes to {DATA_PATH}")
    return DATA_PATH


@app.function(
    image=image,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={"/data": data_volume, "/ckpt": ckpt_volume},
)
def train(smoke: bool = False):
    """SFT training. smoke=True trains on 10 rows / 10 steps; otherwise full."""
    import json
    import time
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    import renderers as R

    if smoke:
        out_dir = "/ckpt/sft_corpus_b_smoke"
        max_steps = 10
        sample_size = 10
        run_name = "sft-corpus-b-smoke"
    else:
        out_dir = "/ckpt/sft_corpus_b"
        max_steps = -1
        sample_size = None
        run_name = "sft-corpus-b-full"

    print(f"=== {run_name} === backbone={BACKBONE} max_length={MAX_LENGTH}")

    # ---- Load and pre-tokenize via renderers ----
    raw = []
    with open(DATA_PATH) as f:
        for line in f:
            raw.append(json.loads(line))
    if sample_size:
        raw = raw[:sample_size]
    print(f"Loaded {len(raw)} rows from {DATA_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max(tokenizer.model_max_length, MAX_LENGTH)
    renderer = R.create_renderer(tokenizer, renderer="qwen3")
    print(f"Renderer: {type(renderer).__name__}")

    def render_row(row):
        # Strip annotation fields.
        msgs = row["messages"]
        tools = row.get("tools") or None
        token_ids, loss_mask = R.build_training_sample(
            renderer, msgs,
            role_to_mask=lambda m: m.get("role") == "assistant",
            tools=tools,
        )
        # Clip to MAX_LENGTH from the end if needed (preserve recommend+terminate).
        if len(token_ids) > MAX_LENGTH:
            token_ids = token_ids[-MAX_LENGTH:]
            loss_mask = loss_mask[-MAX_LENGTH:]
        labels = [
            (token_ids[i] if loss_mask[i] else -100) for i in range(len(token_ids))
        ]
        return {"input_ids": token_ids, "labels": labels,
                "attention_mask": [1] * len(token_ids)}

    print("Pre-tokenizing...")
    t0 = time.time()
    examples = [render_row(r) for r in raw]
    print(f"Done in {time.time() - t0:.1f}s")

    # Diagnostics: assistant-token fraction.
    total_tok = sum(len(e["input_ids"]) for e in examples)
    total_mask = sum(sum(1 for l in e["labels"] if l != -100) for e in examples)
    print(f"Total tokens: {total_tok}")
    print(f"Assistant (trained) tokens: {total_mask} ({100*total_mask/max(total_tok,1):.1f}%)")
    if total_mask == 0:
        print("FATAL: zero assistant tokens — loss mask is broken. Aborting.")
        return {"status": "fatal_no_assistant_tokens"}

    ds = Dataset.from_list(examples)

    # ---- Model + LoRA ----
    print(f"Loading {BACKBONE}...")
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=MAX_LENGTH,
        max_steps=max_steps,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        run_name=run_name,
        # We pre-tokenized; tell TRL not to retokenize.
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=peft_config,
    )

    print("Starting training...")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")
    print(f"Train loss: {train_result.training_loss:.4f}")

    trainer.save_model(out_dir)
    ckpt_volume.commit()
    print(f"Saved checkpoint to {out_dir}")

    return {
        "status": "ok",
        "out_dir": out_dir,
        "train_loss": float(train_result.training_loss),
        "elapsed_sec": elapsed,
        "examples": len(examples),
        "total_tokens": total_tok,
        "assistant_tokens": total_mask,
    }


@app.local_entrypoint()
def smoke():
    import pathlib
    local = pathlib.Path("data/filtration/sft_corpus_B.jsonl")
    print(f"Uploading {local} -> Modal volume...")
    upload_corpus.remote(local.read_bytes())
    print("Running smoke train (10 rows, 10 steps)...")
    result = train.remote(smoke=True)
    print(result)


@app.local_entrypoint()
def full():
    import pathlib
    local = pathlib.Path("data/filtration/sft_corpus_B.jsonl")
    print(f"Uploading {local} -> Modal volume...")
    upload_corpus.remote(local.read_bytes())
    print("Running full train...")
    result = train.remote(smoke=False)
    print(result)
