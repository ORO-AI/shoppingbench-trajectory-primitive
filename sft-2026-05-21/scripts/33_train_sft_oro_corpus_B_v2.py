"""SFT v2: Corpus B at per-problem cap=5 (2,416 rows), 3 epochs, rank-32 LoRA.

Adapted from 27_train_sft_oro_corpus_B.py with three changes:
  1. Data: data/filtration/sft_corpus_B_cap5.jsonl (was sft_corpus_B.jsonl, 1052 → 2416 rows)
  2. Epochs: 3 (was 1)
  3. LoRA: rank=32, alpha=64 (was rank=16, alpha=32)

Everything else identical: Qwen3-4B-Instruct-2507, max_length=14336, renderers package
for correct loss masking, LR=3e-5 cosine, batch=1×grad_accum=8.

Usage:
    modal run --detach scripts/33_train_sft_oro_corpus_B_v2.py::smoke
    modal run --detach scripts/33_train_sft_oro_corpus_B_v2.py::full
"""
from __future__ import annotations

import modal

app = modal.App("oro-sft-corpus-b-v2")

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
        "renderers==0.1.7",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

data_volume = modal.Volume.from_name("oro-distill-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("oro-distill-checkpoints", create_if_missing=True)

BACKBONE = "Qwen/Qwen3-4B-Instruct-2507"
MAX_LENGTH = 14336
DATA_PATH = "/data/sft_corpus_B_cap5.jsonl"


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=10 * 60,
)
def upload_corpus(local_jsonl_bytes: bytes):
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
    timeout=8 * 60 * 60,  # 8h cap for safety; expecting ~2.5h
    volumes={"/data": data_volume, "/ckpt": ckpt_volume},
)
def train(smoke: bool = False):
    import json
    import time
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    import renderers as R

    if smoke:
        out_dir = "/ckpt/sft_corpus_b_v2_smoke"
        max_steps = 10
        sample_size = 10
        run_name = "sft-corpus-b-v2-smoke"
    else:
        out_dir = "/ckpt/sft_corpus_b_v2"
        max_steps = -1
        sample_size = None
        run_name = "sft-corpus-b-v2-full"

    print(f"=== {run_name} === backbone={BACKBONE} max_length={MAX_LENGTH}")
    print(f"  3 epochs, LoRA r=32 α=64, LR=3e-5 cosine")

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
        msgs = row["messages"]
        tools = row.get("tools") or None
        token_ids, loss_mask = R.build_training_sample(
            renderer, msgs,
            role_to_mask=lambda m: m.get("role") == "assistant",
            tools=tools,
        )
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

    total_tok = sum(len(e["input_ids"]) for e in examples)
    total_mask = sum(sum(1 for l in e["labels"] if l != -100) for e in examples)
    print(f"Total tokens: {total_tok}")
    print(f"Assistant (trained) tokens: {total_mask} ({100*total_mask/max(total_tok,1):.1f}%)")
    if total_mask == 0:
        return {"status": "fatal_no_assistant_tokens"}

    ds = Dataset.from_list(examples)

    print(f"Loading {BACKBONE}...")
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=32,           # ↑ from 16
        lora_alpha=64,  # ↑ from 32 (kept α/r=2)
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=3,       # ↑ from 1
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=MAX_LENGTH,
        max_steps=max_steps,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        run_name=run_name,
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
    local = pathlib.Path("data/filtration/sft_corpus_B_cap5.jsonl")
    print(f"Uploading {local} ({local.stat().st_size} bytes)...")
    upload_corpus.remote(local.read_bytes())
    print("Running smoke train...")
    print(train.remote(smoke=True))


@app.local_entrypoint()
def full():
    import pathlib
    local = pathlib.Path("data/filtration/sft_corpus_B_cap5.jsonl")
    print(f"Uploading {local} ({local.stat().st_size} bytes)...")
    upload_corpus.remote(local.read_bytes())
    print("Running full train (3 epochs)...")
    print(train.remote(smoke=False))
