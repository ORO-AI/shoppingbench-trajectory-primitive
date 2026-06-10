"""Eval the KTO checkpoint on Seth's 7-tool ShoppingBench harness (75-set).

Mirrors Seth's 72b driver but runs inference LOCALLY on Modal H100 using:
  Qwen3-4B → rejection_v2 LoRA (merged) → opd_renderers LoRA (merged) → KTO LoRA (active)

The 7-tool agent code from our fork (ORO-AI/ShoppingBench HEAD ba608ce) is baked
into the Modal image so the agent_main function can call our local inference()
patch and use the full 7-tool surface + per-bucket strategy prompts.

The local search proxy (forwards /search/* to oro-ai--oro-search-server-serve)
runs as a subprocess inside the container so the agent's ProxyClient at
localhost:8080 works unmodified.

Output:
  /data/eval_kto_opd_renderers_sonnet_v1/rollouts.jsonl
  /data/eval_kto_opd_renderers_sonnet_v1/scored.json

Usage:
    modal run --detach sft-2026-05-21/scripts/52_eval_kto_7tool_harness.py::full
"""
from __future__ import annotations

import modal

app = modal.App("oro-eval-kto-7tool")

BACKBONE = "Qwen/Qwen3-4B"
REJECTION_V2_PATH = "/ckpt/sft_rejection_v2_from_alpaca_v2_full/final"
OPD_RENDERERS_PATH = "/ckpt/sft_opd_renderers_from_rejection_v2/final"
KTO_PATH = "/ckpt/sft_kto_opd_renderers_sonnet_v4/final"

# Datasets — we pass these in as bytes from local at upload time
HOLDOUT_PATH = "/data/v3_holdout_problems.json"
SHOPPINGBENCH_PATH = "/data/shoppingbench_750_non_knowledge.jsonl"

# OUT subdir is passed as a kwarg now; default /data/eval_kto_opd_renderers_sonnet_v3

MODAL_SEARCH_URL = "https://oro-ai--oro-search-server-serve.modal.run"

# ShoppingBench fork at HEAD ba608ce (our 7-tool surface)
SHOPPINGBENCH_LOCAL = "/path/to/ShoppingBench-oro"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "peft>=0.14.0",
        "accelerate>=1.2.0",
        "hf_transfer>=0.1.6",
        "requests>=2.31.0",
        "ujson",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(SHOPPINGBENCH_LOCAL + "/src", remote_path="/shoppingbench/src")
)

data_volume = modal.Volume.from_name("oro-distill-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("oro-sft-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=5 * 60,
)
def upload_holdout(holdout_bytes: bytes, sb_bytes: bytes):
    import pathlib
    pathlib.Path(HOLDOUT_PATH).write_bytes(holdout_bytes)
    pathlib.Path(SHOPPINGBENCH_PATH).write_bytes(sb_bytes)
    data_volume.commit()
    print(f"uploaded holdout ({len(holdout_bytes)}B) + sb ({len(sb_bytes)}B)")


@app.function(
    image=image,
    gpu="H200",  # H200 has separate pool from H100 — dodges workspace H100 preemption
    timeout=4 * 60 * 60,  # 4hr — 75 problems × ~150s avg ≈ 3hr, +buffer
    volumes={"/data": data_volume, "/ckpt": ckpt_volume},
)
def run_eval(n_problems: int | None = None, temperature: float = 0.0,
             skip_kto: bool = False, skip_all_adapters: bool = False,
             only_rejection_v2: bool = False, only_opd_renderers: bool = False,
             out_subdir: str = "eval_kto_opd_renderers_sonnet_v3",
             commit_every: int = 3,
             lora_path: str = ""):  # override KTO_PATH per-call (e.g., drgrpo checkpoints)
    """Generate rollouts for the 75-problem held-out set using the KTO checkpoint
    + Seth's 7-tool agent harness. Writes raw rollouts to /data/{out_subdir}/rollouts.jsonl."""
    import json
    import os
    import sys
    import subprocess
    import time
    import pathlib
    from concurrent.futures import ThreadPoolExecutor

    # ── 1. Start the local search proxy as a subprocess ──────────────────────
    proxy_code = f"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import requests, sys

UPSTREAM = "{MODAL_SEARCH_URL}"
PORT = 8080
session = requests.Session()

class H(BaseHTTPRequestHandler):
    def _fwd(self, method):
        try:
            p = urlparse(self.path)
            path = p.path[len('/search'):] if p.path.startswith('/search/') else p.path
            url = f"{{UPSTREAM}}{{path}}" + (f"?{{p.query}}" if p.query else "")
            cl = int(self.headers.get('Content-Length', 0) or 0)
            body = self.rfile.read(cl) if cl else None
            r = session.request(method, url, data=body, timeout=60)
            self.send_response(r.status_code)
            self.send_header('Content-Type', r.headers.get('content-type', 'application/json'))
            self.send_header('Content-Length', str(len(r.content)))
            self.end_headers()
            self.wfile.write(r.content)
        except Exception as e:
            err = (f'{{{{"error":"{{type(e).__name__}}: {{e}}"}}}}').encode()
            self.send_response(502); self.send_header('Content-Length', str(len(err))); self.end_headers()
            self.wfile.write(err)
    def do_GET(self): self._fwd('GET')
    def do_POST(self): self._fwd('POST')
    def log_message(self, *a, **kw): pass

s = ThreadingHTTPServer(('127.0.0.1', PORT), H)
print(f"proxy on 127.0.0.1:{{PORT}} -> {{UPSTREAM}}", flush=True)
s.serve_forever()
"""
    proxy_path = "/tmp/search_proxy.py"
    pathlib.Path(proxy_path).write_text(proxy_code)
    proxy_log = open("/tmp/proxy.log", "w")
    proxy_proc = subprocess.Popen(["python3", proxy_path], stdout=proxy_log, stderr=proxy_log)
    print(f"started proxy pid={proxy_proc.pid}")
    time.sleep(2)
    # Verify proxy
    import requests as _req
    try:
        r = _req.get("http://127.0.0.1:8080/search/find_product?q=test", timeout=10)
        print(f"proxy health: {r.status_code}")
    except Exception as e:
        print(f"proxy health FAILED: {e}")

    # ── 2. Set up env for the ShoppingBench agent ────────────────────────────
    os.environ["SANDBOX_PROXY_URL"] = "http://127.0.0.1:8080"
    os.environ["SANDBOX_MODEL"] = "qwen3-4b-kto"
    sys.path.insert(0, "/shoppingbench")
    # Clear any cached agent modules
    for m in list(sys.modules):
        if m.startswith("src.agent"):
            del sys.modules[m]
    from src.agent import agent as agent_mod  # type: ignore
    print(f"loaded agent_mod from {agent_mod.__file__}")

    # ── 3. Load model: Qwen3-4B + rejection_v2 + opd_renderers + KTO ─────────
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading tokenizer + base {BACKBONE}...")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if skip_all_adapters:
        print("SKIP ALL ADAPTERS — base Qwen3-4B only (diagnostic: is generate() broken?)")
    elif only_rejection_v2:
        print(f"merging ONLY rejection_v2 from {REJECTION_V2_PATH}...")
        model = PeftModel.from_pretrained(model, REJECTION_V2_PATH)
        model = model.merge_and_unload()
    elif only_opd_renderers:
        print(f"loading ONLY opd_renderers from {OPD_RENDERERS_PATH} (directly on base)...")
        model = PeftModel.from_pretrained(model, OPD_RENDERERS_PATH)
        model = model.merge_and_unload()
    else:
        # CORRECTED stack (2026-05-28): opd_renderers was trained from base
        # directly, not from rejection_v2. Stacking rejection_v2 underneath
        # garbles the merged weights.
        print(f"merging opd_renderers from {OPD_RENDERERS_PATH}...")
        model = PeftModel.from_pretrained(model, OPD_RENDERERS_PATH)
        model = model.merge_and_unload()
        if not skip_kto:
            active_path = lora_path or KTO_PATH
            print(f"loading active LoRA adapter from {active_path}...")
            model = PeftModel.from_pretrained(model, active_path)
        else:
            print("SKIPPING active LoRA adapter (diagnostic: opd_renderers alone)")
    model.eval()
    print(f"model ready, params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ── 4. Patch agent_mod.inference for local generation ───────────────────
    # Alias the loaded model so the inner `model=...` kwarg (model NAME string
    # from agent_mod.agent_main) doesn't shadow the actual model object.
    loaded_model = model
    @torch.no_grad()
    def inf(model=None, messages=None, temperature=0.0, **kwargs):
        try:
            chat = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
            input_ids = chat["input_ids"].to(loaded_model.device)
            attention_mask = chat["attention_mask"].to(loaded_model.device)
            do_sample = temperature > 0
            out = loaded_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                temperature=max(temperature, 1e-5),
                do_sample=do_sample,
                top_p=0.95 if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_ids = out[0, input_ids.shape[1]:].tolist()
            content = tokenizer.decode(gen_ids, skip_special_tokens=True)
            # Debug: print first 400 chars of every generation to diagnose format issues
            print(f"  [inf] gen_len={len(gen_ids)}, content[:400]={content[:400]!r}", flush=True)
            return {"content": content, "reasoning_content": "", "tool_calls": None}
        except Exception as e:
            import traceback
            print(f"  [inference err] {type(e).__name__}: {e}")
            print(f"  loaded_model type: {type(loaded_model).__name__}")
            traceback.print_exc()
            return {"content": "", "reasoning_content": "", "tool_calls": None}

    agent_mod.inference = inf
    print("patched agent_mod.inference for local generate")

    # ── 5. Load 75-problem held-out + run ────────────────────────────────────
    holdout_payload = json.load(open(HOLDOUT_PATH))
    holdout_queries = set(q.strip().lower() for q in holdout_payload["holdout_queries"])
    sb_rows = []
    with open(SHOPPINGBENCH_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["problem_query"].strip().lower() in holdout_queries:
                sb_rows.append({
                    "query": r["problem_query"],
                    "category": r["bucket"],
                    "reward": r["reward"],
                    "voucher": r.get("voucher"),
                    "difficulty": r.get("difficulty"),
                })
    if n_problems:
        sb_rows = sb_rows[:n_problems]
    print(f"target: {len(sb_rows)} held-out problems")

    out_dir = f"/data/{out_subdir}"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    rollouts_path = pathlib.Path(out_dir) / "rollouts.jsonl"

    # ── Resume logic: read existing rollouts, skip done queries ─────────────
    done_queries = set()
    if rollouts_path.exists():
        with open(rollouts_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_queries.add(r["query"].strip().lower())
                except Exception:
                    pass
        print(f"resume: {len(done_queries)} problems already done, will skip")

    pending = [p for p in sb_rows if p["query"].strip().lower() not in done_queries]
    if not pending:
        print(f"✓ All {len(sb_rows)} problems already complete — nothing to do")
        proxy_proc.terminate()
        return {"status": "ok", "rollouts_path": str(rollouts_path),
                "n_done": len(done_queries), "n_target": len(sb_rows)}
    print(f"running {len(pending)} pending problems")

    t0 = time.time()
    # Append mode so resumed runs don't clobber prior rollouts
    with open(rollouts_path, "a") as fout:
        for i, problem in enumerate(pending):
            tp = time.time()
            try:
                steps = agent_mod.agent_main({"query": problem["query"]})
                err = None
            except Exception as e:
                steps = []
                err = f"{type(e).__name__}: {e}"
            elapsed = time.time() - tp
            row = {
                "variant": "kto-opd-renderers-sonnet-v1",
                "query": problem["query"],
                "category": problem["category"],
                "difficulty": problem.get("difficulty"),
                "reward": problem["reward"],
                "voucher": problem.get("voucher"),
                "steps": steps,
                "n_steps": len(steps),
                "elapsed_s": round(elapsed, 1),
                "error": err,
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            cat_summary = f"cat={problem['category']} steps={len(steps)} t={elapsed:.0f}s"
            running_eta = (time.time() - t0) / (i + 1) * (len(pending) - i - 1)
            done_overall = len(done_queries) + i + 1
            print(f"  [{done_overall}/{len(sb_rows)}] {cat_summary} eta={running_eta/60:.1f}min err={err or ''}",
                  flush=True)
            # Periodic commit so partial results survive timeout / cancel
            if (i + 1) % commit_every == 0:
                data_volume.commit()
                print(f"  ✓ committed {done_overall}/{len(sb_rows)} rollouts to volume", flush=True)

    data_volume.commit()
    print(f"\nrollouts saved → {rollouts_path}")
    proxy_proc.terminate()
    return {"status": "ok", "rollouts_path": str(rollouts_path), "n": len(sb_rows)}


@app.local_entrypoint()
def upload():
    import pathlib
    repo = pathlib.Path("~/Documents/ORO AI/Engineering/oro-post-train")
    h = repo / "data/filtration/v3_holdout_problems.json"
    sb = repo / "data/filtration/shoppingbench_750_non_knowledge.jsonl"
    print(f"uploading {h.name} + {sb.name}...")
    upload_holdout.remote(h.read_bytes(), sb.read_bytes())


@app.local_entrypoint()
def smoke():
    """3 problems for sanity (with KTO)."""
    print(run_eval.remote(n_problems=3))


@app.local_entrypoint()
def smoke_no_kto():
    """3 problems WITHOUT KTO — diagnostic to verify LoRA stack."""
    print(run_eval.remote(n_problems=3, skip_kto=True))


@app.local_entrypoint()
def full():
    """Full 75-problem held-out eval (with KTO)."""
    print(run_eval.remote(n_problems=None, out_subdir="eval_kto_opd_renderers_sonnet_v4"))


@app.local_entrypoint()
def full_drgrpo_step20():
    """Eval Seth's drgrpo_v17_vllm_rollouts step_0020 on the 75-set."""
    print(run_eval.remote(
        n_problems=None,
        lora_path="/ckpt/drgrpo_v17_vllm_rollouts/step_0020/policy",
        out_subdir="eval_drgrpo_v17_step0020",
    ))


@app.local_entrypoint()
def smoke_drgrpo_step20():
    """3-problem smoke of Seth's drgrpo step_0020 — verify it loads cleanly."""
    print(run_eval.remote(
        n_problems=3,
        lora_path="/ckpt/drgrpo_v17_vllm_rollouts/step_0020/policy",
        out_subdir="eval_drgrpo_v17_step0020_smoke",
    ))


@app.local_entrypoint()
def full_no_kto():
    """Full 75-problem eval WITHOUT KTO — should reproduce Seth's opd_renderers 40%."""
    print(run_eval.remote(n_problems=None, only_opd_renderers=True,
                          out_subdir="eval_opd_renderers_baseline"))


@app.local_entrypoint()
def smoke_base_only():
    """3 problems with NO adapters at all — diagnostic for generate() pipeline."""
    print(run_eval.remote(n_problems=3, skip_all_adapters=True))


@app.local_entrypoint()
def smoke_rejection_only():
    """3 problems with ONLY rejection_v2 — diagnostic for stacking vs base adapter."""
    print(run_eval.remote(n_problems=3, only_rejection_v2=True))


@app.local_entrypoint()
def smoke_opd_only():
    """3 problems with ONLY opd_renderers on base — diagnostic for whether
    opd_renderers was actually trained from base, not from rejection_v2."""
    print(run_eval.remote(n_problems=3, only_opd_renderers=True))
