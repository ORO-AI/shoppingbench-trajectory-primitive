"""Fast vLLM eval client: threaded ShoppingBench agent → vLLM server.

Runs the SAME 7-tool agent harness as the working HF eval (script 52), but
delegates generation to the vLLM server (script 56) over HTTP. N worker threads
each run `agent_main` for one problem; the server's continuous batching
coalesces their concurrent /v1/completions requests. Target: ~20-30 min on the
75-set vs ~3 hr for HF in-process.

Parity with script 52 (the canonical scored harness):
  * prompt = the SAME `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`
  * skip_special_tokens=True, max_tokens=2048, temperature default 0.0 (greedy)
  * returns {content, reasoning_content:"", tool_calls:None} — agent_main parses
    tool calls out of `content` exactly as in the HF path.
The only added step is `_strip_qwen_specials` (Seth's sanitizer) for the rare
case the model emits literal <|im_start|>/orphan </think> as TEXT (which
skip_special_tokens does not catch). On clean output it is a no-op, so parity
with 52 holds.

This function needs NO GPU — it is an HTTP client + the tokenizer + agent_main.
The GPU lives in the script-56 server.

Output: /data/{out_subdir}/rollouts.jsonl  (schema-identical to script 52, so
score with sft-2026-05-21/scripts/53_score_rollouts.py unchanged).

Usage:
    modal deploy sft-2026-05-21/scripts/56_serve_eval_vllm.py    # server first
    modal run sft-2026-05-21/scripts/57_eval_vllm_client.py::smoke         # 3 problems, opd_kto_v4
    modal run sft-2026-05-21/scripts/57_eval_vllm_client.py::parity_v4     # 3 problems, opd alone (cmp vs 52)
    modal run --detach sft-2026-05-21/scripts/57_eval_vllm_client.py::full_v4
    modal run --detach sft-2026-05-21/scripts/57_eval_vllm_client.py::full_v3
"""
from __future__ import annotations

import modal

app = modal.App("oro-eval-vllm-client")

BACKBONE = "Qwen/Qwen3-4B"
SERVER_URL = "https://oro-ai--oro-vllm-eval-serve.modal.run"
MODAL_SEARCH_URL = "https://oro-ai--oro-search-server-serve.modal.run"

HOLDOUT_PATH = "/data/v3_holdout_problems.json"
SHOPPINGBENCH_PATH = "/data/shoppingbench_750_non_knowledge.jsonl"

SHOPPINGBENCH_LOCAL = "/path/to/ShoppingBench-oro"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "transformers==4.53.1",   # tokenizer / apply_chat_template parity with script 52
        "jinja2",                  # required by apply_chat_template (no torch in this image)
        "requests>=2.31.0",
        "ujson",
    )
    .add_local_dir(SHOPPINGBENCH_LOCAL + "/src", remote_path="/shoppingbench/src")
)

data_volume = modal.Volume.from_name("oro-distill-data", create_if_missing=True)

# Seth's sanitizer (src/drgrpo/rollout_harness.py). No-op on clean output.
_QWEN_SPECIALS_RE = None


def _strip_qwen_specials(text: str) -> str:
    import re
    global _QWEN_SPECIALS_RE
    if _QWEN_SPECIALS_RE is None:
        _QWEN_SPECIALS_RE = re.compile(r"<\|im_start\|>(?:user|assistant|system|tool)?\s*|<\|im_end\|>\s*")
    text = _QWEN_SPECIALS_RE.sub("", text)
    text = re.sub(r"\A\s*(?:<think>.*?</think>|</think>)\s*", "", text, count=1, flags=re.DOTALL)
    return text


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=2 * 60 * 60,   # client has no GPU; server holds the model
    cpu=4.0,
    memory=8192,
)
def run_eval(
    adapter: str = "opd_kto_v4",
    out_subdir: str = "eval_kto_v4_vllm",
    n_problems: int | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    concurrency: int = 32,
    commit_every: int = 10,
    server_url: str = SERVER_URL,
) -> dict:
    import json
    import os
    import sys
    import subprocess
    import time
    import pathlib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    import requests

    # ── 1. Local search proxy (identical to script 52) ───────────────────────
    proxy_code = f"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import requests

UPSTREAM = "{MODAL_SEARCH_URL}"
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

ThreadingHTTPServer(('127.0.0.1', 8080), H).serve_forever()
"""
    pathlib.Path("/tmp/search_proxy.py").write_text(proxy_code)
    proxy_log = open("/tmp/proxy.log", "w")
    proxy_proc = subprocess.Popen(["python3", "/tmp/search_proxy.py"], stdout=proxy_log, stderr=proxy_log)
    time.sleep(2)
    try:
        print(f"proxy health: {requests.get('http://127.0.0.1:8080/search/find_product?q=test', timeout=10).status_code}")
    except Exception as e:
        print(f"proxy health FAILED: {e}")

    # ── 2. Load tokenizer (CPU) for chat-template parity with script 52 ──────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 3. ShoppingBench agent + vLLM-backed inference patch ─────────────────
    os.environ["SANDBOX_PROXY_URL"] = "http://127.0.0.1:8080"
    os.environ["SANDBOX_MODEL"] = adapter
    sys.path.insert(0, "/shoppingbench")
    for m in list(sys.modules):
        if m.startswith("src.agent"):
            del sys.modules[m]
    from src.agent import agent as agent_mod  # type: ignore
    print(f"loaded agent_mod from {agent_mod.__file__}")

    completions_url = server_url.rstrip("/") + "/v1/completions"
    http = requests.Session()
    # Match the connection pool to the worker count so the session doesn't
    # discard/recreate connections under concurrency (the "pool is full" warning).
    _adapter = requests.adapters.HTTPAdapter(pool_connections=concurrency, pool_maxsize=concurrency)
    http.mount("https://", _adapter)
    http.mount("http://", _adapter)

    def inf(model=None, messages=None, temperature=temperature, **kwargs):
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        payload = {
            "model": adapter,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.0),
            "top_p": 0.95 if temperature > 0 else 1.0,
            "skip_special_tokens": True,
        }
        # Retry-with-backoff so a transient timeout/5xx under concurrency never
        # leaks an empty turn to the agent (which would terminate the rollout
        # early and deflate the score vs HF, which never times out). Makes the
        # vLLM client as "patient" as in-process model.generate().
        last_err = None
        for attempt in range(5):
            try:
                r = http.post(completions_url, json=payload, timeout=600)
                r.raise_for_status()
                content = _strip_qwen_specials(r.json()["choices"][0]["text"])
                return {"content": content, "reasoning_content": "", "tool_calls": None}
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 20))
        print(f"  [inference err after retries] {type(last_err).__name__}: {last_err}", flush=True)
        return {"content": "", "reasoning_content": "", "tool_calls": None}

    agent_mod.inference = inf
    print(f"patched agent_mod.inference → {completions_url} (model={adapter})")

    # ── 4. Held-out problems + resume (identical schema to script 52) ────────
    holdout = json.load(open(HOLDOUT_PATH))
    holdout_queries = {q.strip().lower() for q in holdout["holdout_queries"]}
    sb_rows = []
    with open(SHOPPINGBENCH_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["problem_query"].strip().lower() in holdout_queries:
                sb_rows.append({
                    "query": r["problem_query"], "category": r["bucket"],
                    "reward": r["reward"], "voucher": r.get("voucher"),
                    "difficulty": r.get("difficulty"),
                })
    if n_problems:
        sb_rows = sb_rows[:n_problems]

    out_dir = f"/data/{out_subdir}"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    rollouts_path = pathlib.Path(out_dir) / "rollouts.jsonl"

    # Resume on GOOD rows only. A rollout that produced no steps (e.g. an infra
    # failure caught inside inf → empty content → agent terminates at 0 steps)
    # must NOT count as done — otherwise transient errors silently under-count
    # the eval. Rewrite the file keeping only good rows so retries don't
    # duplicate and 53_score_rollouts sees one clean row per query.
    done = set()
    if rollouts_path.exists():
        good_rows = []
        for line in open(rollouts_path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("n_steps", 0) > 0 and not r.get("error"):
                good_rows.append(r)
                done.add(r["query"].strip().lower())
        with open(rollouts_path, "w") as fw:
            for r in good_rows:
                fw.write(json.dumps(r) + "\n")
    pending = [p for p in sb_rows if p["query"].strip().lower() not in done]
    print(f"target={len(sb_rows)} done={len(done)} pending={len(pending)} workers={concurrency}")
    if not pending:
        proxy_proc.terminate()
        return {"status": "ok", "n_done": len(done), "n_target": len(sb_rows)}

    # ── 5. Run problems concurrently; server batches the inference calls ─────
    file_lock = Lock()
    counter = {"n": 0}
    t0 = time.time()

    def run_one(problem):
        tp = time.time()
        try:
            steps = agent_mod.agent_main({"query": problem["query"]})
            err = None
        except Exception as e:
            steps, err = [], f"{type(e).__name__}: {e}"
        row = {
            "variant": out_subdir, "query": problem["query"], "category": problem["category"],
            "difficulty": problem.get("difficulty"), "reward": problem["reward"],
            "voucher": problem.get("voucher"), "steps": steps, "n_steps": len(steps),
            "elapsed_s": round(time.time() - tp, 1), "error": err,
        }
        with file_lock:
            with open(rollouts_path, "a") as fout:
                fout.write(json.dumps(row) + "\n")
            counter["n"] += 1
            n_done = len(done) + counter["n"]
            print(f"  [{n_done}/{len(sb_rows)}] cat={problem['category']} steps={len(steps)} "
                  f"t={row['elapsed_s']:.0f}s err={err or ''}", flush=True)
            if counter["n"] % commit_every == 0:
                data_volume.commit()
        return row

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in as_completed([ex.submit(run_one, p) for p in pending]):
            pass

    data_volume.commit()
    total = time.time() - t0
    print(f"\nrollouts → {rollouts_path}\nWall clock: {total:.0f}s ({total/60:.1f} min)")
    proxy_proc.terminate()
    return {
        "status": "ok", "rollouts_path": str(rollouts_path),
        "n_target": len(sb_rows), "wall_clock_s": round(total, 1),
        "wall_clock_min": round(total / 60, 1), "adapter": adapter,
    }


@app.local_entrypoint()
def run(adapter: str, out_subdir: str, n_problems: int = 0, concurrency: int = 16):
    """Generic launcher for the eval sweep: any registered adapter → its rollouts."""
    print(run_eval.remote(adapter=adapter, out_subdir=out_subdir,
                          n_problems=(n_problems or None), concurrency=concurrency))


@app.local_entrypoint()
def smoke():
    print(run_eval.remote(adapter="opd_kto_v4", out_subdir="eval_kto_v4_vllm_smoke",
                          n_problems=3, concurrency=3))


@app.local_entrypoint()
def parity_v4():
    """opd alone, 3 problems — directly comparable to script 52 --only_opd_renderers."""
    print(run_eval.remote(adapter="opd", out_subdir="eval_opd_vllm_parity",
                          n_problems=3, concurrency=3))


@app.local_entrypoint()
def full_v4():
    print(run_eval.remote(adapter="opd_kto_v4", out_subdir="eval_kto_v4_vllm"))


@app.local_entrypoint()
def full_v3():
    print(run_eval.remote(adapter="opd_kto_v3", out_subdir="eval_kto_v3_vllm"))


@app.local_entrypoint()
def full_opd():
    print(run_eval.remote(adapter="opd", out_subdir="eval_opd_vllm"))
