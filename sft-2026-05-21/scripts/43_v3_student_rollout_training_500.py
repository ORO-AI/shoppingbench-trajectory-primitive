"""V3 SFT student rollouts on Seth's 498 training problems.

Same harness as script 40 (native Qwen3 tool_calls + paper nudges), pointed at
the 498 training-set problems Seth used for his OPD Sonnet teacher rollouts
(s3://oro-evaluation-trajectories-<ACCOUNT_ID>/sft-research/2026-05-23/).

Output is the rejected side of DPO pairs. Teacher trajectories (in paper XML
format) live in `opd_sonnet_500.jsonl` from Seth; teacher-success mapping is
cached locally at `data/filtration/training_500_teacher_success.json`.

MAX_STEPS=8 matches what v3 was effectively trained against; we are not trying
to lift the student here, just capture its current behavior on these problems
so DPO has a real rejected-side signal.

Usage:
    modal run --detach scripts/43_v3_student_rollout_training_500.py::v3
"""
from __future__ import annotations

import modal

app = modal.App("oro-v3-student-rollout-training-500")

SEARCH_URL = "https://oro-ai--oro-search-server-serve.modal.run"
MAX_STEPS = 8
BACKBONE = "Qwen/Qwen3-4B"
SFT_PATH = "/ckpt/sft_v3"

HOLDOUT_PATH = "/data/training_500_holdout.json"
SHOPPINGBENCH_PATH = "/data/training_500_problems.jsonl"
OUT_BASE = "/data/v3_student_rollouts_training_500_base.json"
OUT_V3 = "/data/v3_student_rollouts_training_500_v3.json"

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
        "renderers==0.1.7",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

data_volume = modal.Volume.from_name("oro-distill-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("oro-distill-checkpoints", create_if_missing=True)

# Seth's converter prompt — what v3 was trained on
SYSTEM_PROMPT = (
    "You are Teacher, a shopping assistant in a synthetic data gate. "
    "Complete the user task using only the available tools and observed tool results. "
    "Do not answer with prose instead of tools. "
    "Search the catalog with find_product. "
    "When the request says verify, inspect, check, compare, SKU, variant, option, "
    "attribute, ingredient, material, description, or product information, call "
    "view_product_information before recommending. "
    "Required finalization: once you have enough evidence, call recommend_product "
    "exactly once with product_ids as a comma-separated string, then call terminate. "
    "For product tasks recommend one ID; for shop, voucher, and multi-item tasks "
    "recommend comma-separated IDs in the requested order. "
    "Do not mention private rewards, benchmark answers, split metadata, source lineage, "
    "or evaluation metadata. Prefer careful verification over speed. "
    "If candidate titles are ambiguous, inspect product information."
)

TOOLS = [
    {"type": "function", "name": "find_product",
     "description": "Search the ShoppingBench product catalog.",
     "parameters": {"type": "object", "additionalProperties": True,
                    "properties": {"page": {"type": "integer"}, "price": {"type": "string"},
                                   "q": {"type": "string"}, "query": {"type": "string"},
                                   "service": {"type": "string"}, "shop_id": {"type": "string"},
                                   "sort": {"type": "string"}}}},
    {"type": "function", "name": "view_product_information",
     "description": "Inspect product details for one or more product IDs.",
     "parameters": {"type": "object", "additionalProperties": True,
                    "properties": {"product_id": {"type": "string"},
                                   "product_ids": {"items": {"type": "string"}, "type": "array"}}}},
    {"type": "function", "name": "recommend_product",
     "description": "Submit final ShoppingBench product recommendation IDs.",
     "parameters": {"type": "object", "additionalProperties": True,
                    "properties": {"product_id": {"type": "string"},
                                   "product_ids": {"items": {"type": "string"}, "type": "array"}}}},
    {"type": "function", "name": "terminate",
     "description": "End the shopping episode after final recommendation.",
     "parameters": {"type": "object", "additionalProperties": True, "properties": {}}},
]


def _inject_id_into_titles(body):
    if isinstance(body, list):
        for p in body:
            if isinstance(p, dict):
                pid = p.get("product_id")
                title = p.get("title")
                if pid and isinstance(title, str) and not title.startswith("[product_id="):
                    p["title"] = f"[product_id={pid}] {title}"
    return body


def dispatch_tool(name, args, session):
    import json
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if name == "find_product":
        try:
            r = session.get(f"{SEARCH_URL}/find_product",
                            params={"q": args.get("q") or args.get("query") or "",
                                    "page": str(args.get("page", 1)),
                                    "shop_id": args.get("shop_id", ""),
                                    "price": args.get("price", ""),
                                    "sort": args.get("sort", ""),
                                    "service": args.get("service", "")},
                            timeout=15)
            body = r.json() if r.ok else {"error": f"http_{r.status_code}"}
            return json.dumps(_inject_id_into_titles(body))[:4000]
        except Exception as e:
            return json.dumps({"error": str(e)[:200]})
    if name == "view_product_information":
        ids = args.get("product_ids", args.get("product_id", ""))
        if isinstance(ids, list):
            ids = ",".join(str(x) for x in ids if str(x).strip())
        try:
            r = session.get(f"{SEARCH_URL}/view_product_information",
                            params={"product_ids": ids}, timeout=15)
            body = r.json() if r.ok else {"error": f"http_{r.status_code}"}
            return json.dumps(_inject_id_into_titles(body))[:4000]
        except Exception as e:
            return json.dumps({"error": str(e)[:200]})
    if name == "recommend_product":
        return json.dumps({"status": "submitted"})
    if name == "terminate":
        return json.dumps({"status": "terminated"})
    return json.dumps({"error": f"unknown tool: {name}"})


def extract_recommend_ids(args):
    import json
    import re
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return []
    if not isinstance(args, dict):
        return []
    raw = args.get("product_ids") or args.get("product_id") or ""
    out = []
    if isinstance(raw, list):
        out = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        out = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, (int, float)):
        out = [str(int(raw))]
    return [p for p in out if re.match(r"^\d{4,}$", p)]


def reward_expected_ids(reward):
    if isinstance(reward, dict) and reward.get("product_id"):
        return [str(reward["product_id"])]
    if isinstance(reward, list):
        return [str(x["product_id"]) for x in reward if isinstance(x, dict) and x.get("product_id")]
    return []


def score_problem(rec_ids, expected, bucket):
    rec, exp = set(rec_ids), set(expected)
    if not exp:
        return False
    if bucket == "product":
        return next(iter(exp)) in rec
    return exp.issubset(rec)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=5 * 60,
)
def upload_artifacts(holdout_bytes: bytes, shoppingbench_bytes: bytes):
    import pathlib
    pathlib.Path(HOLDOUT_PATH).write_bytes(holdout_bytes)
    pathlib.Path(SHOPPINGBENCH_PATH).write_bytes(shoppingbench_bytes)
    data_volume.commit()


@app.function(
    image=image,
    gpu="H100",
    timeout=4 * 60 * 60,
    volumes={"/data": data_volume, "/ckpt": ckpt_volume},
)
def evaluate(use_sft: bool = True, max_turns: int = MAX_STEPS, temperature: float = 0.7):
    import json
    import time
    import requests
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import renderers as R

    label = "v3-LoRA" if use_sft else "base-Qwen3-4B"
    out_path = OUT_V3 if use_sft else OUT_BASE
    print(f"\n=== Native+nudges eval: {label} ===", flush=True)

    session = requests.Session()
    try:
        r = session.get(f"{SEARCH_URL}/health", timeout=10)
        print(f"Search server /health: {r.status_code} {r.text[:120]}", flush=True)
    except Exception as e:
        print(f"WARN: search server probe failed: {e}", flush=True)

    with open(HOLDOUT_PATH) as f:
        holdout_payload = json.load(f)
    holdout_queries = set(q.strip().lower() for q in holdout_payload["holdout_queries"])

    sb_rows = []
    with open(SHOPPINGBENCH_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row["problem_query"].strip().lower() in holdout_queries:
                sb_rows.append(row)
    print(f"Eval rows: {len(sb_rows)}", flush=True)

    print(f"Loading {BACKBONE}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BACKBONE, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    if use_sft:
        print(f"Loading LoRA adapter from {SFT_PATH}...", flush=True)
        model = PeftModel.from_pretrained(base_model, SFT_PATH)
    else:
        print("[base eval] Skipping LoRA; using raw Qwen3-4B", flush=True)
        model = base_model
    model.eval()
    renderer = R.create_renderer(tokenizer, renderer="qwen3")
    stop_token_ids = renderer.get_stop_token_ids()
    print(f"Stop tokens: {stop_token_ids}", flush=True)

    def gen_native(messages, max_new_tokens=1024):
        """Native rendering via renderers package — what v3 was trained with."""
        rendered = renderer.render(messages, tools=TOOLS, add_generation_prompt=True)
        input_ids = list(rendered.token_ids)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        with torch.no_grad():
            out = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids if stop_token_ids else tokenizer.eos_token_id,
            )
        gen_ids = out[0, input_tensor.shape[1]:].tolist()
        while gen_ids and gen_ids[-1] in (stop_token_ids or [tokenizer.eos_token_id]):
            gen_ids = gen_ids[:-1]
        return gen_ids

    def parse_native(gen_ids):
        """Parse via renderer — gives us content + tool_calls list."""
        try:
            parsed = renderer.parse_response(gen_ids)
            content = parsed.content or ""
            tcs = parsed.tool_calls or []
        except Exception:
            content = tokenizer.decode(gen_ids, skip_special_tokens=False)
            tcs = []
        return content, tcs

    def nudge_prefill_recommend(messages):
        """Nudge 1 (paper-harness style, native-format): when the model emits no
        tool_calls, append an assistant message that's already started a
        recommend_product native tool_call, then have the model autocomplete it.

        Qwen3 native format wraps each tool call as:
            <tool_call>
            {"name": "recommend_product", "arguments": {"product_ids": "...
            </tool_call>
        """
        # Append a partial assistant message with prefilled native tool_call start
        prefill_text = (
            "<tool_call>\n"
            '{"name": "recommend_product", "arguments": {"product_ids": "'
        )
        # Render with this partial message at the end, then generate continuation
        prefilled_msgs = messages + [{"role": "assistant", "content": prefill_text}]
        rendered = renderer.render(prefilled_msgs, tools=TOOLS, add_generation_prompt=False)
        input_ids = list(rendered.token_ids)
        # Strip trailing stop tokens (renderer may add im_end after our partial)
        # so the model continues mid-message.
        while input_ids and input_ids[-1] in (stop_token_ids or [tokenizer.eos_token_id]):
            input_ids = input_ids[:-1]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        with torch.no_grad():
            out = model.generate(
                input_tensor,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids if stop_token_ids else tokenizer.eos_token_id,
            )
        gen_ids = out[0, input_tensor.shape[1]:].tolist()
        completion_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        # Combine prefill + completion as the model's assistant content
        full_text = prefill_text + completion_text
        # Try to parse it via the renderer-free path (regex) since the format is fixed
        import json
        import re
        out_tcs = []
        # Pattern: <tool_call>\n{...json...}\n</tool_call>
        for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", full_text, re.DOTALL):
            try:
                tc = json.loads(m.group(1))
                if isinstance(tc, dict) and "name" in tc:
                    out_tcs.append(tc)
            except json.JSONDecodeError:
                continue
        # If the model didn't close </tool_call>, try to salvage by appending and parsing
        if not out_tcs and "</tool_call>" not in completion_text:
            salvage = full_text + '"}}</tool_call>'
            for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", salvage, re.DOTALL):
                try:
                    tc = json.loads(m.group(1))
                    if isinstance(tc, dict) and "name" in tc:
                        out_tcs.append(tc)
                except json.JSONDecodeError:
                    continue
        return full_text, out_tcs

    results = []
    t0 = time.time()

    for idx, row in enumerate(sb_rows):
        query = row["problem_query"]
        bucket = row["bucket"]
        expected_ids = reward_expected_ids(row["reward"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        terminated = False
        find_count = 0
        view_count = 0
        recommend_seen = False
        recommend_nudged = False
        prefill_used = False
        last_step = 0

        for step in range(max_turns):
            last_step = step + 1
            try:
                gen_ids = gen_native(messages)
            except Exception as e:
                print(f"  [{idx}] gen error step={step}: {e}", flush=True)
                break

            content, tcs = parse_native(gen_ids)

            # Build assistant message in the training format
            assistant_msg = {"role": "assistant", "content": content}
            if tcs:
                assistant_msg["tool_calls"] = []
                for j, tc in enumerate(tcs):
                    name = tc.get("name") or (tc.get("function", {}) or {}).get("name")
                    args = tc.get("arguments") if "arguments" in tc \
                        else (tc.get("function", {}) or {}).get("arguments", {})
                    assistant_msg["tool_calls"].append({
                        "id": f"call_{step}_{j}",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
            messages.append(assistant_msg)

            # Nudge 1: no tool_calls → prefill native recommend_product start
            if not tcs:
                prefill_used = True
                print(f"  [{idx}] step {step}: nudge-1 prefill (no tool_calls)", flush=True)
                # Replace the placeholder assistant message with the prefilled one
                messages.pop()
                full_text, salvaged_tcs = nudge_prefill_recommend(messages)
                # Add the salvaged assistant message
                if salvaged_tcs:
                    assistant_msg = {"role": "assistant", "content": full_text}
                    assistant_msg["tool_calls"] = []
                    for j, tc in enumerate(salvaged_tcs):
                        name = tc.get("name")
                        args = tc.get("arguments", {})
                        assistant_msg["tool_calls"].append({
                            "id": f"call_{step}_nudge_{j}",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        })
                    messages.append(assistant_msg)
                    tcs = salvaged_tcs
                else:
                    # Nudge failed; stop the loop
                    break

            # Count tool calls
            for tc in tcs:
                name = tc.get("name") or (tc.get("function", {}) or {}).get("name")
                if name == "find_product":
                    find_count += 1
                elif name == "view_product_information":
                    view_count += 1
                elif name == "recommend_product":
                    args = tc.get("arguments") if "arguments" in tc \
                        else (tc.get("function", {}) or {}).get("arguments", {})
                    recommend_seen = True
                elif name == "terminate":
                    terminated = True

            # Dispatch + append tool responses
            tool_msgs = []
            for j, tc in enumerate(tcs):
                name = tc.get("name") or (tc.get("function", {}) or {}).get("name")
                args = tc.get("arguments") if "arguments" in tc \
                    else (tc.get("function", {}) or {}).get("arguments", {})
                tc_id = (assistant_msg.get("tool_calls", [{}]) or [{}])[j].get("id", f"call_{step}_{j}") \
                    if assistant_msg.get("tool_calls") and j < len(assistant_msg["tool_calls"]) \
                    else f"call_{step}_{j}"
                tool_msgs.append({
                    "role": "tool",
                    "name": name,
                    "tool_call_id": tc_id,
                    "content": dispatch_tool(name, args, session),
                })
            messages.extend(tool_msgs)

            if terminated:
                break

            # Nudge 2: 3+ find_product without recommend → force commit
            if find_count >= 3 and not recommend_seen and not recommend_nudged:
                recommend_nudged = True
                messages.append({
                    "role": "user",
                    "content": (
                        "You have searched enough. Pick ONE product_id (or the right "
                        "combination for shop/voucher tasks) from the find_product results "
                        "you've already seen that best matches the request, then call "
                        "recommend_product followed by terminate(status=\"success\"). "
                        "Do NOT call find_product again."
                    )
                })

        # Extract last recommend_product args
        rec_ids = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name")
                if fn == "recommend_product":
                    args = (tc.get("function") or {}).get("arguments", {})
                    rec_ids = extract_recommend_ids(args)

        score_ok = score_problem(rec_ids, expected_ids, bucket)

        results.append({
            "bucket": bucket,
            "query": query,  # full query (not truncated) so we can join to teacher set
            "expected_ids": expected_ids,
            "rec_ids": rec_ids,
            "any_match": bool(set(rec_ids) & set(expected_ids)),
            "score_ok": score_ok,
            "find_count": find_count,
            "view_count": view_count,
            "recommend_emitted": recommend_seen,
            "terminated": terminated,
            "turns": last_step,
            "prefill_used": prefill_used,
            "recommend_nudged": recommend_nudged,
            "messages": messages,  # full student trajectory for DPO rejected-side construction
        })

        if (idx + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1)
            eta = (len(sb_rows) - idx - 1) / max(rate, 0.001)
            n_ok = sum(r["score_ok"] for r in results)
            n_any = sum(r["any_match"] for r in results)
            n_rec = sum(r["recommend_emitted"] for r in results)
            n_pre = sum(r["prefill_used"] for r in results)
            print(f"  {idx+1}/{len(sb_rows)}  ASR={n_ok}/{idx+1}={100*n_ok/(idx+1):.1f}%  "
                  f"any_match={n_any}  rec_emit={n_rec}  prefill_used={n_pre}  "
                  f"({rate:.2f}/s, ETA {eta:.0f}s)", flush=True)

    n = len(results)
    summary = {
        "label": label,
        "n": n,
        "overall": {
            "ASR_pct": round(100 * sum(r["score_ok"] for r in results) / max(n, 1), 2),
            "any_match_pct": round(100 * sum(r["any_match"] for r in results) / max(n, 1), 2),
            "recommend_emitted_pct": round(
                100 * sum(r["recommend_emitted"] for r in results) / max(n, 1), 2
            ),
            "terminated_pct": round(100 * sum(r["terminated"] for r in results) / max(n, 1), 2),
            "prefill_used_pct": round(
                100 * sum(r["prefill_used"] for r in results) / max(n, 1), 2
            ),
            "recommend_nudged_pct": round(
                100 * sum(r["recommend_nudged"] for r in results) / max(n, 1), 2
            ),
            "mean_find_count": round(sum(r["find_count"] for r in results) / max(n, 1), 2),
            "mean_view_count": round(sum(r["view_count"] for r in results) / max(n, 1), 2),
            "mean_turns": round(sum(r["turns"] for r in results) / max(n, 1), 2),
        },
        "by_bucket": {},
        "max_turns": max_turns,
        "temperature": temperature,
        "harness": "native Qwen3 (renderers) + paper-style nudges",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    for bucket in sorted({r["bucket"] for r in results}):
        rows = [r for r in results if r["bucket"] == bucket]
        summary["by_bucket"][bucket] = {
            "n": len(rows),
            "ASR_pct": round(100 * sum(r["score_ok"] for r in rows) / len(rows), 2),
            "any_match_pct": round(100 * sum(r["any_match"] for r in rows) / len(rows), 2),
            "recommend_emitted_pct": round(
                100 * sum(r["recommend_emitted"] for r in rows) / len(rows), 2
            ),
        }

    out = {"summary": summary, "per_problem": results,  # full 498 needed for DPO pair construction
           "checkpoint": SFT_PATH if use_sft else "NONE (base model)",
           "backbone": BACKBONE,
           "search_url": SEARCH_URL}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    data_volume.commit()
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def base():
    import pathlib
    upload_artifacts.remote(
        pathlib.Path("data/filtration/training_500_holdout.json").read_bytes(),
        pathlib.Path("data/filtration/training_500_problems.jsonl").read_bytes(),
    )
    summary = evaluate.remote(use_sft=False)
    print(summary)


@app.local_entrypoint()
def v3():
    import pathlib
    upload_artifacts.remote(
        pathlib.Path("data/filtration/training_500_holdout.json").read_bytes(),
        pathlib.Path("data/filtration/training_500_problems.jsonl").read_bytes(),
    )
    summary = evaluate.remote(use_sft=True)
    print(summary)
