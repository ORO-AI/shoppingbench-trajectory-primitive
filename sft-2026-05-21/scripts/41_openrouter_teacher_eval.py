"""Teacher-model bake-off via OpenRouter.

Same harness skeleton as 40_native_harness_nudges_eval.py — same TOOLS,
SYSTEM_PROMPT, search-server dispatch, bucket-aware ASR scoring, MAX_STEPS=8 —
but swaps the local vLLM/Qwen3 backend for OpenRouter's OpenAI-compatible
chat-completions endpoint so we can run frontier models end-to-end.

The native renderer + prefill nudge are gone. OpenRouter returns OpenAI-format
tool_calls directly, and we enforce tool use via `tool_choice="required"` until
`recommend_product` fires. Nudge 2 (force-commit after 3 find_product calls)
is kept because it's a prompt-layer signal, not a format hack.

Usage:
    # Smoke (1 model, 3 problems, local-ish run via Modal CPU)
    modal run scripts/41_openrouter_teacher_eval.py::smoke

    # One model, full 75-problem holdout
    modal run --detach scripts/41_openrouter_teacher_eval.py::single --model openai/gpt-4.1

    # Full bake-off, all models in parallel
    modal run --detach scripts/41_openrouter_teacher_eval.py::bakeoff
"""
from __future__ import annotations

import modal

app = modal.App("oro-teacher-bakeoff")

SEARCH_URL = "https://oro-ai--oro-search-server-serve.modal.run"
MAX_STEPS = 8
HOLDOUT_PATH = "/data/v3_holdout_problems.json"
SHOPPINGBENCH_PATH = "/data/shoppingbench_750_non_knowledge.jsonl"
OUT_DIR = "/data/teachers"

MODELS = [
    "openai/gpt-5",
    "openai/gpt-4.1",
    "openai/gpt-chat-latest",
    "anthropic/claude-opus-4.7",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-2.5-pro",
    "moonshotai/kimi-k2",
    "qwen/qwen3-235b-a22b",
    "deepseek/deepseek-chat",
]

PER_MODEL_USD_CAP = 50.0

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "openai>=1.40.0",
        "requests>=2.31.0",
        "tenacity>=8.2.0",
    )
)

data_volume = modal.Volume.from_name("oro-distill-data", create_if_missing=True)
openrouter_secret = modal.Secret.from_name("oro-openrouter-api-key")

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
    {
        "type": "function",
        "function": {
            "name": "find_product",
            "description": "Search the ShoppingBench product catalog.",
            "parameters": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "page": {"type": "integer"},
                    "price": {"type": "string"},
                    "q": {"type": "string"},
                    "query": {"type": "string"},
                    "service": {"type": "string"},
                    "shop_id": {"type": "string"},
                    "sort": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_product_information",
            "description": "Inspect product details for one or more product IDs.",
            "parameters": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "product_id": {"type": "string"},
                    "product_ids": {
                        "items": {"type": "string"},
                        "type": "array",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_product",
            "description": "Submit final ShoppingBench product recommendation IDs.",
            "parameters": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "product_id": {"type": "string"},
                    "product_ids": {
                        "items": {"type": "string"},
                        "type": "array",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminate",
            "description": "End the shopping episode after final recommendation.",
            "parameters": {
                "type": "object",
                "additionalProperties": True,
                "properties": {},
            },
        },
    },
]

ANTHROPIC_PROVIDER_PREFS = {
    "order": ["Anthropic"],
    "ignore": ["Bedrock", "Google", "Azure"],
    "allow_fallbacks": False,
}


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
            r = session.get(
                f"{SEARCH_URL}/find_product",
                params={
                    "q": args.get("q") or args.get("query") or "",
                    "page": str(args.get("page", 1)),
                    "shop_id": args.get("shop_id", ""),
                    "price": args.get("price", ""),
                    "sort": args.get("sort", ""),
                    "service": args.get("service", ""),
                },
                timeout=15,
            )
            body = r.json() if r.ok else {"error": f"http_{r.status_code}"}
            return json.dumps(_inject_id_into_titles(body))[:4000]
        except Exception as e:
            return json.dumps({"error": str(e)[:200]})
    if name == "view_product_information":
        ids = args.get("product_ids", args.get("product_id", ""))
        if isinstance(ids, list):
            ids = ",".join(str(x) for x in ids if str(x).strip())
        try:
            r = session.get(
                f"{SEARCH_URL}/view_product_information",
                params={"product_ids": ids},
                timeout=15,
            )
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
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    data_volume.commit()


@app.function(
    image=image,
    secrets=[openrouter_secret],
    volumes={"/data": data_volume},
    timeout=4 * 60 * 60,
    max_containers=20,
)
def evaluate_model(
    model: str,
    max_problems: int = 75,
    temperature: float = 0.7,
    max_turns: int = MAX_STEPS,
    per_model_usd_cap: float = PER_MODEL_USD_CAP,
):
    import json
    import os
    import pathlib
    import re
    import time

    import requests
    from openai import OpenAI
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", model)
    out_path = f"{OUT_DIR}/eval_{slug}.json"
    print(f"\n=== Teacher eval: {model} → {out_path} ===", flush=True)
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    api_key = os.environ["OPENROUTER_API_KEY"]
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/shardi-b/oro-post-train",
            "X-Title": "ORO teacher bake-off",
        },
    )

    session = requests.Session()
    try:
        r = session.get(f"{SEARCH_URL}/health", timeout=10)
        print(f"Search server /health: {r.status_code} {r.text[:120]}", flush=True)
    except Exception as e:
        print(f"WARN: search server probe failed: {e}", flush=True)

    with open(HOLDOUT_PATH) as f:
        holdout_payload = json.load(f)
    holdout_queries = {q.strip().lower() for q in holdout_payload["holdout_queries"]}

    sb_rows = []
    with open(SHOPPINGBENCH_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row["problem_query"].strip().lower() in holdout_queries:
                sb_rows.append(row)
    sb_rows = sb_rows[:max_problems]
    print(f"Eval rows: {len(sb_rows)}", flush=True)

    is_anthropic = model.startswith("anthropic/")
    is_gpt5 = model.startswith("openai/gpt-5")
    extra_body_base = {"usage": {"include": True}}
    if is_anthropic:
        extra_body_base["provider"] = ANTHROPIC_PROVIDER_PREFS
    if is_gpt5:
        extra_body_base["reasoning"] = {"effort": "low"}

    class HardSchemaReject(Exception):
        """4xx from provider — schema/format incompatible. Skip this model."""

    supports_required = {"value": True}  # mutable so the closure can flip it

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((requests.RequestException, TimeoutError, ConnectionError)),
        reraise=True,
    )
    def call_model(messages, tool_choice):
        effective_tool_choice = tool_choice
        if tool_choice == "required" and not supports_required["value"]:
            effective_tool_choice = "auto"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice=effective_tool_choice,
                temperature=temperature,
                max_tokens=2048,
                extra_body=extra_body_base,
            )
        except Exception as e:
            msg = str(e)
            if "tool_choice" in msg and ("404" in msg or "No endpoints" in msg):
                # Provider doesn't support tool_choice=required — fall back to "auto" for the rest of the run.
                if supports_required["value"]:
                    print(f"  [tool_choice fallback] {model} → auto", flush=True)
                    supports_required["value"] = False
                if effective_tool_choice != "auto":
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=temperature,
                        max_tokens=2048,
                        extra_body=extra_body_base,
                    )
            if any(s in msg for s in ("400", "schema", "Invalid", "is not supported")):
                raise HardSchemaReject(msg) from e
            raise
        return resp

    def extract_cost(resp) -> float:
        try:
            usage = getattr(resp, "usage", None)
            if usage is None:
                return 0.0
            cost = getattr(usage, "cost", None)
            if cost is None and hasattr(usage, "model_dump"):
                cost = usage.model_dump().get("cost")
            return float(cost) if cost is not None else 0.0
        except Exception:
            return 0.0

    results = []
    total_cost = 0.0
    early_exit = None
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
        last_step = 0
        error_reason = None
        problem_cost = 0.0

        for step in range(max_turns):
            last_step = step + 1
            tool_choice = "auto" if recommend_seen else "required"
            try:
                resp = call_model(messages, tool_choice)
            except HardSchemaReject as e:
                error_reason = f"hard_reject: {str(e)[:200]}"
                print(f"  [{idx}] HARD REJECT: {e}", flush=True)
                early_exit = "schema_reject"
                break
            except Exception as e:
                error_reason = f"api_error: {str(e)[:200]}"
                print(f"  [{idx}] api error step={step}: {e}", flush=True)
                break

            problem_cost += extract_cost(resp)

            msg = resp.choices[0].message
            content = msg.content or ""
            raw_tcs = msg.tool_calls or []

            assistant_msg = {"role": "assistant", "content": content}
            if raw_tcs:
                assistant_msg["tool_calls"] = []
                for tc in raw_tcs:
                    name = tc.function.name if tc.function else None
                    args = tc.function.arguments if tc.function else "{}"
                    assistant_msg["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
            messages.append(assistant_msg)

            if not raw_tcs:
                if not recommend_seen:
                    messages.append({
                        "role": "user",
                        "content": (
                            "You must call a tool. If you have enough evidence, call "
                            "recommend_product with the appropriate product_ids and "
                            "then terminate. Otherwise call find_product."
                        ),
                    })
                    continue
                break

            for tc in raw_tcs:
                name = tc.function.name if tc.function else None
                if name == "find_product":
                    find_count += 1
                elif name == "view_product_information":
                    view_count += 1
                elif name == "recommend_product":
                    recommend_seen = True
                elif name == "terminate":
                    terminated = True

            for tc in raw_tcs:
                name = tc.function.name if tc.function else None
                args = tc.function.arguments if tc.function else "{}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": dispatch_tool(name, args, session),
                })

            if terminated:
                break

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
                    ),
                })

        rec_ids = []
        for m in messages:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name")
                if fn == "recommend_product":
                    args = (tc.get("function") or {}).get("arguments", {})
                    rec_ids = extract_recommend_ids(args)

        score_ok = score_problem(rec_ids, expected_ids, bucket)
        total_cost += problem_cost

        results.append({
            "bucket": bucket,
            "query": query[:200],
            "expected_ids": expected_ids,
            "rec_ids": rec_ids,
            "any_match": bool(set(rec_ids) & set(expected_ids)),
            "score_ok": score_ok,
            "find_count": find_count,
            "view_count": view_count,
            "recommend_emitted": recommend_seen,
            "terminated": terminated,
            "turns": last_step,
            "recommend_nudged": recommend_nudged,
            "cost_usd": round(problem_cost, 5),
            "error_reason": error_reason,
        })

        if (idx + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1)
            eta = (len(sb_rows) - idx - 1) / max(rate, 0.001)
            n_ok = sum(r["score_ok"] for r in results)
            n_rec = sum(r["recommend_emitted"] for r in results)
            print(
                f"  {idx+1}/{len(sb_rows)}  ASR={n_ok}/{idx+1}={100*n_ok/(idx+1):.1f}%  "
                f"rec_emit={n_rec}  cost=${total_cost:.2f}  "
                f"({rate:.2f}/s, ETA {eta:.0f}s)",
                flush=True,
            )

        if total_cost >= per_model_usd_cap:
            print(f"  COST CAP HIT: ${total_cost:.2f} >= ${per_model_usd_cap}", flush=True)
            early_exit = "cost_cap"
            break
        if early_exit:
            break

    n = len(results)
    summary = {
        "model": model,
        "n": n,
        "n_target": len(sb_rows),
        "early_exit": early_exit,
        "overall": {
            "ASR_pct": round(100 * sum(r["score_ok"] for r in results) / max(n, 1), 2),
            "any_match_pct": round(100 * sum(r["any_match"] for r in results) / max(n, 1), 2),
            "recommend_emitted_pct": round(
                100 * sum(r["recommend_emitted"] for r in results) / max(n, 1), 2
            ),
            "terminated_pct": round(100 * sum(r["terminated"] for r in results) / max(n, 1), 2),
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
        "harness": "openrouter native tool_calls + tool_choice=required + force-commit nudge",
        "total_cost_usd": round(total_cost, 4),
        "per_model_usd_cap": per_model_usd_cap,
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

    out = {
        "summary": summary,
        "per_problem": results,
        "search_url": SEARCH_URL,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    data_volume.commit()
    print("=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _upload_local_artifacts():
    import pathlib
    here = pathlib.Path(__file__).resolve()
    for candidate in (
        here.parent.parent / "data" / "filtration",
        here.parent.parent.parent / "data" / "filtration",
    ):
        holdout_p = candidate / "v3_holdout_problems.json"
        sb_p = candidate / "shoppingbench_750_non_knowledge.jsonl"
        if holdout_p.exists() and sb_p.exists():
            upload_artifacts.remote(holdout_p.read_bytes(), sb_p.read_bytes())
            return
    raise FileNotFoundError(
        "Could not find v3_holdout_problems.json + shoppingbench_750_non_knowledge.jsonl "
        f"under {here.parent.parent} or {here.parent.parent.parent}"
    )


@app.local_entrypoint()
def smoke(model: str = "openai/gpt-4.1"):
    """3-problem smoke test on one model — sanity-check the loop end-to-end."""
    _upload_local_artifacts()
    summary = evaluate_model.remote(model=model, max_problems=3)
    print("\nSmoke summary:")
    import json
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def single(model: str):
    """Full 75-problem run on a single model."""
    _upload_local_artifacts()
    summary = evaluate_model.remote(model=model)
    print("\nSingle-model summary:")
    import json
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def bakeoff():
    """Full 75-problem run across all candidate teachers, in parallel."""
    _upload_local_artifacts()
    print(f"Fanning out {len(MODELS)} models in parallel...")
    summaries = list(evaluate_model.map(MODELS))
    print("\n=== BAKE-OFF LEADERBOARD ===")
    ranked = sorted(summaries, key=lambda s: s["overall"]["ASR_pct"], reverse=True)
    print(f"{'rank':>4}  {'model':<35}  {'ASR':>6}  {'rec_emit':>8}  {'cost':>8}  {'exit':>14}")
    for i, s in enumerate(ranked, 1):
        print(
            f"{i:>4}  {s['model']:<35}  "
            f"{s['overall']['ASR_pct']:>5.1f}%  "
            f"{s['overall']['recommend_emitted_pct']:>7.1f}%  "
            f"${s['total_cost_usd']:>6.2f}  "
            f"{str(s.get('early_exit', '')):>14}"
        )
