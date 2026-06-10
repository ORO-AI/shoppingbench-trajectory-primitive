"""Paper's XML ReAct harness over OpenRouter — apples-to-apples vs paper's 48.2%.

Combines the harness from `39_paper_harness_eval.py` (paper's SYSTEM_PROMPT,
XML <tool_call>[...]</tool_call> format, regex parser, two nudges) with the
OpenRouter backend from `41_openrouter_teacher_eval.py`. No `tools` param,
no `tool_choice`, no native tool-calls. Pure text completions; we parse XML
out of the assistant content.

Why this script exists: bake-off (script 41) measured frontier teachers on
our native-tools harness with Seth's converter prompt. The paper measured
GPT-4.1 = 48.2% on its XML harness with its own prompt. Those are not
directly comparable. This script gives us GPT-4.1, Sonnet 4.6, and Opus 4.7
on the paper's exact harness so we can tell whether the frontier ceiling is
real (~25-30%) or harness-induced (closer to 48%).

Usage:
    modal run scripts/42_openrouter_paper_harness_eval.py::smoke --model openai/gpt-4.1
    modal run --detach scripts/42_openrouter_paper_harness_eval.py::single --model anthropic/claude-sonnet-4.6
    modal run --detach scripts/42_openrouter_paper_harness_eval.py::bakeoff
"""
from __future__ import annotations

import modal

app = modal.App("oro-paper-harness-teacher-bakeoff")

SEARCH_URL = "https://oro-ai--oro-search-server-serve.modal.run"
MAX_STEPS = 30  # Paper's actual default (ShoppingBench/src/agent/run_rollout.py:16)
HOLDOUT_PATH = "/data/v3_holdout_problems.json"
SHOPPINGBENCH_PATH = "/data/shoppingbench_750_non_knowledge.jsonl"
OUT_DIR = "/data/teachers_paper_harness_30steps"

MODELS = [
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.7",
    "openai/gpt-4.1",
]

PER_MODEL_USD_CAP = 25.0

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

# Paper's actual SYSTEM_PROMPT (verbatim from
# ShoppingBench/src/agent/prompt/rollout.ablation_react.md with toolkit
# description rendered in for our 4 tools).
SYSTEM_PROMPT = """# Role
You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

# Available Tools
1. find_product(query: string, page: int = 1, shop_id: string = "", price: string = "", sort: string = "", service: string = ""): Search the product catalog. Returns matching products with product_id and title.
2. view_product_information(product_id: string): Get full details on one or more product IDs (comma-separated).
3. recommend_product(product_ids: string): Submit final product recommendation IDs as a comma-separated string.
4. terminate(status: string): End the dialogue. status="success" if the request is satisfied.

# Tools Rules
1. Use the "tool_call_id" field to link tool calls to their results in `<obs>...</obs>`.
2. Don't blindly trust the tool call results. Carefully evaluate whether they align with the user's needs, and use additional tools for verification if necessary.
3. Use the `find_product` tool to search for products. If the results do not meet expectations, you can:
    - Modify the parameter `q` and reuse the tool to get results related to the modified query.
    - Keep the parameter `q` the same, but change the parameter `page` to get new results.
    - Set the parameter `shop_id` to get results within the specified shop.
4. To check product information such as color, size, weight, model, material, pattern and so on, use the `view_product_information` tool.
5. When you identify products that fulfill the user's needs, use the `recommend_product` tool to recommend them to the user.
6. When the request is met or you can't proceed further with the task, use the `terminate` tool to end the dialogue.
7. Complete the task progressively without asking the user for external information.

# Output Format
1. Your output must only include `<tool_call>...</tool_call>` and nothing else.
2. Tool calls must be included within `<tool_call>...</tool_call>` and structured as a JSON array. Each tool call must have a "name" field and a "parameters" field as a dictionary. If no parameters are required, the dictionary can be empty.
3. Below is a template of your output:
```plaintext
<tool_call>[
{"name": "tool name", "parameters": {"parameter1": "value1", "parameter2": "value2", ...}},
{"name": "...", "parameters": {...}},
...
]</tool_call>
```"""


ANTHROPIC_PROVIDER_PREFS = {
    "order": ["Anthropic"],
    "ignore": ["Bedrock", "Google", "Azure"],
    "allow_fallbacks": False,
}


def parse_tool_calls(text: str) -> list[dict]:
    """Paper's XML parser, extended to accept the natural-language variants frontier
    models emit (`tool` for `name`, `arguments` for `parameters`). Also accepts a
    single-object `<tool_call>{...}</tool_call>` payload, which Claude tends to emit
    when there's only one call.
    """
    import json
    import re
    out: list[dict] = []
    # Try array form first: <tool_call>[{...}, {...}]</tool_call>
    for m in re.finditer(r"<tool_call>\s*(\[.*?\])\s*</tool_call>", text, flags=re.DOTALL):
        try:
            calls = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(calls, list):
            out.extend(c for c in calls if isinstance(c, dict))
    # Also try object form: <tool_call>{...}</tool_call>
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL):
        try:
            c = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(c, dict):
            out.append(c)
    # Normalize keys: tool → name, arguments → parameters
    normalized = []
    for c in out:
        name = c.get("name") or c.get("tool") or c.get("function")
        params = c.get("parameters") or c.get("arguments") or c.get("args") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        if name:
            normalized.append({"name": name, "parameters": params if isinstance(params, dict) else {}})
    return normalized


def _inject_id_into_titles(body):
    if isinstance(body, list):
        for p in body:
            if isinstance(p, dict):
                pid = p.get("product_id")
                title = p.get("title")
                if pid and isinstance(title, str) and not title.startswith("[product_id="):
                    p["title"] = f"[product_id={pid}] {title}"
    return body


def dispatch_tool(call, session):
    """Paper's dispatch with parameters-style args (not arguments-style)."""
    import json
    name = call.get("name", "")
    params = call.get("parameters", {}) or {}
    if name == "find_product":
        try:
            r = session.get(
                f"{SEARCH_URL}/find_product",
                params={
                    "q": params.get("query", params.get("q", "")),
                    "page": str(params.get("page", "1")),
                    "shop_id": params.get("shop_id", ""),
                    "price": params.get("price", ""),
                    "sort": params.get("sort", ""),
                    "service": params.get("service", ""),
                },
                timeout=15,
            )
            body = r.json() if r.ok else []
            return json.dumps(_inject_id_into_titles(body))
        except Exception as e:
            return json.dumps({"error": str(e)[:200]})
    if name == "view_product_information":
        try:
            r = session.get(
                f"{SEARCH_URL}/view_product_information",
                params={"product_ids": params.get("product_id", params.get("product_ids", ""))},
                timeout=15,
            )
            body = r.json() if r.ok else []
            return json.dumps(_inject_id_into_titles(body))
        except Exception as e:
            return json.dumps({"error": str(e)[:200]})
    if name == "recommend_product":
        return json.dumps({"status": "submitted", "product_ids": params.get("product_ids", "")})
    if name == "terminate":
        return json.dumps({"status": "terminated"})
    return json.dumps({"error": f"unknown tool: {name}"})


def has_terminate(calls):
    return any(c.get("name") == "terminate" for c in calls)


def extract_recommended_ids(messages):
    """Last recommend_product's product_ids — paper convention."""
    last_ids = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in parse_tool_calls(msg.get("content", "")):
            if call.get("name") == "recommend_product":
                pid_raw = (call.get("parameters") or {}).get("product_ids", "")
                if isinstance(pid_raw, str):
                    last_ids = [p.strip() for p in pid_raw.split(",") if p.strip()]
                elif isinstance(pid_raw, list):
                    last_ids = [str(p).strip() for p in pid_raw if str(p).strip()]
    return last_ids


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
    timeout=2 * 60 * 60,
    max_containers=10,
)
def evaluate_model(
    model: str,
    max_problems: int = 75,
    temperature: float = 0.7,
    max_steps: int = MAX_STEPS,
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
    print(f"\n=== Paper-harness eval (XML): {model} → {out_path} ===", flush=True)
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    api_key = os.environ["OPENROUTER_API_KEY"]
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/shardi-b/oro-post-train",
            "X-Title": "ORO teacher paper-harness eval",
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
    extra_body_base = {"usage": {"include": True}}
    if is_anthropic:
        extra_body_base["provider"] = ANTHROPIC_PROVIDER_PREFS

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((requests.RequestException, TimeoutError, ConnectionError)),
        reraise=True,
    )
    def call_model(messages, max_tokens=2048, temp=temperature):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
            extra_body=extra_body_base,
        )

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
        find_product_count = 0
        recommend_nudged = False
        steps_taken = 0
        prefill_used = False
        error_reason = None
        problem_cost = 0.0

        for step in range(max_steps):
            steps_taken = step + 1
            try:
                resp = call_model(messages)
            except Exception as e:
                error_reason = f"api_error: {str(e)[:200]}"
                print(f"  [{idx}] api error step={step}: {e}", flush=True)
                break

            problem_cost += extract_cost(resp)
            assistant_text = resp.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_text})
            calls = parse_tool_calls(assistant_text)

            for c in calls:
                if c.get("name") == "find_product":
                    find_product_count += 1
            has_recommend_call = any(c.get("name") == "recommend_product" for c in calls)

            # Nudge 1 (paper-style): no tool_calls → prefill recommend in the assistant turn.
            # API can't continue mid-message, so we re-prompt with an assistant prefill
            # and let the model autocomplete. Anthropic supports prefill via the last
            # assistant message; OpenAI ignores trailing assistant content but the
            # follow-up user nudge still works.
            if not calls:
                prefill_used = True
                # Replace placeholder assistant message with a prefilled one
                prefill_text = (
                    '<tool_call>[{"name": "recommend_product", "parameters": {"product_ids": "'
                )
                messages[-1]["content"] = (messages[-1]["content"] or "") + "\n" + prefill_text
                try:
                    cont = call_model(messages, max_tokens=256, temp=0.2)
                    problem_cost += extract_cost(cont)
                    completion = cont.choices[0].message.content or ""
                    full_assistant = messages[-1]["content"] + completion
                    if "</tool_call>" not in full_assistant:
                        full_assistant = (
                            full_assistant
                            + '"}}, {"name": "terminate", "parameters": {"status": "success"}}]</tool_call>'
                        )
                    messages[-1]["content"] = full_assistant
                    calls = parse_tool_calls(full_assistant)
                except Exception as e:
                    print(f"  [{idx}] prefill error: {e}", flush=True)
                if not calls:
                    break

            # Nudge 2: 3+ find_product without recommend → force commit
            if find_product_count >= 3 and not has_recommend_call and not recommend_nudged:
                recommend_nudged = True
                messages.append({
                    "role": "user",
                    "content": (
                        "You have searched enough. Pick ONE product_id from the find_product "
                        "results above that best matches the user's requirements, then call "
                        "recommend_product(product_ids=\"<that_id>\") followed by terminate("
                        "status=\"success\") in the SAME tool_call block. Do NOT call find_product again."
                    ),
                })

            if has_terminate(calls):
                terminated = True
                break

            tool_results = []
            for c in calls:
                res = dispatch_tool(c, session)
                tool_results.append(f"[{c.get('name')}] {res[:1500]}")
            messages.append({"role": "user", "content": "\n".join(tool_results)})

        rec_ids = extract_recommended_ids(messages)
        score_ok = score_problem(rec_ids, expected_ids, bucket)
        any_match = bool(set(rec_ids) & set(expected_ids))

        total_cost += problem_cost

        results.append({
            "bucket": bucket,
            "query": query[:200],
            "expected_ids": expected_ids,
            "rec_ids": rec_ids,
            "any_match": any_match,
            "score_ok": score_ok,
            "find_count": find_product_count,
            "recommend_emitted": any(
                c.get("name") == "recommend_product"
                for msg in messages if msg.get("role") == "assistant"
                for c in parse_tool_calls(msg.get("content", ""))
            ),
            "terminated": terminated,
            "steps_taken": steps_taken,
            "recommend_nudged": recommend_nudged,
            "prefill_used": prefill_used,
            "cost_usd": round(problem_cost, 5),
            "error_reason": error_reason,
        })

        if (idx + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1)
            eta = (len(sb_rows) - idx - 1) / max(rate, 0.001)
            n_ok = sum(r["score_ok"] for r in results)
            n_any = sum(r["any_match"] for r in results)
            n_rec = sum(r["recommend_emitted"] for r in results)
            print(
                f"  {idx+1}/{len(sb_rows)}  ASR={n_ok}/{idx+1}={100*n_ok/(idx+1):.1f}%  "
                f"any_match={n_any}  rec_emit={n_rec}  cost=${total_cost:.2f}  "
                f"({rate:.2f}/s, ETA {eta:.0f}s)",
                flush=True,
            )

        if total_cost >= per_model_usd_cap:
            print(f"  COST CAP HIT: ${total_cost:.2f} >= ${per_model_usd_cap}", flush=True)
            early_exit = "cost_cap"
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
            "prefill_used_pct": round(
                100 * sum(r["prefill_used"] for r in results) / max(n, 1), 2
            ),
            "mean_find_count": round(sum(r["find_count"] for r in results) / max(n, 1), 2),
            "mean_steps_taken": round(sum(r["steps_taken"] for r in results) / max(n, 1), 2),
        },
        "by_bucket": {},
        "max_steps": max_steps,
        "temperature": temperature,
        "harness": "paper-11_multi_turn_rollout (XML + nudges) via OpenRouter",
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
        f"Could not find data files under {here.parent.parent} or {here.parent.parent.parent}"
    )


@app.local_entrypoint()
def smoke(model: str = "openai/gpt-4.1"):
    """3-problem smoke."""
    _upload_local_artifacts()
    summary = evaluate_model.remote(model=model, max_problems=3)
    import json
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def single(model: str):
    """Full 75-problem run on one model."""
    _upload_local_artifacts()
    summary = evaluate_model.remote(model=model)
    import json
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def bakeoff():
    """All 3 top teachers in parallel."""
    _upload_local_artifacts()
    print(f"Fanning out {len(MODELS)} models in parallel...")
    summaries = list(evaluate_model.map(MODELS))
    print("\n=== PAPER-HARNESS BAKE-OFF LEADERBOARD ===")
    ranked = sorted(summaries, key=lambda s: s["overall"]["ASR_pct"], reverse=True)
    print(f"{'rank':>4}  {'model':<35}  {'ASR':>6}  {'rec_emit':>8}  {'cost':>8}")
    for i, s in enumerate(ranked, 1):
        print(
            f"{i:>4}  {s['model']:<35}  "
            f"{s['overall']['ASR_pct']:>5.1f}%  "
            f"{s['overall']['recommend_emitted_pct']:>7.1f}%  "
            f"${s['total_cost_usd']:>6.2f}"
        )
