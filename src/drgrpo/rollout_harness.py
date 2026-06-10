"""Multi-turn ReAct rollout harness.

Drives one episode: model emits <tool_call>, dispatch hits the deployed
search-server, results feed back into chat history. Up to `max_steps` turns.

Adapted from shardul_multi_turn_rollout_reference.py (the eval-only version)
into a callable suitable for online RL: returns dialogue + per-turn token spans
so the trainer can build an assistant-token mask later.

Public surface:
  - SYSTEM_PROMPT
  - parse_tool_calls(text) -> list[dict]
  - dispatch_tool(call, session) -> str
  - extract_recommended_ids(messages) -> list[str]
  - run_episode(model, tokenizer, problem, *, max_steps, temperature, ...)
      -> RolloutResult
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


import os as _os
SEARCH_URL = _os.environ.get("SEARCH_SERVER_URL", "https://oro-ai--oro-search-server-serve.modal.run")

# ShoppingBench prompts vendored verbatim — opd_rebal was SFT'd against these.
# Using a different system prompt makes the model emit text that doesn't parse as
# a tool_call, which collapses every rollout to 1 turn and zeros the reward.
from src.drgrpo._shoppingbench_prompts import (  # noqa: E402
    BASE_SYSTEM_PROMPT as _SB_BASE,
    PRODUCT_STRATEGY as _SB_PRODUCT,
    SHOP_STRATEGY as _SB_SHOP,
    VOUCHER_STRATEGY as _SB_VOUCHER,
)


def _detect_task(query: str) -> str:
    q = (query or "").lower()
    if "voucher" in q or "budget" in q:
        return "voucher"
    if "shop" in q and ("both" in q or "these" in q or "offering" in q or "sells" in q):
        return "shop"
    return "product"


def build_system_prompt(query: str) -> str:
    task = _detect_task(query)
    if task == "voucher":
        return _SB_BASE + _SB_VOUCHER
    if task == "shop":
        return _SB_BASE + _SB_SHOP
    return _SB_BASE + _SB_PRODUCT

SYSTEM_PROMPT = """You are a shopping agent helping users find products. \
You have access to the following tools:

- find_product(query, filters): search the product catalog. Returns matching product IDs.
- view_product_information(product_id): get full details on a specific product.
- recommend_product(product_ids): submit your final product recommendation(s) to the user.
- terminate(status): end the session. status="success" if you found a satisfactory answer.

For each turn, think step-by-step inside <think>...</think> tags about what to do next, \
then emit one or more tool calls inside <tool_call>[...]</tool_call> tags as JSON. \
The user's response will follow.

EVERY ASSISTANT TURN MUST END WITH A <tool_call> BLOCK. If you have enough information \
to answer, call recommend_product with the chosen product_id, then call terminate(status="success") \
in the same <tool_call> block."""


# ---------- tool-call parsing / dispatch ----------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\[.*?\])\s*</tool_call>", flags=re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    out: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            calls = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(calls, list):
            for c in calls:
                if isinstance(c, dict) and "name" in c:
                    out.append(c)
    return out


def _inject_id_into_titles(body: Any) -> Any:
    if isinstance(body, list):
        for p in body:
            if isinstance(p, dict):
                pid = p.get("product_id")
                title = p.get("title")
                if pid and isinstance(title, str) and not title.startswith("[product_id="):
                    p["title"] = f"[product_id={pid}] {title}"
    return body


def dispatch_tool(call: dict, session) -> str:
    name = call.get("name", "")
    params = call.get("parameters", {}) or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            params = {}
    if not isinstance(params, dict):
        params = {}

    if name == "find_product":
        # Model emits "q" not "query" — match ShoppingBench tool signature.
        try:
            r = session.get(
                f"{SEARCH_URL}/find_product",
                params={
                    "q": params.get("q", params.get("query", "")),
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
            return json.dumps({"error": str(e)})

    if name == "view_product_information":
        try:
            r = session.get(
                f"{SEARCH_URL}/view_product_information",
                params={"product_ids": params.get("product_ids", params.get("product_id", ""))},
                timeout=15,
            )
            body = r.json() if r.ok else []
            return json.dumps(_inject_id_into_titles(body))
        except Exception as e:
            return json.dumps({"error": str(e)})

    if name == "check_product_match":
        # Client-side helper: fetch product, substring-match requirements against attrs/desc.
        try:
            req_str = params.get("requirements", "{}")
            reqs = json.loads(req_str) if isinstance(req_str, str) else req_str
        except Exception:
            return json.dumps({"matched": False, "error": "Invalid requirements JSON"})
        pid = params.get("product_id", "")
        try:
            r = session.get(f"{SEARCH_URL}/view_product_information",
                            params={"product_ids": pid}, timeout=15)
            info_list = r.json() if r.ok else []
        except Exception as e:
            return json.dumps({"matched": False, "error": str(e)})
        if not info_list:
            return json.dumps({"matched": False, "error": "Product not found",
                               "matches": [], "mismatches": list(reqs.keys())})
        info = info_list[0] if isinstance(info_list, list) else info_list
        attrs = info.get("attributes", {}) or {}
        sku_opts = info.get("sku_options", []) or []
        all_values = []
        for v in attrs.values():
            all_values.extend(str(x).lower() for x in (v if isinstance(v, list) else [v]))
        for opt in (sku_opts if isinstance(sku_opts, list) else sku_opts.values() if isinstance(sku_opts, dict) else []):
            if isinstance(opt, dict):
                for v in opt.values():
                    all_values.extend(str(x).lower() for x in (v if isinstance(v, list) else [v]))
        searchable = " ||| ".join(all_values)
        desc = (str(info.get("short_description", "")) + " " + str(info.get("description", ""))).lower()
        matches, mismatches = [], []
        for key, val in reqs.items():
            req_lower = str(val).lower()
            if req_lower in searchable or req_lower in desc:
                matches.append(key)
            else:
                mismatches.append(key)
        return json.dumps({"matched": len(mismatches) == 0, "matches": matches,
                           "mismatches": mismatches,
                           "product_summary": {"product_id": info.get("product_id", pid)}})

    if name == "find_products_in_same_shop":
        # Run multiple find_product searches, group by shop_id, return shops that have all queries.
        try:
            q_raw = params.get("product_queries", "[]")
            queries = json.loads(q_raw) if isinstance(q_raw, str) else q_raw
        except Exception:
            return json.dumps({"found": False, "error": "Invalid product_queries JSON"})
        if not isinstance(queries, list) or not queries:
            return json.dumps({"found": False, "error": "Empty queries"})
        # Per-query results
        shop_to_products: dict = {}
        for qi, q in enumerate(queries):
            if not isinstance(q, dict): continue
            try:
                r = session.get(f"{SEARCH_URL}/find_product",
                                params={"q": q.get("q", ""), "page": "1",
                                        "shop_id": q.get("shop_id", ""),
                                        "price": q.get("price", ""),
                                        "service": q.get("service", "")},
                                timeout=15)
                rows = r.json() if r.ok else []
            except Exception:
                rows = []
            for row in rows[:10]:
                sid = row.get("shop_id")
                if not sid: continue
                shop_to_products.setdefault(sid, {}).setdefault(qi, []).append(row)
        # Find shops that cover ALL queries
        for sid, by_q in shop_to_products.items():
            if len(by_q) == len(queries):
                products = []
                for qi in range(len(queries)):
                    if by_q.get(qi):
                        p = by_q[qi][0]
                        products.append({"product_id": p.get("product_id"),
                                         "title": p.get("title"),
                                         "price": p.get("price"),
                                         "shop_id": p.get("shop_id")})
                return json.dumps({"found": True, "shop_id": sid, "products": products,
                                   "shops_tried": len(shop_to_products)})
        return json.dumps({"found": False, "shops_tried": len(shop_to_products)})

    if name == "calculate_voucher":
        # Pure math, no API call.
        try:
            prices = params.get("product_prices") or []
            if isinstance(prices, str):
                prices = json.loads(prices)
            prices = [float(p) for p in prices]
            voucher_type = params.get("voucher_type", "fixed")
            discount_value = float(params.get("discount_value", 0))
            threshold = float(params.get("threshold", 0))
            budget = float(params.get("budget", 0))
            cap = params.get("cap")
            cap = float(cap) if cap not in (None, "") else None
            subtotal = sum(prices)
            if subtotal < threshold:
                discount = 0.0
                eligible = False
            else:
                eligible = True
                if voucher_type == "percentage":
                    discount = subtotal * (discount_value / 100.0)
                    if cap is not None: discount = min(discount, cap)
                else:
                    discount = discount_value
                    if cap is not None: discount = min(discount, cap)
            total = subtotal - discount
            return json.dumps({
                "subtotal": subtotal, "discount": discount, "total_after_discount": total,
                "within_budget": total <= budget, "voucher_eligible": eligible,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    if name == "recommend_product":
        return json.dumps({"status": "submitted", "product_ids": params.get("product_ids", "")})

    if name == "terminate":
        return json.dumps({"status": "terminated"})

    return json.dumps({"error": f"unknown tool: {name}"})


def has_terminate(calls: list[dict]) -> bool:
    return any(c.get("name") == "terminate" for c in calls)


def extract_recommended_ids(messages: list[dict]) -> list[str]:
    last_ids: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in parse_tool_calls(msg.get("content", "")):
            if call.get("name") == "recommend_product":
                params = call.get("parameters") or {}
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {}
                if not isinstance(params, dict):
                    params = {}
                pid_raw = params.get("product_ids", "")
                if isinstance(pid_raw, str):
                    last_ids = [p.strip() for p in pid_raw.split(",") if p.strip()]
                elif isinstance(pid_raw, list):
                    last_ids = [str(p).strip() for p in pid_raw if str(p).strip()]
    return last_ids


# ---------- rollout ----------


@dataclass
class AssistantTurn:
    """Token span of one assistant turn in the final tokenized sequence.

    Populated by the trainer after rendering messages with the chat template.
    Stored here so iteration 2 (mask validator) can decode mask==1 spans and
    cross-check them against the recorded assistant text.
    """
    text: str
    token_start: int = -1
    token_end: int = -1


@dataclass
class RolloutResult:
    messages: list[dict]
    assistant_turns: list[AssistantTurn]
    terminated: bool
    n_steps: int
    truncated: bool = False
    error: str | None = None
    meta: dict = field(default_factory=dict)


def _generate(model, tokenizer, messages: list[dict], *, max_new_tokens: int, temperature: float) -> str:
    """Generate the next assistant turn.

    Iteration 3 hardening: snapshot the model's training flag, force eval()
    for the duration of generate(), restore on exit. Belt-and-suspenders with
    the trainer-side model.eval() call.

    BUG MITIGATED: PeftModel.from_pretrained(is_trainable=True) returns the
    model in train mode. Our policy adapter has lora_dropout=0.05; dropout
    during autoregressive generation compounds across turns and corrupts
    late-turn output. If anything upstream forgets eval(), this catches it
    here so generation is never run with dropout active.
    """
    import torch

    was_training = model.training
    model.eval()
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # temperature=0 means greedy → do_sample=False, drop temperature kwarg.
        # transformers rejects temperature=0 with do_sample=True ("must be strictly positive").
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        if was_training:
            model.train()


def run_episode(
    model,
    tokenizer,
    problem: dict,
    *,
    session=None,
    max_steps: int = 8,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    tool_result_clip: int = 1500,
) -> RolloutResult:
    """One episode through the multi-turn loop. Returns the dialogue + metadata.

    Token spans on AssistantTurn are left at -1; the trainer fills them in when
    it materialises the final input_ids (so it can use a single canonical
    tokenization pass).
    """
    import requests

    if session is None:
        session = requests.Session()

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(problem["query"])},
        {"role": "user", "content": problem["query"]},
    ]
    assistant_turns: list[AssistantTurn] = []

    terminated = False
    truncated = False
    err: str | None = None
    step = 0

    for step_idx in range(max_steps):
        step = step_idx + 1
        try:
            assistant_text = _generate(
                model, tokenizer, messages,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
        except Exception as e:
            err = f"generation: {e}"
            break

        messages.append({"role": "assistant", "content": assistant_text})
        assistant_turns.append(AssistantTurn(text=assistant_text))

        calls = parse_tool_calls(assistant_text)
        if not calls:
            # No tool call emitted: end the episode. Reward fn handles the no-recommend case.
            break

        if has_terminate(calls):
            terminated = True
            break

        tool_results: list[str] = []
        for c in calls:
            res = dispatch_tool(c, session)
            tool_results.append(f"[{c.get('name')}] {res[:tool_result_clip]}")
        messages.append({"role": "user", "content": "\n".join(tool_results)})
    else:
        truncated = True

    return RolloutResult(
        messages=messages,
        assistant_turns=assistant_turns,
        terminated=terminated,
        n_steps=step,
        truncated=truncated,
        error=err,
        meta={
            "rec_ids": extract_recommended_ids(messages),
        },
    )


def _generate_batch(model, tokenizer, batch_messages, *, max_new_tokens, temperature):
    """Batched generate: N independent message-lists → N completion strings.

    Pads on the left (so the generated continuation starts at a uniform offset)
    and uses attention_mask so padding tokens don't affect attention.
    Returns list[str] of length N, model-relative completions only.
    """
    import torch

    was_training = model.training
    model.eval()
    saved_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        completions = [
            tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True)
            for i in range(out.shape[0])
        ]
        return completions
    finally:
        tokenizer.padding_side = saved_side
        if was_training:
            model.train()


def run_episode_batch(
    model,
    tokenizer,
    problem: dict,
    K: int,
    *,
    session=None,
    max_steps: int = 8,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    tool_result_clip: int = 1500,
    tool_concurrency: int = 8,
) -> list[RolloutResult]:
    """Run K rollouts for ONE problem in parallel via batched generate.

    All K start from the same (system, user) messages and diverge under
    temperature sampling. At each turn, generate is called once with the
    active (non-terminated) sub-batch. Tool dispatch is parallelized across
    sessions with a thread pool. Returns K RolloutResult in original order.
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor

    if session is None:
        session = requests.Session()

    base_messages = [
        {"role": "system", "content": build_system_prompt(problem["query"])},
        {"role": "user", "content": problem["query"]},
    ]

    sessions = [
        {
            "messages": [dict(m) for m in base_messages],
            "assistant_turns": [],
            "terminated": False,
            "truncated": False,
            "error": None,
            "done": False,
            "n_steps": 0,
        }
        for _ in range(K)
    ]

    pool = ThreadPoolExecutor(max_workers=tool_concurrency)

    for step_idx in range(max_steps):
        active_idx = [i for i, s in enumerate(sessions) if not s["done"]]
        if not active_idx:
            break
        active_msgs = [sessions[i]["messages"] for i in active_idx]
        try:
            completions = _generate_batch(
                model, tokenizer, active_msgs,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
        except Exception as e:
            for i in active_idx:
                sessions[i]["error"] = f"generation: {e}"
                sessions[i]["done"] = True
            break

        # Per-session post-generate processing. Tool dispatch parallelized.
        tool_jobs = []  # list[(sess_idx, calls)]
        for idx, completion in zip(active_idx, completions):
            s = sessions[idx]
            s["n_steps"] = step_idx + 1
            s["messages"].append({"role": "assistant", "content": completion})
            s["assistant_turns"].append(AssistantTurn(text=completion))
            calls = parse_tool_calls(completion)
            if not calls:
                s["done"] = True
                continue
            if has_terminate(calls):
                s["terminated"] = True
                s["done"] = True
                continue
            tool_jobs.append((idx, calls))

        if tool_jobs:
            def _run_session_tools(job):
                idx, calls = job
                results = []
                for c in calls:
                    res = dispatch_tool(c, session)
                    results.append(f"[{c.get('name')}] {res[:tool_result_clip]}")
                return idx, "\n".join(results)

            for idx, joined in pool.map(_run_session_tools, tool_jobs):
                sessions[idx]["messages"].append({"role": "user", "content": joined})

    pool.shutdown(wait=False)

    # Anything still not done after max_steps is truncated.
    for s in sessions:
        if not s["done"]:
            s["truncated"] = True

    out = []
    for s in sessions:
        out.append(RolloutResult(
            messages=s["messages"],
            assistant_turns=s["assistant_turns"],
            terminated=s["terminated"],
            n_steps=s["n_steps"],
            truncated=s["truncated"],
            error=s["error"],
            meta={"rec_ids": extract_recommended_ids(s["messages"])},
        ))
    return out


# ─── v17: vllm-backed rollouts (continuous batching, parallel) ──────────

_QWEN_SPECIALS_RE = None

def _strip_qwen_specials(text: str) -> str:
    """Sanitize Qwen3 vllm `content` so the trainer's mask validator agrees
    with what Qwen3's chat-template renders on re-tokenization.

    Two failure modes seen during v17 training:
      1) Model occasionally echoes `<|im_start|>...` / `<|im_end|>` literals
         in content → spurious phantom spans in the rendered conversation.
      2) Vllm splits reasoning into `reasoning_content` and content but the
         model emits an orphan `</think>` at start of content
         (no opening `<think>`). Qwen3 chat-template still strips up to
         the closing `</think>` on re-render, but our stored
         `messages[i].content` keeps it. Mask span ↔ content drift.
    Both are fixed by removing the special-token strings and any prefix up
    to & including a leading `</think>`."""
    import re
    global _QWEN_SPECIALS_RE
    if _QWEN_SPECIALS_RE is None:
        _QWEN_SPECIALS_RE = re.compile(r"<\|im_start\|>(?:user|assistant|system|tool)?\s*|<\|im_end\|>\s*")
    text = _QWEN_SPECIALS_RE.sub("", text)
    # Strip orphan `</think>` (and any leading `<think>...</think>` block) at
    # the start of content. Vllm separates reasoning into reasoning_content
    # so the content shouldn't have think tags at all — anything present is
    # leakage we should remove to match chat-template re-render behavior.
    text = re.sub(r"\A\s*(?:<think>.*?</think>|</think>)\s*", "", text, count=1, flags=re.DOTALL)
    return text


def _generate_via_vllm(messages, *, vllm_url, model_name, max_new_tokens, temperature, session):
    """Single chat-completion call to vllm. Returns assistant content string,
    sanitized of stray Qwen3 special-token strings."""
    r = session.post(
        f"{vllm_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stream": False,
        },
        timeout=240,
    )
    r.raise_for_status()
    d = r.json()
    content = d["choices"][0]["message"].get("content", "") if d.get("choices") else ""
    return _strip_qwen_specials(content)


def run_episode_via_vllm(
    problem: dict,
    *,
    vllm_url: str,
    model_name: str,
    session=None,
    max_steps: int = 25,
    max_new_tokens: int = 8192,
    temperature: float = 0.7,
    tool_result_clip: int = 1500,
) -> RolloutResult:
    """ReAct loop against vllm. Same semantics as run_episode but generates via HTTP."""
    import requests
    if session is None:
        session = requests.Session()
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(problem["query"])},
        {"role": "user", "content": problem["query"]},
    ]
    assistant_turns: list[AssistantTurn] = []
    terminated = False
    truncated = False
    err: str | None = None
    step = 0
    for step_idx in range(max_steps):
        step = step_idx + 1
        try:
            text = _generate_via_vllm(
                messages, vllm_url=vllm_url, model_name=model_name,
                max_new_tokens=max_new_tokens, temperature=temperature,
                session=session,
            )
        except Exception as e:
            err = f"generation: {type(e).__name__}: {e}"
            break
        messages.append({"role": "assistant", "content": text})
        assistant_turns.append(AssistantTurn(text=text))
        calls = parse_tool_calls(text)
        if not calls:
            break
        if has_terminate(calls):
            terminated = True
            break
        tool_results: list[str] = []
        for c in calls:
            res = dispatch_tool(c, session)
            tool_results.append(f"[{c.get('name')}] {res[:tool_result_clip]}")
        messages.append({"role": "user", "content": "\n".join(tool_results)})
    else:
        truncated = True
    return RolloutResult(
        messages=messages,
        assistant_turns=assistant_turns,
        terminated=terminated,
        n_steps=step,
        truncated=truncated,
        error=err,
        meta={"rec_ids": extract_recommended_ids(messages)},
    )


def run_batch_via_vllm(
    problems: list[dict],
    *,
    K: int,
    vllm_url: str,
    model_name: str,
    max_workers: int = 24,
    max_steps: int = 25,
    max_new_tokens: int = 8192,
    temperature: float = 0.7,
    tool_result_clip: int = 1500,
) -> list[list[RolloutResult]]:
    """K rollouts per problem, all fired in parallel against vllm.
    Returns list-of-lists in (problems × K) order. Each thread gets its own
    requests.Session so HTTP keep-alive + connection pool is per-thread.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from concurrent.futures import ThreadPoolExecutor

    def _mk_session():
        s = requests.Session()
        retry = Retry(total=4, backoff_factor=0.5,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=frozenset(["POST"]))
        a = HTTPAdapter(pool_maxsize=64, pool_connections=64, max_retries=retry)
        s.mount("http://", a); s.mount("https://", a)
        return s

    import threading
    _tls = threading.local()
    def _sess():
        if not hasattr(_tls, "s"):
            _tls.s = _mk_session()
        return _tls.s

    jobs = [(p_idx, k, prob) for p_idx, prob in enumerate(problems) for k in range(K)]
    results = [[None] * K for _ in problems]

    def _one(job):
        p_idx, k, prob = job
        ro = run_episode_via_vllm(
            prob,
            vllm_url=vllm_url, model_name=model_name,
            session=_sess(),
            max_steps=max_steps, max_new_tokens=max_new_tokens,
            temperature=temperature, tool_result_clip=tool_result_clip,
        )
        return p_idx, k, ro

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for p_idx, k, ro in ex.map(_one, jobs):
            results[p_idx][k] = ro
    return results
