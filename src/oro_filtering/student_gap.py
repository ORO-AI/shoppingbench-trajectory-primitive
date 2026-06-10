"""Qwen3-4B student-teacher gap filter.

Task 6a defines the pure-Python contract. Task 6b adds the Modal vLLM rollout
plus a ``run_gap_batch`` orchestrator that scores teacher and student
trajectories using ``score_components`` and returns ``GapResult`` per row.

Modal-specific code is importable without Modal credentials. The Modal class
``Qwen3StudentServer`` only spins up on Modal compute; tests inject a stub
rollout via ``run_gap_batch(student_rollout=...)`` so the suite never needs a
GPU.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import modal

# Note: `oro_filtering.leak_cluster` and `oro_filtering.reward_components` are
# imported inside `run_gap_batch` (orchestrator-side only) so the Modal container
# — which only runs `Qwen3StudentServer.load`/`rollout` — doesn't need the
# `oro_filtering` package installed in its image.


@dataclass(frozen=True)
class GapResult:
    """Result of scoring teacher + student trajectories on the same problem.

    ``teacher_components`` carries all 4 components from
    ``oro_filtering.reward_components.ALL_COMPONENTS``.

    ``student_components`` is a 3-component dict. ``constraint_satisfied`` is
    excluded because the student has no real grader signal and synthesizing
    one from PID-match would double-count ``final_recommendation_correct``
    (see review note I-1). The student score is rescaled by ``4/3`` so it is
    directly comparable to the teacher's 4-component score.

    ``gap`` is a derived property to avoid drift between
    ``teacher_score - student_score`` and a separately-stored field.
    """

    row_id: str
    teacher_score: float
    student_score: float
    teacher_components: dict[str, float]
    student_components: dict[str, float]

    @property
    def gap(self) -> float:
        return round(self.teacher_score - self.student_score, 4)

    def passes_filter(self, *, min_gap: float = 0.05, max_student_score: float = 0.90) -> bool:
        return self.gap >= min_gap and self.student_score <= max_student_score


# ---------------------------------------------------------------------------
# Tool schemas + system prompt
#
# Mirror Seth's scripts/25_convert_to_nemo_sft.py::TOOLS (copy, don't import:
# that script is sys.argv-driven and not import-friendly).
# ---------------------------------------------------------------------------

# Copied verbatim from scripts/25_convert_to_nemo_sft.py::TOOLS so the
# student rollout speaks the exact tool schema Seth's NeMo SFT export
# expects. Keep in sync (see review note I-3).
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "description": "Search the ShoppingBench product catalog.",
        "name": "find_product",
        "parameters": {
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
            "type": "object",
        },
        "type": "function",
    },
    {
        "description": "Inspect product details for one or more product IDs.",
        "name": "view_product_information",
        "parameters": {
            "additionalProperties": True,
            "properties": {
                "product_id": {"type": "string"},
                "product_ids": {"items": {"type": "string"}, "type": "array"},
            },
            "type": "object",
        },
        "type": "function",
    },
    {
        "description": "Submit final ShoppingBench product recommendation IDs.",
        "name": "recommend_product",
        "parameters": {
            "additionalProperties": True,
            "properties": {
                "product_id": {"type": "string"},
                "product_ids": {"items": {"type": "string"}, "type": "array"},
            },
            "type": "object",
        },
        "type": "function",
    },
    {
        "description": "End the shopping episode after final recommendation.",
        "name": "terminate",
        "parameters": {"additionalProperties": True, "properties": {}, "type": "object"},
        "type": "function",
    },
]


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


# ---------------------------------------------------------------------------
# Stub tool environment
#
# Task 8 will wire real catalog access. Here we just need the rollout loop to
# advance: any tool call gets a benign empty result so the model can continue
# its turn budget and either give up or call ``recommend_product``/``terminate``.
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _stub_tool_result(tool_name: str) -> Any:
    """Empty-ish result for each tool; gates the rollout wiring."""
    if tool_name == "find_product":
        return []
    if tool_name == "view_product_information":
        return {}
    if tool_name == "recommend_product":
        return "recorded"
    return {}


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract every ``<tool_call>{...}</tool_call>`` JSON object in ``text``.

    Qwen3 chains multiple ``<tool_call>...</tool_call>`` blocks in a single
    assistant turn ~10-15% of the time (review note I-2); using ``search``
    silently dropped all but the first. We use ``findall`` and skip any block
    whose body fails JSON parsing.
    """
    calls: list[dict[str, Any]] = []
    for body in _TOOL_CALL_RE.findall(text or ""):
        try:
            calls.append(json.loads(body))
        except json.JSONDecodeError:
            continue
    return calls


# ---------------------------------------------------------------------------
# Modal app
#
# This block is importable without Modal credentials — ``modal.App(...)`` and
# ``@app.cls(...)`` register objects but don't reach out to Modal until the
# class is invoked via ``.remote(...)``.
# ---------------------------------------------------------------------------

app = modal.App("oro-sft-student-gap")

# vLLM's flashinfer attention backend JIT-compiles CUDA at first inference, so
# the image needs `nvcc` available. We use Modal's CUDA-devel image, not
# debian_slim. The "12.4.1-devel" tag includes the CUDA toolkit.
_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm>=0.6", "torch>=2.4", "transformers>=4.51", "huggingface_hub")
    .env({"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
)


@app.cls(gpu="A10G", image=_image, scaledown_window=300, max_containers=4)
class Qwen3StudentServer:
    """Hosts Qwen/Qwen3-4B via vLLM and runs the multi-turn tool loop."""

    @modal.enter()
    def load(self) -> None:  # pragma: no cover - runs only on Modal compute
        from vllm import LLM, SamplingParams

        self._llm = LLM(model="Qwen/Qwen3-4B", dtype="bfloat16", max_model_len=8192)
        self._sampling = SamplingParams(
            temperature=0.0, max_tokens=2048, stop=["</tool_call>"]
        )
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    @modal.method()
    def rollout(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tool_schemas: list[dict],
        max_turns: int = 6,
    ) -> dict:  # pragma: no cover - runs only on Modal compute
        """Run a multi-turn tool loop.

        Returns ``{"steps": list[dict], "n_turns": int}`` where each step matches
        the teacher trajectory shape: ``{"think", "tool_calls", "response",
        "content"}``.

        Tool environment is a STUB (see ``_stub_tool_result``). The orchestrator
        in Task 8 will wire real catalog access.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        steps: list[dict[str, Any]] = []

        for turn in range(max_turns):
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tools=tool_schemas,
                add_generation_prompt=True,
                tokenize=False,
            )
            outputs = self._llm.generate([prompt_text], self._sampling)
            generated = outputs[0].outputs[0].text or ""

            # Re-attach the stop sequence the sampler trimmed off so downstream
            # parsing sees a well-formed <tool_call>...</tool_call> block.
            content = generated
            if "<tool_call>" in generated and "</tool_call>" not in generated:
                content = generated + "</tool_call>"

            parsed_calls = _parse_tool_calls(content)
            think = ""
            if "<think>" in content and "</think>" in content:
                think = content.split("<think>", 1)[1].split("</think>", 1)[0].strip()

            if not parsed_calls:
                # No tool call: model gave prose. End the rollout.
                steps.append(
                    {
                        "think": think,
                        "tool_calls": [],
                        "response": content,
                        "content": content,
                    }
                )
                return {"steps": steps, "n_turns": turn + 1}

            # Feed the assistant turn back into the conversation, then execute
            # each parsed tool call in order and append a tool-role message
            # per call. This matches what Qwen3 expects when it chains
            # multiple tool_calls in a single assistant turn.
            messages.append({"role": "assistant", "content": content})

            step_tool_calls: list[dict[str, Any]] = []
            terminator: str | None = None
            for tool_call in parsed_calls:
                name = tool_call.get("name")
                params = (
                    tool_call.get("parameters", {})
                    or tool_call.get("arguments", {})
                    or {}
                )
                result = _stub_tool_result(name or "")
                step_tool_calls.append(
                    {"name": name, "parameters": params, "result": result}
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
                if name in ("recommend_product", "terminate") and terminator is None:
                    terminator = name

            steps.append(
                {
                    "think": think,
                    "tool_calls": step_tool_calls,
                    "response": None,
                    "content": content,
                }
            )

            if terminator is not None:
                return {"steps": steps, "n_turns": turn + 1}

        return {"steps": steps, "n_turns": max_turns}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


# Cap on concurrent in-flight Modal rollout calls. A10G + vLLM batches
# internally, so 8 concurrent client-side requests is a reasonable balance
# between throughput and Modal container fan-out cost.
_DEFAULT_MODAL_MAX_WORKERS = 4


def _default_modal_rollout(
    arg_tuples: list[dict[str, Any]],
) -> list[dict]:  # pragma: no cover - hits Modal compute
    """Default rollout: dispatches a batch to the Modal Qwen3StudentServer.

    Each entry in ``arg_tuples`` is ``{"system_prompt", "user_prompt",
    "tool_schemas", "max_turns"}`` and we fan them out concurrently. The
    ``Qwen3StudentServer`` handle is instantiated ONCE per batch so we don't
    re-create it per row.

    We use a ``ThreadPoolExecutor`` over ``server.rollout.remote(...)`` rather
    than ``server.rollout.map(...)`` because Modal's ``.map()`` does not cleanly
    accept per-call kwargs in the way our rollout signature requires. Threaded
    ``.remote()`` calls still hit Modal's container pool concurrently.

    Lazy so importing this module never reaches Modal. Tests inject a stub
    instead of going through here.
    """
    DeployedServer = modal.Cls.from_name("oro-sft-student-gap", "Qwen3StudentServer")
    server = DeployedServer()

    def _one(args: dict[str, Any]) -> dict:
        return server.rollout.remote(
            system_prompt=args["system_prompt"],
            user_prompt=args["user_prompt"],
            tool_schemas=args["tool_schemas"],
            max_turns=args["max_turns"],
        )

    if not arg_tuples:
        return []
    with ThreadPoolExecutor(max_workers=_DEFAULT_MODAL_MAX_WORKERS) as executor:
        return list(executor.map(_one, arg_tuples))


def run_gap_batch(
    rows: list[dict[str, Any]],
    *,
    student_rollout: Callable[[list[dict[str, Any]]], list[dict]] | None = None,
    max_turns: int = 4,
) -> list[GapResult]:
    """Score teacher + student trajectories and return ``GapResult`` per row.

    ``student_rollout`` is a callable with signature
    ``(arg_tuples: list[dict]) -> list[dict]`` where each input dict is
    ``{"system_prompt", "user_prompt", "tool_schemas", "max_turns"}`` and each
    output dict is ``{"steps", "n_turns"}`` (same shape as the per-row rollout
    return). The default is the Modal ``Qwen3StudentServer``; tests inject a
    stub.

    The batch call lets the implementation fan out concurrently (Modal
    ``.map``/threaded ``.remote``) rather than serial per-row Modal calls,
    which dominates wall-clock at 3K+ rows.

    Rows whose teacher trajectory has no recommended product IDs are skipped
    (returned with ``gap=0.0`` and ``passes_filter=False``) because the gap
    can't be computed without a ground-truth target.
    """
    # Local imports so the Modal container can import `student_gap.py` without
    # `oro_filtering` installed (this function only runs on the orchestrator).
    from oro_filtering.leak_cluster import extract_recommended_product_ids
    from oro_filtering.reward_components import ALL_COMPONENTS, score_components

    rollout_fn = student_rollout or _default_modal_rollout

    # First pass: classify each row into "needs rollout" vs "skip (no ground
    # truth)", and pre-compute teacher components for the ones we will keep.
    # Order is preserved so results[] aligns with rows[].
    plans: list[dict[str, Any]] = []
    rollout_inputs: list[dict[str, Any]] = []
    rollout_index: list[int] = []  # plan index for each rollout_input

    for row in rows:
        row_id = str(row.get("run_id") or row.get("row_id") or "")
        steps = row.get("steps") or []
        ground_truth_pids = extract_recommended_product_ids(steps)

        if not ground_truth_pids:
            plans.append({"kind": "skip", "row_id": row_id})
            continue

        teacher_components = score_components(row, ground_truth_pids=ground_truth_pids)
        teacher_score = round(sum(teacher_components[c] for c in ALL_COMPONENTS), 4)

        plans.append(
            {
                "kind": "rollout",
                "row_id": row_id,
                "row": row,
                "ground_truth_pids": ground_truth_pids,
                "teacher_components": teacher_components,
                "teacher_score": teacher_score,
            }
        )
        rollout_inputs.append(
            {
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": str(row.get("problem_query") or ""),
                "tool_schemas": TOOL_SCHEMAS,
                "max_turns": max_turns,
            }
        )
        rollout_index.append(len(plans) - 1)

    # Batch rollout: single call (concurrent under the hood) instead of N
    # serial Modal hops.
    rollout_results = rollout_fn(rollout_inputs) if rollout_inputs else []
    if len(rollout_results) != len(rollout_inputs):
        raise ValueError(
            f"student_rollout returned {len(rollout_results)} results for "
            f"{len(rollout_inputs)} inputs; expected one-to-one"
        )

    # Stitch rollout outputs back into the plans by recorded index.
    by_plan_idx: dict[int, dict] = {}
    for idx, result in zip(rollout_index, rollout_results, strict=True):
        by_plan_idx[idx] = result

    results: list[GapResult] = []
    for plan_idx, plan in enumerate(plans):
        if plan["kind"] == "skip":
            results.append(
                GapResult(
                    row_id=plan["row_id"],
                    teacher_score=0.0,
                    student_score=0.0,
                    teacher_components={k: 0.0 for k in ALL_COMPONENTS},
                    student_components={
                        k: 0.0 for k in ALL_COMPONENTS if k != "constraint_satisfied"
                    },
                )
            )
            continue

        student_rollout_result = by_plan_idx[plan_idx]
        # Student uses 3 components (no real grader; synthesizing a
        # constraint score from PID-match would double-count
        # final_recommendation_correct, see review note I-1). We rescale by
        # 4/3 so identical behavior to the teacher yields an identical score
        # and gap == 0.
        student_row = {
            "steps": student_rollout_result.get("steps") or [],
            "problem_query": plan["row"].get("problem_query"),
            "per_problem_score": 0.0,
        }
        student_components_full = score_components(
            student_row, ground_truth_pids=plan["ground_truth_pids"]
        )
        student_components = {
            k: v for k, v in student_components_full.items() if k != "constraint_satisfied"
        }
        student_score = round(sum(student_components.values()) * (4.0 / 3.0), 4)

        results.append(
            GapResult(
                row_id=plan["row_id"],
                teacher_score=plan["teacher_score"],
                student_score=student_score,
                teacher_components=plan["teacher_components"],
                student_components=student_components,
            )
        )

    return results
