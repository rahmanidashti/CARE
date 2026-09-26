"""On-the-fly pseudo-reference + rubric generation for rubric-based RL.

Training-time pipeline for a single GRPO/GDPO group (one prompt, ``n`` rollouts):

    1. pseudo-reference : synthesise the group's ``n`` rollouts into one
                          "perfect answer" (conditions on ALL rollouts);
    2. rubric           : generate a grading rubric from the pseudo-reference;
    3. judge            : (unchanged) score each rollout against that rubric.

All three stages are served by the SAME frozen base model -- i.e. the endpoint
the judge already uses (``VLLM_BASE_URL`` / ``VLLM_MODEL``), which is a separate,
frozen model, NOT the policy. This module therefore reuses the judge's sampler
singleton (``get_global_grader``) and only owns stages 1 and 2. The judge /
scoring pipeline in ``rurbichub_v1_Medical.py`` is untouched and simply reads
whatever rubrics land in ``extra_info["reward_model"]["rubrics"]``.

Because the pseudo-reference (and therefore the rubric) is derived from the whole
group and shared across all its rollouts, group consistency -- required for
well-defined GRPO/GDPO advantages -- holds by construction. The batch-level hook
in ``verl/experimental/reward_loop/reward_loop.py`` calls ``generate_group_rubrics``
once per ``uid`` before the per-rollout judging.

On any failure the caller keeps the static dataset rubrics (fallback).

Config via environment variables:
    RUBRIC_GEN_ENABLE            "1"/"true" to turn generation on (default off)
    RUBRIC_GEN_N_ITEMS           target number of rubric items (soft hint)
    RUBRIC_GEN_TEMPERATURE       rubric-stage temperature at train time
    RUBRIC_GEN_TEMPERATURE_VAL   rubric-stage temperature at validation time
    PSEUDO_REF_TEMPERATURE       pseudo-reference-stage temperature (default 0.7)
    (endpoint/model come from VLLM_BASE_URL / VLLM_MODEL, shared with the judge)
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

# Reuse the judge's sampler singleton (the frozen base model) and the shared schema
# so generated rubrics are byte-for-byte compatible with the downstream scoring math.
from .rurbichub_v1_Medical import (
    RubricItem,
    _build_batch_grader_prompt,
    _format_prompt_messages,
    _parse_presence_response,
    calculate_score,
    get_global_grader,
)


def _env_str(key: str, default: str = "") -> str:
    val = os.getenv(key)
    return default if val is None else val


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or str(val).strip() == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or str(val).strip() == "":
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def is_enabled() -> bool:
    return _env_str("RUBRIC_GEN_ENABLE", "0").strip().lower() in ("1", "true", "yes", "on")


def rubric_temperature(is_validate: bool) -> float:
    train_temp = _env_float("RUBRIC_GEN_TEMPERATURE", 1.0)
    val_temp = _env_float("RUBRIC_GEN_TEMPERATURE_VAL", train_temp)
    return val_temp if is_validate else train_temp


def is_v2() -> bool:
    """Pipeline selector. v1 = pseudo-ref -> rubric. v2 = + boundary + exemplars.

    Everything downstream of rubric generation (schema=points, batch judge, scoring)
    is identical between v1 and v2 -- v2 only inserts two extra generation stages that
    feed exemplar *strings* into the same rubric prompt as inspiration.
    """
    return _env_str("RUBRIC_GEN_MODE", "v1").strip().lower() == "v2"


def n_exemplars() -> int:
    return _env_int("RUBRIC_GEN_N_EXEMPLARS", 2)


def _boundary_temperature() -> float:
    return _env_float("BOUNDARY_TEMPERATURE", _env_float("PSEUDO_REF_TEMPERATURE", 0.7))


def _exemplar_temperature(is_validate: bool) -> float:
    return _env_float("EXEMPLAR_TEMPERATURE", rubric_temperature(is_validate))


# --- Stage 1: pseudo-reference synthesis ------------------------------------------

def build_pseudo_reference_prompt(
    prompt: List[Dict[str, str]], responses: List[str], ground_truth: Any
) -> str:
    prompt_str = _format_prompt_messages(prompt) if isinstance(prompt, list) else str(prompt)
    candidates = "\n\n".join(f"[Candidate {i + 1}]\n{r}" for i, r in enumerate(responses) if r and r.strip())
    reference_block = ""
    if ground_truth not in (None, "", {}):
        reference_block = (
            f"\nYou may also consult this reference material, but do not copy it verbatim:\n"
            f"<ReferenceMaterial>\n{ground_truth}\n</ReferenceMaterial>\n"
        )

    return f"""You are an expert. Below is a user prompt and several candidate responses written by \
different models. Synthesise them into ONE ideal, comprehensive, correct reference answer that \
combines the strengths of the candidates and fixes their mistakes.

Output ONLY the reference answer text, with no preamble, commentary, or JSON.

<Prompt>
{prompt_str}
</Prompt>
{reference_block}
<Candidates>
{candidates}
</Candidates>"""


async def generate_pseudo_reference(
    prompt: List[Dict[str, str]], responses: List[str], ground_truth: Any = None
) -> str:
    """Synthesise the group's rollouts into one pseudo-reference answer. '' on failure."""
    valid_responses = [r for r in responses if isinstance(r, str) and r.strip()]
    if not prompt or not valid_responses:
        return ""
    sampler = get_global_grader()  # frozen base model, shared with judge
    gen_prompt = build_pseudo_reference_prompt(prompt, valid_responses, ground_truth)
    temperature = _env_float("PSEUDO_REF_TEMPERATURE", 0.7)
    try:
        resp = await sampler([{"role": "user", "content": gen_prompt}], temperature=temperature)
    except Exception as e:  # never crash the reward batch
        print(f"[pseudo-ref error] synthesis request failed: {e}")
        return ""
    text = resp.response_text or ""
    return text.strip()


# --- Stage 2: rubric generation from the pseudo-reference -------------------------

# --- Stage 2 (v2 only): boundary answer -------------------------------------------
# Prompt copied verbatim from evolving-rubric/.../prompts/boundary.py.

_BOUNDARY_SYSTEM = (
    "You are an expert answer writer. Given a strong correct answer, you produce "
    "a subtly incorrect variant — one that looks plausible but fails on one or two "
    "specific reasoning steps. The wrong answer should be close to the correct one, "
    "not obviously bad."
)


def build_boundary_prompt(prompt: List[Dict[str, str]], pseudo_reference: str) -> tuple[str, str]:
    question = _format_prompt_messages(prompt) if isinstance(prompt, list) else str(prompt)
    user = f'''You are given a QUESTION and a CORRECT ANSWER.

Write a BORDERLINE-WRONG answer: a response that looks plausible and is similar
in style to the correct answer, but fails on one or two specific reasoning steps.

Rules:
- Keep it close to the correct answer — same general approach, similar length.
- Introduce a subtle error (e.g. a wrong conclusion, a flawed reasoning step, a
  missing key condition) — not an obvious or absurd mistake.
- Do NOT make it completely wrong or radically different.

Return ONLY the wrong answer text (no preamble, no labels, no JSON).

QUESTION:
"""{question}"""

CORRECT ANSWER:
"""{pseudo_reference}"""'''
    return _BOUNDARY_SYSTEM, user


async def generate_boundary_answer(prompt: List[Dict[str, str]], pseudo_reference: str) -> str:
    """v2 stage 2: y* -> borderline-wrong variant y-. '' on failure."""
    if not prompt or not pseudo_reference:
        return ""
    system, user = build_boundary_prompt(prompt, pseudo_reference)
    sampler = get_global_grader()  # frozen base model, shared with judge
    try:
        resp = await sampler(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=_boundary_temperature(),
        )
    except Exception as e:
        print(f"[boundary error] generation request failed: {e}")
        return ""
    return (resp.response_text or "").strip()


# --- Stage 3 (v2 only): exemplar criteria (discriminating y* from y-) --------------
# Prompt copied verbatim from evolving-rubric/.../prompts/boundary.py. We keep only the
# `criterion` strings (weight/category discarded) and feed them into the rubric prompt
# as non-binding inspiration -- nothing is merged into the final rubric.

_EXEMPLAR_SYSTEM = (
    "You are an expert evaluator. Given a correct answer and a borderline-wrong "
    "variant, you identify the specific criteria that separate them — the minimal "
    "distinctions that make the correct answer right and the wrong answer fail."
)


def build_exemplars_prompt(
    prompt: List[Dict[str, str]], pseudo_reference: str, boundary: str, num_examples: int
) -> tuple[str, str]:
    question = _format_prompt_messages(prompt) if isinstance(prompt, list) else str(prompt)
    user = f'''You are given a QUESTION, a CORRECT ANSWER, and a BORDERLINE-WRONG answer.

Identify exactly {num_examples} rubric criteria that separate the CORRECT answer
from the BORDERLINE-WRONG answer — criteria the correct answer satisfies but the
wrong answer fails.

Each criterion MUST have three components:
- "criterion": a clear, self-contained positive statement a good answer satisfies.
- "weight": a number in [0, 1] (weights should sum to roughly 1).
- "category": a short label, e.g. "correctness", "reasoning", "completeness".

Keep criteria specific to THIS question. Phrase so that "satisfied" = good.

Return ONLY valid JSON:
{{"rubrics": [{{"criterion": "...", "weight": 0.0, "category": "..."}}]}}

QUESTION:
"""{question}"""

CORRECT ANSWER:
"""{pseudo_reference}"""

BORDERLINE-WRONG ANSWER:
"""{boundary}"""'''
    return _EXEMPLAR_SYSTEM, user


def parse_exemplar_criteria(resp_text: str) -> List[str]:
    """Extract only the `criterion` strings from the exemplar JSON. [] on failure."""
    if not isinstance(resp_text, str) or not resp_text.strip():
        return []
    candidate = None
    m = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", resp_text, re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1)
    else:
        m = re.search(r"(\{.*\}|\[.*\])", resp_text, re.DOTALL)
        candidate = m.group(1) if m else None
    if candidate is None:
        return []
    try:
        data = json.loads(candidate)
    except Exception:
        return []
    items = data.get("rubrics", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    out: List[str] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                crit = it.get("criterion")
                if isinstance(crit, str) and crit.strip():
                    out.append(crit.strip())
    return out


async def generate_exemplars(
    prompt: List[Dict[str, str]],
    pseudo_reference: str,
    boundary: str,
    num_examples: int,
    *,
    is_validate: bool = False,
) -> List[str]:
    """v2 stage 3: (y*, y-) -> list of discriminating criterion strings. [] on failure."""
    if not prompt or not pseudo_reference or not boundary:
        return []
    system, user = build_exemplars_prompt(prompt, pseudo_reference, boundary, num_examples)
    sampler = get_global_grader()
    try:
        resp = await sampler(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=_exemplar_temperature(is_validate),
        )
    except Exception as e:
        print(f"[exemplar error] generation request failed: {e}")
        return []
    return parse_exemplar_criteria(resp.response_text)


# --- Stage 2 (v1) / Stage 4 (v2): rubric generation -------------------------------

def build_rubric_prompt(
    prompt: List[Dict[str, str]],
    pseudo_reference: str,
    n_items: int,
    example_criteria: List[str] | None = None,
) -> str:
    prompt_str = _format_prompt_messages(prompt) if isinstance(prompt, list) else str(prompt)
    inspiration = ""
    if example_criteria:
        examples_block = "\n".join(f"  Example {i + 1}: {c}" for i, c in enumerate(example_criteria))
        inspiration = (
            "\nFor inspiration, here are a few example criteria that capture key distinctions "
            "for this question. You are NOT required to use them — treat them as a hint about "
            "what matters, then write the best rubric you can from the reference answer itself:\n"
            f"{examples_block}\n"
        )
    return f"""You are an expert evaluator designing a grading rubric. Given a user prompt and an \
ideal reference answer, produce a list of about {n_items} rubric criteria that a high-quality \
response should satisfy.

Requirements:
- Each criterion must be a single, atomic, objectively checkable statement (PRESENT / NOT_PRESENT).
- Assign each criterion a positive integer "points" value reflecting its importance.
- Base the criteria on the substance of the reference answer, but phrase them so they can be \
checked against ANY response (do not reference "the reference answer" in the text).
- Cover correctness, completeness, safety, and clarity as relevant to the prompt.
{inspiration}
Start your response with a valid JSON array that starts with "```json" and ends with "```". \
Each element must be an object with keys "criterion" (string) and "points" (number). Do not \
include any extra text or explanations.

Example response:
```json
[
 {{"criterion": "States that the condition requires urgent medical attention", "points": 5}},
 {{"criterion": "Mentions at least two common symptoms", "points": 3}}
]
```

<Prompt>
{prompt_str}
</Prompt>

<ReferenceAnswer>
{pseudo_reference}
</ReferenceAnswer>"""


def parse_generated_rubrics(resp_text: str) -> List[Dict[str, Any]]:
    """Robustly extract a rubric list; returns [] on failure so the caller can fall back.

    Mirrors the defensive parsing in ``_parse_presence_response``: prefer a fenced
    ```json block, then fall back to any bracketed array. Returns normalized dicts
    ({criterion, points, tags}) suitable for ``RubricItem.from_dict``.
    """
    if not isinstance(resp_text, str) or not resp_text.strip():
        return []

    def _coerce(items: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(items, list):
            return out
        for it in items:
            if not isinstance(it, dict):
                continue
            criterion = it.get("criterion")
            if not isinstance(criterion, str) or not criterion.strip():
                continue
            try:
                points = float(it.get("points", 1.0))
            except (TypeError, ValueError):
                points = 1.0
            if points <= 0:
                points = 1.0  # generator must assign positive weight; default otherwise
            tags = it.get("tags", {})
            if not isinstance(tags, (dict, list)):
                tags = {}
            out.append({"criterion": criterion.strip(), "points": points, "tags": tags})
        return out

    match = re.search(r"```json\s*(\[.*?\])\s*```", resp_text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return _coerce(json.loads(match.group(1)))
        except Exception:
            pass

    match = re.search(r"\[.*\]", resp_text, re.DOTALL)
    if match:
        cleaned = re.sub(r",\s*]", "]", match.group(0).strip())
        try:
            return _coerce(json.loads(cleaned))
        except Exception:
            print("[rubric-gen debug] failed to parse generated rubric array")
            print(resp_text)
    return []


async def generate_rubrics_from_reference(
    prompt: List[Dict[str, str]],
    pseudo_reference: str,
    *,
    temperature: float | None = None,
    n_items: int | None = None,
    example_criteria: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Generate a rubric list from a pseudo-reference. Returns [] on failure.

    `example_criteria` (v2 exemplar strings) are injected as non-binding inspiration.
    """
    if not prompt or not pseudo_reference:
        return []
    if n_items is None:
        n_items = _env_int("RUBRIC_GEN_N_ITEMS", 10)

    sampler = get_global_grader()  # frozen base model, shared with judge
    gen_prompt = build_rubric_prompt(prompt, pseudo_reference, n_items, example_criteria)
    try:
        resp = await sampler([{"role": "user", "content": gen_prompt}], temperature=temperature)
    except Exception as e:
        print(f"[rubric-gen error] generation request failed: {e}")
        return []

    rubric_dicts = parse_generated_rubrics(resp.response_text)
    validated: List[Dict[str, Any]] = []
    for d in rubric_dicts:
        try:
            validated.append(RubricItem.from_dict(d).to_dict())
        except Exception:
            continue
    return validated


# --- Orchestration: one group -> (pseudo_reference, rubrics) ----------------------

async def generate_group_rubrics(
    prompt: List[Dict[str, str]],
    responses: List[str],
    ground_truth: Any = None,
    *,
    is_validate: bool = False,
    n_items: int | None = None,
) -> Tuple[str, List[Dict[str, Any]], str, List[str]]:
    """Full generation pipeline for a single prompt-group.

    v1: rollouts -> pseudo-reference -> rubric.
    v2: rollouts -> pseudo-reference -> boundary answer -> exemplars -> rubric
        (exemplars injected as inspiration; identical rubric schema + downstream judging).

    Returns ``(pseudo_reference, rubric_dicts, boundary, example_criteria)``. On any
    failure ``rubric_dicts`` is ``[]`` and the caller falls back to static rubrics.
    ``boundary``/``example_criteria`` are empty in v1 (and in v2 when the guard trips).
    """
    pseudo_reference = await generate_pseudo_reference(prompt, responses, ground_truth)
    if not pseudo_reference:
        return "", [], "", []

    boundary = ""
    example_criteria: List[str] = []
    if is_v2():
        boundary = await generate_boundary_answer(prompt, pseudo_reference)
        # Guard: if the model ignored "make it subtly wrong" and echoed y* verbatim,
        # skip exemplars and fall back to the plain (no-inspiration) rubric prompt.
        if boundary and boundary.strip() != pseudo_reference.strip():
            example_criteria = await generate_exemplars(
                prompt, pseudo_reference, boundary, n_exemplars(), is_validate=is_validate
            )

    rubrics = await generate_rubrics_from_reference(
        prompt,
        pseudo_reference,
        temperature=rubric_temperature(is_validate),
        n_items=n_items,
        example_criteria=example_criteria,
    )
    return pseudo_reference, rubrics, boundary, example_criteria


# --- Checkpoint trace: per-group record of the REAL pipeline artifacts -----------
#
# Written once per traced step (by default the checkpoint steps, save_freq), by the
# driver process only. Everything recorded here is an artifact that was ACTUALLY used
# during training: the pseudo-reference and rubric that graded the group, and the
# judge's real response for each rollout at the real training temperature.
#
# There is deliberately NO judge re-run. An earlier version re-ran the judge at
# temperature 0 purely for display, which meant the scores in the trace were not the
# scores that trained the model. That path is deleted, not flag-gated, so it cannot
# silently come back.
#
# Rollout texts are omitted by default (RUBRIC_TRACE_INCLUDE_ROLLOUTS=1 to include):
# verl's native rollout_log already stores them every step.


def trace_enabled() -> bool:
    return _env_str("RUBRIC_TRACE_ENABLE", "0").strip().lower() in ("1", "true", "yes", "on")


def _trace_every(save_freq: int | None) -> int:
    """Step interval. 0/unset -> follow the trainer's save_freq (checkpoint steps)."""
    every = _env_int("RUBRIC_TRACE_EVERY", 0)
    if every > 0:
        return every
    return int(save_freq) if save_freq and int(save_freq) > 0 else 0


def trace_first_step() -> bool:
    """Also trace the very first training step (default on)."""
    return _env_str("RUBRIC_TRACE_FIRST_STEP", "1").strip().lower() in ("1", "true", "yes", "on")


def should_trace_step(global_step: Any, save_freq: int | None) -> bool:
    """True when this step's groups should be traced.

    Fires on multiples of RUBRIC_TRACE_EVERY (or trainer.save_freq when that is 0),
    plus the first training step so a run can be eyeballed immediately instead of
    after a whole interval.

    Note there is no step 0 here: global_steps is 0 only during the pre-training
    validation, and the rubric pipeline does not run during validation at all, so the
    earliest traceable step is 1.

    Returns False when the step is unknown -- writing an unattributable trace is worse
    than writing none.
    """
    if not trace_enabled():
        return False
    if global_step is None:
        print(
            "[rubric-trace] global_step is None -- skipping. The reward batch is missing "
            "meta_info['global_steps']; see docs_rubricgen/checkpoint_trace_logging_plan.md",
            flush=True,
        )
        return False
    try:
        step = int(global_step)
    except (TypeError, ValueError):
        return False
    if step == 1 and trace_first_step():
        return True
    every = _trace_every(save_freq)
    if every <= 0:
        return False
    return step % every == 0


def trace_max_groups() -> int:
    """Groups to record per traced step. 0 = all of them."""
    return _env_int("RUBRIC_TRACE_MAX_GROUPS", 0)


def trace_include_rollouts() -> bool:
    return _env_str("RUBRIC_TRACE_INCLUDE_ROLLOUTS", "0").strip().lower() in ("1", "true", "yes", "on")


def _trace_dir() -> str:
    return _env_str("RUBRIC_TRACE_DIR", _env_str("RUBRIC_GEN_DUMP_DIR", "log/gen_trace"))


def _trace_format() -> str:
    fmt = _env_str("RUBRIC_TRACE_FORMAT", "jsonl").strip().lower()
    return fmt if fmt in ("jsonl", "txt", "both") else "jsonl"


def build_group_record(
    *,
    global_step: Any,
    uid: str,
    prompt: Any,
    ground_truth: Any,
    pseudo_reference: str,
    rubrics: List[Dict[str, Any]],
    boundary: str = "",
    example_criteria: List[str] | None = None,
    judged: List[Dict[str, Any]] | None = None,
    responses: List[str] | None = None,
) -> Dict[str, Any]:
    """Assemble one group's trace record from artifacts already produced by training."""
    rubric_items = [RubricItem.from_dict(r) for r in (rubrics or [])]
    record: Dict[str, Any] = {
        "global_step": global_step,
        "uid": str(uid),
        "mode": "v2" if (boundary or example_criteria) else "v1",
        "input_prompt": _format_prompt_messages(prompt) if isinstance(prompt, list) else str(prompt),
        "ground_truth": ground_truth,
        "pseudo_reference": pseudo_reference,
        "rubric": [{"criterion": it.criterion, "points": it.points} for it in rubric_items],
        "judge": judged or [],
    }
    if boundary or example_criteria:
        record["boundary_answer"] = boundary or ""
        record["exemplar_criteria"] = list(example_criteria or [])
    if responses is not None:
        record["rollouts"] = responses
    return record


def _record_to_text(rec: Dict[str, Any]) -> str:
    sep, hsep = "=" * 80, "#" * 80
    parts: List[str] = [
        sep,
        f"GENERATION TRACE  |  step={rec.get('global_step')}  |  uid={rec.get('uid')}  |  mode={rec.get('mode')}",
        sep,
        "\n### INPUT PROMPT\n" + str(rec.get("input_prompt", "")),
        "\n" + hsep + "\n# PSEUDO-REFERENCE\n" + hsep,
        str(rec.get("pseudo_reference", "")),
    ]
    if "boundary_answer" in rec:
        parts.append("\n" + hsep + "\n# BOUNDARY ANSWER (v2)\n" + hsep)
        parts.append(str(rec.get("boundary_answer") or "<none>"))
        parts.append("\n" + hsep + "\n# EXEMPLAR CRITERIA (v2)\n" + hsep)
        ex = rec.get("exemplar_criteria") or []
        parts.extend([f"  Example {i + 1}: {c}" for i, c in enumerate(ex)] or ["  <none>"])
    parts.append("\n" + hsep + f"\n# RUBRIC ({len(rec.get('rubric', []))} items)\n" + hsep)
    for i, it in enumerate(rec.get("rubric", [])):
        parts.append(f"{i + 1}. (points: {it.get('points')}) {it.get('criterion')}")
    parts.append("\n" + hsep + "\n# JUDGE (real training output, per rollout)\n" + hsep)
    for j in rec.get("judge", []):
        parts.append(f"\n--- Rollout {j.get('rollout_index')}  |  score={j.get('score')} ---")
        parts.append("JUDGE OUTPUT:\n" + str(j.get("judge_raw", "")))
    if "rollouts" in rec:
        parts.append("\n" + hsep + "\n# ROLLOUTS\n" + hsep)
        for i, r in enumerate(rec["rollouts"]):
            parts.append(f"\n--- Rollout {i + 1} ---\n{r}")
    return "\n".join(parts) + "\n"


def write_step_traces(global_step: Any, records: List[Dict[str, Any]]) -> None:
    """Write all traced groups for one step. Never raises into the training loop."""
    if not records:
        return
    fmt = _trace_format()
    try:
        out_dir = _trace_dir()
        os.makedirs(out_dir, exist_ok=True)
        if fmt in ("jsonl", "both"):
            path = os.path.join(out_dir, f"step{global_step}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            print(f"[rubric-trace] wrote {len(records)} group traces -> {path}", flush=True)
        if fmt in ("txt", "both"):
            for i, rec in enumerate(records):
                uid_tag = str(rec.get("uid", "na"))[:8]
                path = os.path.join(out_dir, f"group_{i:02d}_step{global_step}_{uid_tag}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(_record_to_text(rec))
            print(f"[rubric-trace] wrote {len(records)} group traces (txt) -> {out_dir}", flush=True)
    except Exception as e:
        print(f"[rubric-trace] failed to write traces: {e}", flush=True)
