# The pipeline, stage by stage

One GRPO group = one prompt `q` and its `G` rollouts `o_1 … o_G`. Every stage
below is served by `π₀`, a frozen copy of the policy taken before training, at
`VLLM_BASE_URL`. No human labels, no reference answers, no second model.

```
                                              ┌──────────── v2 only ────────────┐
rollouts ──▶ (1) pseudo-reference y⁺ ──▶ (2) near-miss y⁻ ──▶ (3) contrastive c₁…c_m ──┐
                      │                                                                │
                      └────────────────────────▶ (4) rubric R ◀──── as optional hint ──┘
                                                      │
                                       (5) judge each rollout against R
                                                      │
                             reward = Σ wⱼ·1[criterion met] / Σ wⱼ  −  length penalty
```

`RUBRIC_GEN_MODE` selects the arm, and it is the **only** functional difference
between them:

| | `v1` — Positive-only (ablation) | `v2` — CARE |
|---|---|---|
| Generation stages | 3 | 5 |
| Extra stages | — | near-miss, contrastive criteria |
| Rubric schema | `{criterion, points:int}` | same |
| Judge | batch PRESENT/NOT_PRESENT, one call per rollout | same |
| Score | `Σ points(met) / Σ points` | same |
| Calls per group | `2 + G` | `4 + G` |

At `G = 8` that is a 20% increase in calls to the frozen model, none of which
touch the policy's forward or backward pass. The two extra calls are per *group*,
so the overhead shrinks as `G` grows.

## Stage 1 — pseudo-reference `y⁺`

`build_pseudo_reference_prompt` → `generate_pseudo_reference`. All `G` rollouts go
in; one synthesised answer comes out — the points they agree on, with
contradictions and mistakes dropped. Temperature `PSEUDO_REF_TEMPERATURE`.

Because `y⁺` is derived from the whole group and the rubric is derived from `y⁺`,
every rollout in the group is graded against the same yardstick by construction —
which is what a group-standardised advantage requires.

## Stage 2 — near-miss `y⁻` *(v2)*

`build_boundary_prompt` → `generate_boundary_answer`. Given `q` and `y⁺`, write an
answer of the same shape and similar length that fails on one or two specific
reasoning steps. The instruction is explicit that the error must be subtle, not an
obviously bad answer: a wildly wrong negative only yields criteria every rollout
already passes. Temperature `BOUNDARY_TEMPERATURE`.

## Stage 3 — contrastive criteria *(v2)*

`build_exemplars_prompt` → `generate_exemplars` → `parse_exemplar_criteria`. Given
`q`, `y⁺` and `y⁻`, name the `m` criteria that separate them — statements `y⁺`
satisfies and `y⁻` fails. The stage returns JSON objects but **only the criterion
strings are kept**; any weights or categories it invents are discarded.
`m = RUBRIC_GEN_N_EXEMPLARS`, temperature `EXEMPLAR_TEMPERATURE`.

## Stage 4 — rubric `R`

`build_rubric_prompt` → `generate_rubrics_from_reference` →
`parse_generated_rubrics`. About `k = RUBRIC_GEN_N_ITEMS` atomic, objectively
checkable criteria with positive integer `points`, written from `y⁺`.

The stage-3 criteria are shown as **non-binding inspiration** — the prompt says
the model is not required to use them — and nothing is merged: stage 4
regenerates all `k` criteria from scratch. `y⁻` itself is never passed to stage 4;
it is consumed entirely by stage 3 and discarded.

This coupling is deliberate. `y⁻` is synthetic, so a contrastive criterion may
reflect an artefact of how the model chose to break the answer rather than a real
quality difference, and a criterion with variance but no signal is worse for the
gradient than one with neither. Used as a hint, the criteria steer what the model
attends to while the rubric stays grounded in `y⁺`.

## Guard

In `generate_group_rubrics`: if `y⁻` comes back equal to `y⁺` after stripping, or
stage 3 returns nothing parseable, the contrastive criteria are dropped and stage
4 uses the plain prompt — byte-identical to the v1 prompt minus the inspiration
paragraph. **The method degrades to the ablation, never to anything worse.**

Aligned models do sometimes decline to corrupt an answer they have just endorsed;
this is the path that catches it.

## Stage 5 — judge and score

`_build_batch_grader_prompt` → `_parse_presence_response` → `calculate_score`, all
in `rurbichub_v1_Medical.py`. One call per rollout grades every criterion at once
and returns PRESENT / NOT_PRESENT per item. The score is the weighted fraction met,
clipped to `[0, 1]`, with verl's DAPO overlong-response penalty subtracted on top.

If rubric generation failed for a group, the dataset's static rubrics are used
instead. Validation always uses the static rubrics, so the validation curve means
the same thing in every arm.

## Traces

With `RUBRIC_TRACE_ENABLE=1`, each traced step writes one JSONL record per group to
`RUBRIC_TRACE_DIR`: `pseudo_reference`, `boundary_answer`, `exemplar_criteria`,
`rubric`, and the judge's real per-rollout verdicts. Everything recorded was
actually used during training — there is no judge re-run.

One caveat when reading traces: the `mode` field is set to `v2` whenever a boundary
answer exists, **including when the guard tripped** and no contrastive criteria
shaped the rubric. The reliable signal for "this group got the full treatment" is a
non-empty `exemplar_criteria`, not `mode`.
