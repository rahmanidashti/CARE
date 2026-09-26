# CARE — rubrics that learn from a near-miss

Reference implementation of **contrastively anchored rubric generation** for
reinforcement learning in non-verifiable domains.

The problem this solves: outside math and code there is no checker, so the reward
has to come from a rubric — and every existing way of getting one reaches outside
the policy, for a human-written rubric, a reference answer, or a stronger model to
write and grade with. This pipeline reaches for none of them. Everything — the
criteria, the weights, the verdicts — is produced by `π₀`, a frozen copy of the
policy taken before training starts.

The obstacle, and the fix, in one paragraph. If you ask a model what makes a good
answer good and show it only a good answer, it *describes* that answer: *covers
the main mechanisms*, *is clearly structured*. Criteria like these are satisfied
by every rollout in the group, so the reward sits high and flat and a
group-standardised advantage has nothing to work with. So we add a negative, and
get it the same way we got the positive — by asking. The frozen model writes a
**near-miss**: an answer of the same shape that breaks on one or two specific
steps. The criteria that separate the two then anchor the rubric the reward is
computed from. Two extra generation calls per group; nothing else.

```
                                              ┌──────────── CARE only ──────────┐
rollouts ──▶ (1) pseudo-reference y⁺ ──▶ (2) near-miss y⁻ ──▶ (3) contrastive c₁…c_m ──┐
                      │                                                                │
                      └────────────────────────▶ (4) rubric R ◀──── as optional hint ──┘
                                                      │
                                       (5) judge each rollout against R
                                                      │
                             reward = Σ wⱼ·1[criterion met] / Σ wⱼ  −  length penalty
```

`docs/pipeline.md` walks through the stages; `docs/integration.md` explains the
one place this attaches to verl.

## The ablation is one environment variable

The matched ablation — **Positive-only**, a rubric built from the synthesised
answer with no negative anywhere — is the same code path with stages 2 and 3
switched off:

```bash
bash scripts/train.sh                        # CARE           (RUBRIC_GEN_MODE=v2)
RUBRIC_GEN_MODE=v1 bash scripts/train.sh     # Positive-only  (stages 2–3 removed)
```

Same rubric schema, same judge, same scoring rule, same optimiser settings, same
number of steps. Nothing else differs, which is the point: it isolates the
negative rather than comparing two pipelines.

## What is here

```
verl_overlay/          copy this over a verl checkout
  verl/utils/reward_score/rubric_reward/
    rubric_generator.py        stages 1–4, the guard, the trace writer
    rurbichub_v1_Medical.py    stage 5: judge, scorer, `compute_score` entry point
  verl/experimental/reward_loop/
    reward_loop.py             verl's file, carrying the one hook we add
prompts/               the five stage prompts, verbatim, as plain text
scripts/               train.sh, start_vllm_judge.sh, check_judge.sh, audit_release.sh
data/                  prepare_healthbench.py, prepare_researchqa.py
docs/                  pipeline.md, integration.md
```

Deliberately **not** here: model weights, datasets, training logs, traces,
checkpoints, or experimental results. This is the method's code and nothing else.

> `rurbichub_v1_Medical.py` keeps its original name so this tree diffs cleanly
> against the working repo the experiments were run from. It is the judge and
> scorer, and it is not medicine-specific — it is the generic rubric reward.

## Quick start

**1. Install verl.** This is an overlay, not a fork. Follow
[verl's installation](https://github.com/volcengine/verl) for your hardware
(torch, vLLM, ray, flash-attn), then:

```bash
git clone https://github.com/volcengine/verl.git
cd verl && pip install -e .
```

**2. Overlay this code and install its own dependencies.**

```bash
cp -r /path/to/care-rubric-rl/verl_overlay/verl/. /path/to/verl/verl/
cp -r /path/to/care-rubric-rl/{scripts,prompts,data,docs} /path/to/verl/
cd /path/to/verl && pip install -r /path/to/care-rubric-rl/requirements.txt
```

`reward_loop.py` is overwritten by that copy. If your verl revision differs from
ours, apply the hook by hand instead — `docs/integration.md` says exactly what it
is and where it goes.

**3. Configure.**

```bash
cp /path/to/care-rubric-rl/.env.example .env
$EDITOR .env          # at minimum VLLM_BASE_URL, VLLM_MODEL, MODEL_PATH, TRAIN_FILE
```

**4. Prepare data.** Download HealthBench and ResearchQA from their own sources
into `raw_data/`, then:

```bash
python data/prepare_healthbench.py --local_dir raw_data/healthbench --output_dir data/health_bench
python data/prepare_researchqa.py  --local_dir raw_data/ResearchQA  --output_dir data/research_qa
```

Each writes a train and a validation `.parquet`. The training split's own rubrics
are never shown to the training loop — they exist only as a per-group fallback and
for validation, so the validation curve means the same thing in every arm.

**5. Start the frozen model, then train.**

```bash
bash scripts/start_vllm_judge.sh    # in its own terminal or tmux session
bash scripts/check_judge.sh         # asserts it returns content, not an empty think block
bash scripts/train.sh
```

Start the judge from the **same checkpoint** you are about to train, and leave it
untouched for the whole run. That is what makes this self-improvement rather than
distillation, and it is the one invariant the code cannot enforce for you.

## Configuration

Everything is environment variables; `.env.example` documents all of them.
The ones that change the method:

| Variable | Default | Meaning |
|---|---|---|
| `RUBRIC_GEN_MODE` | `v2` | `v2` = CARE, `v1` = Positive-only ablation |
| `RUBRIC_GEN_N_ITEMS` | `10` | `k`, criteria per rubric (a soft hint to the writer) |
| `RUBRIC_GEN_N_EXEMPLARS` | `2` | `m`, contrastive criteria elicited from the pair |
| `PSEUDO_REF_TEMPERATURE` | `0.7` | stage 1 |
| `BOUNDARY_TEMPERATURE` | `0.7` | stage 2 |
| `EXEMPLAR_TEMPERATURE` | `1.0` | stage 3 |
| `RUBRIC_GEN_TEMPERATURE` | `1.0` | stage 4 at training time |
| `RUBRIC_GEN_TEMPERATURE_VAL` | `0` | stage 4 at validation time |

Reported runs: `G = 8` rollouts per prompt, batch 64 prompts, mini-batch 32, LR
`1e-6` with 10 warm-up steps, clip `[0.2, 0.28]`, no KL term, no entropy bonus,
responses capped at 2048 tokens with the DAPO overlong-buffer penalty over the
last 1024. Five epochs on HealthBench, two on ResearchQA. `scripts/train.sh`
carries these as defaults.

## Reading what the pipeline did

With `RUBRIC_TRACE_ENABLE=1`, traced steps write one JSONL record per group to
`RUBRIC_TRACE_DIR` containing the pseudo-reference, the near-miss, the contrastive
criteria, the rubric, and the judge's real verdicts. Everything recorded was
actually used during training — there is no judge re-run, so the scores in a trace
are the scores that produced the gradient.

One gotcha when analysing them: the `mode` field reads `v2` whenever a near-miss
exists, **including when the guard tripped** and no contrastive criteria reached
the rubric. The reliable signal that a group got the full treatment is a non-empty
`exemplar_criteria`.

## Requirements

Two GPUs is the comfortable configuration — one for the policy, one for the frozen
model serving all five stages — though both fit on one card at 4B with
`gpu_memory_utilization` turned down. The frozen endpoint receives `4 + G` calls
per group per step against the ablation's `2 + G`; at `G = 8` that is 20% more
calls, none of which touch the policy's forward or backward pass.

## Before you push

`scripts/audit_release.sh` re-checks the tree for credential-shaped strings,
routable IPs, absolute home paths, cluster mounts and a hardcoded wandb entity.
Pass your own terms to catch anything specific to you:

```bash
PRIVATE_TERMS='yourname|your-cluster|your-wandb-entity' bash scripts/audit_release.sh
```

Keep that pattern out of the repo — the point is not to commit the strings you
are scanning for.

## Licence

Apache 2.0, inherited from [verl](https://github.com/volcengine/verl). See
`LICENSE` and `NOTICE` for what is ours and what is verl's. Datasets are not
redistributed; `data/prepare_*.py` reads them from their original sources under
their own licences.
