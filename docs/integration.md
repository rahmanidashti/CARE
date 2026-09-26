# How the pipeline attaches to verl

The method is two files of its own plus **one hook** inside verl. If you are
overlaying onto a different verl revision than ours, this page is what you need;
the rest copies over unchanged.

## Files

| File | Status | What it is |
|---|---|---|
| `verl/utils/reward_score/rubric_reward/rubric_generator.py` | ours, new | Stages 1–4: pseudo-reference, near-miss, contrastive criteria, rubric. Plus the guard and the trace writer. |
| `verl/utils/reward_score/rubric_reward/rurbichub_v1_Medical.py` | ours, new | Stage 5: the batch judge, the scorer, the shared `RubricItem` schema, the pooled async client, and `compute_score` — the entry point verl calls. |
| `verl/experimental/reward_loop/reward_loop.py` | verl's, **modified** | Carries the hook described below. |

The two new files are self-contained: they import from each other and from
`aiohttp`/`dotenv`, nothing else of ours.

## The hook

Everything the method needs from verl is that **one rubric is built per prompt
group, before the batch is split across reward workers**. If the batch were
chunked first, rollouts of the same group could be graded against different
rubrics, and a group-standardised advantage would be comparing them on different
yardsticks.

Three additions to `RewardLoopManager`:

**1. `__init__` — a driver-side tokenizer and one persistent event loop.**
Stage 1 needs the *decoded* text of a whole group before any per-rollout judging,
so the driver decodes rather than the workers. The event loop is created once and
reused: the generator's `aiohttp` session is cached on the loop it was first
created under, so `asyncio.run()` per step would strand the session. Both are
built only when `RUBRIC_GEN_ENABLE=1`, and a failure here logs a warning instead
of blocking init.

**2. `_maybe_generate_rubrics(data)` — the pipeline itself.**
Groups rows by `uid`, runs `rubric_generator.generate_group_rubrics` concurrently
over the groups, and writes the result into every row's
`extra_info["reward_model"]["rubrics"]`. It returns early during validation
(`data.meta_info["validate"]`), which is why validation always scores against the
dataset's own static rubrics and means the same thing in every arm. Any group
that fails generation keeps its static rubrics — the fallback is silent and safe.

**3. `_write_group_traces(outputs_flat)` — the trace writer.**
On traced steps, records the artifacts that were *actually used*: the
pseudo-reference, the near-miss, the contrastive criteria, the rubric, and the
judge's real verdicts. There is deliberately no judge re-run, so a trace never
shows scores other than the ones that trained the model.

Both are called from `compute_rm_score`:

```python
def compute_rm_score(self, data: DataProto) -> DataProto:
    ...
    self._trace_stash = None
    self._maybe_generate_rubrics(data)      # <-- BEFORE the chunking below
    chunks = data.chunk(len(self.reward_loop_workers))
    ...
    self._write_group_traces(outputs_flat)  # <-- after the workers return
```

## Porting to another verl revision

1. Copy the two `rubric_reward/` files in unchanged.
2. In your `reward_loop.py`, add the three members above to `RewardLoopManager`
   and the two call sites in `compute_rm_score`. Lifting them out of the copy in
   `verl_overlay/` is the quickest route — search it for `rubric_generator`.
3. Nothing else in verl is touched. The reward function is selected on the
   command line (`custom_reward_function.path=...`), not by patching a registry.
