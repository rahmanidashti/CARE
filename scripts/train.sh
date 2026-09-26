#!/usr/bin/env bash
# Train a policy with GRPO against self-generated, contrastively anchored rubrics.
#
#   CARE (full pipeline)   RUBRIC_GEN_MODE=v2   <- the method
#   Positive-only (ablation) RUBRIC_GEN_MODE=v1 <- stages 2-3 removed
#
# Those two arms differ by this one variable and nothing else: same rubric schema,
# same judge, same scoring rule, same optimiser settings. Everything else in this
# file is shared.
#
# Per GRPO group, all stages served by the frozen model at VLLM_BASE_URL:
#   rollouts -> y+  pseudo-reference      (stage 1)
#            -> y-  near-miss             (stage 2, v2 only)
#            -> c   contrastive criteria  (stage 3, v2 only)
#            -> R   rubric                (stage 4; c is shown as optional inspiration)
#            -> judge each rollout against R, reward = weighted fraction met (stage 5)
#
# Guard: if y- comes back equal to y+, or stage 3 does not parse, the contrastive
# criteria are dropped and stage 4 uses the plain prompt -- i.e. the group degrades
# to the ablation, never to anything worse.
#
# Usage:
#   bash scripts/train.sh                       # CARE, defaults below
#   RUBRIC_GEN_MODE=v1 bash scripts/train.sh    # Positive-only ablation
#   MODEL_PATH=Qwen/Qwen3-8B bash scripts/train.sh
#
# Any variable below can be overridden from the environment or from .env.

set -euo pipefail

# --- load .env if present (see .env.example) ---------------------------------
[ -f ".env" ] && set -a && . .env && set +a

# --- what to train -----------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}        # HF hub id or a local path
TRAIN_FILE=${TRAIN_FILE:-data/healthbench/train.parquet}
VAL_FILE=${VAL_FILE:-data/healthbench/val.parquet}
EXP_NAME=${EXP_NAME:-care-qwen3-4b}
PROJECT_NAME=${PROJECT_NAME:-care-rubric-rl}

# --- the rubric pipeline -----------------------------------------------------
export RUBRIC_GEN_ENABLE=1
export RUBRIC_GEN_MODE=${RUBRIC_GEN_MODE:-v2}                      # v2 = CARE, v1 = ablation
export RUBRIC_GEN_N_ITEMS=${RUBRIC_GEN_N_ITEMS:-10}                # k, criteria per rubric
export RUBRIC_GEN_N_EXEMPLARS=${RUBRIC_GEN_N_EXEMPLARS:-2}         # m, contrastive criteria (v2)
export RUBRIC_GEN_TEMPERATURE=${RUBRIC_GEN_TEMPERATURE:-1.0}       # stage 4, training
export RUBRIC_GEN_TEMPERATURE_VAL=${RUBRIC_GEN_TEMPERATURE_VAL:-0} # stage 4, validation
export PSEUDO_REF_TEMPERATURE=${PSEUDO_REF_TEMPERATURE:-0.7}       # stage 1
export BOUNDARY_TEMPERATURE=${BOUNDARY_TEMPERATURE:-0.7}           # stage 2 (v2)
export EXEMPLAR_TEMPERATURE=${EXEMPLAR_TEMPERATURE:-1.0}           # stage 3 (v2)
export RUBRIC_GEN_DEBUG=${RUBRIC_GEN_DEBUG:-0}

# --- per-group traces (what the pipeline actually produced) ------------------
# Records the real artifacts used during training -- pseudo-reference, near-miss,
# contrastive criteria, rubric, and the judge's real verdicts. No judge re-run, so
# the scores in a trace are the scores that trained the model. Validation never runs
# the pipeline. data_log_freq below must match RUBRIC_TRACE_EVERY: a trace joins back
# to verl's rollout_log by row.
export RUBRIC_TRACE_ENABLE=${RUBRIC_TRACE_ENABLE:-1}
export RUBRIC_TRACE_EVERY=${RUBRIC_TRACE_EVERY:-100}
export RUBRIC_TRACE_FIRST_STEP=${RUBRIC_TRACE_FIRST_STEP:-1}
export RUBRIC_TRACE_MAX_GROUPS=${RUBRIC_TRACE_MAX_GROUPS:-0}             # 0 = all groups
export RUBRIC_TRACE_INCLUDE_ROLLOUTS=${RUBRIC_TRACE_INCLUDE_ROLLOUTS:-0} # rollout_log has these
export RUBRIC_TRACE_FORMAT=${RUBRIC_TRACE_FORMAT:-jsonl}                 # jsonl | txt | both
export RUBRIC_TRACE_DIR=${RUBRIC_TRACE_DIR:-log/gen_trace/${PROJECT_NAME}/${EXP_NAME}}

# --- optimiser ---------------------------------------------------------------
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}                      # G, rollouts per prompt
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}                # paper: 5 on HealthBench, 2 on ResearchQA
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
MODEL_DTYPE=${MODEL_DTYPE:-bf16}
NUM_GPUS=${NUM_GPUS:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
USE_DYNAMIC_BSZ=True

# DAPO overlong-response penalty: linear from MAX_RESPONSE_LENGTH-OVERLONG_BUFFER_LEN,
# reaching -1.0 at the cap.
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-True}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-1024}
OVERLONG_BUFFER_PENALTY_FACTOR=${OVERLONG_BUFFER_PENALTY_FACTOR:-1}

MAX_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_PPO_MAX_TOKEN_LEN=$((MAX_TOKENS * 1))
INFER_PPO_MAX_TOKEN_LEN=$((MAX_TOKENS * 1))
MAX_NUM_BATCHED_TOKENS=$((MAX_TOKENS * 1))
VAL_ONLY=${VAL_ONLY:-False}

# --- the frozen model that writes and grades rubrics -------------------------
# Start it first:  bash scripts/start_vllm_judge.sh
: "${VLLM_BASE_URL:?set VLLM_BASE_URL (see .env.example) -- start scripts/start_vllm_judge.sh first}"
: "${VLLM_MODEL:?set VLLM_MODEL (see .env.example)}"
curl -sf "${VLLM_BASE_URL%/v1}/v1/models" >/dev/null 2>&1 || {
  echo "FATAL: no vLLM endpoint at ${VLLM_BASE_URL}. Run: bash scripts/start_vllm_judge.sh" >&2
  exit 1; }

echo "[care] mode=${RUBRIC_GEN_MODE}  model=${MODEL_PATH}  G=${ROLLOUT_N}  k=${RUBRIC_GEN_N_ITEMS}  m=${RUBRIC_GEN_N_EXEMPLARS}"

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    reward_model.reward_manager=dapo \
    reward_model.num_workers=1 \
    reward_model.use_reward_loop=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${ENABLE_OVERLONG_BUFFER} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_BUFFER_PENALTY_FACTOR} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
    custom_reward_function.path=verl/utils/reward_score/rubric_reward/rurbichub_v1_Medical.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE} \
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=${MAX_TOKENS} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=${MODEL_DTYPE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.rollout_data_dir="log/rollout_log/${PROJECT_NAME}/${EXP_NAME}" \
    trainer.validation_data_dir="log/validation_log/${PROJECT_NAME}/${EXP_NAME}" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    +trainer.data_log_freq=100 \
    trainer.save_freq=350 \
    trainer.test_freq=5 \
    trainer.val_only=${VAL_ONLY} \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    trainer.total_epochs=${TOTAL_EPOCHS} "$@"
