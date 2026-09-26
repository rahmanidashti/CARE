# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
from typing import Any

import aiohttp
import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

from .reward_manager import get_reward_manager_cls
from .reward_model import RewardModelManager

logger = logging.getLogger(__file__)

# Keys that compute_score returns purely so the driver can write a faithful rubric
# trace. They are read directly from the worker outputs and then dropped, so they
# never enter non_tensor_batch / verl's native rollout_log dump.
_TRACE_ONLY_KEYS = frozenset({"judge_raw", "judge_grading", "judge_temperature"})
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_debug_reward_flow_count = 0


def _as_object_array(values):
    arr = np.empty(len(values), dtype=object)
    arr[:] = values
    return arr

def _debug_reward_flow_log(tag: str, payload: dict) -> None:
    if os.getenv("DEBUG_REWARD_FLOW", "0") != "1":
        return
    global _debug_reward_flow_count
    try:
        limit = int(os.getenv("DEBUG_REWARD_FLOW_LIMIT", "5"))
    except ValueError:
        limit = 5
    if _debug_reward_flow_count >= limit:
        return
    _debug_reward_flow_count += 1
    record = {"tag": tag, **payload}
    try:
        with open("reward_flow_debug.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write reward flow log: {e}")


@ray.remote
class RewardLoopWorker:
    def __init__(self, config: DictConfig, reward_router_address: str = None):
        """
        RewardLoopWork can tackle reward computation:
        (1) rule-based reward computation
        (2) reward model-based reward computation (both disrm and genrm)
        (3) high-flexible user-customized reward function (can access rm by posting requests to reward_model_router)

        Reward Computation Logic:
        - if user-customized reward function is provided:
            -> directly use user-customized reward function
        - if user-customized reward function is not provided:
            -> rm is not enabled: use default rule-based reward function
            -> rm is disrm: compute reward score using disrm
            -> rm is genrm: raise error (user-costomized reward func must be provided)

        Args:
            config: DictConfig, the config for reward loop worker.
            reward_router_address: str, the address of reward router.
        """
        self.config = config
        self.reward_router_address = reward_router_address
        self._init_reward_fn()

    def _init_reward_fn(self):
        input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = None
        if self.config.reward_model.enable:
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward_model.model.path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)
        self.reward_fn = get_custom_reward_fn(self.config)

        # Load reward loop manager class
        # Support both registry and importlib loading methods
        reward_loop_source = self.config.reward_model.get("reward_loop_source", "register")

        if reward_loop_source == "register":
            # Load from registry (default behavior)
            reward_manager_cls = get_reward_manager_cls(self.config.reward_model.reward_manager)
        elif reward_loop_source == "importlib":
            # Load from external module using importlib
            from verl.utils.import_utils import load_extern_object

            reward_loop_module_path = self.config.reward_model.get("reward_loop_module_path", None)
            reward_loop_class_name = self.config.reward_model.get("reward_loop_class_name", None)

            assert reward_loop_module_path is not None, (
                "reward_loop_module_path must be set when reward_loop_source='importlib'"
            )
            assert reward_loop_class_name is not None, (
                "reward_loop_class_name must be set when reward_loop_source='importlib'"
            )

            reward_manager_cls = load_extern_object(
                module_path=reward_loop_module_path, object_name=reward_loop_class_name
            )
        else:
            raise ValueError(f"Unknown reward_loop_source: {reward_loop_source}. Must be 'register' or 'importlib'")

        self.reward_loop = reward_manager_cls(
            self.config, self.input_tokenizer, self.reward_fn, self.reward_router_address, self.reward_model_tokenizer
        )

    async def compute_score_batch(self, data: DataProto) -> list[dict]:
        tasks = []
        for i in range(len(data)):
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
        outputs = await asyncio.gather(*tasks)
        return outputs

    async def compute_score(self, data: DataProto) -> dict:
        assert len(data) == 1, "RewardLoopWorker only support single data item"
        if self.config.custom_reward_function.path is not None:
            # directly use user-customized reward function
            return await self.reward_loop.run_single(data)
        else:
            if self.config.reward_model.enable:
                # we assume the rm is disrm
                # genrm must set custom_reward_function
                return await self.compute_score_disrm(data)
            else:
                return await self.reward_loop.run_single(data)

    async def _post_request(self, payload: dict, endpoint: str, max_retries: int = 16):
        url = f"http://{self.reward_router_address}/{endpoint}"
        last_exception = None
        for attempt in range(max_retries):
            try:
                # It's safer to have a timeout instead of None, which can hang indefinitely.
                timeout = aiohttp.ClientTimeout(total=None)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientResponseError as e:
                # Do not retry on 4xx client errors, but retry on 5xx server errors.
                if 400 <= e.status < 500:
                    logger.error(f"Request to {url} failed with client error HTTP {e.status}: {e}. Not retrying.")
                    raise
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with HTTP {e.status}: {e}. "
                    "Retrying..."
                )
            except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:
                last_exception = e
                logger.warning(f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed: {e}. Retrying...")
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with unexpected error: {e}. "
                    "Retrying..."
                )

            if attempt < max_retries - 1:
                # Using exponential backoff is generally better than a fixed sleep.
                backoff_seconds = 2**attempt
                await asyncio.sleep(min(backoff_seconds, 30))

        logger.error(f"Max retries ({max_retries}) reached for request to {url}.")
        if last_exception:
            raise last_exception

    async def _preprocess_reward_inputs(self, data: DataProto) -> str:
        assert len(data) == 1, "RewardLoopWorker only support single data item"
        data_item = data[0]
        assert "raw_prompt" in data_item.non_tensor_batch

        # extract raw prompt
        chat: list = list(data_item.non_tensor_batch["raw_prompt"])

        # extract response
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        rollout_response = self.input_tokenizer.decode(valid_response_ids)
        # remove bos and eos
        rollout_response = rollout_response.replace(self.input_tokenizer.eos_token, "")

        chat.append({"role": "assistant", "content": rollout_response})

        rm_prompt = self.reward_model_tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=False,
            tokenize=False,
        )

        # llama tokenizer will add bos token by default
        # will be removed in vllm >= 0.11.2, where we can add "add_special_tokens" = False
        if self.reward_model_tokenizer.bos_token is not None and rm_prompt.startswith(
            self.reward_model_tokenizer.bos_token
        ):
            rm_prompt = rm_prompt[len(self.reward_model_tokenizer.bos_token) :]

        return rm_prompt

    async def compute_score_disrm(self, data: DataProto) -> dict:
        disrm_prompt = await self._preprocess_reward_inputs(data)
        engine_name = self.config.reward_model.rollout.name
        model_name = self.config.reward_model.model.path
        if engine_name == "vllm":
            # TODO (dyy): the "activation" has been changed to "use_activation" in vllm 0.11.2
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
                "activation": False,
                # "add_special_tokens": False,  # vllm >= 0.11.2
            }
            output = await self._post_request(payloads, "classify")
            rm_score = output["data"][-1]["probs"][-1]
        elif engine_name == "sglang":
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
            }
            output = await self._post_request(payloads, "v1/embeddings")
            rm_score = output["data"][-1]["embedding"][-1]
        else:
            raise NotImplementedError(f"RewardLoopManager does not support {engine_name}")

        return {"reward_score": rm_score}


class RewardLoopManager:
    """
    RewardLoopManager run in single controller.
    This class will create reward loop workers and manage them.
    RewardLoopManager will deprecate fsdp/megatron RewardModelWorker in the future.
    """

    def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
        self.config = config
        if self.config.reward_model.enable:
            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()
        else:
            self.reward_model_manager = None
            self.reward_router_address = None

        # Tokenizer for driver-side decoding of rollouts, used by the optional
        # on-the-fly rubric generation hook (pseudo-reference synthesis needs the
        # decoded responses of a whole prompt-group before per-rollout judging).
        # Only built when generation is enabled to avoid unnecessary I/O.
        self._gen_tokenizer = None
        self._gen_loop = None
        try:
            from verl.utils.reward_score.rubric_reward import rubric_generator

            if rubric_generator.is_enabled():
                input_tokenizer_local_path = copy_to_local(config.actor_rollout_ref.model.path)
                self._gen_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
                # Persistent driver-side event loop: the generator's aiohttp session is
                # cached on the loop it is first created under, so we must reuse ONE loop
                # across steps rather than asyncio.run() (which opens/closes a new loop
                # each call and would strand the session).
                self._gen_loop = asyncio.new_event_loop()
        except Exception as e:  # never block init on the optional feature
            logger.warning(f"rubric generation tokenizer init skipped: {e}")

        self._init_reward_loop_workers()

    def _init_reward_loop_workers(self):
        self.reward_loop_workers = []
        num_workers = self.config.reward_model.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.reward_loop_workers.append(
                RewardLoopWorker.options(
                    name=f"reward_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.reward_router_address)
            )

    def _decode_valid_responses(self, data: DataProto) -> list[str]:
        """Decode each rollout's valid (unpadded) response to text on the driver."""
        responses = data.batch["responses"]
        prompt_length = data.batch["prompts"].size(1)
        valid_lens = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)
        texts = []
        for i in range(len(data)):
            valid_len = int(valid_lens[i].item())
            ids = responses[i][:valid_len] if valid_len > 0 else responses[i][:0]
            texts.append(self._gen_tokenizer.decode(ids, skip_special_tokens=True))
        return texts

    def _maybe_generate_rubrics(self, data: DataProto) -> None:
        """Stage 1+2 of the on-the-fly pipeline, per prompt-group, before chunking.

        For each ``uid`` group: synthesise a pseudo-reference from all its rollouts,
        generate a rubric from it, and inject that rubric into every group row's
        ``extra_info["reward_model"]["rubrics"]``. Groups that fail generation keep
        their static dataset rubrics (fallback). No-op during validation or when the
        feature is disabled.
        """
        from verl.utils.reward_score.rubric_reward import rubric_generator

        if self._gen_tokenizer is None or not rubric_generator.is_enabled():
            return
        if bool(data.meta_info.get("validate", False)):
            return  # validation uses static dataset rubrics (n=1 -> no meaningful synthesis)
        if "uid" not in data.non_tensor_batch or "extra_info" not in data.non_tensor_batch:
            logger.warning("rubric generation skipped: missing 'uid'/'extra_info' in batch")
            return

        uids = data.non_tensor_batch["uid"]
        extra_infos = data.non_tensor_batch["extra_info"]

        # Group row indices by uid (preserve first-seen order).
        groups: dict[Any, list[int]] = {}
        for i, uid in enumerate(uids):
            groups.setdefault(uid, []).append(i)

        responses_str = self._decode_valid_responses(data)

        # Build one generation coroutine per group.
        order = list(groups.keys())
        coros = []
        for uid in order:
            idxs = groups[uid]
            rep = extra_infos[idxs[0]] if isinstance(extra_infos[idxs[0]], dict) else {}
            prompt = rep.get("prompt")
            rm = rep.get("reward_model") if isinstance(rep.get("reward_model"), dict) else {}
            ground_truth = rm.get("ground_truth")
            group_responses = [responses_str[i] for i in idxs]
            coros.append(
                rubric_generator.generate_group_rubrics(
                    prompt, group_responses, ground_truth, is_validate=False
                )
            )

        async def _gather():
            return await asyncio.gather(*coros, return_exceptions=True)

        try:
            results = self._gen_loop.run_until_complete(_gather())
        except Exception as e:
            logger.warning(f"rubric generation batch failed, using static rubrics: {e}")
            return

        n_ok, n_fallback = 0, 0
        for uid, result in zip(order, results, strict=True):
            if isinstance(result, Exception):
                n_fallback += 1
                continue
            pseudo_ref, rubrics, _boundary, _example_criteria = result
            if not rubrics:  # empty -> keep static dataset rubrics
                n_fallback += 1
                continue
            n_ok += 1
            for i in groups[uid]:
                info = extra_infos[i]
                if not isinstance(info, dict):
                    continue
                rm = info.get("reward_model")
                if not isinstance(rm, dict):
                    rm = {}
                    info["reward_model"] = rm
                rm["rubrics"] = rubrics
                info["pseudo_reference"] = pseudo_ref  # for logging/inspection

        # Stash this step's group artifacts for the trace. Nothing is written here:
        # the judge has not run yet at this point in compute_rm_score (it runs in the
        # workers, below), so a trace written now could only contain a *re-run* of the
        # judge rather than the real thing. The trace is assembled after the workers
        # return -- see _write_group_traces.
        global_step = data.meta_info.get("global_steps")
        if rubric_generator.should_trace_step(global_step, data.meta_info.get("save_freq")):
            include_rollouts = rubric_generator.trace_include_rollouts()
            max_groups = rubric_generator.trace_max_groups()
            stash = []
            for uid, result in zip(order, results, strict=True):
                if isinstance(result, Exception):
                    continue
                pseudo_ref, rubrics, boundary, example_criteria = result
                if not rubrics:
                    continue
                if max_groups > 0 and len(stash) >= max_groups:
                    break
                idxs = groups[uid]
                rep = extra_infos[idxs[0]] if isinstance(extra_infos[idxs[0]], dict) else {}
                rm = rep.get("reward_model") if isinstance(rep.get("reward_model"), dict) else {}
                stash.append(
                    {
                        "uid": uid,
                        "idxs": idxs,
                        "prompt": rep.get("prompt"),
                        "ground_truth": rm.get("ground_truth"),
                        "pseudo_reference": pseudo_ref,
                        "rubrics": rubrics,
                        "boundary": boundary,
                        "example_criteria": example_criteria,
                        "responses": [responses_str[i] for i in idxs] if include_rollouts else None,
                    }
                )
            self._trace_stash = {"global_step": global_step, "groups": stash}

        _debug_reward_flow_log(
            "rubric_generation",
            {"groups": len(order), "generated": n_ok, "fallback": n_fallback},
        )

    def _write_group_traces(self, outputs_flat: list) -> None:
        """Join stashed per-group artifacts with the REAL per-rollout judge output.

        No-op unless this step was selected for tracing. Everything written here was
        actually used to produce this step's rewards -- there is no judge re-run.
        """
        from verl.utils.reward_score.rubric_reward import rubric_generator

        stash = getattr(self, "_trace_stash", None)
        if not stash or not stash.get("groups"):
            return

        def _jsonable_score(v):
            """reward_score arrives as a 0-d torch tensor (or a token-level sequence).
            Serialise it as a real number, not as the string 'tensor(0.45)'."""
            try:
                if hasattr(v, "item"):
                    return float(v.item())
                if isinstance(v, (list, tuple)):
                    return [float(x) for x in v]
                return float(v)
            except Exception:
                return None

        try:
            records = []
            for g in stash["groups"]:
                judged = []
                for local_i, row in enumerate(g["idxs"]):
                    if row >= len(outputs_flat):
                        continue
                    out = outputs_flat[row] or {}
                    info = out.get("reward_extra_info", {}) or {}
                    judged.append(
                        {
                            "rollout_index": local_i + 1,
                            # Global batch row. verl's native rollout_log writes one
                            # line per batch row in order, so this doubles as the line
                            # number in rollout_log/<step>.jsonl -- the join key for
                            # recovering this rollout's text, which the trace omits.
                            "row": int(row),
                            "score": _jsonable_score(out.get("reward_score")),
                            "judge_raw": info.get("judge_raw", ""),
                            "judge_grading": info.get("judge_grading", []),
                            "judge_temperature": info.get("judge_temperature"),
                        }
                    )
                records.append(
                    rubric_generator.build_group_record(
                        global_step=stash["global_step"],
                        uid=g["uid"],
                        prompt=g["prompt"],
                        ground_truth=g["ground_truth"],
                        pseudo_reference=g["pseudo_reference"],
                        rubrics=g["rubrics"],
                        boundary=g["boundary"],
                        example_criteria=g["example_criteria"],
                        judged=judged,
                        responses=g["responses"],
                    )
                )
            rubric_generator.write_step_traces(stash["global_step"], records)
        except Exception as e:
            logger.warning(f"rubric trace write failed: {e}")
        finally:
            self._trace_stash = None

    # this func is used to replace the legacy fsdp/megatron RewardModelWorker.compute_rm_score
    def compute_rm_score(self, data: DataProto) -> DataProto:
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        # Reset every step: the stash holds one step's trace artifacts and must not
        # accumulate across the run.
        self._trace_stash = None

        # Optional: generate per-group rubrics on the fly (pseudo-reference -> rubric)
        # BEFORE chunking, so every rollout of a group shares one rubric regardless of
        # how the batch is split across workers. Mutates extra_info in place; the
        # per-rollout judge downstream then reads the injected rubrics.
        self._maybe_generate_rubrics(data)

        chunks = data.chunk(len(self.reward_loop_workers))
        outputs = ray.get(
            [
                worker.compute_score_batch.remote(chunk)
                for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
            ]
        )
        outputs_flat = [item for sublist in outputs for item in sublist]

        # Now that the real judge has run, write the trace for this step (no-op unless
        # this is a traced step). outputs_flat is row-aligned with `data` -- the same
        # assumption the reward_extra_info scatter below already relies on.
        self._write_group_traces(outputs_flat)

        # compute rm score
        scores = [item["reward_score"] for item in outputs_flat]
        prompt_length = data.batch["prompts"].size(1)
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)
        rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        seq_count = 0
        scalar_count = 0
        for i, output in enumerate(outputs_flat):
            reward_score = output.get("reward_score")
            extra_info = output.get("reward_extra_info", {})
            seq = None
            if isinstance(reward_score, (list, tuple, np.ndarray, torch.Tensor)):
                seq = reward_score
            elif isinstance(extra_info.get("token_level_rewards"), (list, tuple, np.ndarray, torch.Tensor)):
                seq = extra_info.get("token_level_rewards")

            valid_len = int(valid_response_length[i].item())
            if valid_len <= 0:
                continue

            if seq is not None:
                seq_count += 1
                seq_tensor = torch.tensor(seq, dtype=torch.float32).flatten()
                if seq_tensor.numel() == 0:
                    continue
                use_len = min(valid_len, seq_tensor.numel())
                rm_scores[i, :use_len] = seq_tensor[:use_len]
            else:
                scalar_count += 1
                rm_scores[i, valid_len - 1] = float(reward_score)

        _debug_reward_flow_log(
            "reward_loop_rm_scores",
            {"batch_size": len(outputs_flat), "seq_count": seq_count, "scalar_count": scalar_count},
        )
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))

        reward_extra_infos = [output.get("reward_extra_info", {}) for output in outputs_flat]
        # Trace-only keys stay out of the published allowlist. They are consumed above
        # by _write_group_traces straight from outputs_flat; publishing them would send
        # the full judge text into verl's native rollout_log on EVERY step (see
        # ray_trainer._dump_generations, which dumps every allowlisted key), roughly
        # doubling those files for data we only want on checkpoint steps.
        reward_extra_keys = sorted(
            {key for info in reward_extra_infos for key in info.keys()} - _TRACE_ONLY_KEYS
        )
        non_tensor_batch = {}
        for key in reward_extra_keys:
            non_tensor_batch[key] = _as_object_array([info.get(key) for info in reward_extra_infos])

        if self.reward_model_manager is not None:
            self.reward_model_manager.sleep()

        return DataProto(
            batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"reward_extra_keys": reward_extra_keys}
        )

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())
