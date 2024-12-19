import warnings
from typing import List, Optional, Dict
from dataclasses import dataclass
from collections import deque
import time
import math
from itertools import chain

import torch
from trl.core import (
    PPODecorators,
    logprobs_from_logits,
    stack_dicts,
    WANDB_PADDING,
    stats_to_np,
    convert_to_scalar,
    clip_by_value,
    masked_mean,
    entropy_from_logits,
    masked_var,
    flatten_dict,
)
from trl.models import PreTrainedModelWrapper
from trl.trainer import RunningMoments
from accelerate.utils import broadcast, gather_object
import numpy as np

from red_teaming.trainer.red_teaming_ppo_trainer import RedTeamPPOTrainer
from red_teaming.trainer.lagrange import LagrangeMultiplier


class ConstrainedRedTeamingPPOTrainer(RedTeamPPOTrainer):
    def __init__(self, *args, cost_module_names: List[str] = [], **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_module_names = cost_module_names
        self.running_costs = {cost_module_name: RunningMoments(self.accelerator) for cost_module_name in cost_module_names}

    def setup_lagrange(self, trainer_config: "dataclass", blue_config: "dataclass"):
        self.episode_costs = {}
        self.lagrange_multipliers = {}
        self.lagrange_update_delay_steps = trainer_config.lagrange_update_delay_steps
        for cost_name, cost_fn in blue_config.cost_configs.to_dict().items():
            if cost_fn is None:
                continue
            lagrange_multiplier = LagrangeMultiplier(
                name=cost_name,
                initial_value=trainer_config.lagrange_init,
                lr=trainer_config.lagrange_lr,
                momentum=trainer_config.lagrange_momentum,
                max_value=trainer_config.lagrange_max,
                threshold=cost_fn.threshold,
                transform_fn=trainer_config.lagrange_transform,
                inverse_fn=trainer_config.lagrange_inverse,
                device=self.accelerator.device,
                # only rank 0 process should update lagrange multiplier
                # and then broadcast to other processes
                is_main_process=self.accelerator.is_main_process,
            )
            self.lagrange_multipliers[cost_name] = lagrange_multiplier
            self.episode_costs[cost_name] = deque(maxlen=trainer_config.episode_cost_window_size)

    def _step_safety_checker(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        costs: Dict[str, List[torch.FloatTensor]],
        masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            masks (List[`torch.LongTensor`], *optional*):
                list of optional tensors containing the masks of shape (`query_length` + `response_length`)
        Returns:
            `tuple`: The input processed data.
        """
        for name, tensor_list in zip(["queries", "responses", "scores"], [queries, responses, scores]):
            if not isinstance(tensor_list, list):
                raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(f"Elements in {name} must be tensors - got {type(tensor_list[0])}")
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
                )
        for cost_name in self.cost_module_names:
            if not isinstance(costs['cost_' + cost_name], list):
                raise ValueError(f"{cost_name} must be a list of tensors - got {type(costs['cost_' + cost_name])}")
            if not isinstance(costs['cost_' + cost_name][0], torch.Tensor):
                raise ValueError(f"Elements in {cost_name} must be tensors - got {type(costs['cost_' + cost_name][0])}")
            if batch_size is not None and len(costs['cost_' + cost_name]) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but got {len(costs['cost_' + cost_name])} for: {cost_name}"
                )

        # add queries, scores and responses on the correct device
        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = [tensor.to(self.current_device) for tensor in scores]
        masks = [tensor.to(self.current_device) for tensor in masks] if masks is not None else None
        costs = {cost_name: [cost.to(self.current_device) for cost in costs['cost_' + cost_name]] for cost_name in self.cost_module_names}

        # squeeze scores if needed
        for i, score in enumerate(scores):
            if score.dim() > 1:
                raise ValueError(f"Scores must be 1-dimensional - got {score.dim()} for {score}")
            elif score.dim() == 1:
                scores[i] = score.squeeze()
        for cost_name in self.cost_module_names:
            for i, cost in enumerate(costs[cost_name]):
                if cost.dim() > 1:
                    raise ValueError(f"Costs must be 1-dimensional - got {cost.dim()} for {cost}")
                elif cost.dim() == 1:
                    costs[cost_name][i] = cost.squeeze()

        for cost_name in self.cost_module_names:
            self.episode_costs[cost_name].extend(costs[cost_name])

        return queries, responses, scores, costs, masks

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        return_values: bool = True,
        response_masks: Optional[torch.Tensor] = None,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []
        all_cost_values = {}

        model.eval()

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            logits, _, values, cost_values = model(**input_kwargs, return_values=return_values)
            if self.config.scale_logits:
                logits = logits / self.config.red_generation_kwargs['temperature']

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1  # logprobs starts from the second query token
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])
                    if response_masks is not None:
                        response_masks_batch[j] = torch.cat(
                            (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                        )[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)
            if return_values:
                for cost_name in self.cost_module_names:
                    if cost_name not in all_cost_values:
                        all_cost_values[cost_name] = []
                    all_cost_values[cost_name].append(cost_values[cost_name])

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1] if return_values else None,
            {cost_name: torch.cat(all_cost_values[cost_name])[:, :-1] for cost_name in all_cost_values} if return_values else None,
            torch.cat(all_masks)[:, :-1],
        )

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        cost_scores: Dict[str, torch.FloatTensor],
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        # rewards, non_score_rewards, kls = [], [], []
        rewards, non_score_rewards, kls, costs = [], [], [], {}

        # going over each trajectory in the batch
        for i, (score, logprob, ref_logprob, mask) in enumerate(zip(scores, logprobs, ref_logprobs, masks)):
            # compute KL penalty (from difference in logprobs)
            kl = self._kl_penalty(logprob, ref_logprob)
            kls.append(self._kl_penalty(logprob, ref_logprob))

            entropy = self._entropy_bonus(logprob)

            non_score_reward = -self.kl_ctl.value * kl + self.config.entropy_coeff * entropy
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            # token level costs are zero, only has sequence level costs
            cost = {cost_name: torch.zeros_like(reward) for cost_name in self.cost_module_names}
            last_non_masked_index = mask.nonzero()[-1]

            # reward is preference model score + KL penalty
            reward[last_non_masked_index] += score
            for cost_name in self.cost_module_names:
                cost[cost_name][last_non_masked_index] += cost_scores[cost_name][i]
                if cost_name not in costs:
                    costs[cost_name] = []
                costs[cost_name].append(cost[cost_name])
            rewards.append(reward)
        return (
            torch.stack(rewards),
            {cost_name: torch.stack(costs[cost_name]) for cost_name in costs.keys()},
            torch.stack(non_score_rewards),
            torch.stack(kls)
        )

    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        cost_scores: List[torch.FloatTensor],
        scores_dict: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        # setups
        bs = self.config.batch_size
        queries, responses, scores, cost_scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, cost_scores, response_masks
        )

        # update lagrange first
        for cost_name, lagrange in self.lagrange_multipliers.items():
            episode_cost = torch.tensor(self.episode_costs[cost_name]).mean().to(self.accelerator.device)
            episode_cost = self.accelerator.reduce(episode_cost, reduction="mean")
            if self.accelerator.is_main_process and self.current_step >= self.lagrange_update_delay_steps:
                lagrange.update_lambda(episode_cost)
            broadcast(lagrange.transformed_lambda, from_process=0)
        self.accelerator.wait_for_everyone()

        # ppo step
        scores = torch.tensor(scores, device=self.current_device)
        cost_scores = {cost_name: torch.tensor(cost_scores[cost_name], device=self.current_device) for cost_name in self.cost_module_names}
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= score_scaling_factor
            # Cost scaling
            for cost_name in self.cost_module_names:
                cost_mean, cost_std = self.running_cost[cost_name].update(cost_scores[cost_name])
                tensor_to_kwargs = dict(dtype=cost_scores[cost_name].dtype, device=cost_scores[cost_name].device)
                score_scaling_factor = self.running_cost[cost_name].std.to(**tensor_to_kwargs) + torch.finfo(cost_scores[cost_name].dtype).eps
                if self.config.use_score_norm:
                    cost_scores[cost_name] = (cost_scores[cost_name] - self.running_cost[cost_name].mean.to(**tensor_to_kwargs)) / score_scaling_factor
                else:
                    cost_scores[cost_name] /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)
            # Cost clipping
            for cost_name in self.cost_module_names:
                costs_dtype = cost_scores[cost_name].dtype
                cost_scores[cost_name] = torch.clip(cost_scores[cost_name].float(), -self.config.cost_clip, self.config.cost_clip).to(dtype=costs_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, cost_values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
                return_values=True,
            )
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                    return_values=False,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, costs, non_score_reward, kls = self.compute_rewards(
                    scores, cost_scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, costs, non_score_reward, kls = self.compute_rewards(
                    scores, cost_scores, all_logprobs, ref_logprobs, masks
                )
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            cost_advantages, cost_returns = {}, {}
            for cost_name in self.cost_module_names:
                cost_values[cost_name], cost_advantages[cost_name], cost_returns[cost_name] = self.compute_advantages(
                    cost_values[cost_name], costs[cost_name], masks
                )
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
            **{f'{cost_name}_returns' : cost_returns[cost_name] for cost_name in self.cost_module_names},
            **{f'{cost_name}_advantages' : cost_advantages[cost_name] for cost_name in self.cost_module_names},
            **{f'{cost_name}_values' : cost_values[cost_name] for cost_name in self.cost_module_names},
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                        **{f'cost_{cost_name}_returns' : batch_dict[f'{cost_name}_returns'][mini_batch_inds] for cost_name in self.cost_module_names},
                        **{f'cost_{cost_name}_advantages' : batch_dict[f'{cost_name}_advantages'][mini_batch_inds] for cost_name in self.cost_module_names},
                        **{f'cost_{cost_name}_values' : batch_dict[f'{cost_name}_values'][mini_batch_inds] for cost_name in self.cost_module_names},
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, cpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                            return_values=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                            cpreds,
                            {k.replace('cost_', '').replace('_values', ''): v for k, v in mini_batch_dict.items() if k.startswith('cost_') and 'values' in k},
                            {k.replace('cost_', '').replace('_advantages', ''): v for k, v in mini_batch_dict.items() if k.startswith('cost_') and 'advantages' in k},
                            {k.replace('cost_', '').replace('_returns', ''): v for k, v in mini_batch_dict.items() if k.startswith('cost_') and 'returns' in k},
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)
        for cost_name in self.cost_module_names:
            train_stats[f"policy/{cost_name}_disadvantages"] = torch.flatten(train_stats[f"policy/{cost_name}_advantages"]).unsqueeze(0)
            train_stats[f"policy/{cost_name}_disadvantages"] = torch.nan_to_num(train_stats[f"policy/{cost_name}_advantages"], WANDB_PADDING)

        stats = self.record_step_stats(
            costs=costs,
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,    
            queries=queries,
            responses=responses,
            kls=kls,
            scores_dict=scores_dict,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        for cost_name in self.cost_module_names:
            stats[f"lagrange/{cost_name}_lambda"] = self.lagrange_multipliers[cost_name].get_multiplier().item()

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
        # cost stuff
        cpreds: Dict[str, torch.FloatTensor],
        cost_values: Dict[str, torch.FloatTensor],
        cost_advantages: Dict[str, torch.FloatTensor],
        cost_returns: Dict[str, torch.FloatTensor],
    ):
        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        # cost value losses
        if cost_values != {}: # if there are cost values
            cpredclipped = {cost_name: clip_by_value(
                cpreds[cost_name],
                cost_values[cost_name] - self.config.cliprange_value,
                cost_values[cost_name] + self.config.cliprange_value,
            ) for cost_name in self.cost_module_names}
            cf_losses1 = {cost_name: (cpreds[cost_name] - cost_returns[cost_name]) ** 2 for cost_name in self.cost_module_names}
            cf_losses2 = {cost_name: (cpredclipped[cost_name] - cost_returns[cost_name]) ** 2 for cost_name in self.cost_module_names}
            cf_loss = {cost_name: 0.5 * masked_mean(torch.max(cf_losses1[cost_name], cf_losses2[cost_name]), mask) for cost_name in self.cost_module_names}
            cf_clipfrac = {cost_name: masked_mean(torch.gt(cf_losses2[cost_name], cf_losses1[cost_name]).float(), mask) for cost_name in self.cost_module_names}
            cf_loss_w_coef = sum(cf_loss.values()) * self.config.cost_coef
        else:
            # if there are no cost values, then no loss on it
            cf_loss = {}
            cf_clipfrac = {}
            cf_loss_w_coef = 0.0

        ratio = torch.exp(logprobs - old_logprobs)

        combined_advantages = advantages
        normalizer = 1.0
        if cost_values != {}:
            for cost_name, cost_advantage in cost_advantages.items():
                multiplier = self.lagrange_multipliers[cost_name].get_multiplier().item()
                # cost is negative advantage
                combined_advantages -= multiplier * cost_advantage
                normalizer += multiplier
        if self.config.normalize_advantages:
            combined_advantages /= normalizer

        pg_losses = -combined_advantages * ratio
        pg_losses2 = -combined_advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss + cf_loss_w_coef

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            cf_loss = {cost_name: cf_loss[cost_name] * 0.0 for cost_name in self.cost_module_names}
            loss = loss * 0.0

        entropy = masked_mean(entropy_from_logits(logits), mask)

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)
        cost_return_mean, cost_return_var = (
            {cost_name: masked_mean(cost_returns[cost_name], mask) for cost_name in self.cost_module_names},
            {cost_name: masked_var(cost_returns[cost_name], mask) for cost_name in self.cost_module_names}
        )
        cost_value_mean, cost_value_var = (
            {cost_name: masked_mean(cost_values[cost_name], mask) for cost_name in self.cost_module_names},
            {cost_name: masked_var(cost_values[cost_name], mask) for cost_name in self.cost_module_names}
        )

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach(), **{cost_name + '_cost': cf_loss[cost_name].detach() for cost_name in self.cost_module_names}),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                **{cost_name + '_advantages': cost_advantages[cost_name].detach() for cost_name in self.cost_module_names},
                **{cost_name + '_advantages_mean': masked_mean(cost_advantages[cost_name], mask).detach() for cost_name in self.cost_module_names},
                combined_advantages=combined_advantages.detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(
                mean=return_mean.detach(),
                var=return_var.detach(),
                **{k + '_mean': v.detach() for k, v in cost_return_mean.items()},
                **{k + '_var': v.detach() for k, v in cost_return_var.items()},
            ),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
            cost_val=dict(
                **{cost_name + '_vpred': masked_mean(cpreds[cost_name], mask).detach() for cost_name in self.cost_module_names},
                **{cost_name + '_error': masked_mean((cpreds[cost_name] - cost_returns[cost_name]) ** 2, mask).detach() for cost_name in self.cost_module_names},
                **{cost_name + '_clipfrac': cf_clipfrac[cost_name].detach() for cost_name in self.cost_module_names},
                **{cost_name + '_mean': cost_value_mean[cost_name].detach() for cost_name in self.cost_module_names},
                **{cost_name + '_var': cost_value_var[cost_name].detach() for cost_name in self.cost_module_names},
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, cf_loss_w_coef, flatten_dict(stats)

    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
        # cost stuff
        cpreds: Dict[str, torch.FloatTensor],
        cost_values: Dict[str, torch.FloatTensor],
        cost_advantages: Dict[str, torch.FloatTensor],
        cost_returns: Dict[str, torch.FloatTensor],
    ):
        self.model.train()
        loss_p, loss_v, loss_c, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns,
            cpreds, cost_values, cost_advantages, cost_returns,
        )
        loss = loss_p + loss_v + loss_c
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model_params,
                    self.config.max_grad_norm,
                )
        self.optimizer.step()
        # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
        # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.optimizer.zero_grad()
        return train_stats

    def record_step_stats(self, costs: torch.Tensor, kl_coef: float, scores_dict: Dict[str, torch.Tensor], **data):
        stats = super().record_step_stats(kl_coef, scores_dict, **data)
        return stats

    def log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
        costs: Dict[str, List[torch.FloatTensor]],
        columns_to_log: List[str] = ["query", "response"],
        log_freq = 1000,
    ):
        # all gather stats
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards).to(self.current_device)
        rewards = self.accelerator.gather(rewards).flatten()
        self.accelerator.wait_for_everyone()
        # check if any NaN
        if torch.isnan(rewards).any():
            warnings.warn("NaN values found in rewards.!!!!!!!")
        if torch.isinf(rewards).any():
            warnings.warn("Inf values found in rewards.!!!!!!!")

        if costs is not None: # if there are costs
            for cost_name, cost in costs.items():
                if not isinstance(cost, torch.Tensor):
                    costs[cost_name] = torch.tensor(cost).to(self.current_device)
                    if torch.isnan(costs[cost_name]).any():
                        warnings.warn(f"NaN values found in {cost_name} costs.!!!!!!!")
                    if torch.isinf(costs[cost_name]).any():
                        warnings.warn(f"Inf values found in {cost_name} costs.!!!!!!!")
            costs = {cost_name: self.accelerator.gather(costs[cost_name]).flatten() for cost_name in costs.keys()}

        if self.config.log_with == "wandb":
            import wandb

            if any([column_to_log not in batch.keys() for column_to_log in columns_to_log]):
                raise ValueError(f"Columns to log {columns_to_log} are not present in the batch {batch.keys()}.")

            batch_list = [batch[column_to_log] for column_to_log in columns_to_log]
            if self.is_distributed:
                gathered_batch_list = []
                for b in batch_list:
                    flattened = gather_object(b)
                    gathered_batch_list.append(flattened)
                batch_list = gathered_batch_list

        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            logs = {}

            # Log stats
            # if "query" not in batch.keys() and "response" not in batch.keys():
            # if "red_prompt" not in batch.keys() and "blue_response" not in batch.keys():
            #     # warn the user that the game logs will not be logged
            #     warnings.warn(
            #         "The game logs will not be logged because the batch does not contain the keys 'query' and "
            #         "'response'. "
            #     )
            # elif self.config.log_with == "wandb":
            #     # log every 1000 steps
            #     if batch['iteration'][0] % log_freq == 0:
            #         table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
            #         logs.update({"game_log": wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)})

            logs.update(stats)

            # manually cast in fp32 for bf16 torch tensors
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()
                    # check for NaN
                    if torch.isnan(v).any():
                        warnings.warn(f"NaN values found in {k}.!!!!!!")
                    if torch.isinf(v).any():
                        warnings.warn(f"Inf values found in {k}.!!!!!!")

            if self.config.log_with == "tensorboard":
                # update the current step
                self.current_step += 1

            self.accelerator.log(
                logs,
                step=self.current_step if self.config.log_with == "tensorboard" else None,
            )
