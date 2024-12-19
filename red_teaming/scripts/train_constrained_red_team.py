from dataclasses import dataclass, field
from typing import Optional
import os
import time
import wandb 

import torch
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, GenerationConfig
from trl import set_seed
from trl.import_utils import is_npu_available, is_xpu_available
from trl.trainer.utils import disable_dropout_in_model

from red_teaming.trainer import ConstrainedRedTeamingPPOTrainer
from red_teaming.configs import BlueConfig, RedTeamPPOLagrangianConfig, EvaluationConfig
from red_teaming.environment import BlueTeamCostEnvironment
from red_teaming import utility_functions
from red_teaming.models.utils import freeze_bottom_causal_layers, get_model_cls
from red_teaming.evaluation.evaluate_run import evaluate_runs
from red_teaming.evaluation.metrics.utils import (
    flatten_scores_dict, save_obs_jsonl, convert_jsonl_to_pkl
)


tqdm.pandas()


@dataclass
class ScriptArguments:
    run_name: str = field(default="red_teaming", metadata={"help": "The name of the run"})
    wandb_tags: str = field(default=None, metadata={"help": "The tags for the run"})

    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})
    target_modules: Optional[str] = field(default="all-linear", metadata={"help": "the target modules for lora"})
    lora_dropout: float = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    use_rslora: bool = field(default=False, metadata={"help": "whether to use rslora"})

    # model_config
    summary_dropout_prob: float = field(default=0.1, metadata={"help": "the dropout probability for the summary layer"})
    bfloat16: bool = field(default=False, metadata={"help": "whether to use bfloat16"})


def build_dataset(model_name: "str", query_dataset: "str"):
    """
    Build a dataset by loading and concatenating multiple datasets.

    Args:
        config (object): Configuration object.
        query_dataset (str): Comma-separated string of dataset file paths.

    Returns:
        concatenated_ds (object): Concatenated dataset object.
    """
    dataset_paths = query_dataset.split(",")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load all datasets and store them in a list
    datasets = [load_dataset('json', data_files=dataset, split="train") for dataset in dataset_paths]

    # Concatenate all datasets
    concatenated_ds = concatenate_datasets(datasets)

    def tokenize(sample: str):
        sample["input_ids"] = tokenizer.encode(sample["instruction"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    concatenated_ds = concatenated_ds.map(tokenize, batched=False)
    concatenated_ds.set_format(type="torch")
    return concatenated_ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def setup_generation(ppo_config, tokenizer, model) -> "GenerationConfig":
    red_generation_configs = ppo_config.red_generation_kwargs
    red_generation_configs['eos_token_id'] = tokenizer.eos_token_id
    red_generation_configs['bos_token_id'] = tokenizer.bos_token_id
    # only support gpt red team for now
    if 'gpt2' in ppo_config.model_name:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        red_generation_configs['pad_token_id'] = tokenizer.eos_token_id
        # make gpt2 stop at newline, other models stop at eos
        red_generation_configs['eos_token_id'] = 198
    return red_generation_configs


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RedTeamPPOLagrangianConfig, BlueConfig, EvaluationConfig))

    args, ppo_config, blue_config, evaluation_config = parser.parse_args_into_dataclasses()
    # set the cost window to the size of the current batch size (only costs for current episodes)
    if ppo_config.episode_cost_window_size is None:
        ppo_config.episode_cost_window_size = ppo_config.batch_size
    trl_model_class = get_model_cls(seq2seq=False, cost_head=True)

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(ppo_config.model_name, ppo_config.query_dataset)

    os.environ["WANDB_PROJECT"] = "red-teaming"
    os.environ['WANDB_NAME'] = args.run_name
    if args.wandb_tags is not None:
        os.environ["WANDB_TAGS"] = args.wandb_tags + ",constrained"
    else:
        os.environ["WANDB_TAGS"] = "constrained"

    # set seed before initializing value head for deterministic eval
    set_seed(ppo_config.seed)

    # get the cost module names, an utility can either be a reward or a cost
    cost_module_names = [k for k, v in blue_config.cost_configs.to_dict().items() if v is not None]    

    if args.bfloat16:
        dtype = {'torch_dtype': torch.bfloat16}
    else:
        dtype = {}
    # Now let's build the model, the reference model, and the tokenizer.
    if not args.use_peft:
        ref_model = trl_model_class.from_pretrained(
            ppo_config.model_name,
            cost_module_names=cost_module_names,
            trust_remote_code=args.trust_remote_code,
            **dtype,
        )
        device_map = None
        peft_config = None
    else:
        if args.target_modules != 'all-linear' and args.target_modules is not None:
            target_modules = args.target_modules.split(",")
        else:
            target_modules = args.target_modules
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            use_rslora=args.use_rslora,
        )
        ref_model = None
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}

    model = trl_model_class.from_pretrained(
        ppo_config.model_name,
        cost_module_names=cost_module_names,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
        summary_dropout_prob=args.summary_dropout_prob,
        **ppo_config.red_team_model_config.model_kwargs,
        **dtype,
    )
    if not args.use_peft:
        freeze_bottom_causal_layers(
            model=model.pretrained_model.base_model,
            num_layers_unfrozen=ppo_config.red_team_model_config.num_layers_unfrozen,
        )
    disable_dropout_in_model(model)
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

    # setting up red generation config
    red_generation_config: GenerationConfig = setup_generation(ppo_config, tokenizer, model)

    # We then build the RedTeamPPOTrainer, passing the model, the reference model, the tokenizer
    red_teaming_ppo_lagrangian_trainer = ConstrainedRedTeamingPPOTrainer(
        ppo_config, model, ref_model, tokenizer, cost_module_names=cost_module_names,
        dataset=dataset, data_collator=collator,
    )

    # init reward/cost modules
    device = red_teaming_ppo_lagrangian_trainer.accelerator.device
    if red_teaming_ppo_lagrangian_trainer.accelerator.num_processes == 1:
        if is_xpu_available():
            device = "xpu:0"
        elif is_npu_available():
            device = "npu:0"
        else:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    ds_plugin = red_teaming_ppo_lagrangian_trainer.accelerator.state.deepspeed_plugin

    if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
        with ds_plugin.zero3_init_context_manager(enable=False):
            reward_modules = utility_functions.get_utility_modules(
                blue_config.reward_configs, device=device, utility_type='reward'
            )
            cost_modules = utility_functions.get_utility_modules(
                blue_config.cost_configs, device=device, utility_type='cost'
            )
    else:
        reward_modules = utility_functions.get_utility_modules(
            blue_config.reward_configs, device=device, utility_type='reward'
        )
        cost_modules = utility_functions.get_utility_modules(
            blue_config.cost_configs, device=device, utility_type='cost'
        )

    # Init Blue Team
    blue_team = BlueTeamCostEnvironment(config=blue_config, max_steps=ppo_config.steps)
    blue_trl_model_class = get_model_cls(seq2seq=False, cost_head=False)
    extra_blue_kwargs = dict(attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16) if blue_config.save_memory else {}
    blue_model = blue_trl_model_class.from_pretrained(
        blue_config.blue_model_name,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        **extra_blue_kwargs,
    )
    disable_dropout_in_model(blue_model)
    blue_model.eval().requires_grad_(False)
    blue_tokenizer = AutoTokenizer.from_pretrained(blue_config.blue_model_name)
    blue_tokenizer.padding_side = "left"
    blue_team.prepare_blue_team(
        accelerator=red_teaming_ppo_lagrangian_trainer.accelerator,
        blue_model=blue_model,
        blue_tokenizer=blue_tokenizer,
        red_tokenizer=tokenizer,
        dataloader=red_teaming_ppo_lagrangian_trainer.dataloader,
        reward_modules=reward_modules,
        cost_modules=cost_modules,
    )

    # prepare lagrange multipliers
    red_teaming_ppo_lagrangian_trainer.setup_lagrange(ppo_config, blue_config)

    # Start training
    continue_training = True
    
    # Create a folder to save the observations
    save_path = f'{ppo_config.log_train_path}/{args.run_name}+{time.strftime("%Y_%m_%d-%H")}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    #progressbar
    if red_teaming_ppo_lagrangian_trainer.accelerator.is_main_process:
        pbar = tqdm(total=ppo_config.steps, desc=args.run_name)
        
    while continue_training:
        if red_teaming_ppo_lagrangian_trainer.accelerator.is_main_process:
            pbar.update(1)
        # Get batch from blue team
        observation, continue_training = blue_team.reset()

        # If the red team is done training, we close the environment and break the loop
        if not continue_training:
            blue_team.close()
            break
        query_tensors = observation["input_ids"]

        with torch.no_grad():
            response_tensors = red_teaming_ppo_lagrangian_trainer.generate(
                query_tensors,
                return_prompt=False,
                generate_ref_response=False,
                batch_size=ppo_config.batch_size,
                **red_generation_config,
            )

        # Run PPO step
        with torch.no_grad():
            step = blue_team.step(query_tensors, response_tensors)
        observations, rewards, costs = step.observations, step.rewards, step.costs
        stats = red_teaming_ppo_lagrangian_trainer.step(query_tensors, response_tensors, rewards, costs, scores_dict=step.info)

        # logging
        red_teaming_ppo_lagrangian_trainer.log_stats(
            stats, observations.to_dict(), rewards, costs,
            columns_to_log=['decoded_prompt', 'red_prompt', 'blue_response', 'safety', 'iteration', 'gibberish'],
            log_freq=ppo_config.log_freq,
        )

        # Locally save the observations
        save_obs_jsonl(red_teaming_ppo_lagrangian_trainer, tokenizer, blue_tokenizer, observations, save_path)

    convert_jsonl_to_pkl(os.path.join(save_path, f'obs_{os.getpid()}.jsonl'))
    red_teaming_ppo_lagrangian_trainer.accelerator.wait_for_everyone()
    if ppo_config.need_eval:
        # Evaluating the run
        if red_teaming_ppo_lagrangian_trainer.accelerator.is_main_process:
            evaluation_config.log_path = save_path
            scores = evaluate_runs(args = evaluation_config)

            # Log to wandb
            scores = flatten_scores_dict(scores)
            columns = list(scores.keys())
            values = [list(scores.values())]
            eval_scores = {"evaluation": wandb.Table(columns=columns, rows=values)}
            red_teaming_ppo_lagrangian_trainer.accelerator.log(
                eval_scores,
                step=red_teaming_ppo_lagrangian_trainer.current_step if red_teaming_ppo_lagrangian_trainer.config.log_with == "tensorboard" else None,
            )
