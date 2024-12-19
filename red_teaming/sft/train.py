# copied from trl examples
import os
from typing import Optional
import numpy as np

from trl import DataCollatorForCompletionOnlyLM

import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments, HfArgumentParser
from trl import (
    ModelConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
import json
import numpy as np
import dataclasses
from dataclasses import field


@dataclasses.dataclass
class ScriptArgs:
    safety_threshold: float = dataclasses.field(default=0.5)
    constraint: bool = dataclasses.field(default=False)
    safe_response_pool_path: str = dataclasses.field(default="red_teaming/data/safe_responses.json")
    alpaca_template_path: str = dataclasses.field(default="red_teaming/data/Alpaca.yaml")
    supplement_dataset_path: str = dataclasses.field(default="tatsu-lab/alpaca")
    ratio_unsafe: float = dataclasses.field(default=0.5)

    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "The maximum sequence length"})
    packing: Optional[bool] = field(default=False, metadata={"help": "Use packing for training"})


tqdm.pandas()

if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, ModelConfig, ScriptArgs))
    training_args, model_config, script_args = parser.parse_args_into_dataclasses()

    os.environ["WANDB_PROJECT"] = "red-teaming"
    os.environ['WANDB_NAME'] = training_args.run_name

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        import yaml
        with open(script_args.alpaca_template_path) as file:
            tokenizer.chat_template = yaml.safe_load(file)['instruction_template']

    ################
    # Dataset
    ################
    raw_datasets = pd.DataFrame(pd.read_pickle(script_args.dataset_name))
    safe_response_pool = json.load(open(script_args.safe_response_pool_path))

    if script_args.constraint:
        process_dataset = raw_datasets[raw_datasets.apply(lambda row: row['safety'] <= script_args.safety_threshold, axis=1)].copy()
    else:
        process_dataset = raw_datasets[raw_datasets.apply(lambda row: row['safety'] >= script_args.safety_threshold, axis=1)].copy()
    process_dataset['safe_response'] = np.random.choice(safe_response_pool, len(process_dataset), replace=True)
    process_dataset['text'] = process_dataset.apply(
        lambda row: tokenizer.apply_chat_template(
            [{'role': 'user', 'content': row['red_prompt'].strip()}, {'role': 'assistant', 'content': row['safe_response'].strip()}], tokenize=False
        ), axis=1
    )
    supplement_dataset = load_dataset(script_args.supplement_dataset_path)['train']
    supplement_dataset = supplement_dataset.map(lambda example: {'text': example['text']})
    # only take a ratio of unsafe examples, 0.5 means the 0.5*len(supplement_dataset) hf_dataset examples
    # seed the random choice for reproducibility
    random_idx = np.random.choice(len(process_dataset), int(script_args.ratio_unsafe*len(supplement_dataset)), replace=False, seed=42)
    hf_dataset = Dataset.from_pandas(process_dataset)
    hf_dataset = hf_dataset.select(random_idx)
    hf_dataset = hf_dataset.map(lambda example: {'text': example['text']})
    hf_dataset = concatenate_datasets([hf_dataset, supplement_dataset])
    print(f'Supplement dataset size: {len(supplement_dataset)}')
    print(f'HF dataset size: {len(hf_dataset)}')
    ################
    # Optional rich context managers
    ###############
    collator = DataCollatorForCompletionOnlyLM('### Response:\n', tokenizer=tokenizer)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=hf_dataset,
        dataset_text_field='text',
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=script_args.packing,
        data_collator=collator,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
