from typing import List
import logging
import os

import torch
from transformers import PreTrainedModel
from trl.import_utils import is_peft_available
from trl.models import (
    AutoModelForCausalLMWithValueHead,
)
from trl.models.modeling_value_head import ValueHead
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HFValidationError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from safetensors.torch import load_file as safe_load_file

if is_peft_available():
    from peft import (
        PeftConfig,
        PeftModel,
        PromptLearningConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )


class AutoModelForCausalLMWithValueCostHeads(AutoModelForCausalLMWithValueHead):
    def __init__(self, config, *args, cost_module_names: List[str] = [], **kwargs):
        super().__init__(config, *args, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)
        self.c_heads = torch.nn.ModuleDict()
        for cost_module_name in cost_module_names:
            self.c_heads[cost_module_name] = ValueHead(self.pretrained_model.config, **v_head_kwargs)

        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        r"""
        Initializes the weights of the value/cost heads. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `v_head_init_strategy` argument
        when calling `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class. These arguments
                can contain the `v_head_init_strategy` argument as well as the `v_head_initializer_range`
                argument.
        """
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()
            for c_head in self.c_heads.values():
                c_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
                c_head.summary.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_values=True,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if return_values:
            if last_hidden_state.device != self.v_head.summary.weight.device:
                last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)
            value = self.v_head(last_hidden_state).squeeze(-1)
            costs = {
                cost_name: c_head(last_hidden_state).squeeze(-1) for cost_name, c_head in self.c_heads.items()
            }
        else:
            value, costs = None, None

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value, costs)

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        for cost_name, c_head in self.c_heads.items():
            c_head_state_dict = c_head.state_dict(*args, **kwargs)
            for k, v in c_head_state_dict.items():
                pretrained_model_state_dict[f"{cost_name}.c_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, "v_head", self.v_head)
        for cost_name, c_head in self.c_heads.items():
            setattr(self.pretrained_model, f"{cost_name}.c_head", c_head)

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
            for cost_name, c_head in self.c_heads.items():
                if cost_name + '.c_head.' in k:
                    state_dict[k.replace(cost_name + '.c_head.', "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        for c_head in self.c_heads.values():
            c_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]

            self.v_head = self.v_head.to(first_device)
            for cost_name, c_head in self.c_heads.items():
                self.c_heads[cost_name] = c_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, cost_module_names: List[str] = [], **kwargs):
        if kwargs is not None:
            peft_config = kwargs.pop("peft_config", None)
            reward_adapter = kwargs.pop("reward_adapter", None)
            reward_adapter_name = kwargs.pop("reward_adapter_name", "reward_adapter")
            is_trainable = kwargs.pop("is_trainable", False)
            trl_model_args, pretrained_kwargs, peft_quantization_kwargs = cls._split_kwargs(kwargs)
            token = pretrained_kwargs.get("token", None)
        else:
            peft_config = None
            is_trainable = False
            trl_model_args = {}
            pretrained_kwargs = {}
            peft_quantization_kwargs = {}
            token = None

        if reward_adapter is not None and not isinstance(reward_adapter, str):
            raise ValueError(
                "The `reward_adapter` argument should be a string representing the name of local path or the Hub id to the Reward Modeling adapter."
            )

        is_peft_model = False

        current_device = cls._get_current_device()
        if isinstance(pretrained_model_name_or_path, str):
            is_loaded_in_8bit = pretrained_kwargs["load_in_8bit"] if "load_in_8bit" in pretrained_kwargs else False
            is_loaded_in_4bit = pretrained_kwargs["load_in_4bit"] if "load_in_4bit" in pretrained_kwargs else False
        else:
            is_loaded_in_8bit = getattr(pretrained_model_name_or_path, "is_loaded_in_8bit", False)
            is_loaded_in_4bit = getattr(pretrained_model_name_or_path, "is_loaded_in_4bit", False)

        if (is_loaded_in_8bit or is_loaded_in_4bit) and "device_map" not in pretrained_kwargs:
            # warn users
            logging.warning(
                "The `device_map` argument is not provided. We will override the device_map argument."
                " to set the entire"
                " model on the current device. If you want to set the model on multiple devices, please provide"
                " a custom `device_map` argument."
            )
            pretrained_kwargs["device_map"] = {"": current_device}

        if is_peft_available() and peft_config is not None and not isinstance(peft_config, PeftConfig):
            raise ValueError("The `peft_config` argument should be an instance of `peft.PeftConfig` class.")

        # First, load the pre-trained model using the parent-class
        # either `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
        if isinstance(pretrained_model_name_or_path, str):
            if is_peft_available():
                try:
                    # If there is a trained peft adapter in the hub, load its config.
                    remote_adapter_config = hf_hub_download(
                        pretrained_model_name_or_path,
                        "adapter_config.json",
                        token=token,
                    )
                except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
                    remote_adapter_config = None
            else:
                remote_adapter_config = None

            local_adapter_present = os.path.exists(os.path.join(pretrained_model_name_or_path, "adapter_config.json"))

            if (local_adapter_present or remote_adapter_config is not None) and is_peft_available():
                if peft_config is not None:
                    logging.warning(
                        "`peft_config` argument ignored since a peft config file was found in "
                        f"{pretrained_model_name_or_path}"
                    )

                # Load the trained peft adapter config
                if local_adapter_present:
                    trained_adapter_config = PeftConfig.from_pretrained(pretrained_model_name_or_path)
                else:
                    remote_adapter_dir = os.path.dirname(remote_adapter_config)
                    trained_adapter_config = PeftConfig.from_pretrained(remote_adapter_dir)

                # Load the pretrained base model
                pretrained_model = cls.transformers_parent_class.from_pretrained(
                    trained_adapter_config.base_model_name_or_path, *model_args, **pretrained_kwargs
                )

                # Wrap the pretrained model with the trained peft adapter
                pretrained_model = PeftModel.from_pretrained(
                    pretrained_model, pretrained_model_name_or_path, is_trainable=is_trainable
                )
                logging.info("Trained peft adapter loaded")
            else:
                pretrained_model = cls.transformers_parent_class.from_pretrained(
                    pretrained_model_name_or_path, *model_args, **pretrained_kwargs
                )

                if peft_config is not None:
                    # Initialize a new peft adapter with the given config
                    if is_loaded_in_8bit or is_loaded_in_4bit:
                        pretrained_model = prepare_model_for_kbit_training(
                            pretrained_model,
                            **peft_quantization_kwargs,
                        )
                    pretrained_model = get_peft_model(pretrained_model, peft_config)
                    logging.info("peft adapter initialised")

        elif isinstance(pretrained_model_name_or_path, cls.supported_pretrained_model_architectures):
            pretrained_model = pretrained_model_name_or_path

            if peft_config is not None and isinstance(pretrained_model, PreTrainedModel):
                # Initialize a new peft adapter with the given config
                if is_loaded_in_8bit or is_loaded_in_4bit:
                    pretrained_model = prepare_model_for_kbit_training(
                        pretrained_model,
                        **peft_quantization_kwargs,
                    )
                pretrained_model = get_peft_model(pretrained_model, peft_config)
                logging.info("peft adapter initialised")
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string or a PreTrainedModel, "
                f"but is {type(pretrained_model_name_or_path)}"
            )

        if is_peft_available():
            if isinstance(pretrained_model, PeftModel):
                is_peft_model = True
                # for backward compatibility
                if hasattr(pretrained_model, "active_peft_config") and isinstance(
                    pretrained_model.active_peft_config, PromptLearningConfig
                ):
                    raise ValueError("PromptLearningConfig is not supported for PPO training.")

        # Add reward modeling adapter if specified
        if not is_peft_model and reward_adapter is not None:
            raise ValueError("reward_adapter can only be used with a PeftModel. ")
        elif is_peft_model and reward_adapter is not None:
            score_module = cls.add_and_load_reward_modeling_adapter(
                pretrained_model, reward_adapter, reward_adapter_name, token=token
            )
            multi_adapter_args = {
                "score_module": score_module,
                "supports_rm_adapter": True,
                "rm_adapter_name": reward_adapter_name,
            }
        else:
            multi_adapter_args = {"supports_rm_adapter": False}

        # Then, create the full model by instantiating the wrapper class
        model = cls(pretrained_model, cost_module_names=cost_module_names, **multi_adapter_args, **trl_model_args)

        # if resume_training, load the state_dict again - this is ok since the
        # state_dict is removed from the model after loading it.
        is_resuming_training = True
        if isinstance(pretrained_model_name_or_path, str):
            safe_filename = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")

            sharded_index_filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin.index.json")
            safe_sharded_index_filename = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
            is_sharded = False
            use_safe = os.path.exists(safe_filename)

            if not (os.path.exists(filename) or os.path.exists(safe_filename)):
                # Try with `pytorch_model.bin`
                filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(
                    pretrained_model,
                    pretrained_model_name_or_path,
                    sharded_index_filename,
                    token=token,
                )
                # Try with safetensors
                if filename is None and files_to_download is None:
                    safe_filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(
                        pretrained_model,
                        pretrained_model_name_or_path,
                        safe_sharded_index_filename,
                        token=token,
                        model_name="model.safetensors",
                        model_index_name="model.safetensors.index.json",
                    )
                    use_safe = True
                else:
                    use_safe = False

            loading_func = safe_load_file if use_safe else torch.load
            load_kwargs = {} if use_safe else {"map_location": "cpu"}

            if is_resuming_training:
                if is_sharded:
                    # download each file and add it to the state_dict
                    state_dict = {}

                    for shard_file in files_to_download:
                        filename = hf_hub_download(
                            pretrained_model_name_or_path,
                            shard_file,
                            token=token,
                        )
                        state_dict.update(loading_func(filename, **load_kwargs))
                else:
                    state_dict = loading_func(filename if not use_safe else safe_filename, **load_kwargs)

        else:
            state_dict = pretrained_model_name_or_path.state_dict()

        model.is_peft_model = is_peft_model
        model.current_device = current_device

        if is_resuming_training:
            model.post_init(state_dict=state_dict)

        return model
