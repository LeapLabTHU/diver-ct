from typing import Tuple, Union
import functools
import torch.nn as nn
from transformers import PreTrainedModel
from trl.models import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
)

from red_teaming.models.modeling_multi_value_heads import (
    AutoModelForCausalLMWithValueCostHeads,
    AutoModelForSeq2SeqLMWithValueCostHeads,
)


# Freeze model tools from trlx

def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])

    
def rgetattr(obj, attr: str, *args) -> object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs: Tuple[str]) -> Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")
    
    
def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
    """
    hidden_layers_attrs = (
        "h",
        "layers",
        "model.layers",
        "decoder.layers",
        "transformer.h",
        "transformer.blocks",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
    )
    return findattr(model, hidden_layers_attrs)


def freeze_bottom_causal_layers(model: PreTrainedModel, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)

    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
        hidden_layers_to_freeze += [model.get_input_embeddings(), model.get_output_embeddings()]
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
        hidden_layers_to_freeze += [model.get_input_embeddings()]
        if model.config.tie_word_embeddings:
            hidden_layers_to_freeze += [model.get_output_embeddings()]
    else:
        hidden_layers_to_freeze = []

    for layer in hidden_layers_to_freeze:
        if layer is None:
            continue
        layer.requires_grad_(False)


def get_model_cls(seq2seq: bool, cost_head: bool):
    if cost_head and seq2seq:
        return AutoModelForSeq2SeqLMWithValueCostHeads
    if not cost_head and seq2seq:
        return AutoModelForSeq2SeqLMWithValueHead
    if cost_head and not seq2seq:
        return AutoModelForCausalLMWithValueCostHeads
    if not cost_head and not seq2seq:
        return AutoModelForCausalLMWithValueHead
