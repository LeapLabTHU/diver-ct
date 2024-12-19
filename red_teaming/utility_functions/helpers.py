from typing import Callable, List, Union
from dataclasses import dataclass, field

import numpy as np
import torch

from red_teaming.utility_functions import *


@dataclass
class RewardModule:
    module: Callable = field(default=None, metadata={"help": "The reward module"})
    name: str = field(default="reward", metadata={"help": "The reward module name"})
    _coef: float = field(default=1.0, metadata={"help": "The reward coefficient"})
    team: str = field(default="red", metadata={"help": "The reward team"})
    device: str = field(default="cpu", metadata={"help": "The device to use"})
    clip_value: float = field(default=None, metadata={"help": "The clip value"})
    is_dynamic: bool = field(default=False, metadata={"help": "Whether the reward is dynamic"})
    amplitude: float = field(default=1.0, metadata={"help": "The amplitude of the wave"})
    frequency: float = field(default=0.025, metadata={"help": "The frequency of the wave"})
    bias: float = field(default=0.5, metadata={"help": "The bias of the wave"})
    scale: float = field(default=1.0, metadata={"help": "The scale of the wave"})

    def __call__(self, *args, **kwargs):
        if self.clip_value is None:
            return self.module.get_score(*args, **kwargs)
        return torch.clamp(self.module.get_score(*args, **kwargs), min=-self.clip_value, max=self.clip_value)

    def __post_init__(self):
        if self.is_dynamic:
            self.counter = 0

    @property
    def coef(self):
        if self.is_dynamic:
            coef = (self.amplitude * np.sin(self.frequency * self.counter) + self.bias) * self.scale
            self.counter += 1
            return coef
        return self._coef

    def to_dict(self):
        # skipping device
        return {
            "module": self.module,
            "name": self.name,
            "_coef": self.coef,
            "team": self.team,
            "is_dynamic": self.is_dynamic,
            "amplitude": self.amplitude,
            "frequency": self.frequency,
            "bias": self.bias,
            "scale": self.scale,
            "clip_value": self.clip_value,
        }


@dataclass
class CostModule:
    module: Callable
    name: str
    constraint: float
    team: str
    clip_value: float = field(default=None, metadata={"help": "The clip value"})
    device: str = field(default="cpu", metadata={"help": "The device to use"})
    transform: Callable = field(default=lambda x: -x, metadata={"help": "The transform function"})

    def __call__(self, *args, **kwargs):
        if self.clip_value is None:
            return self.transform(self.module.get_score(*args, **kwargs))
        return self.transform(
            np.clip(self.module.get_score(*args, **kwargs), -self.clip_value, self.clip_value)
        )

    def to_dict(self):
        # skipping device
        return {
            "module": self.module,
            "name": self.name,
            "constraint": self.constraint,
            "team": self.team,
            "transform": self.transform,
        }


def get_utility_modules(utility_module_config: dataclass, device: str, utility_type: str) -> List[Union[RewardModule, CostModule]]:
    utility_modules = []
    for utility_name, utility_dict in utility_module_config.to_dict().items():
        if utility_dict is None:
            continue
        utility_fn = eval(utility_dict.module)(**utility_dict.kwargs)
        if hasattr(utility_fn, 'model'):
            utility_fn.model.to(device)
            utility_fn.model.eval()
        if utility_type == 'reward':
            reward_kwrags = dict(
                module=utility_fn,
                name=utility_name,
                _coef=utility_dict.coef,
                team=utility_dict.team,
                device=device,
                clip_value=utility_dict.clip_value,
            )
            # Add dynamic reward parameters
            if hasattr(utility_dict, 'is_dynamic'):
                reward_kwrags.update(
                    dict(
                        is_dynamic=utility_dict.is_dynamic,
                        amplitude=utility_dict.amplitude,
                        frequency=utility_dict.frequency,
                        bias=utility_dict.bias,
                        scale=utility_dict.scale,
                    )
                )
            else:
                reward_kwrags['is_dynamic'] = False
            utility = RewardModule(**reward_kwrags)
        elif utility_type == 'cost':
            utility = CostModule(
                    module=utility_fn,
                    name=utility_name,
                    constraint=utility_dict.threshold,
                    team=utility_dict.team,
                    device=device,
                    transform=utility_dict.transformation,
                )
        else:
            raise ValueError(f"Utility type {utility_type} not recognized")
        utility_modules.append(utility)
    return utility_modules
