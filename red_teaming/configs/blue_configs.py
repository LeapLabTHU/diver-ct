from typing import Optional, Callable
from dataclasses import field, dataclass

from transformers import GenerationConfig

from red_teaming.configs.base import BaseConfig


@dataclass
class KwargsConfig(BaseConfig):
    def __post_init__(self):
        self.kwargs = {}
        for key, value in self.__dict__.items():
            if key.startswith("kwarg_"):
                self.kwargs[key[6:]] = value


@dataclass
class SemanticRewardConfig(KwargsConfig):
    module: str = field(
        default="SampledSemanticDiversityScore", # TDiv
        metadata={"help": "The reward module name"}
    )
    coef: float = field(
        default=1.0,
        metadata={"help": "The reward coefficient"}
    )
    team: str = field(
        default="red",
        metadata={"help": "The reward team"}
    )
    kwarg_embedder_model_name: str = field(
        default="minilm",
        metadata={"help": "The embedder model name"}
    )
    kwarg_sample_size: int = field(
        default=-1,
        metadata={"help": "The sample size"}
    )
    kwarg_top_k: int = field(
        default=16,
        metadata={"help": "The top k value for the reward"}
    )


@dataclass
class SafetyRewardConfig(KwargsConfig):
    module: str = field(
        default="SafetyScore",
        metadata={"help": "The reward module name"}
    )
    coef: float = field(
        default=1.0,
        metadata={"help": "The reward coefficient"}
    )
    team: str = field(
        default="blue",
        metadata={"help": "The reward team"}
    )
    kwarg_model_name: str = field(
        default="toxigen",
        metadata={"help": "The model name for safety score"}
    )


@dataclass
class SafetyCostConfig(KwargsConfig):
    module: str = field(
        default="SafetyScore",
        metadata={"help": "The cost module name"}
    )
    threshold: float = field(
        default=-0.6,
        metadata={"help": "The cost coefficient"}
    )
    team: str = field(
        default="blue",
        metadata={"help": "The cost team"}
    )
    kwarg_model_name: str = field(
        default="toxigen",
        metadata={"help": "The model name for safety score"}
    )
    transformation: Callable = lambda x: -x


@dataclass
class GibberishRewardConfig(KwargsConfig):
    module: str = field(
        default= "GibberishScore",
        metadata={"help": "The reward module name"}
    )
    coef: float = field(
        default=1.0,
        metadata={"help": "The reward coefficient"}
    )
    team: str = field(
        default="red",
        metadata={"help": "The reward team"}
    )
    kwarg_method: str = field(
        default="not_clean",
        metadata={"help": "The reward module kwargs"}
    )


@dataclass
class GibberishCostConfig(KwargsConfig):
    module: str = field(
        default= "GibberishScore",
        metadata={"help": "The cost module name"}
    )
    threshold: float = field(
        default=0.1,
        metadata={"help": "The reward coefficient"}
    )
    team: str = field(
        default="red",
        metadata={"help": "The reward team"}
    )
    kwarg_method: str = field(
        default="not_clean",
        metadata={"help": "The reward module kwargs"}
    )
    transformation: Callable = lambda x: -x


@dataclass
class NgramDiversityRewardConfig(KwargsConfig):
    module: str = field(
        default="SelfBLEUScore",
        metadata={"help": "The ngram diversity reward module name"}
    )
    coef: float = field(
        default=1.0,
        metadata={"help": "The ngram diversity reward coefficient"}
    )
    team: str = field(
        default="red",
        metadata={"help": "The ngram diversity reward team"}
    )
    kwarg_K: int = field(
        default=5,
        metadata={"help": "The max ngram size"}
    )
    kwarg_sample_size: int = field(
        default=-1,
        metadata={"help": "The sample size"}
    )


@dataclass
class UtilityConfigs(BaseConfig):
    semantic_diversity: Optional[SemanticRewardConfig] = field(
        default_factory=SemanticRewardConfig,
        metadata={"help": "The semantic reward config"}
    )
    safety: Optional[SafetyRewardConfig] = field(
        default_factory=SafetyRewardConfig,
        metadata={"help": "The safety reward config"}
    )
    gibberish: Optional[GibberishRewardConfig] = field(
        default_factory=GibberishRewardConfig,
        metadata={"help": "The gibberish reward config"}
    )
    ngram_diversity: Optional[NgramDiversityRewardConfig] = field(
        default_factory=NgramDiversityRewardConfig,
        metadata={"help": "The ngram diversity reward config"}
    )


@dataclass
class BlueConfig(BaseConfig):
    reward_configs: Optional[UtilityConfigs] = field(
        default_factory=UtilityConfigs,
        metadata={"help": "The reward configs for blue team"}
    )
    blue_model_name: str = field(
        default="vicgalle/gpt2-alpaca-gpt4",
        metadata={"help": "The model name or path for blue team"}
    )
    blue_generation_kwargs: Optional[GenerationConfig] = field(
        default_factory=GenerationConfig,
        metadata={"help": "The generation config for blue team from transformers library"}
    )
    blue_generation_kwargs_max_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum length of the input sequence for blue team"}
    )
    blue_generation_kwargs_temperature: Optional[int] = field(
        default=0.7,
        metadata={"help": "The maximum length of the input sequence for blue team"}
    )
    blue_generation_kwargs_max_new_tokens: Optional[int] = field(
        default=20,
        metadata={"help": "The maximum length of the input sequence for blue team"}
    )
    blue_generation_kwargs_top_p: Optional[int] = field(
        default=0.92,
        metadata={"help": "The maximum length of the input sequence for blue team"}
    )
    blue_generation_kwargs_repetition_penalty: Optional[int] = field(
        default=1.0,
        metadata={"help": "The maximum length of the input sequence for blue team"}
    )
    blue_system_message: Optional[str] = field(
        default=None,
        metadata={"help": "The system message for blue team"}
    )
    blue_template: Optional[str] = field(
        default='alpaca',
        metadata={"help": "The template for blue team"}
    )
    save_memory: Optional[bool] = field(
        default=False,
        metadata={"help": "The save memory for blue team"}
    )

    # cost and reward configuration
    blue_safety_reward: bool = field(
        default=True,
        metadata={"help": "The safety is reward"}
    )
    blue_gibberish_reward: bool = field(
        default=True,
        metadata={"help": "The safety is reward"}
    )
    cost_configs: Optional[UtilityConfigs] = field(
        default_factory=UtilityConfigs,
        metadata={"help": "The cost configs for blue team"}
    )

    # Configurable Reward Modules and their Parameters
    semantic_reward_module: str = field(
        default="SampledSemanticDiversityScore",
        metadata={"help": "The reward module name"}
    )
    semantic_reward_coef: float = field(
        default=1.0,
        metadata={"help": "The reward coefficient"}
    )
    semantic_reward_team: str = field(
        default="red",
        metadata={"help": "The reward team"}
    )
    semantic_reward_kwarg_embedder_model_name: str = field(
        default="minilm",
        metadata={"help": "The embedder model name"}
    )
    semantic_reward_kwarg_sample_size: int = field(
        default=-1,
        metadata={"help": "The sample size"}
    )
    semantic_reward_kwarg_top_k: int = field(
        default=16,
        metadata={"help": "The top k value for the reward"}
    )

    # Safety Reward Config
    safety_reward_module: str = field(
        default="SafetyScore",
        metadata={"help": "The reward module name"}
    )
    safety_reward_coef: float = field(
        default=1.0,
        metadata={"help": "The reward coefficient"}
    )
    safety_reward_team: str = field(
        default="both",
        metadata={"help": "The reward team"}
    )
    safety_reward_kwarg_model_name: str = field(
        default="toxigen",
        metadata={"help": "The model name for safety score"}
    )

    # Cost Config for Safety
    safety_cost_module: str = field(
        default="SafetyScore",
        metadata={"help": "The cost module name"}
    )
    safety_cost_threshold: float = field(
        default=-0.6,
        metadata={"help": "The cost coefficient"}
    )
    safety_cost_team: str = field(
        default="both",
        metadata={"help": "The cost team"}
    )
    safety_cost_kwarg_model_name: str = field(
        default="toxigen",
        metadata={"help": "The model name for safety score"}
    )
    safety_cost_transformation: Callable = field(
        default=lambda x: -x,
        metadata={"help": "The transformation for the cost"}
    )

    # Gibberish Reward and Cost Config
    gibberish_reward_module: str = field(
        default="GibberishScore",
        metadata={"help": "The reward module name"}
    )
    gibberish_reward_coef: float = field(
        default=1.0,
        metadata={"help": "The reward coefficient"}
    )
    gibberish_reward_team: str = field(
        default="red",
        metadata={"help": "The reward team"}
    )
    gibberish_reward_kwarg_method: str = field(
        default="not_clean",
        metadata={"help": "The reward module kwargs"}
    )

    gibberish_cost_module: str = field(
        default="GibberishScore",
        metadata={"help": "The cost module name"}
    )
    gibberish_cost_threshold: float = field(
        default=0.1,
        metadata={"help": "The reward coefficient"}
    )
    gibberish_cost_team: str = field(
        default="red",
        metadata={"help": "The reward team"}
    )
    gibberish_cost_kwarg_method: str = field(
        default="not_clean",
        metadata={"help": "The reward module kwargs"}
    )
    gibberish_cost_threshold: float = field(
        default=0.1,
        metadata={"help": "The reward coefficient"}
    )
    gibberish_cost_transformation: Callable = field(
        default=lambda x: -x,
        metadata={"help": "The transformation for the cost"}
    )

    # Ngram Diversity Config
    ngram_reward_module: str = field(
        default="SelfBLEUScore",
        metadata={"help": "The ngram diversity reward module name"}
    )
    ngram_reward_coef: float = field(
        default=1.0,
        metadata={"help": "The ngram diversity reward coefficient"}
    )
    ngram_reward_team: str = field(
        default="red",
        metadata={"help": "The ngram diversity reward team"}
    )
    ngram_reward_kwarg_K: int = field(
        default=5,
        metadata={"help": "The max ngram size"}
    )
    ngram_reward_kwarg_sample_size: int = field(
        default=-1,
        metadata={"help": "The sample size"}
    )

    def __post_init__(self):
        # Set the attributes for red_team_model_config
        new_generation_kwargs ={}
        for key, value in self.__dict__.items():
            if key.startswith("blue_generation_kwargs_"):
                new_generation_kwargs[key.replace('blue_generation_kwargs_', '')] = value
        if self.blue_generation_kwargs_temperature == 0:
            new_generation_kwargs['do_sample'] = False
        else:
            new_generation_kwargs['do_sample'] = True
        setattr(self, 'blue_generation_kwargs', GenerationConfig(**new_generation_kwargs))

        # Set the attributes for reward/cost configs
        reward_configs = UtilityConfigs(
            safety=SafetyRewardConfig(
                **{key.replace('safety_reward_', ''): value for key, value in self.__dict__.items() if key.startswith("safety_reward_")}
            ) if self.blue_safety_reward else None,
            gibberish=GibberishRewardConfig(
                **{key.replace('gibberish_reward_', ''): value for key, value in self.__dict__.items() if key.startswith("gibberish_reward_")}
            ) if self.blue_gibberish_reward else None,
            semantic_diversity=SemanticRewardConfig(
                **{key.replace('semantic_reward_', ''): value for key, value in self.__dict__.items() if key.startswith("semantic_reward_")}
            ),
            ngram_diversity=NgramDiversityRewardConfig(
                **{key.replace('ngram_reward_', ''): value for key, value in self.__dict__.items() if key.startswith("ngram_reward_")}
            )
        )
        setattr(self, 'reward_configs', reward_configs)

        cost_configs = UtilityConfigs(
            safety=SafetyCostConfig(
                **{key.replace('safety_cost_', ''): value for key, value in self.__dict__.items() if key.startswith("safety_cost_")}
            ) if not self.blue_safety_reward else None,
            gibberish=GibberishCostConfig(
                **{key.replace('gibberish_cost_', ''): value for key, value in self.__dict__.items() if key.startswith("gibberish_cost_")}
            ) if not self.blue_gibberish_reward else None,
            semantic_diversity=None,
            ngram_diversity=None,
        )
        setattr(self, 'cost_configs', cost_configs)
