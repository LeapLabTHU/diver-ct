from typing import Optional, Literal
from dataclasses import field, dataclass

from trl.trainer.ppo_config import PPOConfig
from transformers import GenerationConfig
from torch.nn import functional as F
import torch

from red_teaming.configs.red_teaming_model_config import RedTeamModelConfig


@dataclass
class RedTeamPPOConfig(PPOConfig):
    """
    Configuration class for RedTeamPPOTrainer
    """

    red_team_model_config: Optional[RedTeamModelConfig] = field(
        default_factory=RedTeamModelConfig,
        metadata={"help": "The model config for red team"}
    )

    red_generation_kwargs: Optional[GenerationConfig] = field(
        default_factory=GenerationConfig,
        metadata={"help": "The generation config for blue team from transformers library"}
    )
    # Set the default values for red_team_model_config
    red_generation_kwargs_max_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum length of the input sequence"}
    )
    red_generation_kwargs_temperature: Optional[int] = field(
        default=0.7,
        metadata={"help": "The maximum length of the input sequence for blue team"}
    )
    red_generation_kwargs_max_new_tokens: Optional[int] = field(
        default=50,
        metadata={"help": "The maximum length of the input sequence for blue team"}
    )
    red_generation_kwargs_top_p: Optional[int] = field(
        default=0.92,
        metadata={"help": "The maximum length of the input sequence for blue team"}
    )
    red_generation_kwargs_repetition_penalty: Optional[int] = field(
        default=1.0,
        metadata={"help": "The maximum length of the input sequence for blue team"}
    )

    entropy_coeff: float = field(
        default=0.001,
        metadata={"help": "The entropy bonus coefficient"}
    )
    # From PPOConfig    
    query_dataset: str = field(
        default="",
        metadata={"help": "Comma-separated string of dataset file paths"}
    )
    seed: int = field(
        default=69,
        metadata={"help": "The random seed"}
    )
    kl_penalty: Literal["kl", "abs", "mse", "full", "unbiased-kl"] = field(
        default="kl",
        metadata={"help": "kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution, 'unbiased-kl': ratio - log ratio - 1 "}
    )   
    log_with: Optional[Literal["wandb", "tensorboard", None]] = field(
        default=None,
        metadata={"help": "Log with either 'wandb' or 'tensorboard', check  https://huggingface.co/docs/accelerate/usage_guides/tracking for more details"}
    )
    log_freq: int = field(
        default=100,
        metadata={"help": "Logging frequency"}
    )
    log_train_path: Optional[str] = field(
        default='logs/train',
        metadata={"help": "Path to log the training observations"}
    )
    need_eval: bool = field(
        default=True,
        metadata={"help": "Whether to evaluate the model"}
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of task to use - used only for tracking purposes"}
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "The reward model to use - used only for tracking purposes"}
    )
    
    ## hyperparameters
    steps: int = field(
        default=10000,
        metadata={"help": "Number of training steps"}
    )
    learning_rate: float = field(
        default=3e-5,
        metadata={"help": "Adam learning rate"}
    )
    adap_kl_ctrl: bool = field(
        default=False,
        metadata={"help": "Use adaptive KL control, otherwise linear"}
    )
    init_kl_coef: float = field(
        default=0.001,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"}
    )
    target: Optional[float] = field(
        default=None,
        metadata={"help": "Target KL value for adaptive KL control"}
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Range for clipping in PPO policy gradient loss"}
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Range for clipping values in loss calculation"}
    )
    vf_coef: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for value loss"}
    )
    batch_size: int = field(
        default=128,
        metadata={"help": "The batch size"}
    )
    mini_batch_size: int = field(
        default=8,
        metadata={"help": "The mini batch size"}
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "Number of optimisation epochs per batch of samples"}
    )
    score_clip: Optional[float] = field(
        default=None,
        metadata={"help": "Use score normalization. Only applicable if use_score_scaling is True"}
    )

    # optimization configs
    train_use_8bit_adam: bool = field(
        default=False,
        metadata={"help": "Whether to use the 8bit Adam optimizer from bitsandbytes."}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "adam beta1"}
    )
    """Adam beta1."""
    adam_beta2: float = field(
        default=0.95,
        metadata={"help": "adam beta2"}
    )
    """Adam beta2."""
    weight_decay: float = field(
        default=1.0e-6,
        metadata={"help": "weight decay"}
    )
    """Adam weight decay."""
    adam_epsilon: float = field(
        default=1.0e-8,
        metadata={"help": "adam epsilon"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler [linear, cosine, cosine_with_restarts, \
            polynomial, constant, constant_with_warmup]"},
    )
    lr_warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "warmup ratio"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "grad norm clipping"}
    )
    scale_logits: bool = field(
        default=True,
        metadata={"help": "Whether to scale logits"}
    )

    def __post_init__(self):
        super().__post_init__()
        # Set the attributes for red_team_model_config
        new_generation_kwargs ={}
        for key, value in self.__dict__.items():
            if key.startswith("red_generation_kwargs_"):
                new_generation_kwargs[key.replace('red_generation_kwargs_', '')] = value
        if self.red_generation_kwargs_temperature == 0:
            new_generation_kwargs['do_sample'] = False
        else:
            new_generation_kwargs['do_sample'] = True
        setattr(self, 'red_generation_kwargs', new_generation_kwargs)


@dataclass
class RedTeamPPOLagrangianConfig(RedTeamPPOConfig):
    # Constrained RL stuff
    lagrange_lr: float = field(
        default=0.1,
        metadata={"help": "lagrange multiplier learning rate"}
    )
    lagrange_momentum: float = field(
        default=0.1,
        metadata={"help": "lagrange momentum"}
    )
    lagrange_init: float = field(
        default=1.0,
        metadata={"help": "lagrange init value"}
    )
    lagrange_max: float = field(
        default=None,
        metadata={"help": "lagrange max value"}
    )
    lagrange_transform: str = field(
        default="exp",
        metadata={"help": "lagrange transform"}
    )
    langrange_untransform: bool = field(
        default=True,
        metadata={"help": "lagrange untransform"}
    )
    lagrange_update_delay_steps: int = field(
        default=0,
        metadata={"help": "lagrange update delay steps"}
    )
    episode_cost_window_size: int = field(
        default=None,
        metadata={"help": "episode cost window size"}
    )
    cost_coef: float = field(
        default=1.0,
        metadata={"help": "cost loss coefficient"}
    )
    normalize_advantages: bool = field(
        default=True,
        metadata={"help": "normalize advantages"}
    )

    def __post_init__(self):
        super().__post_init__()
        # check lagrange config
        if self.lagrange_transform not in ["softplus", "exp", "sigmoid", "tanh"]:
            raise ValueError("lagrange_tranform must be one of 'softplus', 'exp', 'sigmoid', 'tanh'")
        if self.lagrange_max is not None and self.lagrange_max < 0:
            raise ValueError("lagrange_max must be positive")

        # inverse transform
        if self.langrange_untransform:
            # if we are untransforming, we cannot use softplus
            if self.lagrange_transform == "softplus":
                raise ValueError("softplus transform cannot be untransformed")
            if self.lagrange_transform == "exp":
                setattr(self, 'lagrange_inverse', torch.log)
            elif self.lagrange_transform == "sigmoid":
                setattr(self, 'lagrange_inverse', F.logit)
            elif self.lagrange_transform == "tanh":
                setattr(self, 'lagrange_inverse', F.atanh)

        # transform
        if self.lagrange_transform == "softplus":
            setattr(self, 'lagrange_transform', F.softplus)
        elif self.lagrange_transform == "exp":
            setattr(self, 'lagrange_transform', torch.exp)
        elif self.lagrange_transform == "sigmoid":
            setattr(self, 'lagrange_transform', F.sigmoid)
        elif self.lagrange_transform == "tanh":
            setattr(self, 'lagrange_transform', F.tanh)
