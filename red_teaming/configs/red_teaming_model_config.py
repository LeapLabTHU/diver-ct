from dataclasses import field, dataclass

from trl.trainer.model_config import ModelConfig

@dataclass
class RedTeamModelConfig(ModelConfig):
    """
    Arguments which define the model and tokenizer to load.
    """
    model_kwargs: dict = field(
        default_factory=lambda: {},
        metadata={"help": "Model kwargs for initializing from pretrained"}
    )
    num_layers_unfrozen : int = field(
        default=2,
        metadata={"help": "if > 0, the number of layers to unfreeze, else all layers are unfrozen"}
    )
    model_name_or_path: str = field(
        default="gpt2",
        metadata={"help": "Model name or path"}
    )

    def __post_init__(self):
        super().__post_init__()
        if 'gpt2' not in self.model_name_or_path:
            self.model_kwargs = field(
                default=lambda: {},
                metadata={"help": "Model kwargs for initializing from pretrained"}
            )
