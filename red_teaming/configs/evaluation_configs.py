
from dataclasses import dataclass, field
from typing import List


@dataclass
class EvaluationConfig:
    log_path: str             = field(default=None,metadata={"help": "Path to the FOLDER of the logs to evaluate."})
    min_log_idx: int          = field(default=0,metadata={"help": "Minimum index of the logs to load. Default is 0."})
    max_log_idx: int          = field(default=-1,metadata={"help": "Maximum index of the logs to load. Default is -1 (all)."})
    toxicity_threshold: float = field(default=0.5,metadata={"help": "Toxicity threshold to filter the loaded logs. Default is 0."})
    success_threshold: float  = field(default=0.5,metadata={"help": "Success threshold to filter the loaded logs. Default is 0."})
    device: str               = field(default='cuda',metadata={"help": "Device to use for computation. Default is cuda."})
    dataset_names: List[str]  = field(default_factory=lambda: ["PKU-Alignment/PKU-SafeRLHF","skg/toxigen-data"],metadata={"help": "List of dataset names to evaluate."})
    split: str                = field(default='test',metadata={"help": "Split of the dataset to use. Default is test."})
    num_dataset_samples: int  = field(default=-1,metadata={"help": "Number of samples from the dataset to compare to. Default is -1 (all)."})
    toxicity_models_name: str  = field(default='toxigen',metadata={"help": "Name of the toxicity model to use. Default is toxigen."})
    metrics_type: str         = field(default='self',metadata={"help": "Type of metrics to compute."})
    eval_num_samples: int     = field(default=1000,metadata={"help": "For sampled eval, number of data to sample. Default is 100."})
    eval_num_sampling: int    = field(default=25,metadata={"help": "For sampled eval, number of times to sample the data. Default is 1000."})
    save_name: str            = field(default=None,metadata={"help": "Name of the file to save the evaluation results. Default is evaluation."})