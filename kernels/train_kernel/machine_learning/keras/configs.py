from typing import Optional, Generic

from pydantic import PositiveInt
from research_flow.types.comon_types import ModelConfig
from research_flow.types.configs.hpo_config_base_model import HPOConfigBaseModel
from research_flow.types.configs.train_config_base_model import TrainConfigBaseModel


class KerasTrainConfigs(TrainConfigBaseModel):
    epochs: int
    log_dir: str = "tb_logs/train"
    verbose: PositiveInt = 1
    with_early_stopping: bool = False
    batch_size: Optional[int] = None
    num_of_workers: PositiveInt = 12
    patience: PositiveInt = 10


class KerasHPOConfig(HPOConfigBaseModel[ModelConfig], Generic[ModelConfig]):
    pass
