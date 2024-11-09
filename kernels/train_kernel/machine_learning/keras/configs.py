from typing import Optional, Generic

from pydantic import PositiveInt
from research_flow.types.comon_types import ModelConfig
from research_flow.types.configs.hpo_config_base_model import HPOConfigBaseModel
from research_flow.types.configs.train_config_base_model import TrainConfigBaseModel


class KerasTrainConfigs(TrainConfigBaseModel):
    """
    Class representing the configuration for training a Keras model.

    Attributes:
        epochs (int): The number of training epochs.
        log_dir (str, optional): The directory where training logs will be stored. Defaults to "tb_logs/train".
        verbose (PositiveInt, optional): The verbosity level of the training process. Defaults to 1.
        with_early_stopping (bool, optional): Whether to use early stopping during training. Defaults to False.
        batch_size (Optional[int], optional): The size of each training batch. Defaults to None.
        num_of_workers (PositiveInt, optional): The number of workers to use for training. Defaults to 12.
        patience (PositiveInt, optional): The number of epochs to wait before early stopping if no improvement is seen. Defaults to 10.
    """

    epochs: int
    log_dir: str = "tb_logs/train"
    verbose: PositiveInt = 1
    with_early_stopping: bool = False
    batch_size: Optional[int] = None
    num_of_workers: PositiveInt = 12
    patience: PositiveInt = 10


class KerasHPOConfig(HPOConfigBaseModel[ModelConfig], Generic[ModelConfig]):
    pass
