from gloe import transformer, partial_transformer
from research_flow.machine_learning.base_machine_learning_algorithm import (
    DataType,
    MachineLearningAlgorithm,
)
from research_flow.types.comon_types import ModelConfig, ModelType
from research_flow.types.configs.train_config_base_model import TrainConfigBaseModel
from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)
from research_flow.types.machine_learning.machine_learning_data_parts_with_scalers_model import (
    MachineLearningDataPartsWithScalersModel,
)


@transformer
def pick_train(
    data: MachineLearningDataPartsWithScalersModel,
) -> MachineLearningDataModel:
    return data.train


@transformer
def pick_validation(
    data: MachineLearningDataPartsWithScalersModel,
) -> MachineLearningDataModel:
    return data.validation


@transformer
def pick_test(
    data: MachineLearningDataPartsWithScalersModel,
) -> MachineLearningDataModel:
    return data.test


@partial_transformer
def parse_information_to_train(
    data: tuple[DataType, DataType],
    model_config: ModelConfig,
) -> tuple[ModelConfig, tuple[DataType, DataType]]:
    train, val = data

    return model_config, (train, val)


@partial_transformer
def train_ml_alg(
    data: tuple[ModelConfig, tuple[DataType, DataType]],
    ml_alg: MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
    train_config: TrainConfigBaseModel,
) -> MachineLearningAlgorithm[ModelType, ModelConfig, DataType]:
    model_config, (train, val) = data

    model = ml_alg.fit(
        data=train,
        train_config=train_config,
        model_config=model_config,
        val_data=val,
    )

    return model
