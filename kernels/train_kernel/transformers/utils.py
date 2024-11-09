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
    """
    A transformer that takes a MachineLearningDataPartsWithScalersModel and returns the MachineLearningDataModel that is stored in the "train" attribute of the input.

    Args:
        data (MachineLearningDataPartsWithScalersModel): The input data model

    Returns:
        MachineLearningDataModel: The train data model
    """
    return data.train


@transformer
def pick_validation(
    data: MachineLearningDataPartsWithScalersModel,
) -> MachineLearningDataModel:
    """
    A transformer that takes a MachineLearningDataPartsWithScalersModel and returns the MachineLearningDataModel that is stored in the "validation" attribute of the input.

    Args:
        data (MachineLearningDataPartsWithScalersModel): The input data model

    Returns:
        MachineLearningDataModel: The validation data model
    """
    return data.validation


@transformer
def pick_test(
    data: MachineLearningDataPartsWithScalersModel,
) -> MachineLearningDataModel:
    """
    A transformer that extracts the test data model from a MachineLearningDataPartsWithScalersModel.

    Args:
        data (MachineLearningDataPartsWithScalersModel): The input data model containing train, validation, and test parts.

    Returns:
        MachineLearningDataModel: The test data model extracted from the input.
    """
    return data.test


@partial_transformer
def parse_information_to_train(
    data: tuple[DataType, DataType],
    model_config: ModelConfig,
) -> tuple[ModelConfig, tuple[DataType, DataType]]:
    """
    A partial transformer that takes a tuple of two data models and a model configuration model.
    It returns a tuple containing the model configuration and the two data models.

    This transformer is used to parse the information needed to train a machine learning algorithm.

    Args:
        data (tuple[DataType, DataType]): A tuple containing two data models.
        model_config (ModelConfig): The configuration model for the machine learning algorithm.

    Returns:
        tuple[ModelConfig, tuple[DataType, DataType]]: A tuple containing the model configuration and the two data models.
    """
    train, val = data

    return model_config, (train, val)


@partial_transformer
def train_ml_alg(
    data: tuple[ModelConfig, tuple[DataType, DataType]],
    ml_alg: MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
    train_config: TrainConfigBaseModel,
) -> MachineLearningAlgorithm[ModelType, ModelConfig, DataType]:
    """
    A partial transformer that takes a tuple of two data models, a machine learning algorithm instance,
    and a training configuration model. It returns the trained machine learning algorithm instance.

    This transformer is used to train a machine learning algorithm using the given data and configuration.

    Args:
        data (tuple[ModelConfig, tuple[DataType, DataType]]): A tuple containing two data models.
        ml_alg (MachineLearningAlgorithm[ModelType, ModelConfig, DataType]): The machine learning algorithm instance.
        train_config (TrainConfigBaseModel): The training configuration model.

    Returns:
        MachineLearningAlgorithm[ModelType, ModelConfig, DataType]: The trained machine learning algorithm instance.
    """
    model_config, (train, val) = data

    model = ml_alg.fit(
        data=train,
        train_config=train_config,
        model_config=model_config,
        val_data=val,
    )

    return model
