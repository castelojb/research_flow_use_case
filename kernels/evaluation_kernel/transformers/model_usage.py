from gloe import transformer
from research_flow.machine_learning.base_machine_learning_algorithm import (
    DataType,
    MachineLearningAlgorithm,
)
from research_flow.types.comon_types import ModelType, ModelConfig


@transformer
def do_predictions(
    data: tuple[DataType, MachineLearningAlgorithm[ModelType, ModelConfig, DataType]]
) -> DataType:
    """
    A transformer that takes a tuple of a DataType object and a MachineLearningAlgorithm object as input.
    It makes a prediction on the DataType object using the MachineLearningAlgorithm object.
    The prediction is then returned as a DataType object.

    :param data: A tuple containing a DataType object and a MachineLearningAlgorithm object.
    :return: A DataType object with the prediction.
    """
    x, model = data

    preds = model.predict(x)

    return preds
