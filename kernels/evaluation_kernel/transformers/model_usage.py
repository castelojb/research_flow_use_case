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
    x, model = data

    preds = model.predict(x)

    return preds
