import pandas as pd
from gloe import transformer
from research_flow.machine_learning.base_machine_learning_algorithm import (
    MachineLearningAlgorithm,
    DataType,
)
from research_flow.types.comon_types import ModelType, ModelConfig
from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)
from research_flow.types.machine_learning.machine_learning_data_parts_with_scalers_model import (
    MachineLearningDataPartsWithScalersModel,
)
from research_flow.types.metrics.metric_score_model import MetricScoreModel


@transformer
def pick_data(
    data: tuple[
        MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
        MachineLearningDataPartsWithScalersModel,
    ]
) -> MachineLearningDataPartsWithScalersModel:
    return data[1]


@transformer
def pick_test(
    data: MachineLearningDataPartsWithScalersModel,
) -> MachineLearningDataModel:
    return data.test


@transformer
def pick_model(
    data: tuple[
        MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
        MachineLearningDataPartsWithScalersModel,
    ]
) -> MachineLearningAlgorithm[ModelType, ModelConfig, DataType]:
    return data[0]


@transformer
def build_score_dataframe(data: list[list[MetricScoreModel]]) -> pd.DataFrame:
    rows = []

    for row in data:
        rows.extend(row)

    dicts = [x.dict() for x in rows]

    df = pd.DataFrame(dicts)

    return df
