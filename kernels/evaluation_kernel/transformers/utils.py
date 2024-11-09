from itertools import chain

import pandas as pd
from gloe import transformer, partial_transformer
from research_flow.machine_learning.base_machine_learning_algorithm import (
    MachineLearningAlgorithm,
    DataType,
)
from research_flow.types.comon_types import ModelType, ModelConfig, Prediction, Real
from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)
from research_flow.types.machine_learning.machine_learning_data_parts_with_scalers_model import (
    MachineLearningDataPartsWithScalersModel,
)

from kernels.evaluation_kernel.metrics.score_model import MetricScore


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
def build_score_dataframe(data: list[list[MetricScore]]) -> pd.DataFrame:
    rows = []

    for row in data:
        rows.extend(row)

    dicts = [x.dict() for x in rows]

    df = pd.DataFrame(dicts)

    return df


@transformer
def flat_signals(data: DataType) -> tuple[Prediction, Real]:
    preds = data.prediction

    real = data.output_series

    preds_flatten = list(chain.from_iterable(preds))

    real_flatten = list(chain.from_iterable(real))

    return preds_flatten, real_flatten


@partial_transformer
def build_score_report(metric_result: float, metric_name: str) -> MetricScore:

    return MetricScore(metric_name=metric_name, score=metric_result)
