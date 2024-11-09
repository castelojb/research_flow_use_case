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
    """
    A transformer that extracts the data parts with scalers from a tuple containing
    a machine learning algorithm and its associated data.

    Args:
        data (tuple): A tuple containing:
            - MachineLearningAlgorithm: The machine learning algorithm.
            - MachineLearningDataPartsWithScalersModel: The data parts with scalers.

    Returns:
        MachineLearningDataPartsWithScalersModel: The data parts with scalers extracted from the input tuple.
    """
    return data[1]


@transformer
def pick_test(
    data: MachineLearningDataPartsWithScalersModel,
) -> MachineLearningDataModel:
    """
    A transformer that extracts the test data part from a MachineLearningDataPartsWithScalersModel.

    Args:
        data (MachineLearningDataPartsWithScalersModel): The data parts with scalers.

    Returns:
        MachineLearningDataModel: The test data part extracted from the input.
    """
    return data.test


@transformer
def pick_model(
    data: tuple[
        MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
        MachineLearningDataPartsWithScalersModel,
    ]
) -> MachineLearningAlgorithm[ModelType, ModelConfig, DataType]:
    """
    A transformer that extracts the machine learning algorithm from a tuple containing
    a machine learning algorithm and its associated data parts with scalers.

    Args:
        data (tuple): A tuple containing:
            - MachineLearningAlgorithm: The machine learning algorithm.
            - MachineLearningDataPartsWithScalersModel: The data parts with scalers.

    Returns:
        MachineLearningAlgorithm: The machine learning algorithm extracted from the input tuple.
    """
    return data[0]


@transformer
def build_score_dataframe(data: list[list[MetricScore]]) -> pd.DataFrame:
    """
    This function takes a list of lists of MetricScore and builds a pandas DataFrame
    where each row is a MetricScore and the columns are the attributes of the MetricScore.

    Args:
        data (list[list[MetricScore]]): A list of lists of MetricScore.

    Returns:
        pd.DataFrame: A pandas DataFrame with the MetricScore.
    """
    rows = []

    for row in data:
        rows.extend(row)

    dicts = [x.dict() for x in rows]

    df = pd.DataFrame(dicts)

    return df


@transformer
def flat_signals(data: DataType) -> tuple[Prediction, Real]:
    """
    This function takes a DataType object and returns a tuple of two lists.
    The first element of the tuple is a list of all the predictions in the DataType object,
    flattened into a single list. The second element of the tuple is a list of all the
    real values in the DataType object, flattened into a single list.

    Args:
        data (DataType): A DataType object.

    Returns:
        tuple[Prediction, Real]: A tuple of two lists. The first list is a list of all the
        predictions in the DataType object, flattened into a single list. The second list
        is a list of all the real values in the DataType object, flattened into a single list.
    """
    preds = data.prediction

    real = data.output_series

    preds_flatten = list(chain.from_iterable(preds))

    real_flatten = list(chain.from_iterable(real))

    return preds_flatten, real_flatten


@partial_transformer
def build_score_report(metric_result: float, metric_name: str) -> MetricScore:
    """
    Constructs a `MetricScore` object from a metric result and its corresponding name.

    This function takes a float value representing the result of a metric calculation and a string
    representing the name of the metric. It returns a `MetricScore` object containing these values.

    Args:
        metric_result (float): The result of the metric calculation.
        metric_name (str): The name of the metric.

    Returns:
        MetricScore: A `MetricScore` object containing the metric name and the calculated score.
    """
    return MetricScore(metric_name=metric_name, score=metric_result)


@transformer
def denomalize(
    data: tuple[
        DataType,
        tuple[
            MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
            MachineLearningDataPartsWithScalersModel,
        ],
    ]
) -> DataType:
    """
    The denomalize function reverses the normalization process for the given data.

    This function takes a tuple containing the data predictions and a tuple with the machine learning algorithm
    and the data parts with scalers. It applies the inverse transform using the scalers to denormalize the
    input series, output series, and predictions.

    Args:
        data (tuple): A tuple containing:
            - DataType: The data predictions.
            - tuple: A tuple containing:
                - MachineLearningAlgorithm: The machine learning algorithm used.
                - MachineLearningDataPartsWithScalersModel: The data parts with their associated scalers.

    Returns:
        DataType: The denormalized data predictions, including the denormalized input series, output series,
        and predictions.
    """
    data_preds, (_, data_with_scalers) = data

    input_test_scaler = data_with_scalers.test_scaler_input_series
    output_test_scaler = data_with_scalers.test_scaler_output_series

    denomalized_output = output_test_scaler.inverse_transform(data_preds.output_series)
    denomalized_input = input_test_scaler.inverse_transform(data_preds.input_series)
    denomalized_pred = output_test_scaler.inverse_transform(data_preds.prediction)

    return data_preds.copy(
        update={
            "prediction": denomalized_pred,
            "input_series": denomalized_input,
            "output_series": denomalized_output,
        }
    )
