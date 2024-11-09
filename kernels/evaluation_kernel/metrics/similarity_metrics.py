from typing import Generic

import numpy as np
from fastdtw import fastdtw
from gloe import transformer, Transformer
from research_flow.machine_learning.base_machine_learning_algorithm import DataType
from research_flow.types.comon_types import Prediction, Real
from research_flow.types.metrics.metric_base_model import MetricBaseModel
from scipy.stats import pearsonr

from kernels.evaluation_kernel.metrics.score_model import MetricScore
from kernels.evaluation_kernel.transformers.utils import (
    flat_signals,
    build_score_report,
)


@transformer
def person_transformer(data: tuple[Prediction, Real]) -> float:
    """
    A transformer that takes a tuple of two lists, the first one being the predictions and the second one being the real values.
    It first flattens the two lists into a single tuple, then computes the Person correlation between the two lists and finally
    builds a score report from the result.

    Args:
        data (tuple[Prediction, Real]): A tuple containing two lists, the first one being the predictions and the second one being the real values.

    Returns:
        float: The Person correlation between the two lists.
    """
    preds, real = data

    correlation, p_valor = pearsonr(preds, real)

    return correlation


@transformer
def dtw_transformer(data: tuple[Prediction, Real]) -> float:
    """
    A transformer that takes a tuple of two lists, the first one being the predictions and the second one being the real values.
    It first flattens the two lists into a single tuple, then computes the DTW distance between the two lists and finally
    builds a score report from the result.

    Args:
        data (tuple[Prediction, Real]): A tuple containing two lists, the first one being the predictions and the second one being the real values.

    Returns:
        float: The DTW distance between the two lists.
    """
    preds, real = data

    distance, _ = fastdtw(np.array(preds), np.array(real))

    return distance


class PersonPerPatient(MetricBaseModel[DataType], Generic[DataType]):
    """
    A class that evaluates the Person correlation metric for a given model.

    The Person correlation metric measures the correlation between the predicted and actual signals for a given patient.
    This class provides a transformer that takes in a DataType and returns a MetricScore object, which contains the
    correlation value and the name of the metric.

    Attributes:
        None

    Methods:
        get_metric_name: Returns the name of the metric, which is "Person".
        get_metric: Returns a transformer that evaluates the Person correlation metric.

    Returns:
        A MetricScore object containing the correlation value and the name of the metric.
    """

    def get_metric_name(self) -> str:
        return "Person"

    def get_metric(
        self,
    ) -> Transformer[DataType, MetricScore]:
        """
        Returns a transformer that evaluates the Person correlation metric for a given model.

        The transformer takes in a DataType and returns a MetricScore.

        The transformer first flattens the prediction and ground truth signals, then calculates the Pearson correlation
        between the two signals. The correlation is then used to construct a MetricScore object, which contains the
        correlation value and the name of the metric.

        :return: A transformer that evaluates the Person correlation metric
        :rtype: Transformer[DataType, MetricScore]
        """
        return (
            flat_signals
            >> person_transformer
            >> build_score_report(self.get_metric_name())
        )


class DTWPerPatient(MetricBaseModel[DataType], Generic[DataType]):
    """
    A class that represents the Dynamic Time Warping (DTW) metric for evaluating model performance on a per-patient basis.

    The DTW metric measures the similarity between two time series signals by calculating the minimum cost of transforming one signal into another.

    This class provides a transformer that takes in a DataType and returns a MetricScore object, which contains the DTW distance value and the name of the metric.

    Attributes:
        None

    Methods:
        get_metric_name: Returns the name of the metric, which is "DTW".
        get_metric: Returns a transformer that evaluates the DTW metric for a given model.

    Returns:
        A MetricScore object containing the DTW distance value and the name of the metric.
    """

    def get_metric_name(self) -> str:
        return "DTW"

    def get_metric(
        self,
    ) -> Transformer[DataType, MetricScore]:
        """
        Returns a transformer that evaluates the DTW metric for a given model.

        The transformer takes in a DataType and returns a MetricScore.

        The transformer first flattens the prediction and ground truth signals, then calculates the DTW distance between the two signals. The distance is then used to construct a MetricScore object, which contains the distance value and the name of the metric.

        :return: A transformer that evaluates the DTW metric
        :rtype: Transformer[DataType, MetricScore]
        """
        return (
            flat_signals
            >> dtw_transformer
            >> build_score_report(self.get_metric_name())
        )
