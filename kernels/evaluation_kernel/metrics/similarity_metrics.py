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
    preds, real = data

    correlation, p_valor = pearsonr(preds, real)

    return correlation


@transformer
def dtw_transformer(data: tuple[Prediction, Real]) -> float:
    preds, real = data

    distancia, _ = fastdtw(np.array(preds), np.array(real))

    return distancia


class PersonPerPatient(MetricBaseModel[DataType], Generic[DataType]):
    def get_metric_name(self) -> str:
        return "Person"

    def get_metric(
        self,
    ) -> Transformer[DataType, MetricScore]:
        return (
            flat_signals
            >> person_transformer
            >> build_score_report(self.get_metric_name())
        )


class DTWPerPatient(MetricBaseModel[DataType], Generic[DataType]):
    def get_metric_name(self) -> str:
        return "DTW"

    def get_metric(
        self,
    ) -> Transformer[DataType, MetricScore]:
        return (
            flat_signals
            >> dtw_transformer
            >> build_score_report(self.get_metric_name())
        )
