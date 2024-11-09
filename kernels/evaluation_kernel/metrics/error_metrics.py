from typing import Generic

from gloe import transformer, Transformer
from research_flow.machine_learning.base_machine_learning_algorithm import DataType
from research_flow.types.comon_types import Prediction, Real
from research_flow.types.metrics.metric_base_model import MetricBaseModel
from sklearn.metrics import mean_squared_error

from kernels.evaluation_kernel.metrics.score_model import MetricScore
from kernels.evaluation_kernel.transformers.utils import (
    flat_signals,
    build_score_report,
)


@transformer
def mean_squared_error_transformer(data: tuple[Prediction, Real]) -> float:
    preds, real = data

    return mean_squared_error(real, preds)


class MeanSquareErrorPerPatient(MetricBaseModel[DataType], Generic[DataType]):
    def get_metric_name(self) -> str:
        return "RMSE"

    def get_metric(
        self,
    ) -> Transformer[DataType, MetricScore]:
        return (
            flat_signals
            >> mean_squared_error_transformer
            >> build_score_report(self.get_metric_name())
        )
