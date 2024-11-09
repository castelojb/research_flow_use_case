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
    """
    Calculates the mean squared error between predictions and real values.

    This function takes a tuple containing two lists: predictions and real values,
    and returns the mean squared error between them.

    Args:
        data (tuple[Prediction, Real]): A tuple containing predictions and real values.

    Returns:
        float: The mean squared error.
    """
    preds, real = data

    return mean_squared_error(real, preds)


class MeanSquareErrorPerPatient(MetricBaseModel[DataType], Generic[DataType]):
    """
    A metric class that calculates the Root Mean Square Error (RMSE) per patient.

    This class provides a transformer that takes a tuple of two lists as input,
    where the first list contains the predicted values and the second list contains
    the actual values. The transformer calculates the mean square error between
    the two lists and returns a score report.

    Attributes
    ----------
    None

    Methods
    -------
    get_metric_name(): str
        Returns the name of the metric, which is 'RMSE'.
    get_metric(): Transformer[DataType, MetricScore]
        Returns a transformer that calculates the mean square error and builds a score report.
    """

    def get_metric_name(self) -> str:
        return "RMSE"

    def get_metric(
        self,
    ) -> Transformer[DataType, MetricScore]:
        """
        This function returns a transformer that will be used to compute the mean square error of the predictions
        for the given data. It takes a tuple of two lists, the first one being the predictions and the second one
        being the real values. It first flattens the two lists into a single tuple, then computes the mean square
        error between the two lists and finally builds a score report from the result.

        Returns
        -------
        Transformer[DataType, MetricScore]
            A transformer that takes a tuple of two lists as input and returns a score report.
        """
        return (
            flat_signals
            >> mean_squared_error_transformer
            >> build_score_report(self.get_metric_name())
        )
