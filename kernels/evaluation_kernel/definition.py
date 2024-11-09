import pandas as pd
from gloe import Transformer
from gloe.collection import Map
from gloe.utils import forward, attach
from pydantic import ConfigDict
from research_flow.kernels.base_kernel import BaseKernel
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
from research_flow.types.metrics.metric_base_model import MetricBaseModel
from research_flow.types.pydantic_utils.subclass_type_resolution import ModelInstance

from kernels.evaluation_kernel.transformers.model_usage import do_predictions
from kernels.evaluation_kernel.transformers.utils import (
    pick_data,
    pick_test,
    pick_model,
    build_score_dataframe,
    denomalize,
)


class ModelEvaluateKernel(
    BaseKernel[
        list[
            tuple[
                MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
                MachineLearningDataPartsWithScalersModel,
            ]
        ],
        pd.DataFrame,
    ]
):
    """
    A kernel for evaluating machine learning models with given metrics on specific data.

    This class represents a pipeline that takes a list of models and their corresponding data as input,
    and returns a score dataframe with the evaluation results.

    Attributes:
        model_config (ConfigDict): The configuration for the models.
        metrics (list[ModelInstance[MetricBaseModel[DataType]]]): A list of metric transformers.
        data_converter (Transformer[MachineLearningDataModel, DataType]): A transformer for converting data.

    Properties:
        pipeline_graph: A pipeline graph that evaluates the given models with the given metrics on the given data.

    Returns:
        pd.DataFrame: A score dataframe with the evaluation results.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: list[ModelInstance[MetricBaseModel[DataType]]]

    data_converter: Transformer[MachineLearningDataModel, DataType]

    @property
    def pipeline_graph(
        self,
    ) -> Transformer[
        list[
            tuple[
                MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
                MachineLearningDataPartsWithScalersModel,
            ]
        ],
        pd.DataFrame,
    ]:
        """
        A pipeline graph that evaluates the given models with the given metrics on the given data.

        The pipeline will first convert the data to the correct format, then it will make predictions
        with the given model on the test data. After that, it will calculate the metrics with the given
        metric transformers and then it will build a score dataframe with the results.

        Args:
            models: A list of tuples containing a model and its corresponding data.

        Returns:
            A score dataframe with the results of the evaluation.
        """
        evaluate_calculations = tuple(
            metric_transformer.get_metric() for metric_transformer in self.metrics
        )

        pipeline = (
            Map(
                attach(
                    forward[
                        tuple[
                            MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
                            MachineLearningDataPartsWithScalersModel,
                        ]
                    ]()
                    >> (pick_data >> pick_test >> self.data_converter, pick_model)
                    >> do_predictions
                )
                >> denomalize
                >> evaluate_calculations
            )
            >> build_score_dataframe
        )

        return pipeline
