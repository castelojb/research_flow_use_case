import pandas as pd
from gloe import Transformer
from gloe.collection import Map
from gloe.utils import forward
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
        evaluate_calculations = tuple(
            metric_transformer.get_metric() for metric_transformer in self.metrics
        )

        pipeline = (
            Map(
                forward[
                    tuple[
                        MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
                        MachineLearningDataPartsWithScalersModel,
                    ]
                ]()
                >> (pick_data >> pick_test >> self.data_converter, pick_model)
                >> do_predictions
                >> evaluate_calculations
            )
            >> build_score_dataframe
        )

        return pipeline
