from gloe import Transformer
from gloe.collection import Map
from gloe.utils import forward, attach
from pydantic import ConfigDict
from research_flow.kernels.base_kernel import BaseKernel
from research_flow.machine_learning.base_machine_learning_algorithm import (
    DataType,
    MachineLearningAlgorithm,
)
from research_flow.types.comon_types import ModelConfig, ModelType
from research_flow.types.configs.train_config_base_model import TrainConfigBaseModel
from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)
from research_flow.types.machine_learning.machine_learning_data_parts_with_scalers_model import (
    MachineLearningDataPartsWithScalersModel,
)
from research_flow.types.pydantic_utils.subclass_type_resolution import ModelInstance

from kernels.train_kernel.transformers.utils import (
    parse_information_to_train,
    pick_train,
    pick_validation,
    train_ml_alg,
)


class PatientSpecificTrainKernel(
    BaseKernel[
        list[MachineLearningDataPartsWithScalersModel],
        list[
            tuple[
                MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
                MachineLearningDataPartsWithScalersModel,
            ]
        ],
    ],
):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ml_alg: ModelInstance[MachineLearningAlgorithm[ModelType, ModelConfig, DataType]]

    train_config: ModelInstance[TrainConfigBaseModel]

    ml_config: ModelConfig

    data_converter: Transformer[MachineLearningDataModel, DataType]

    @property
    def pipeline_graph(
        self,
    ) -> Transformer[
        list[MachineLearningDataPartsWithScalersModel],
        list[
            tuple[
                MachineLearningAlgorithm[ModelType, ModelConfig, DataType],
                MachineLearningDataPartsWithScalersModel,
            ]
        ],
    ]:
        train_pipeline = (
            forward[MachineLearningDataPartsWithScalersModel]()
            >> (
                pick_train >> self.data_converter,
                pick_validation >> self.data_converter,
            )
            >> parse_information_to_train(self.ml_config)
            >> train_ml_alg(self.ml_alg, self.train_config)
        )

        pipeline = forward[list[MachineLearningDataPartsWithScalersModel]]() >> Map(
            attach(train_pipeline)
        )

        return pipeline
