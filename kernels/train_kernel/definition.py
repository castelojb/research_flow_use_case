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
    """
    A training kernel for machine learning algorithms that is specific to a patient.

    The `PatientSpecificTrainKernel` class represents a training pipeline for machine learning algorithms. It takes a list of `MachineLearningDataPartsWithScalersModel` objects as input, processes them into training and validation data, trains a machine learning algorithm, and returns a list of tuples containing the trained algorithm and its corresponding data parts.

    Attributes:
        model_config (ConfigDict): A configuration dictionary for the machine learning model.
        ml_alg (ModelInstance[MachineLearningAlgorithm[ModelType, ModelConfig, DataType]]): An instance of the machine learning algorithm.
        train_config (ModelInstance[TrainConfigBaseModel]): An instance of the training configuration.
        ml_config (ModelConfig): The configuration for the machine learning model.
        data_converter (Transformer[MachineLearningDataModel, DataType]): A transformer that converts the data into the desired format.

    Methods:
        pipeline_graph(): A transformer that converts input data into trained machine learning models and their associated data parts.
    """

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
        """
        Constructs and returns a training pipeline graph for machine learning algorithms.

        The pipeline processes a list of `MachineLearningDataPartsWithScalersModel` by separating
        it into training and validation data, converting the data into the desired format,
        and training the machine learning algorithm. The result is a list of tuples, each containing
        a trained `MachineLearningAlgorithm` and the corresponding data parts.

        Returns:
            Transformer: A transformer that converts input data into trained machine learning models
            and their associated data parts.
        """
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
