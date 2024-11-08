from gloe import transformer
from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)

from kernels.train_kernel.machine_learning.keras.data_type import KerasDataModel


@transformer
def create_keras_data_model(data: MachineLearningDataModel) -> KerasDataModel:
    return KerasDataModel(**data.dict())
