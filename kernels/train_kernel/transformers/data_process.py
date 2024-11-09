from gloe import transformer
from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)

from kernels.train_kernel.machine_learning.keras.data_type import KerasDataModel


@transformer
def create_keras_data_model(data: MachineLearningDataModel) -> KerasDataModel:
    """
    Converts a MachineLearningDataModel into a KerasDataModel.

    Args:
    data (MachineLearningDataModel): The MachineLearningDataModel to be converted.

    Returns:
    KerasDataModel: The converted KerasDataModel.
    """
    return KerasDataModel(**data.dict())
