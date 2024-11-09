import numpy as np
from pydantic import ConfigDict
from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)


class KerasDataModel(MachineLearningDataModel):
    """
    A data model class specifically designed for use with Keras machine learning models.

    This class inherits from MachineLearningDataModel and provides additional properties to convert the input and output series into numpy arrays suitable for use with Keras models.

    Attributes:
        model_config (ConfigDict): A configuration dictionary with arbitrary types allowed.

    Properties:
        input_series_for_keras (np.ndarray): A 3D numpy array with shape (batch_size, sequence_length, 1) representing the input series.
        output_series_for_keras (np.ndarray): A 2D numpy array where each row corresponds to a sequence in the output series.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def input_series_for_keras(self) -> np.ndarray:
        """
        Converts the input series of the data model into a 3D numpy array
        suitable for use with Keras models.

        Returns:
            np.ndarray: A 3D numpy array with shape (batch_size, sequence_length, 1),
            where 'batch_size' is the number of input series and 'sequence_length'
            is the length of each series.
        """
        x = np.array(self.input_series)
        return x.reshape([x.shape[0], x.shape[1], 1])

    @property
    def output_series_for_keras(self) -> np.ndarray:
        """
        Converts the output series of the data model into a numpy array
        suitable for use with Keras models.

        Returns:
            np.ndarray: A 2D numpy array where each row corresponds to a sequence
            in the output series.
        """
        x = np.array(self.output_series)
        return x
