import numpy as np
from pydantic import ConfigDict
from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)


class KerasDataModel(MachineLearningDataModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def input_series_for_keras(self) -> np.ndarray:
        x = np.array(self.input_series)
        return x.reshape([x.shape[0], x.shape[1], 1])

    @property
    def output_series_for_keras(self) -> np.ndarray:
        x = np.array(self.output_series)
        return x
