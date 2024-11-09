import os
import uuid
from datetime import datetime
from typing import Generic, Optional

from keras import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.saving.save import load_model
from research_flow.machine_learning.base_machine_learning_algorithm import (
    MachineLearningAlgorithm,
)
from research_flow.types.comon_types import ModelConfig

from kernels.train_kernel.machine_learning.keras.configs import (
    KerasTrainConfigs,
    KerasHPOConfig,
)
from kernels.train_kernel.machine_learning.keras.data_type import KerasDataModel


class KerasMachineLearningAlgorithm(
    MachineLearningAlgorithm[Model, ModelConfig, KerasDataModel],
    Generic[ModelConfig],
):
    def save_alg(self, path: str):
        """
        Saves the trained Keras model to the specified file path.

        Parameters:
        path (str): The file path where the trained model will be saved.
        """
        self.alg_trained.save(path)
        self.alg_trained.save(path)

    def load_alg(self, path: str):
        """
        Loads a saved Keras model from the specified file path.

        Parameters:
        path (str): The file path from which the trained model will be loaded.
        """
        self.alg_trained = load_model(path)

    def fit(
        self,
        data: KerasDataModel,
        train_config: KerasTrainConfigs,
        model_config: ModelConfig,
        val_data: Optional[KerasDataModel] = None,
    ) -> "KerasMachineLearningAlgorithm":
        """
        Trains a Keras model using the given data and configuration.

        Parameters:
        data (KerasDataModel): The data to be used for training.
        train_config (KerasTrainConfigs): The configuration for the training process.
        model_config (ModelConfig): The configuration for the model.
        val_data (Optional[KerasDataModel]): The data to be used for validation. Defaults to None.

        Returns:
        KerasMachineLearningAlgorithm: The instance of the class with the trained model.
        """
        x = data.input_series_for_keras
        y = data.output_series_for_keras

        val = None
        if val_data is not None:
            val_x = val_data.input_series_for_keras
            val_y = val_data.output_series_for_keras
            val = (val_x, val_y)

        file_name = datetime.now().strftime(f"%d_%m_%Y__%H_%M_%S_{uuid.uuid4()}")
        file_path = os.path.join(train_config.log_dir, file_name)

        tensorboard_callback = TensorBoard(
            log_dir=file_path,
            histogram_freq=1,
        )
        callbacks = [tensorboard_callback]
        if train_config.with_early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=train_config.patience,
                mode="min",
                verbose=train_config.verbose,
                restore_best_weights=True,
            )
            callbacks.append(early_stopping)

        model = self.alg_factory(model_config)

        model.fit(
            x=x,
            y=y,
            validation_data=val,
            epochs=train_config.epochs,
            batch_size=train_config.batch_size,
            callbacks=callbacks,
            workers=train_config.num_of_workers,
            use_multiprocessing=True,
            verbose=train_config.verbose,
        )

        result = self.copy()

        result.alg_trained = model

        return result

    def predict(self, data: KerasDataModel) -> KerasDataModel:
        """
        This function takes a KerasDataModel object and makes a prediction on its input series.
        If the model has not been trained yet, the function returns the input data as is.
        Otherwise, it makes a prediction on the input series and returns a new KerasDataModel object
        with the prediction as a separate field.

        :param data: KerasDataModel: The data to make a prediction on.
        :return: KerasDataModel: The input data with a prediction if the model has been trained.
        """
        if self.alg_trained is None:
            return data

        x = data.input_series_for_keras

        y_hat = self.alg_trained.predict(x).reshape(x.shape[0], -1).tolist()

        return KerasDataModel(
            input_series=data.input_series,
            output_series=data.output_series,
            prediction=y_hat,
        )

    def tune_model_parameters(
        self,
        data: KerasDataModel,
        val_data: KerasDataModel,
        hpo_config: KerasHPOConfig[ModelConfig],
    ) -> ModelConfig:
        pass
