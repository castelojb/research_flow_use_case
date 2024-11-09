from kernels.data_kernel.definition import PatientSpecificDataKernel
from kernels.data_kernel.transformers.read_data import parse_from_mimic_mat_file
from kernels.evaluation_kernel.definition import ModelEvaluateKernel
from kernels.evaluation_kernel.metrics.error_metrics import MeanSquareErrorPerPatient
from kernels.evaluation_kernel.metrics.similarity_metrics import (
    PersonPerPatient,
    DTWPerPatient,
)
from kernels.train_kernel.definition import PatientSpecificTrainKernel
from kernels.train_kernel.machine_learning.keras.configs import KerasTrainConfigs
from kernels.train_kernel.machine_learning.keras.data_type import KerasDataModel
from kernels.train_kernel.machine_learning.keras.keras_machine_learning_algorithm import (
    KerasMachineLearningAlgorithm,
)
from kernels.train_kernel.machine_learning.keras.models.bilstm import (
    KerasBiLSTMConfig,
    keras_bilstm,
)
from kernels.train_kernel.transformers.data_process import create_keras_data_model

if __name__ == "__main__":
    data_kernel = PatientSpecificDataKernel(
        read_data_transformer=parse_from_mimic_mat_file
    )

    train_config = KerasTrainConfigs(epochs=1)
    ml_config = KerasBiLSTMConfig(n_neurons=10, n_features=125)
    model_trainer_kernel = PatientSpecificTrainKernel(
        ml_alg=KerasMachineLearningAlgorithm[KerasBiLSTMConfig](
            alg_factory=keras_bilstm
        ),
        train_config=train_config,
        data_converter=create_keras_data_model,
        ml_config=ml_config,
    )

    evaluate_kernel = ModelEvaluateKernel(
        metrics=[
            MeanSquareErrorPerPatient[KerasDataModel](),
            PersonPerPatient[KerasDataModel](),
            DTWPerPatient[KerasDataModel](),
        ],
        data_converter=create_keras_data_model,
    )

    experiment = data_kernel >> model_trainer_kernel >> evaluate_kernel

    results = experiment("data/Records.mat")

    print(results)
