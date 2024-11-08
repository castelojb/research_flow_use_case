import asyncio

from gloe import transformer
from keras import Model
from optuna import Study, Trial

from src.definitions.data_model.train.keras_machine_learning_algorithm import (
    KerasDataModel,
    KerasMachineLearningAlgorithm,
    create_keras_data_model,
)
from src.definitions.data_model.train.machine_learning_algorithm import (
    MachineLearningAlgorithmHPOConfigs,
    MachineLearningAlgorithmTrainConfigs,
)
from src.recipes.signal_reconstruction.patient_specific import (
    PatientSpecific,
    PatientSpecificDataKernel,
    PatientSpecificModelTrainerKernel,
)
from src.toolkit.biosignals.data_read import parse_from_bidmc_directory
from src.toolkit.biosignals.ml_algs.bilstm_ml_alg import BiLSTMConfig, bilstm
from src.toolkit.biosignals.processing import (
    align_signals,
    clean_ecg_ppg_signals,
    get_r_and_s_signals_peaks,
)
from src.toolkit.utils import async_forward_incoming

default_config_model = dict(
    n_features=125,
    l1=0.0001,
    l2=0.0001,
    learning_rate=0.001,
    loss_func_name="mean_squared_error",
)

clean_data_pipeline = (
    clean_ecg_ppg_signals()
    >> async_forward_incoming(get_r_and_s_signals_peaks())
    >> align_signals
)


@transformer
def search_space(trial: Trial) -> BiLSTMConfig:
    space = dict(
        n_neurons=trial.suggest_int("n_neurons", 10, 100, 2),
        dropout=trial.suggest_uniform("dropout", 0.0, 0.9),
    )

    args = {**default_config_model, **space}

    config = BiLSTMConfig(**args)

    return config


@transformer
def get_best_config(study: Study) -> BiLSTMConfig:
    args = {**default_config_model, **study.best_params}

    config = BiLSTMConfig(**args)

    return config


async def main():
    data_kernel = PatientSpecificDataKernel(
        read_data_transformer=parse_from_bidmc_directory,
        data_processing_transformer=clean_data_pipeline,
    )

    hpo_config = MachineLearningAlgorithmHPOConfigs[BiLSTMConfig](
        epochs_per_trial=1,
        n_trials=1,
        search_space_setup=search_space,
        get_best_config=get_best_config,
    )

    train_config = MachineLearningAlgorithmTrainConfigs(epochs=1)

    model_trainer_kernel = PatientSpecificModelTrainerKernel[
        Model, BiLSTMConfig, KerasDataModel
    ](
        ml_alg=KerasMachineLearningAlgorithm[BiLSTMConfig](alg_factory=bilstm),
        hpo_config=hpo_config,
        train_config=train_config,
        data_converter=create_keras_data_model,
    )

    pck = PatientSpecific[Model, BiLSTMConfig, KerasDataModel](
        data_kernel=data_kernel,
        ml_trainer_kernel=model_trainer_kernel,
        base_report_dir="./preds",
    )

    await pck.graph("../../data/bidmc")


if __name__ == "__main__":
    print(asyncio.run(main()))
