from gloe import Transformer
from gloe.collection import Map
from gloe.utils import attach
from pydantic import ConfigDict
from research_flow.kernels.base_kernel import BaseKernel
from research_flow.types.machine_learning.machine_learning_data_parts_model import (
    MachineLearningDataPartsModel,
)
from research_flow.types.machine_learning.machine_learning_data_parts_with_scalers_model import (
    MachineLearningDataPartsWithScalersModel,
)
from research_flow.types.scalars.min_max_scaler import MinMaxScalerModel
from research_flow.types.series.multiple_series_to_series_signal_model import (
    MultipleSeriesToSeriesSignalModel,
)
from research_flow.types.series.single_series_to_series_signal_model import (
    SingleSeriesToSeriesSignalModel,
)

from kernels.data_kernel.transformers.data_processing import (
    align_signals,
    clean_ecg_ppg_signals,
    get_r_and_s_signals_peaks,
    normalize_machine_learning_data_parts,
    segment_signals,
    split_signal_data_by_time,
    union_signals,
)
from kernels.data_kernel.transformers.utils import (
    assemble_machine_learning_data_parts,
    format_to_experiments,
    pick_test,
    pick_train,
    pick_validation,
)


class PatientSpecificDataKernel(
    BaseKernel[str, list[MachineLearningDataPartsWithScalersModel]]
):
    """
    A class representing a data processing pipeline for patient-specific data.

    The `PatientSpecificDataKernel` class defines a pipeline that reads data from a file,
    applies various transformations to clean and process the data, segments the data into
    overlapping segments, splits the data into training, validation, and test sets, and
    normalizes the data using scalers. The resulting output is a list of
    `MachineLearningDataPartsWithScalersModel` objects, which contain the processed data
    and scalers.

    The pipeline is defined as a composition of several transformers, which are applied to
    each patient's data. The `pipeline_graph` property returns a `Transformer` object that
    represents the entire pipeline, which can be used to convert a path to a patient's data
    file into a list of processed data objects.

    Attributes:
        model_config (ConfigDict): A configuration dictionary for the model.
        read_data_transformer (Transformer[str, list[SingleSeriesToSeriesSignalModel]]): A transformer that reads data from a file.
        data_processing_transformer (Transformer[SingleSeriesToSeriesSignalModel, SingleSeriesToSeriesSignalModel]): A transformer that cleans and processes the data.
        data_segmented_transformer (Transformer[SingleSeriesToSeriesSignalModel, list[SingleSeriesToSeriesSignalModel]]): A transformer that segments the data into overlapping segments.
        data_split_data_into_train_and_test (Transformer[MultipleSeriesToSeriesSignalModel, tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]]): A transformer that splits the data into a training and test set.
        data_split_train_into_validation_and_train (Transformer[MultipleSeriesToSeriesSignalModel, tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]]): A transformer that splits the training set into a validation and training set.
        normalize_data_transformer (Transformer[MachineLearningDataPartsModel, MachineLearningDataPartsWithScalersModel]): A transformer that normalizes the data using scalers.
        clean_data_transformer (Transformer[SingleSeriesToSeriesSignalModel, SingleSeriesToSeriesSignalModel]): A transformer that cleans the signal and detects R- and S-peaks.

    Properties:
        pipeline_graph (Transformer[str, list[MachineLearningDataPartsWithScalersModel]]): A property that returns a `Transformer` object representing the entire pipeline.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    read_data_transformer: Transformer[str, list[SingleSeriesToSeriesSignalModel]]

    data_processing_transformer: Transformer[
        SingleSeriesToSeriesSignalModel, SingleSeriesToSeriesSignalModel
    ] = (clean_ecg_ppg_signals >> attach(get_r_and_s_signals_peaks) >> align_signals)

    data_segmented_transformer: Transformer[
        SingleSeriesToSeriesSignalModel, list[SingleSeriesToSeriesSignalModel]
    ] = segment_signals(segment_time=1, overlap=0)

    data_split_data_into_train_and_test: Transformer[
        MultipleSeriesToSeriesSignalModel,
        tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel],
    ] = split_signal_data_by_time(split_percentage=0.8)

    data_split_train_into_validation_and_train: Transformer[
        MultipleSeriesToSeriesSignalModel,
        tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel],
    ] = split_signal_data_by_time(split_percentage=0.9)

    normalize_data_transformer: Transformer[
        MachineLearningDataPartsModel, MachineLearningDataPartsWithScalersModel
    ] = normalize_machine_learning_data_parts(
        train_scaler_input_series=MinMaxScalerModel(min_feature=-1, max_feature=1),
        train_scaler_output_series=MinMaxScalerModel(min_feature=-1, max_feature=1),
        test_scaler_input_series=MinMaxScalerModel(min_feature=-1, max_feature=1),
        test_scaler_output_series=MinMaxScalerModel(min_feature=-1, max_feature=1),
    )

    clean_data_transformer: Transformer[
        SingleSeriesToSeriesSignalModel, SingleSeriesToSeriesSignalModel
    ] = (clean_ecg_ppg_signals >> attach(get_r_and_s_signals_peaks) >> align_signals)

    @property
    def pipeline_graph(
        self,
    ) -> Transformer[str, list[MachineLearningDataPartsWithScalersModel]]:
        """
        The `pipeline_graph` property returns a Transformer object that can be used to convert a path to a patient's data file into a list of MachineLearningDataPartsWithScalersModel objects.

        The pipeline first reads the data from the specified path, then applies the following steps to each signal:
        - applies the `data_processing_transformer` to clean the signal and detect R- and S-peaks
        - segments the signal into overlapping segments
        - unions the segments
        - splits the data into a training and test set
        - splits the training set into a validation and training set
        - formats each of the three sets into a MachineLearningDataModel object
        - assembles the three sets into a single MachineLearningDataPartsWithScalersModel object
        - normalizes the data using the given scalers

        The pipeline then applies the same steps to each patient's data and returns the resulting list of MachineLearningDataPartsWithScalersModel objects.

        :return: A Transformer object that can be used to convert a path to a patient's data file into a list of MachineLearningDataPartsWithScalersModel objects
        """
        per_patient_processing = (
            self.data_processing_transformer
            >> self.data_segmented_transformer
            >> union_signals
            >> self.data_split_data_into_train_and_test
            >> (
                pick_train
                >> self.data_split_train_into_validation_and_train
                >> (
                    pick_train >> format_to_experiments,
                    pick_validation >> format_to_experiments,
                ),
                pick_test >> format_to_experiments,
            )
            >> assemble_machine_learning_data_parts
            >> self.normalize_data_transformer
        )

        pipeline = self.read_data_transformer >> Map(per_patient_processing)

        return pipeline
