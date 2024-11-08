from gloe import Transformer
from gloe.collection import Map
from gloe.utils import forward_incoming
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

from kernels.data_kernel.transformers.data_processing import segment_signals, split_signal_data_by_time, \
    normalize_machine_learning_data_parts, clean_ecg_ppg_signals, get_r_and_s_signals_peaks, align_signals, \
    union_signals
from kernels.data_kernel.transformers.utils import pick_train, format_to_experiments, pick_validation, pick_test, \
    assemble_machine_learning_data_parts


class PatientSpecificDataKernel(
    BaseKernel[str, list[MachineLearningDataPartsWithScalersModel]]
):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    read_data_transformer: Transformer[str, list[SingleSeriesToSeriesSignalModel]]

    data_processing_transformer: Transformer[
        SingleSeriesToSeriesSignalModel, SingleSeriesToSeriesSignalModel
    ]

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
    ] = (
        clean_ecg_ppg_signals
        >> forward_incoming(get_r_and_s_signals_peaks)
        >> align_signals
    )

    @property
    def pipeline_graph(
        self,
    ) -> Transformer[str, list[MachineLearningDataPartsWithScalersModel]]:
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
