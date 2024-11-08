from gloe import transformer
from research_flow.types.machine_learning.machine_learning_data_model import MachineLearningDataModel
from research_flow.types.machine_learning.machine_learning_data_parts_model import MachineLearningDataPartsModel
from research_flow.types.series.multiple_series_to_series_signal_model import MultipleSeriesToSeriesSignalModel


@transformer
def pick_train(
    data: tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]
) -> MultipleSeriesToSeriesSignalModel:
    return data[0]

@transformer
def pick_validation(
    data: tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]
) -> MultipleSeriesToSeriesSignalModel:
    return data[1]

@transformer
def format_to_experiments(
    signals: MultipleSeriesToSeriesSignalModel,
) -> MachineLearningDataModel:
    """
    The format_to_experiments function takes a MultipleSeriesToSeriesSignalModel and returns a MachineLearningDataModel.

    :param signals: MultipleSeriesToSeriesSignalModel: Pass the input and output series to the function
    :param : Pass in the signals that we want to use for our experiments
    :return: A machinelearningdatamodel object
    :doc-author: Trelent
    """
    return MachineLearningDataModel(
        input_series=signals.input_series, output_series=signals.output_series
    )

@transformer
def pick_test(
    data: tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]
) -> MultipleSeriesToSeriesSignalModel:
    return data[1]

@transformer
def assemble_machine_learning_data_parts(
    data: tuple[
        tuple[MachineLearningDataModel, MachineLearningDataModel],
        MachineLearningDataModel,
    ]
) -> MachineLearningDataPartsModel:
    (train, validation), test = data

    return MachineLearningDataPartsModel(train=train, validation=validation, test=test)
