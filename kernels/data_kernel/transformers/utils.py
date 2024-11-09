from gloe import transformer
from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)
from research_flow.types.machine_learning.machine_learning_data_parts_model import (
    MachineLearningDataPartsModel,
)
from research_flow.types.series.multiple_series_to_series_signal_model import (
    MultipleSeriesToSeriesSignalModel,
)


@transformer
def pick_train(
    data: tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]
) -> MultipleSeriesToSeriesSignalModel:
    """
    The pick_train function takes a tuple of two MultipleSeriesToSeriesSignalModel objects
    and returns the first one.

    :param data: tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]: A tuple of two
        MultipleSeriesToSeriesSignalModel objects
    :return: MultipleSeriesToSeriesSignalModel: The first element of the tuple
    """
    return data[0]


@transformer
def pick_validation(
    data: tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]
) -> MultipleSeriesToSeriesSignalModel:
    """
    The pick_validation function takes a tuple of two MultipleSeriesToSeriesSignalModel objects
    and returns the second one.

    :param data: tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]: A tuple of two
        MultipleSeriesToSeriesSignalModel objects
    :return: MultipleSeriesToSeriesSignalModel: The second element of the tuple
    """
    return data[1]


@transformer
def format_to_experiments(
    signals: MultipleSeriesToSeriesSignalModel,
) -> MachineLearningDataModel:
    """
    The format_to_experiments function takes a MultipleSeriesToSeriesSignalModel and returns a MachineLearningDataModel.

    :param signals: MultipleSeriesToSeriesSignalModel: Pass the input and output series to the function
    :return: A MachineLearningDataModel object
    """
    return MachineLearningDataModel(
        input_series=signals.input_series, output_series=signals.output_series
    )


@transformer
def pick_test(
    data: tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]
) -> MultipleSeriesToSeriesSignalModel:
    """
    The pick_test function takes a tuple of two MultipleSeriesToSeriesSignalModel objects
    and returns the second one, which is considered as the test set.

    :param data: tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]: A tuple of two
        MultipleSeriesToSeriesSignalModel objects
    :return: MultipleSeriesToSeriesSignalModel: The second element of the tuple, representing the test data
    """
    return data[1]


@transformer
def assemble_machine_learning_data_parts(
    data: tuple[
        tuple[MachineLearningDataModel, MachineLearningDataModel],
        MachineLearningDataModel,
    ]
) -> MachineLearningDataPartsModel:
    """
    The assemble_machine_learning_data_parts function takes a tuple of two elements, where the first element is a tuple of two MachineLearningDataModel objects, and the second element is a MachineLearningDataModel object.

    The function returns a MachineLearningDataPartsModel where the train attribute is the first element of the tuple, the validation attribute is the second element of the tuple, and the test attribute is the second element of the outer tuple.

    :param data: tuple[tuple[MachineLearningDataModel, MachineLearningDataModel], MachineLearningDataModel]: Pass the train, validation, and test data to the function
    :return: A MachineLearningDataPartsModel object
    """
    (train, validation), test = data

    return MachineLearningDataPartsModel(train=train, validation=validation, test=test)
