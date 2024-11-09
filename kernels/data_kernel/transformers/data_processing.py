import numpy as np
from gloe import partial_transformer, transformer
from neurokit2 import ecg_clean, ppg_clean, ecg_findpeaks, ppg_findpeaks
from research_flow.types.machine_learning.machine_learning_data_model import (
    MachineLearningDataModel,
)
from research_flow.types.machine_learning.machine_learning_data_parts_model import (
    MachineLearningDataPartsModel,
)
from research_flow.types.machine_learning.machine_learning_data_parts_with_scalers_model import (
    MachineLearningDataPartsWithScalersModel,
)
from research_flow.types.scalars.base_class import Scaler
from research_flow.types.series.multiple_series_to_series_signal_model import (
    MultipleSeriesToSeriesSignalModel,
)
from research_flow.types.series.single_series_peaks_model import SingleSeriesPeaksModel
from research_flow.types.series.single_series_to_series_signal_model import (
    SingleSeriesToSeriesSignalModel,
)


def __make_segmentation(
    signal: list[float],
    segment_time: int,
    sampling_rate: int,
    overlap: float,
) -> list[list[float]]:
    """
    The __make_segmentation function takes a signal and splits it into segments of length segment_length.
    The function returns a list of lists, where each sublist is one segment.


    :param signal: list[float]: Pass the signal to be segmented
    :param segment_time: int: Determine the length of each segment in seconds
    :param sampling_rate: int: Determine the length of a segment in seconds
    :param overlap: float: Determine the amount of overlap between segments
    :param segment_length: Optional[int]: Specify the length of a segment
    :return: A list of lists
    """
    signal_lenght = len(signal)

    segment_length = segment_time * sampling_rate

    data_per_second = int(segment_length / (1 - overlap))

    shape_of_result = (signal_lenght // data_per_second, data_per_second)

    start = 0
    segments = []

    for _ in range(shape_of_result[0]):
        end = start + segment_length
        segments.append(signal[start:end])
        start = end

    return segments


@partial_transformer
def segment_signals(
    signals: SingleSeriesToSeriesSignalModel,
    segment_time: int,
    overlap: float,
) -> list[SingleSeriesToSeriesSignalModel]:
    """
    The segment_signals function takes a SingleSeriesToSeriesSignalModel object and returns a list of
    SingleSeriesToSeriesSignalModel objects. Each element in the returned list is an individual segment of the original
    signals, with each segment having its own input_series and output_series attributes. The length of each segment is
    determined by the overlap parameter (which determines how much overlap there should be between segments) and either:

    :param signals: SingleSeriesToSeriesSignalModel: Pass the data to be segmented
    :param segment_time: int: Define the time in seconds that each segment will have
    :param overlap: float: Determine how much overlap there should be between segments
    :param : Define the time in seconds that each segment will have
    :return: A list of singleseriestoseriessignalmodel objects
    """
    input_segments = __make_segmentation(
        signals.input_series,
        segment_time,
        signals.input_series_frequency,
        overlap,
    )

    output_segments = __make_segmentation(
        signals.output_series,
        segment_time,
        signals.output_series_frequency,
        overlap,
    )

    arr = [
        SingleSeriesToSeriesSignalModel(
            input_series=input_series,
            output_series=output_series,
            input_series_frequency=signals.input_series_frequency,
            output_series_frequency=signals.output_series_frequency,
        )
        for input_series, output_series in zip(input_segments, output_segments)
    ]

    return arr


def __apply_and_fit_custom_scalar(
    signals: MachineLearningDataModel,
    scaler_input_series: Scaler,
    scaler_output_series: Scaler,
    fit_transformer: bool = True,
) -> MachineLearningDataModel:
    """
    Normalizes the input and output series of a machine learning data model using custom scalers.

    Args:
        signals (MachineLearningDataModel): Machine learning data model containing the input and output series.
        scaler_input_series (Scaler): Scaler to normalize the input series.
        scaler_output_series (Scaler): Scaler to normalize the output series.
        fit_transformer (bool, optional): If True, fits the scalers to the data before normalizing. Defaults to True.

    Returns:
        MachineLearningDataModel: Machine learning data model with the normalized input and output series.
    """

    if fit_transformer:
        scaler_input_series.fit(signals.input_series)
        scaler_output_series.fit(signals.output_series)

    input_normalized = scaler_input_series.transform(signals.input_series)
    output_normalized = scaler_output_series.transform(signals.output_series)

    normalized = MachineLearningDataModel(
        input_series=input_normalized, output_series=output_normalized  # type: ignore
    )

    return normalized


@partial_transformer
def split_signal_data_by_time(
    signal: MultipleSeriesToSeriesSignalModel, split_percentage: float
) -> tuple[MultipleSeriesToSeriesSignalModel, MultipleSeriesToSeriesSignalModel]:
    """
    The split_signal_data_by_time function splits a signal into two parts.
    The split is done by time, so the first part will contain the first X% of the data and
    the second part will contain all remaining data. The split percentage is given as an argument to this function.


    :param signal: MultipleSeriesToSeriesSignalModel: Pass the signal to be split
    :param split_percentage: float: Determine how much of the data should be used for training and how much for testing
    :return: A tuple of two multipleseriestoseriessignalmodel objects
    """
    number_of_segments = len(signal.input_series)
    split_size = int(number_of_segments * split_percentage)

    first_part_input = signal.input_series[0:split_size]
    first_part_output = signal.output_series[0:split_size]

    second_part_input = signal.input_series[split_size:]
    second_part_output = signal.output_series[split_size:]

    first_part_split = MultipleSeriesToSeriesSignalModel(
        input_series=first_part_input,
        output_series=first_part_output,
        input_series_frequency=signal.input_series_frequency,
        output_series_frequency=signal.output_series_frequency,
    )

    second_part_split = MultipleSeriesToSeriesSignalModel(
        input_series=second_part_input,
        output_series=second_part_output,
        input_series_frequency=signal.input_series_frequency,
        output_series_frequency=signal.output_series_frequency,
    )

    return first_part_split, second_part_split


@partial_transformer
def normalize_machine_learning_data_parts(
    data_parts: MachineLearningDataPartsModel,
    train_scaler_input_series: Scaler,
    train_scaler_output_series: Scaler,
    test_scaler_input_series: Scaler,
    test_scaler_output_series: Scaler,
) -> MachineLearningDataPartsWithScalersModel:
    """
    The normalize_machine_learning_data_parts function takes a MachineLearningDataPartsModel, as well as four scalers, and returns a MachineLearningDataPartsWithScalersModel.

    The function will normalize all the input and output series of the given data parts using the given scalers. The scalers are given as arguments to the function. The function will also return the scalers used for normalization, so that the same scalers can be used to normalize new data.

    :param data_parts: MachineLearningDataPartsModel: Pass the data to be normalized
    :param train_scaler_input_series: Scaler: Scaler to normalize the input series of the training data
    :param train_scaler_output_series: Scaler: Scaler to normalize the output series of the training data
    :param test_scaler_input_series: Scaler: Scaler to normalize the input series of the test data
    :param test_scaler_output_series: Scaler: Scaler to normalize the output series of the test data
    :return: A MachineLearningDataPartsWithScalersModel containing the normalized data and the used scalers
    """
    train_scaler_input_series_scaler = train_scaler_input_series.copy_empty_like()
    train_scaler_output_series_scaler = train_scaler_output_series.copy_empty_like()
    test_scaler_input_series_scaler = test_scaler_input_series.copy_empty_like()
    test_scaler_output_series_scaler = test_scaler_output_series.copy_empty_like()

    normalized_train = __apply_and_fit_custom_scalar(
        data_parts.train,
        scaler_input_series=train_scaler_input_series_scaler,
        scaler_output_series=train_scaler_output_series_scaler,
    )

    normalized_val = __apply_and_fit_custom_scalar(
        data_parts.validation,
        scaler_input_series=train_scaler_input_series_scaler,
        scaler_output_series=train_scaler_output_series_scaler,
        fit_transformer=False,
    )

    normalized_test = __apply_and_fit_custom_scalar(
        data_parts.test,
        scaler_input_series=test_scaler_input_series_scaler,
        scaler_output_series=test_scaler_output_series_scaler,
    )

    return MachineLearningDataPartsWithScalersModel(
        train=normalized_train,
        validation=normalized_val,
        test=normalized_test,
        train_scaler_input_series=train_scaler_input_series_scaler,
        train_scaler_output_series=train_scaler_output_series_scaler,
        test_scaler_input_series=test_scaler_input_series_scaler,
        test_scaler_output_series=test_scaler_output_series_scaler,
    )


@transformer
def clean_ecg_ppg_signals(
    signals: SingleSeriesToSeriesSignalModel,
) -> SingleSeriesToSeriesSignalModel:
    """
    The clean_ecg_ppg_signals function takes in a SingleSeriesToSeriesSignalModel object and returns the same object with cleaned ECG and PPG signals.

    :param signals: SingleSeriesToSeriesSignalModel: Pass the data into the function
    :param input_series_is_ecg: bool: Determine which signal is the ecg and which one is the ppg
    :return: A singleseriestoseriessignalmodel object
    """

    ecg_cleaned = ecg_clean(
        signals.output_series, sampling_rate=signals.output_series_frequency
    )

    ppg_cleaned = ppg_clean(
        signals.input_series, sampling_rate=signals.input_series_frequency
    )

    out = SingleSeriesToSeriesSignalModel(
        output_series=ecg_cleaned,
        output_series_frequency=signals.output_series_frequency,
        input_series=ppg_cleaned,
        input_series_frequency=signals.input_series_frequency,
    )

    return out


def __get_ecg_r_picks(ecg: list[float], ecg_sampling_rate: int) -> list[int]:
    """
    The __get_ecg_r_picks function takes in an ECG signal and the sampling rate of that signal,
    and returns a list of indices corresponding to the R-peaks in that ECG signal.


    :param ecg: list[float]: Pass in the ecg data, and the ecg_sampling_rate: int parameter is used to pass in the sampling rate of that data
    :param ecg_sampling_rate: int: Set the sampling rate of the ecg signal
    :return: A list of integers
    """
    r_peak = ecg_findpeaks(
        np.array(ecg),
        sampling_rate=ecg_sampling_rate,
        # method='pantompkins1985'
    )["ECG_R_Peaks"]

    return list(r_peak)


def __get_ppg_s_peaks(
    ppg: list[float],
    ppg_sampling_rate: int,
) -> list[int]:
    """
    The __get_ppg_s_peaks function takes in a list of PPG values and the sampling rate of those values.
    It then uses the ppg_findpeaks function to find all peaks in that signal, and returns them as a list.

    :param ppg: list[float]: Pass in the ppg signal
    :param ppg_sampling_rate: int: Determine the sampling rate of the ppg data
    :param : Get the peaks of the ppg signal
    :return: A list of the indices of the peaks in
    """

    peaks = ppg_findpeaks(
        np.array(ppg), sampling_rate=ppg_sampling_rate, method="elgendi"
    )["PPG_Peaks"]

    return list(peaks)


@transformer
def get_r_and_s_signals_peaks(
    signals: SingleSeriesToSeriesSignalModel,
) -> SingleSeriesPeaksModel:
    """
    The get_r_and_s_signals_peaks function takes in a SingleSeriesToSeriesSignalModel object and returns a
    SingleSeriesPeaksModel object. The input_series of the SingleSeriesToSeriesSignalModel is assumed to be an ECG signal,
    and the output_series is assumed to be a PPG signal. The function uses the ecg-kit library's get_rpeaks function on
    the input series, and then uses our own __get_ppg_s_peaks function on the output series.

    :param signals: SingleSeriesToSeriesSignalModel: Pass in the input and output series, as well as their frequencies
    :param input_series_is_ecg: bool: Determine whether the input series is an ecg or a ppg
    :return: A singleseriespeaksmodel object, which contains the r_peaks and s_peaks
    """

    r_peaks = __get_ecg_r_picks(signals.output_series, signals.output_series_frequency)
    s_peaks = __get_ppg_s_peaks(signals.input_series, signals.input_series_frequency)

    out = SingleSeriesPeaksModel(
        output_series_peaks=r_peaks, input_series_peaks=s_peaks
    )

    return out


def __align_peaks_and_signal(
    peaks: list[int], signal: list[float], window_size=15
) -> list[int]:
    """
    The __align_peaks_and_signal function takes in a list of peak locations and the signal,
    and returns a new list of peak locations that are aligned with the signal.
    The function does this by taking each peak location and finding the maximum value within
    a window around it. The size of this window is determined by the parameter 'window_size'.
    This function is used to align peaks that were found using different methods.

    :param peaks: list[int]: Store the peak locations
    :param signal: list[float]: Pass in the signal data
    :param window_size: Define the window size around each peak to look for a better peak
    :return: A list of integers
    """
    peak_fixed = []

    for peak in peaks:
        start_loc = max(1, peak - window_size)
        end_loc = min(len(signal), peak + window_size)
        segment = signal[start_loc:end_loc]
        loc = np.argmax(segment)
        peak_fixed.append(loc + start_loc)

    return peak_fixed


def __obtain_previous_peak(
    r_peak: list[int], s_peak: list[int], index: int
) -> list[int]:
    """
    The __obtain_previous_peak function takes in a list of R-peaks and a list of S-peaks
    and an index into the S-peaks list, and returns a list of all R-peaks that are
    before the R-peak at the given index. The function is used to align peaks that were
    found using different methods.

    :param r_peak: list[int]: Store the R-peak locations
    :param s_peak: list[int]: Store the S-peak locations
    :param index: int: Determine which S-peak to look for R-peaks for
    :return: A list of integers
    """
    return list(filter(lambda x: x < s_peak[index], r_peak))


def __align_signals(
    signals: SingleSeriesToSeriesSignalModel, r_peak: list[int], s_peak: list[int]
) -> SingleSeriesToSeriesSignalModel:
    """
    The __align_signals function takes in a SingleSeriesToSeriesSignalModel object and two lists of peaks (R-peaks and S-peaks)
    and returns a new SingleSeriesToSeriesSignalModel object where the input and output series are aligned.
    The function does this by finding the first R-peak after the first S-peak, and then aligning the two series based on that.
    The function is used to align peaks that were found using different methods.

    :param signals: SingleSeriesToSeriesSignalModel: Pass in the input and output series, as well as their frequencies
    :param r_peak: list[int]: Store the R-peak locations
    :param s_peak: list[int]: Store the S-peak locations
    :return: A singleseriestoseriessignalmodel object
    """
    all_previous_peak = [
        (index, __obtain_previous_peak(r_peak, s_peak, index))
        for index in range(2, len(s_peak))
    ]

    for index, previous_peak in all_previous_peak:

        i2_candidates = list(
            filter(
                lambda x: abs(
                    (s_peak[index + 1] - s_peak[index]) - (r_peak[x + 1] - r_peak[x])
                )
                <= 0.05 * signals.input_series_frequency,
                range(len(previous_peak) - 1, 0, -1),
            )
        )

        if len(i2_candidates) > 0:
            i2 = i2_candidates[0]
            break

    shift_point = s_peak[index] - r_peak[i2]
    input_aligned = signals.input_series[shift_point + 1 : len(signals.input_series)]
    output_aligned = signals.output_series[1 : len(signals.output_series) - shift_point]

    return SingleSeriesToSeriesSignalModel(
        input_series=input_aligned,
        output_series=output_aligned,
        input_series_frequency=signals.input_series_frequency,
        output_series_frequency=signals.output_series_frequency,
    )


@transformer
def align_signals(
    data: tuple[SingleSeriesPeaksModel, SingleSeriesToSeriesSignalModel]
) -> SingleSeriesToSeriesSignalModel:
    """
    The align_signals function takes a tuple of a SingleSeriesPeaksModel object and a SingleSeriesToSeriesSignalModel object,
    and returns a new SingleSeriesToSeriesSignalModel object where the input and output series are aligned.
    The function does this by first finding the best R-peak and S-peak values for each beat, and then aligning the two series based on those.
    The function is used to align peaks that were found using different methods.

    :param data: tuple[SingleSeriesPeaksModel, SingleSeriesToSeriesSignalModel]: Pass in the peaks and signals to be aligned
    :return: A SingleSeriesToSeriesSignalModel object
    """
    peaks, signals = data

    if signals.input_series_frequency != signals.output_series_frequency:
        raise Exception("Sampling frequencies must be equal between signals")

    s_peak_fixed = __align_peaks_and_signal(
        peaks.input_series_peaks, signals.input_series
    )
    r_peak_fixed = __align_peaks_and_signal(
        peaks.output_series_peaks, signals.output_series
    )

    return __align_signals(signals, r_peak_fixed, s_peak_fixed)


@transformer
def union_signals(
    signals: list[SingleSeriesToSeriesSignalModel],
) -> MultipleSeriesToSeriesSignalModel:
    """
    The union_signals function takes a list of SingleSeriesToSeriesSignalModel objects and returns a MultipleSeriesToSeriesSignalModel object.

    :param signals: list[SingleSeriesToSeriesSignalModel]: Pass a list of MultipleSeriesToSeriesSignalModel objects
    :return: A multipleseriestoseriessignalmodel
    """
    inputs = [segment.input_series for segment in signals]
    outputs = [segment.output_series for segment in signals]

    input_series_frequencys = {segment.input_series_frequency for segment in signals}
    output_series_frequency = {segment.output_series_frequency for segment in signals}

    return MultipleSeriesToSeriesSignalModel(
        input_series=inputs,
        output_series=outputs,
        input_series_frequency=input_series_frequencys.pop(),
        output_series_frequency=output_series_frequency.pop(),
    )
