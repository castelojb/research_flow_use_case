import numpy as np
from gloe import transformer
from research_flow.types.series.single_series_to_series_signal_model import SingleSeriesToSeriesSignalModel
from scipy.io import loadmat


@transformer
def parse_from_mimic_mat_file(
    file_path: str,
) -> list[SingleSeriesToSeriesSignalModel]:
    data = loadmat(file_path)

    records = data["records"]

    arr = []

    for idx in range(records.size):
        ecg = records[idx, 0]["ecg_II"][:, 0].astype(np.float32).reshape(-1).tolist()
        ppg = records[idx, 0]["ppg"][:, 0].astype(np.float32).reshape(-1).tolist()

        arr.append(
            SingleSeriesToSeriesSignalModel(
                input_series=ppg,
                output_series=ecg,
                input_series_frequency=125,
                output_series_frequency=125,
            )
        )

    return arr