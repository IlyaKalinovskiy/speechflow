import numpy as np
import pytest

from tqdm import tqdm

from speechflow.data_pipeline.core import DataPipeline, DataSample
from speechflow.data_pipeline.core.dataset import Dataset
from speechflow.data_server.helpers import LoaderParams, init_data_loader
from speechflow.io import Config
from speechflow.logging.server import LoggingServer


@pytest.mark.parametrize("drop_non_full", [True, False])
@pytest.mark.parametrize("non_stop", [True, False])
def test_server(
    drop_non_full: bool,
    non_stop: bool,
    dataset_size: int = 100,
    batch_size: int = 32,
    num_epoch: int = 2,
    n_processes: int = 2,
):
    cfg = Config({"dataset": {"subsets": ["test"]}})
    data_pipeline = DataPipeline(cfg)
    data_pipeline.init_components()

    data = [DataSample(label=str(i)) for i in range(dataset_size)]
    data_pipeline["test"].set_dataset(Dataset(data))

    with LoggingServer.ctx():
        with init_data_loader(
            loader_params=LoaderParams(
                batch_size=batch_size, drop_non_full=drop_non_full, non_stop=non_stop
            ),
            data_pipeline=data_pipeline,
            n_processes=n_processes,
        ) as loaders:
            loader = list(loaders.values())[0]
            label_counter: dict = {}
            for i in tqdm(range(num_epoch * len(loader))):
                batch = next(loader)
                print(i)
                for sample in batch.data_samples:
                    label = f"L_{sample.label}"
                    label_counter.setdefault(label, 0)
                    label_counter[label] += 1

            val = np.asarray([value for value in label_counter.values()])
            assert dataset_size - len(label_counter) < batch_size
            assert abs(val.sum() - num_epoch * dataset_size) < num_epoch * batch_size
            assert round(val.mean()) == num_epoch


if __name__ == "__main__":
    test_server(True, True)
