import os

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from speechflow.data_server.client import DataClient
from speechflow.data_server.loader import DataLoader
from speechflow.data_server.pool import WorkerPool
from speechflow.data_server.server import DataServer
from speechflow.logging import track_process

torch.set_num_threads(1)


@dataclass
class DataConfig:
    cfg_path: str
    batch_size: int
    debug: bool = False


class VocosDataModule(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()

        is_main_proc = int(os.environ.get("LOCAL_RANK", 0)) == 0

        data_config_path = Path(config.cfg_path)
        value_select = ["debug"] if config.debug else None

        if data_config_path and is_main_proc:
            try:
                self.server = DataServer.init_from_config(
                    file_path=data_config_path,
                    value_select=value_select,
                )
                self.server.start()

                self.workers = WorkerPool(
                    server_addr=self.server.address,
                    n_processes=self.server.num_processes,
                )
                self.workers.start()

                os.environ["DATASERVER_ADDR"] = self.server.address  # type: ignore
            except Exception as e:
                print(e)
                raise e

        server_addr = os.environ.get("DATASERVER_ADDR")
        if not server_addr:
            raise ValueError("Address of DataServer is not set!")

        self.data_client = DataClient(server_addr=server_addr)
        subsets = self.data_client.find_info("subsets", [])
        self.hparams["data_config_raw"] = self.data_client.info["data_config_raw"]
        self.hparams["data_config"] = self.data_client.info["data_config"]

        sp_id = self.data_client.find_info("SpeakerIDSetter")
        self.hparams["lang2id"] = sp_id.lang2id
        self.hparams["speaker2id"] = sp_id.speaker2id

        track_process("MAIN", os.getpid())

        try:
            self.data_loaders = []
            for name in subsets:
                loader = DataLoader(
                    server_addr=server_addr,
                    subset_name=name,
                    batch_size=config.batch_size,
                )
                self.data_loaders.append(loader)

            for loader in reversed(self.data_loaders):
                loader.start()

        except Exception as e:
            print(e)
            self.server.finish()
            self.workers.finish()

    def train_dataloader(self) -> DataLoader:
        return self.data_loaders[0]

    def val_dataloader(self) -> DataLoader:
        return self.data_loaders[1]


if __name__ == "__main__":
    cfg = DataConfig(
        cfg_path="../configs/whisp_data_24khz.yml",
        batch_size=4,
        debug=True,
    )
    dataset = VocosDataModule(cfg)
    dl = dataset.train_dataloader()
    print(next(dl))
