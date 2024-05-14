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
    data_cfg_path: str
    batch_size: int
    debug: bool = False


# class VocosDataModule(LightningDataModule):
#     def __init__(self, train_params: DataConfig, val_params: DataConfig):
#         super().__init__()
#         self.train_config = train_params
#         self.val_config = val_params
#
#     def _get_dataloder(self, cfg: DataConfig, train: bool):
#         dataset = VocosDataset(cfg, train=train)
#         dataloader = DataLoader(
#             dataset,
#             batch_size=cfg.batch_size,
#             num_workers=cfg.num_workers,
#             shuffle=train,
#             pin_memory=True,
#         )
#         return dataloader
#
#     def train_dataloader(self) -> DataLoader:
#         return self._get_dataloder(self.train_config, train=True)
#
#     def val_dataloader(self) -> DataLoader:
#         return self._get_dataloder(self.val_config, train=False)


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

        is_main_proc = int(os.environ.get("LOCAL_RANK", 0)) == 0

        data_config_path = Path(train_params.data_cfg_path)
        value_select = ["debug"] if train_params.debug else None

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
                    batch_size=train_params.batch_size,
                )
                self.data_loaders.append(loader)

            for loader in reversed(self.data_loaders):
                loader.start()

        except Exception as e:
            print(e)
            self.server.finish()
            self.workers.finish()

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        return self.data_loaders[0 if train else 1]

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(
            y, sr, [["norm", f"{gain:.2f}"]]
        )
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(
                y, orig_freq=sr, new_freq=self.sampling_rate
            )
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return y[0]


if __name__ == "__main__":

    cfg = DataConfig(
        data_cfg_path="../configs/data_24khz.yml",
        batch_size=4,
        debug=True,
    )

    dataset = VocosDataModule(cfg, cfg)
    dl = dataset.train_dataloader()
    print(next(dl))
