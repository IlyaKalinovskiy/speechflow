import math
import typing as tp
import logging
import argparse

from collections import deque
from threading import Event, Thread

from speechflow.data_pipeline.core import Batch
from speechflow.data_server.client import DataClient
from speechflow.io import Config, check_path, tp_PATH
from speechflow.logging import log_to_file, trace
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler
from speechflow.utils.serialize import Serialize

__all__ = ["DataLoader"]

LOGGER = logging.getLogger("root")


class DataLoader:
    def __init__(
        self,
        server_addr: str,
        subset_name: str,
        epoch_len: int = 0,
        batch_size: int = 1,
        min_batch_size: int = 0,
        drop_non_full: bool = False,
        non_stop: bool = True,
        pin_memory: bool = False,
        prefetch_on_gpu: bool = False,
        min_prefetch_factor: int = 50,
        max_prefetch_factor: int = 150,
        ddp_reduce_factor: int = 1,
    ):
        self._info_client = DataClient(server_addr)
        self._data_client = DataClient(
            server_addr, sub_type="loader", uid=self._info_client.uid
        )
        self._request_task = Thread(target=self._batch_request)
        self._receive_tasks = [Thread(target=self._batch_receive) for _ in range(2)]

        if subset_name not in self._data_client.info["subsets"]:
            raise ValueError(f"subset {subset_name} not provided by data server!")

        self.subset_name = subset_name
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.drop_non_full = drop_non_full
        self.non_stop = non_stop
        self.pin_memory = pin_memory
        self.prefetch_on_gpu = prefetch_on_gpu
        assert self.min_batch_size <= self.batch_size

        assert max_prefetch_factor >= min_prefetch_factor
        self.min_prefetch_factor = min_prefetch_factor
        self.max_prefetch_factor = max_prefetch_factor
        self.prefetch_factor = min_prefetch_factor

        if epoch_len:
            self.epoch_len = epoch_len
        else:
            self.epoch_len = math.ceil(self.epoch_size / batch_size)

        assert ddp_reduce_factor > 0
        self.epoch_len = math.ceil(self.epoch_len / ddp_reduce_factor)

        self._stop_event = Event()
        self._epoch_complete_event = Event()
        self._batch_queue: tp.Deque = deque()
        self._async_supported = self.client.find_info("async_supported", False)

    def __len__(self) -> int:
        return int(self.epoch_len)

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        return self.next_batch()

    def __del__(self):
        self.finish()

    @property
    def client(self) -> DataClient:
        return self._data_client

    @property
    def epoch_size(self) -> int:
        return self._data_client.info["epoch_size"][self.subset_name]

    @property
    def dataset_size(self) -> int:
        return len(self._data_client.info["dataset"][self.subset_name])

    @staticmethod
    @check_path(assert_file_exists=True)
    def init_from_config(
        file_path: tp_PATH,
        value_select: tp.Optional[tp.List[str]] = None,
        server_addr: tp.Optional[str] = None,
        subset_name: tp.Optional[str] = None,
        config_section: str = "data_loaders",
    ) -> "DataLoader":
        cfg = Config.create_from_file(
            file_path, section=config_section, value_select=value_select
        )

        if server_addr:
            cfg["server_addr"] = server_addr

        if subset_name:
            cfg["subset_name"] = subset_name

        return init_class_from_config(DataLoader, cfg)()

    def _batch_request(self):
        while not self._stop_event.is_set():
            try:
                free_slots = self.prefetch_factor - len(self._batch_queue)
                if free_slots < self.prefetch_factor // 4:
                    continue

                if self._async_supported:
                    response = self._info_client.request(
                        message={"message": "is_ready"},
                        deserialize=False,
                        timeout=10000,  # in milliseconds
                    )
                    if not response:
                        continue
                else:
                    response = [b"info: true"]

                for _bytes in response:
                    if _bytes.startswith(b"info: true") or _bytes.startswith(
                        b"info: epoch complete"
                    ):
                        message = {
                            "message": "batch",
                            "subset_name": self.subset_name,
                            "batch_size": self.batch_size,
                            "batch_num": free_slots,
                        }
                        self._data_client.send(message)

            except KeyboardInterrupt:
                LOGGER.error(trace(self, "Interrupt received, stopping ..."))
                break
            except Exception as e:
                LOGGER.error(trace(self, e))
            finally:
                Profiler.sleep(0.5)

    def _batch_receive(self):
        while not self._stop_event.is_set():
            try:
                response = self._data_client.recv(
                    deserialize=False, timeout=100  # in milliseconds
                )
                if not response:
                    continue

                batch_list = []
                is_epoch_complete = False
                for _bytes in response:
                    if _bytes.startswith(b"info: epoch complete"):
                        is_epoch_complete = True
                        continue
                    elif _bytes == b"" or _bytes.startswith(b"info:"):
                        continue
                    else:
                        batch_list.append(_bytes)

                if batch_list:
                    batch_list = Serialize.loads(batch_list)

                for batch in batch_list:
                    if not isinstance(batch, Batch):
                        continue

                    if not batch.is_last and (
                        (self.drop_non_full and batch.size != self.batch_size)
                        or batch.size < self.min_batch_size
                    ):
                        message = (
                            f"batch size mismatch "
                            f"(expected size {self.batch_size} but received {batch.size})"
                        )
                        LOGGER.info(trace(self, message=message))
                    else:
                        if self.pin_memory and batch.collated_samples is not None:
                            batch.collated_samples.pin_memory()

                        self._batch_queue.append(batch)

                if is_epoch_complete:
                    if self.non_stop or len(self._batch_queue) == 0:
                        self._data_client.send({"message": "receiving_completed"})
                        self._epoch_complete_event.set()
                        log_to_file(
                            trace(self, self.subset_name, message="epoch_complete")
                        )

            except KeyboardInterrupt:
                LOGGER.error(trace(self, "Interrupt received, stopping ..."))
                break
            except Exception as e:
                LOGGER.error(trace(self, e))

    def start(self):
        self._stop_event.clear()
        self._epoch_complete_event.clear()
        self._request_task.start()
        [t.start() for t in self._receive_tasks]

    def finish(self):
        self._stop_event.set()
        try:
            if self._request_task.is_alive():
                self._request_task.join(timeout=1)
            [t.join(timeout=1) for t in self._receive_tasks if t.is_alive()]
        except RuntimeError:
            pass
        except Exception as e:
            LOGGER.error(trace(self, e))

    def next_batch(self, sleep: float = 0) -> Batch:
        if len(self._batch_queue) == 0:
            assert self._request_task.is_alive(), "DataLoader has not been started!"
            if sleep > 1:
                LOGGER.warning(
                    trace(self, self.subset_name, message="Batches receive too slowly!")
                )
                Profiler.sleep(sleep)
            else:
                self.prefetch_factor = int(
                    min(self.prefetch_factor * 1.2, self.max_prefetch_factor)
                )
                log_to_file(
                    trace(
                        self,
                        message=f"increase prefetch factor for {self.subset_name}: {self.prefetch_factor}",
                    )
                )
                Profiler.sleep(1.0)

            if sleep > 0 and sleep % 15 == 0:
                self.abort_processing()
            elif sleep > 300:
                raise RuntimeError("DataServer stopped responding!")

            return self.next_batch(sleep + 3)

        batch = self._batch_queue.popleft()

        if self.prefetch_on_gpu and len(self._batch_queue) > 0:
            next_batch = self._batch_queue[0]
            if next_batch.collated_samples is not None:
                next_batch.collated_samples.cuda()

        return batch

    def abort_processing(self):
        self._data_client.send(
            {"message": "abort_processing", "subset_name": self.subset_name}
        )
        LOGGER.info(trace(self, self.subset_name, message="abort_processing"))

    def reset(self):
        self._batch_queue.clear()
        self._epoch_complete_event.clear()
        self._data_client.send({"message": "reset", "subset_name": self.subset_name})
        self.prefetch_factor = self.min_prefetch_factor
        LOGGER.info(trace(self, self.subset_name, message="reset"))

    def get_epoch_iterator(self) -> tp.Iterator[Batch]:
        self.reset()

        class EpochIterator:
            def __init__(self, dl: "DataLoader"):
                self._dl = dl

            def __iter__(self):
                return self

            def __next__(self):
                if self._dl._epoch_complete_event.is_set():
                    self._dl._epoch_complete_event.clear()
                    log_to_file(
                        trace(
                            self._dl, message=f"stop iteration for {self._dl.subset_name}"
                        )
                    )
                    raise StopIteration

                return next(self._dl)

        return EpochIterator(self)


def test_connection(data_loader: DataLoader, max_time: int = 10):
    number = batch_size = time = size = 0
    test_time = Profiler()

    while test_time.get_time() < max_time:
        with Profiler(auto_logging=False) as timer:
            batch = next(data_loader)

        if isinstance(batch, Batch):
            number += 1
            batch_size = batch.size
            time += timer.get_time()
            size += Serialize.get_obj_size(batch, format=Serialize.Format.MB)

    assert size > 0, "Not batch received!"
    speed = round(number / time, 3)
    size = round(size / number, 3)

    LOGGER.info(
        trace(
            "test",
            message=f"GET {data_loader.subset_name} {speed} batches/s, "
            f"batch size {batch_size}, "
            f"packet size {size} MB",
        )
    )


if __name__ == "__main__":
    """
    example:
        client.py -a=localhost:8000
    """

    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-a", "--addr", help="address of data server", type=str, required=True
    )
    arguments_parser.add_argument(
        "-ss", "--subset_name", help="subset name", type=str, default="train"
    )
    arguments_parser.add_argument(
        "-bs", "--batch_size", help="batch size", type=int, default=48
    )
    arguments_parser.add_argument(
        "-s", "--sleep", help="batch request interval", type=float, default=0.5
    )
    arguments_parser.add_argument(
        "-n", "--number", help="number of clients", type=int, default=1
    )
    arguments_parser.add_argument(
        "-v",
        "--visualize",
        help="draw wave and spectrogram segmentations",
        action="store_true",
    )
    args = arguments_parser.parse_args()

    def run_client(client_index):
        loader = DataLoader(
            args.addr,
            "train",
            epoch_len=100,
            batch_size=args.batch_size,
            drop_non_full=True,
        )
        loader.start()
        test_connection(loader)

        try:
            for batch_index in range(len(loader)):
                Profiler.sleep(args.sleep)
                with Profiler(auto_logging=False) as timer:
                    batch = next(loader)

                if batch:
                    packet_size = Serialize.get_obj_size(batch, Serialize.Format.MB)
                    print(
                        f"client {client_index}, "
                        f"batch {batch_index}, "
                        f"batch size {batch.size}, "
                        f"packet size {round(packet_size, 3)} MB, "
                        f"time {timer.get_time()} ms"
                    )
                else:
                    print("load dataset completed")

                # if args.visualize:
                #    visialize_batch(batch, batch_index)
        finally:
            loader.finish()

    tasks = [Thread(target=run_client, args=(index,)) for index in range(args.number)]
    for task in tasks:
        task.start()
    for task in tasks:
        task.join()

    print("TEST COMPLETED")
