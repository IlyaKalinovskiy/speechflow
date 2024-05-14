import uuid
import typing as tp
import logging
import argparse
import multiprocessing as mp

from collections import defaultdict
from dataclasses import dataclass

import psutil

from speechflow.concurrency import ProcessWorker
from speechflow.data_pipeline.core import DataPipeline
from speechflow.data_server.patterns import ZMQPatterns, ZMQServer
from speechflow.data_server.pool import WorkerPool
from speechflow.io import Config, check_path, tp_PATH
from speechflow.logging import log_to_file, trace
from speechflow.utils.gpu import get_freer_gpu
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler
from speechflow.utils.serialize import Serialize
from speechflow.utils.sockopt import find_free_port

__all__ = ["DataServer"]

LOGGER = logging.getLogger("root")


@dataclass
class SamplingStatus:
    num_batch_in_processing: int = 0
    is_last_batch: bool = False
    batches: tp.List = None  # type: ignore
    async_mode: bool = True
    subset: str = None  # type: ignore

    def __post_init__(self):
        self.batches = []


class DataServer(ProcessWorker):
    def __init__(
        self,
        data_pipeline: DataPipeline,
        n_processes: int = 0,
        n_gpus: tp.Union[int, tp.List[int]] = 0,
        server_addr: tp.Optional[str] = None,
        memory_save: bool = False,
        use_profiler: bool = False,
    ):
        ProcessWorker.__init__(self)
        self._addr_for_clients = (
            server_addr if server_addr else f"127.0.0.1:{find_free_port()}"
        )
        self._addr_for_workers = f"127.0.0.1:{find_free_port()}"
        self._pipe = data_pipeline
        self._n_processes = n_processes if n_processes else mp.cpu_count()
        self._zmq_server: ZMQServer = None  # type: ignore
        self._async_supported = True
        self._work_queues: tp.Dict[str, SamplingStatus] = defaultdict(SamplingStatus)
        self._uid_map: tp.Dict[bytes, str] = {}

        self._subscribers: tp.Dict[str, int] = {}
        self._info_for_worker = None
        self._info_for_loader = None

        self._batch_counter = 0
        self._total_batch_in_processing = 0
        self._timer = Profiler(auto_logging=False)

        self._gpus = self.init_gpus(n_gpus) if isinstance(n_gpus, int) else n_gpus

        self._memory_save = memory_save
        self._use_profiler = use_profiler

    @property
    def address(self) -> str:
        return self._addr_for_clients

    @property
    def num_processes(self) -> int:
        return self._n_processes

    @property
    def num_workers(self) -> int:
        return self._subscribers.get("worker", 0)

    @staticmethod
    def init_gpus(num_gpu: int) -> tp.List[int]:
        import torch

        gpus = []
        for _ in range(num_gpu):
            gpus.append(get_freer_gpu())
            torch.tensor([0.0], device=f"cuda:{gpus[-1]}")

        return gpus

    @staticmethod
    @check_path(assert_file_exists=True)
    def init_from_config(
        file_path: tp_PATH,
        value_select: tp.Optional[tp.List[str]] = None,
        server_addr: tp.Optional[str] = None,
        config_section: str = "data_server",
    ) -> "DataServer":
        cfg = Config.create_from_file(
            file_path, section=config_section, value_select=value_select
        )

        if server_addr:
            cfg["server_addr"] = server_addr

        data_pipeline: DataPipeline = DataPipeline.init_from_config(
            file_path=file_path,
            value_select=value_select,
        )

        return init_class_from_config(DataServer, cfg)(data_pipeline=data_pipeline)

    def on_start(self):
        self._zmq_server = ZMQPatterns.server(
            self._addr_for_clients, self._addr_for_workers
        )
        self._pipe.init_components()
        self._pipe.load_data(n_processes=self._n_processes)

        self._info_for_worker = self._pipe.get_info()
        self._info_for_loader = self._pipe.get_info(object_size_limit=0)
        LOGGER.info(trace(self, message=f"Start DataServer {self._addr_for_clients}"))

    def on_finish(self):
        self._zmq_server.close()
        LOGGER.info(trace(self, message=f"Finish DataServer {self._addr_for_clients}"))

    def status_info(self, timeout: float = 600):
        if self._timer.get_time() > timeout:
            mem = psutil.virtual_memory()
            info = (
                "\n"
                f"\tsubscribers: {list(self._subscribers.items())}\n"
                f"\tbatches_prepared: {self._batch_counter}\n"
                f"\tcpu_utilization: {psutil.cpu_percent()}\n"
                f"\tavailable_memory: {round(mem.available * 100 / mem.total, 1)}%\n"
                f"\tused_virtual_memory: {round(mem.used / 1024 ** 3, 1)}GB\n"
                f"\tused_swap_memory: {round(psutil.swap_memory().used / 1024 ** 3, 1)}GB"
            )
            log_to_file(trace(self, info))
            self._timer.reset()

    def send_info_message(self, message, text: str, subset: tp.Optional[str] = None):
        info = f"info: {text}"
        message = [message[0], b"", info.encode()]
        self._zmq_server.frontend.send_multipart(message)
        if text not in ["true", "request queue exceeded"]:
            log_to_file(trace(self, f"{subset}: {info}" if subset else info))

    def is_reject_request(self, message, queue_info: SamplingStatus) -> bool:
        if self.num_workers == 0:
            self.send_info_message(message, "workers not found", queue_info.subset)
            return True

        if self._total_batch_in_processing >= 4 * self.num_workers:
            self.send_info_message(message, "server overload", queue_info.subset)
            return True

        if queue_info.num_batch_in_processing > 0:
            self.send_info_message(message, "request queue exceeded", queue_info.subset)
            return True

        if queue_info.is_last_batch:
            self.send_info_message(message, "epoch complete", queue_info.subset)
            return True

        return False

    def gen_response(self, message):
        request = Serialize.load(message[-1])
        if message[0] not in self._uid_map:
            self._uid_map[message[0]] = request.get("client_uid", uuid.uuid4().hex)

        if request["message"] == "info":
            response = {
                "subscriber_id": self._subscribers.setdefault(request["sub_type"], 0),
                "async_supported": self._async_supported,
                "addr_for_workers": self._addr_for_workers,
            }
            if request["sub_type"] == "loader":
                response.update(self._info_for_loader)
            else:
                response.update(self._info_for_worker)
                response.update(
                    {"memory_save": self._memory_save, "use_profiler": self._use_profiler}
                )
                if self._gpus:
                    idx = response["subscriber_id"] % len(self._gpus)
                    response.update({"device": f"cuda:{self._gpus[idx]}"})

            message[-1] = Serialize.dump(response)
            self._zmq_server.frontend.send_multipart(message)
            self._subscribers[request["sub_type"]] += 1

        elif request["message"] == "is_ready":
            queue_info = self._work_queues[self._uid_map[message[0]]]
            if not self.is_reject_request(message, queue_info):
                self.send_info_message(message, "true", queue_info.subset)

        elif request["message"] == "batch":
            subset = request["subset_name"]
            batch_size = request["batch_size"]
            batch_num = request.get("batch_num", 1)

            queue_info = self._work_queues[self._uid_map[message[0]]]
            queue_info.async_mode = request.get("async_mode", True)
            queue_info.subset = subset
            if self.is_reject_request(message, queue_info):
                return

            batch_list = []
            sampler = self._pipe[subset].sampler
            for _ in range(batch_num):
                if self._total_batch_in_processing >= 4 * self.num_workers:
                    break

                batch_list.append(sampler.sampling(batch_size))
                self._total_batch_in_processing += 1

                if sampler.is_last_batch:
                    queue_info.is_last_batch = True
                    break

            message.insert(1, b"")
            queue_info.num_batch_in_processing += len(batch_list)

            for samples in batch_list:
                self._zmq_server.backend.send_multipart(
                    message + Serialize.dumps(samples)
                )

        elif request["message"] in ["receiving_completed", "abort_processing", "reset"]:
            status = self._work_queues[self._uid_map[message[0]]]
            if status.batches:
                self._zmq_server.frontend.send_multipart(message + status.batches)

            status.is_last_batch = False
            status.batches = []
            status.num_batch_in_processing = 0
            self.send_info_message(message, "queue cleared", status.subset)

            if request["message"] == "abort_processing":
                self._total_batch_in_processing = 0
                self.send_info_message(
                    message, "abort processing the current batch", status.subset
                )

            if request["message"] == "reset":
                self._total_batch_in_processing = 0
                self._pipe[request["subset_name"]].sampler.reset()
                self.send_info_message(message, "reset sampler state", status.subset)

    def do_work_once(self):
        try:
            self._zmq_server.pool(timeout=10)

            if self._zmq_server.is_frontend_ready():
                message = self._zmq_server.frontend.recv_multipart()
                self.gen_response(message)

            if self._zmq_server.is_backend_ready():
                message = self._zmq_server.backend.recv_multipart()
                self._total_batch_in_processing = max(
                    0, self._total_batch_in_processing - 1
                )

                queue_info = self._work_queues.get(self._uid_map[message[0]])
                if queue_info is None:
                    self.send_info_message(message, "batch skipped")
                    return

                queue_info.num_batch_in_processing = max(
                    0, queue_info.num_batch_in_processing - 1
                )

                if queue_info.async_mode:
                    self._zmq_server.frontend.send_multipart(message)
                else:
                    if queue_info.num_batch_in_processing == 0:
                        self._zmq_server.frontend.send_multipart(
                            message + queue_info.batches
                        )
                        queue_info.batches = []
                    else:
                        queue_info.batches.append(message[2])

                self._batch_counter += 1

            self.status_info()

        except KeyboardInterrupt:
            LOGGER.error(trace(self, "Interrupt received, stopping ..."))
            self.finish()
        except Exception as e:
            LOGGER.error(trace(self, e))


if __name__ == "__main__":
    """
    example:
        server.py -c=../../../tts_data/config_example.yml
    """

    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-c", "--config_path", help="path to yaml config", type=str, required=True
    )
    arguments_parser.add_argument(
        "-vs", "--value_select", help="select specific values", nargs="+", type=str
    )
    args = arguments_parser.parse_args()

    server = DataServer.init_from_config(**args.__dict__)
    worker_pool = WorkerPool(server_addr=server.address, n_processes=server.num_processes)

    server.start()
    worker_pool.start()

    server.join()

    worker_pool.finish()
    server.finish()
