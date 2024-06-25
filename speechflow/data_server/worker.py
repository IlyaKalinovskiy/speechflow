import typing as tp
import logging

from os import environ as env

from speechflow.concurrency import ProcessWorker
from speechflow.data_pipeline.core import DataPipeline
from speechflow.data_pipeline.core.data_processor import DataProcessor
from speechflow.data_server.patterns import ZMQClient, ZMQPatterns, ZMQWorker
from speechflow.logging import trace
from speechflow.utils.serialize import Serialize

__all__ = ["BatchWorker"]

LOGGER = logging.getLogger("root")


class BatchWorker(ProcessWorker):
    def __init__(self, server_addr: str):
        ProcessWorker.__init__(self)
        self._server_addr = server_addr
        self._zmq_client: ZMQClient = None  # type: ignore
        self._zmq_worker: ZMQWorker = None  # type: ignore
        self._data_pipeline: DataPipeline = None  # type: ignore
        self._data_processor: tp.Dict = {}

    def on_start(self):
        self._zmq_client = ZMQPatterns.client(self._server_addr)
        info = self._zmq_client.request({"message": "info", "sub_type": "worker"})

        addr_for_workers = info["addr_for_workers"]
        self._zmq_worker = ZMQPatterns.worker(addr_for_workers)

        if info.get("device"):
            env["DEVICE"] = info["device"]

        if "data_pipeline" in info:
            self._data_pipeline = Serialize.load(info["data_pipeline"])
        else:
            self._data_pipeline = DataPipeline(info["data_config"])
            self._data_pipeline.init_components(
                preinit_handlers=info.get("singleton_handlers")
            )

        for subset_name in self._data_pipeline.subsets:
            components = self._data_pipeline[subset_name]
            self._data_processor[subset_name]: DataProcessor = components.data_processor

        message = f"Start Data Worker-{info['subscriber_id']} {self._server_addr}"
        LOGGER.debug(trace(self, message=message))

    def on_finish(self):
        self._zmq_worker.close()
        LOGGER.info(trace(self, message=f"Finish Batch Worker {self._server_addr}"))

    def do_work_once(self):
        batch = None
        try:
            request = self._zmq_worker.socket.recv_multipart()
            if request[0] == b"":
                request = request[1:]
            message = Serialize.load(request[0])
            samples = Serialize.loads(request[1:])

            data_processor = self._data_processor[message["subset_name"]]
            batch = data_processor.process(samples)

            if batch is not None:
                batch.tag = self._data_pipeline.tag

        except KeyboardInterrupt:
            LOGGER.error(trace(self, "Interrupt received, stopping ..."))
            self.finish()
        except Exception as e:
            LOGGER.error(trace(self, e))

        finally:
            self._zmq_worker.socket.send_pyobj(batch)
