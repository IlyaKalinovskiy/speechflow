import typing as tp
import logging
import itertools

from speechflow.concurrency import ProcessWorker
from speechflow.data_pipeline.core import Batch, DataPipeline
from speechflow.data_server.patterns import ZMQPatterns, ZMQProxy
from speechflow.io import Config, tp_PATH
from speechflow.logging import trace
from speechflow.utils.init import init_class_from_config
from speechflow.utils.serialize import Serialize
from speechflow.utils.sockopt import find_free_port

__all__ = ["Proxy"]

LOGGER = logging.getLogger("root")


class Proxy(ProcessWorker):
    def __init__(
        self,
        server_addrs: tp.List[str],
    ):
        ProcessWorker.__init__(self)
        self._server_addrs = server_addrs
        self._proxy_addr = f"127.0.0.1:{find_free_port()}"
        self._zmq_proxy: ZMQProxy = None  # type: ignore
        self._async_supported = False

    @property
    def address(self):
        return self._proxy_addr

    @staticmethod
    def init_from_config(
        file_path: tp_PATH,
        value_select: tp.Optional[tp.List[str]] = None,
        server_addr: tp.Optional[str] = None,
        config_section: str = "proxy_server",
    ) -> "Proxy":
        cfg = Config.create_from_file(
            file_path, section=config_section, value_select=value_select
        )

        if server_addr:
            cfg["server_addrs"] = [server_addr]
        else:
            if cfg.is_empty:
                raise RuntimeError("Missing proxy settings!")

        clsmembers = {v.__name__: v for v in Proxy.__subclasses__()}
        custom_proxy_cls = clsmembers[cfg["type"]]
        custom_proxy_cls = init_class_from_config(custom_proxy_cls, cfg)
        return custom_proxy_cls()

    def on_start(self):
        self._zmq_proxy = ZMQPatterns.proxy(self._proxy_addr, self._server_addrs)
        message = f"Start {self.__class__.__name__} Server {self._proxy_addr}"
        LOGGER.info(trace(self, message=message))

    def on_finish(self):
        self._zmq_proxy.close()
        LOGGER.info(
            trace(
                self,
                message=f"Finish {self.__class__.__name__} Server {self._proxy_addr}",
            )
        )

    def do_preprocessing(self, request: tp.Dict) -> tp.List[tp.Any]:
        batches = self.get_batches(request, decode_batch=False)
        return batches

    def get_batches(
        self,
        request: dict,
        batch_size: tp.Optional[int] = None,
        batch_num: tp.Optional[int] = None,
        decode_batch: bool = True,
    ) -> tp.List[Batch]:
        if batch_size:
            request["batch_size"] = max(1, batch_size)

        if batch_num:
            request["batch_num"] = max(1, batch_num)
        else:
            request["batch_num"] = max(1, request["batch_num"] // len(self._server_addrs))

        request["async_mode"] = False

        response = self._zmq_proxy.request(request)
        response = list(itertools.chain(*response))

        batches = []
        for resp in response:
            if resp.startswith(b"info: epoch complete"):
                self._zmq_proxy.request({"message": "receiving_completed"})
                continue
            elif resp == b"" or resp.startswith(b"info:"):
                continue

            if decode_batch:
                batch: Batch = Serialize.load(resp)
                if not isinstance(batch, Batch):
                    continue
            else:
                batch = resp

            batches.append(batch)

        return batches

    def do_work_once(self):
        try:
            self._zmq_proxy.pool(timeout=10)

            if self._zmq_proxy.is_frontend_ready():
                message = self._zmq_proxy.frontend.recv_multipart()
                request = Serialize.load(message[-1])

                if request["message"] == "info":
                    all_response = self._zmq_proxy.request(request)

                    all_info = []
                    for resp in all_response:
                        info = Serialize.load(resp[0])
                        info["async_supported"] = self._async_supported
                        all_info.append(info)

                    message[-1] = Serialize.dump(DataPipeline.aggregate_info(all_info))
                    self._zmq_proxy.frontend_send_multipart(message)
                elif request["message"] == "batch":
                    try:
                        message.pop(-1)
                        message += Serialize.dumps(self.do_preprocessing(request))
                    except Exception as e:
                        LOGGER.error(trace(self, e))

                    self._zmq_proxy.frontend_send_multipart(message)
                else:
                    message[-1] = self._zmq_proxy.request(request)[0][-1]
                    self._zmq_proxy.frontend_send_multipart(message)

        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            LOGGER.error(trace(self, e))
