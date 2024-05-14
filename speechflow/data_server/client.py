import uuid
import typing as tp
import logging

from threading import Lock as ThreadLock

from speechflow.data_server.patterns import ZMQPatterns
from speechflow.logging import trace
from speechflow.utils.dictutils import flatten_dict

__all__ = ["DataClient"]

LOGGER = logging.getLogger("root")


class DataClient:
    def __init__(self, server_addr: str, sub_type: str = "client", uid: str = None):
        self._uid = uid if uid else uuid.uuid4().hex
        self._server_addr = server_addr
        self._request_lock = ThreadLock()
        self._send_lock = ThreadLock()
        self._recv_lock = ThreadLock()
        self._zmq_client = ZMQPatterns.async_client(server_addr)
        self._info = self.request(
            {"message": "info", "sub_type": sub_type}, timeout=600_000
        )
        if self._info is None:
            raise RuntimeError("DataServer not responding!")

        LOGGER.info(trace(self, message=f"Start DataClient {self._server_addr}"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._zmq_client.close()
        LOGGER.info(trace(self, message=f"Finish DataClient {self._server_addr}"))

    @property
    def uid(self):
        return self._uid

    @property
    def server_address(self):
        return self._server_addr

    @property
    def info(self) -> tp.Dict:
        return self._info

    def find_info(self, name: str, default: tp.Any = None, section: str = None) -> tp.Any:
        if section is None:
            flatten_info = flatten_dict(self.info)
        else:
            flatten_info = flatten_dict(self.info["singleton_handlers"])

        found = []
        for key, field in flatten_info.items():
            if key.endswith(name):
                if field not in [None, {}]:
                    found.append(field)
        if len(found):
            return found[0]
        else:
            return default

    def request(
        self,
        message,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
    ):
        message["client_uid"] = self._uid
        with self._request_lock:
            return self._zmq_client.request(
                message, deserialize=deserialize, timeout=timeout
            )

    def send(self, message):
        message["client_uid"] = self._uid
        with self._send_lock:
            self._zmq_client.send(message)

    def recv(
        self,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
    ):
        with self._recv_lock:
            self._zmq_client.pool(timeout=timeout)
            if self._zmq_client.is_ready():
                return self._zmq_client.recv(deserialize)

        return []
