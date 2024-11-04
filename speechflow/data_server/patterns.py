import sys
import pickle
import typing as tp
import asyncio

from dataclasses import dataclass

import zmq
import zmq.asyncio

from speechflow.logging import log_to_file, trace

__all__ = [
    "ZMQPatterns",
    "ZMQServer",
    "ZMQClient",
    "ZMQAsyncClient",
    "ZMQWorker",
    "ZMQProxy",
]


@dataclass
class ZMQServer:
    context: zmq.Context
    frontend: zmq.Socket
    backend: zmq.Socket
    poller: zmq.Poller
    socks: tp.Dict[zmq.Socket, tp.Any] = None  # type: ignore

    def pool(self, timeout: tp.Optional[int] = None):  # in milliseconds
        self.socks = dict(self.poller.poll(timeout))

    def is_frontend_ready(self) -> bool:
        if self.frontend:
            return self.socks.get(self.frontend) == zmq.POLLIN
        else:
            return False

    def is_backend_ready(self) -> bool:
        if self.backend:
            return self.socks.get(self.backend) == zmq.POLLIN
        else:
            return False

    def close(self):
        if self.frontend:
            self.frontend.close()
        if self.backend:
            self.backend.close()

    def frontend_send_multipart(self, message):
        self.frontend.send_multipart(message, flags=zmq.NOBLOCK)

    def backend_send_multipart(self, message):
        self.backend.send_multipart(message, flags=zmq.NOBLOCK)


@dataclass
class ZMQClient:
    context: zmq.Context
    socket: zmq.Socket

    def close(self):
        self.socket.close()

    def send(self, message, serialize: bool = True):
        self.socket.send_pyobj(
            message, flags=zmq.NOBLOCK
        ) if serialize else self.socket.send(message, flags=zmq.NOBLOCK)

    def recv(
        self,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
    ):
        if timeout is not None and self.socket.poll(timeout=timeout) == 0:
            return None
        else:
            list_bytes = self.socket.recv_multipart()
            if deserialize:
                list_obj = [pickle.loads(item) for item in list_bytes if item != b""]
                if len(list_obj) == 0:
                    return None
                elif len(list_obj) == 1:
                    return list_obj[0]
                else:
                    return list_obj
            else:
                return list_bytes

    def request(
        self,
        message,
        serialize: bool = True,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
    ):
        self.send(message, serialize)
        return self.recv(deserialize, timeout)

    def send_string(self, message: str):
        self.socket.send_string(message, flags=zmq.NOBLOCK)

    def recv_string(self, timeout: tp.Optional[int] = None):  # in milliseconds
        if timeout is not None and self.socket.poll(timeout=timeout) == 0:  # wait
            return None  # timeout reached before any events were queued
        else:
            return self.socket.recv_string()  # events queued within our time limit

    def request_as_string(
        self, message: str, timeout: tp.Optional[int] = None  # in milliseconds
    ):
        self.send_string(message)
        return self.recv_string(timeout)


@dataclass
class ZMQAsyncClient(ZMQClient):
    poller: zmq.Poller
    socks: tp.Dict[zmq.Socket, tp.Any] = None  # type: ignore

    def close(self):
        self.socket.close()

    def pool(self, timeout: tp.Optional[int] = None):  # in milliseconds
        self.socks = dict(self.poller.poll(timeout=timeout))

    def is_ready(self) -> bool:
        return self.socks.get(self.socket) == zmq.POLLIN


@dataclass
class ZMQWorker:
    context: zmq.Context
    socket: zmq.Socket

    def close(self):
        self.socket.close()

    def send(self, message, serialize: bool = True):
        self.socket.send_pyobj(message) if serialize else self.socket.send(message)


@dataclass
class ZMQProxy:
    context: zmq.Context
    frontend: zmq.Socket
    backend: tp.List[zmq.Socket]
    poller: zmq.Poller
    socks: tp.Dict[zmq.Socket, tp.Any] = None  # type: ignore

    def close(self):
        self.frontend.close()
        [client.close() for client in self.backend]

    def pool(self, timeout: tp.Optional[int] = None):  # in milliseconds
        self.socks = dict(self.poller.poll(timeout))

    def is_frontend_ready(self) -> bool:
        return self.socks.get(self.frontend) == zmq.POLLIN

    async def _send(self, sock, message):
        return await sock.send_pyobj(message, flags=zmq.NOBLOCK)

    async def _send_from_all(self, message):
        tasks = [self._send(sock, message) for sock in self.backend]
        return await asyncio.gather(*tasks)

    async def _request(self, sock, message):
        await sock.send_pyobj(message, flags=zmq.NOBLOCK)
        return await sock.recv_multipart()

    async def _request_from_all(self, message):
        tasks = [self._request(sock, message) for sock in self.backend]
        return await asyncio.gather(*tasks)

    def request(self, message):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._request_from_all(message))

    def frontend_send_multipart(self, message):
        self.frontend.send_multipart(message, flags=zmq.NOBLOCK)

    def backend_send_multipart(self, message):
        self.backend.send_multipart(message, flags=zmq.NOBLOCK)


class ZMQPatterns:
    @staticmethod
    def __create_socket_and_bind(
        context: zmq.Context, addr: str, socket_type
    ) -> zmq.Socket:
        socket = context.socket(socket_type)
        socket.bind(f"tcp://{addr}")
        socket.setsockopt(zmq.LINGER, 0)
        return socket

    @staticmethod
    def __create_socket_and_connect(
        context: zmq.Context, addr: str, socket_type
    ) -> zmq.Socket:
        socket = context.socket(socket_type)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(f"tcp://{addr}")
        return socket

    @staticmethod
    def __get_req(context: zmq.Context, addr: str) -> zmq.Socket:
        return ZMQPatterns.__create_socket_and_connect(context, addr, zmq.REQ)

    @staticmethod
    def __get_rep(context: zmq.Context, addr: str) -> zmq.Socket:
        return ZMQPatterns.__create_socket_and_connect(context, addr, zmq.REP)

    @staticmethod
    def __get_sub(context: zmq.Context, addr: str) -> zmq.Socket:
        return ZMQPatterns.__create_socket_and_connect(context, addr, zmq.SUB)

    @staticmethod
    def __get_pub(context: zmq.Context, addr: str) -> zmq.Socket:
        return ZMQPatterns.__create_socket_and_bind(context, addr, zmq.PUB)

    @staticmethod
    def __get_router(context: zmq.Context, addr: str) -> zmq.Socket:
        return ZMQPatterns.__create_socket_and_bind(context, addr, zmq.ROUTER)

    @staticmethod
    def __get_dealer(context: zmq.Context, addr: str, bind: bool = True) -> zmq.Socket:
        if bind:
            return ZMQPatterns.__create_socket_and_bind(context, addr, zmq.DEALER)
        else:
            return ZMQPatterns.__create_socket_and_connect(context, addr, zmq.DEALER)

    @staticmethod
    def __get_poller(sockets: tp.List[zmq.Socket]) -> zmq.Poller:
        poller = zmq.Poller()
        for s in sockets:
            poller.register(s, zmq.POLLIN)
        return poller

    @classmethod
    def server(
        cls,
        addr_for_clients: tp.Optional[str] = None,
        addr_for_workers: tp.Optional[str] = None,
    ) -> ZMQServer:
        if addr_for_clients is None and addr_for_workers is None:
            raise AttributeError("Least one socket address must be specified")

        sockets = []
        context = zmq.Context.instance()

        if addr_for_clients:
            log_to_file(trace(cls, f"bind socket {addr_for_clients} for clients"))
            frontend = cls.__get_router(context, addr_for_clients)
            sockets.append(frontend)
        else:
            frontend = None

        if addr_for_workers:
            log_to_file(trace(cls, f"bind socket {addr_for_workers} for workers"))
            backend = cls.__get_dealer(context, addr_for_workers)
            sockets.append(backend)
        else:
            backend = None

        poller = cls.__get_poller(sockets)

        return ZMQServer(
            context=context, frontend=frontend, backend=backend, poller=poller
        )

    @classmethod
    def client(cls, server_addr: str) -> ZMQClient:
        log_to_file(trace(cls, f"connection to {server_addr}"))

        context = zmq.Context()
        socket = cls.__get_req(context, server_addr)

        return ZMQClient(context=context, socket=socket)

    @classmethod
    def async_client(cls, server_addr: str) -> ZMQAsyncClient:
        log_to_file(trace(cls, f"connection to {server_addr}"))

        context = zmq.Context()
        socket = cls.__get_dealer(context, server_addr, bind=False)
        poller = cls.__get_poller([socket])

        return ZMQAsyncClient(context=context, socket=socket, poller=poller)

    @classmethod
    def worker(cls, server_addr: str) -> ZMQWorker:
        log_to_file(trace(cls, f"connection to {server_addr}"))

        context = zmq.Context()
        socket = cls.__get_rep(context, server_addr)

        return ZMQWorker(context=context, socket=socket)

    @classmethod
    def proxy(cls, proxy_addr: str, server_addrs: tp.List[str]) -> ZMQProxy:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        try:
            log_to_file(trace(cls, f"bind socket {proxy_addr}"))

            context = zmq.Context()
            frontend = cls.__get_router(context, proxy_addr)
            poller = cls.__get_poller([frontend])

            async_context = zmq.asyncio.Context()
            backend = []
            for addr in server_addrs:
                log_to_file(trace(cls, f"connection to {addr}"))
                backend.append(cls.__get_req(async_context, addr))

        except zmq.error.ZMQError as e:
            raise e

        return ZMQProxy(
            context=context,
            frontend=frontend,
            backend=backend,
            poller=poller,
        )
