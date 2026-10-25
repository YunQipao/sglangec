"""Transport interface shared by TCP debug and native verbs implementations."""

from abc import ABC, abstractmethod


class Transport(ABC):
    name: str

    @abstractmethod
    def send(self, host: str, port: int, payload: bytes) -> float:
        """Transfer a payload and return elapsed seconds after completion."""


def create_transport(name):
    if name == "tcp":
        from .tcp_transport import TCPTransport

        return TCPTransport()
    if name == "rdma":
        from .rdma_transport import RDMATransport

        return RDMATransport()
    raise ValueError(f"unsupported transport: {name}")
