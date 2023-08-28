import logging
import sys
import uuid

import zmq
from zmq import devices
from queue import Queue
import threading

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout,
)

logger = logging.getLogger("tensorshare")
logger.setLevel(logging.WARNING)
LOCALHOST = "tcp://localhost"


class TensorConsumer:
    def __init__(
        self,
        port: int = 5555,
        ack_port: int = 5556,
        heart_ports: (int, int) = (4444, 4445),
        max_buffer_size: int = 10,
    ):
        """Data loader (iterator) that receives inputs and labels over tcp.

        Args:
            port (int, optional): Data transmission port. Defaults to 5555.
            ack_port (int, optional): Acknowledgement port. Defaults to 5556.
            heart_ports (int, int, optional): Life pulse ports. Defaults to (4444, 4445).
            max_buffer_size (int, optional): How many batches of data to hold in consumer buffer. Defaults to 10.
        """
        self.port = port
        self.ack_port = ack_port
        self.heart_ports = heart_ports

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"{LOCALHOST}:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.consumer_id = uuid.uuid4()

        # Ack
        self.ack_socket = self.context.socket(zmq.PUB)
        self.ack_socket.connect(f"{LOCALHOST}:{self.ack_port}")

        # Heartbeat
        self.dev = devices.ThreadDevice(zmq.FORWARDER, zmq.SUB, zmq.DEALER)
        self.dev.setsockopt_in(zmq.SUBSCRIBE, b"")
        self.dev.connect_in(f"{LOCALHOST}:{self.heart_ports[0]}")
        self.dev.connect_out(f"{LOCALHOST}:{self.heart_ports[1]}")
        self.dev.start()

        # On spawn, fetch payloads on socket until we get one with the data loader length
        while True:
            data = self.socket.recv_pyobj()
            if data.get("data_loader_len"):
                self.data_loader_len = data.get("data_loader_len")
                break

        # Buffer setup
        self.buffer = Queue(maxsize=max_buffer_size)
        self.fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.fetch_thread.start()

        # Logic
        self.batch_count = 0
        self.epoch = 0

    def _fetch_loop(self):
        while True:
            cuda_tensor_info = self.socket.recv_pyobj()
            self.buffer.put(cuda_tensor_info)

            self.ack_socket.send_multipart(
                [
                    bytes(str(self.consumer_id).encode("utf-8")),
                    bytes(str(self.batch_count).encode("utf-8")),
                ]
            )

    def __iter__(self):
        return self

    def __len__(self):
        return self.data_loader_len

    def __next__(self):
        while True:
            cuda_tensor_info = self.buffer.get()  # This will block if buffer is empty
            if cuda_tensor_info.get("data_loader_len"):
                continue

            current_epoch = cuda_tensor_info["current_epoch"]
            batch_idx = cuda_tensor_info["current_batch_index"]
            inputs = cuda_tensor_info["inputs"].tensor
            labels = cuda_tensor_info["labels"].tensor

            if current_epoch != self.epoch:
                self.epoch = current_epoch
                self.batch_count = 0

            logger.info(
                f"Epoch: {self.epoch}, batch_idx: {batch_idx}, batch count: {self.batch_count}"
            )
            if batch_idx == self.batch_count:
                self.batch_count += 1
                return (inputs, labels)
