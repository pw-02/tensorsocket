import logging
import threading
import time
import zmq

from collections import deque
from dataclasses import dataclass, field
from torch import Tensor, cuda
from tornado import ioloop
from typing import Any, Iterator, Optional
from zmq.eventloop import zmqstream

from .heartbeat import HeartBeater
from .payload import TensorPayload

logger = logging.getLogger("tensorsocket")
logger.setLevel(logging.WARNING)
LOCALHOST = "tcp://*"


@dataclass
class ConsumerProgress:
    """Tracks progress of individual consumers through dataset.

    Maintains ordered queues for batch IDs and payloads to ensure sequential processing
    and proper synchronization between producer and consumer.

    Attributes:
        id_queue: Queue storing sequential batch IDs
        payload_queue: Queue storing corresponding tensor payloads
    """

    id_queue: deque[int]
    payload_queue: deque[dict]

    def add_batch(self, id: int, payload: dict) -> None:
        """Add a new batch to tracking queues.

        Args:
            id: Sequential batch identifier
            payload: Associated tensor data and metadata

        Raises:
            AssertionError: If batch ID is not sequential
        """
        assert id == self.batch_max
        self.id_queue.append(id)
        self.payload_queue.append(payload)

    def remove_batch(self, id: int) -> None:
        """Remove the leftmost ID/payload pair. ID must match arg0.

        Args:
            id: Sequential batch identifier

        Raises:
            AssertionError: If batch ID does not match the leftmost ID
        """
        assert id == self.batch_count
        self.id_queue.popleft()
        self.payload_queue.popleft()

    @property
    def batch_count(self) -> int:
        """Get the current batch count.

        Returns:
            The leftmost batch ID in the queue.
        """
        return self.id_queue[0]

    @property
    def batch_max(self) -> int:
        """Get the maximum batch ID.

        Returns:
            The rightmost batch ID in the queue plus one, or 0 if the queue is empty.
        """
        return self.id_queue[-1] + 1 if self.id_queue else 0


def process_tensor(tensor: Any) -> TensorPayload:
    """Process single tensor for transmission.

    Args:
        tensor: PyTorch tensor to process

    Returns:
        Processed tensor wrapped in TensorPayload
    """
    if isinstance(tensor, Tensor):
        if cuda.is_available():
            tensor = tensor.to(device="cuda")
        tensor = TensorPayload(tensor)
    return tensor


def pack(data: tuple) -> tuple:
    """Pack multiple tensors for transmission.

    Args:
        data: Tuple of tensors to process

    Returns:
        Tuple of processed tensors
    """
    return tuple((process_tensor(t) for t in data))


class TensorProducer:
    """Distributes tensor batches to multiple training processes over ZMQ.

    Handles:
    - Dataset iteration and batch distribution
    - Consumer health monitoring via heartbeat
    - Acknowledgement tracking
    - Rubberbanding for consumer synchronization
    - Automatic cleanup of disconnected consumers
    """

    def __init__(
        self,
        data_loader: Iterator,
        port: int = 5555,
        ack_port: int = 5556,
        heart_ports: tuple[int, int] = (4444, 4445),
        rubber_band_pct: float = 0.02,
        pack_fn: callable = pack,
        consumer_max_buffer_size: int = 10,
    ) -> None:
        """Initialize producer with configuration.

        Args:
            data_loader: Source of tensor batches
            port: Main data transmission port
            ack_port: Consumer acknowledgement port
            heart_ports: (Pub, Sub) ports for heartbeat monitoring
            rubber_band_pct: Max allowed consumer lag as % of dataset
            pack_fn: Function to prepare tensors for transmission
            consumer_max_buffer_size: Max queued batches per consumer
        """
        self.port = port
        self.ack_port = ack_port
        self.heart_ports = heart_ports

        self.data_loader = data_loader
        self.data_loader_len = len(self.data_loader)
        try:
            self.data_loader_iter = iter(self.data_loader)
        except TypeError:
            logger.warning("TensorSocket: DataLoader is already iterable")
            self.data_loader_iter = self.data_loader

        self.pack_fn = pack_fn

        self.index = 0
        self.context = zmq.Context()
        self.consumers = {}
        self.consumer_max_buffer_size = consumer_max_buffer_size

        # Send batches via
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"{LOCALHOST}:{self.port}")

        # Ack
        self.ack_socket = self.context.socket(zmq.SUB)
        self.ack_socket.bind(f"{LOCALHOST}:{self.ack_port}")
        self.ack_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Heartbeat monitor
        self.hb = None
        self.heartbeat_thread = threading.Thread(
            target=self._start_heartbeat, args=(), daemon=True
        )
        self.heartbeat_thread.start()

        while self.hb is None:
            time.sleep(0.2)

        self._heartbeat_monitor_alive = True
        self.heartbeat_monitor_thread = threading.Thread(
            target=self._heartbeat_monitor, args=(), daemon=True
        )
        self.heartbeat_monitor_thread.start()

        # Dataset logic
        self.dataset_is_reset = True
        self.epoch = 0

        # Rubberbanding
        self.rb_buffer = list()
        self.rb_max_len = rubber_band_pct * self.data_loader_len
        self.rb_running = False

        self.active_payloads = []

    def _start_heartbeat(self) -> None:
        """Initialize and start heartbeat monitoring system.

        Sets up ZMQ pub/sub socket pair for tracking consumer connections
        and initializes event loop for async heartbeat processing.
        """
        self.loop = ioloop.IOLoop()
        context = zmq.Context()
        pub = context.socket(zmq.PUB)
        pub.bind(f"{LOCALHOST}:{self.heart_ports[0]}")
        sub = context.socket(zmq.SUB)
        sub.bind(f"{LOCALHOST}:{self.heart_ports[1]}")
        sub.subscribe(b"")

        outstream = zmqstream.ZMQStream(pub, self.loop)
        instream = zmqstream.ZMQStream(sub, self.loop)

        self.hb = HeartBeater(self.loop, outstream, instream)
        self.loop.start()

    def _heartbeat_monitor(self) -> None:
        """Monitor heartbeat in background thread.

        Runs continuously until producer shutdown, checking consumer health.
        """
        while True:
            if not self._heartbeat_monitor_alive:
                break
            time.sleep(1)

    def join(self) -> None:
        """Clean shutdown of producer threads and event loops.

        Stops heartbeat monitoring and joins all background threads.
        """
        self.loop.stop()
        self.heartbeat_thread.join()
        self._heartbeat_monitor_alive = False
        self.heartbeat_monitor_thread.join()

    def __iter__(self) -> Iterator:
        """Return an iterator for the data loader.

        Returns:
            An iterator for the data loader.
        """
        self.data_loader_iter = iter(self.data_loader)
        return self

    def __next__(self) -> tuple:
        """Get the next sample from the data loader.

        Returns:
            The next sample from the data loader.
        """
        return self.get_sample()

    def get_sample(self) -> Optional[tuple]:
        """Core batch distribution method.

        Handles:
        1. Dead consumer cleanup
        2. Epoch boundaries and dataset reset
        3. Consumer synchronization via rubberbanding
        4. Batch transmission and acknowledgement processing

        Returns:
            Tensor batch or None if no active consumers

        Raises:
            StopIteration: At end of epoch
        """
        # clean up dead consumers
        for consumer in list(self.consumers.keys()):
            if consumer not in self.hb.consumers:
                self.consumers.pop(consumer)

        if self.index >= self.data_loader_len:
            self.index = 0
            self.epoch += 1
            self.rb_buffer = []
            raise StopIteration

        # idle when no consumers attached
        elif not len(self.hb.consumers):
            logger.info("No consumers, waiting ...")
            time.sleep(0.5)
            return

        current_batch_index = self.index

        # TODO add flex rubberbanding midrun

        try:
            send_len = False
            expected = [str(x) for x in self.hb.consumers]

            for consumer in expected:
                if str(consumer) not in self.consumers:
                    self.consumers[str(consumer)] = ConsumerProgress(
                        deque(maxlen=self.consumer_max_buffer_size + 1),
                        deque(maxlen=self.consumer_max_buffer_size + 1),
                    )
                    send_len = True

            if send_len:
                self._send_consumer_len()

            # # if we are relatively early in the epoch, allow for new proc to catch up (rubberbanding)
            if (
                (self.rb_buffer)
                and (
                    (min_batch := min([x.batch_max for x in self.consumers.values()]))
                    < current_batch_index
                )
                and (current_batch_index < self.rb_max_len)
            ):
                current_batch_index, data = self.rb_buffer[min_batch]
            else:
                data = next(self.data_loader_iter)

                # add CPU tensors to rubberband buffer TODO: make sure not gpu and dont always pull from this
                self.rb_buffer.append((current_batch_index, data))

                # if buffer full, pop from end
                if len(self.rb_buffer) > self.rb_max_len:
                    _ = self.rb_buffer.pop(-1)

                self.index += 1

                # current_batch_index, data = self.rb_buffer[
                #     min([c.batch_max for c in self.consumers.values()])
                # ]  # TODO: fix when rband is full?

            expected = [x for x in expected]
            payload = self._broadcast(self.epoch, current_batch_index, data)
            self._handle_acks(expected, current_batch_index, payload)

        except StopIteration:
            self.socket.send_pyobj({"stop_iteration": 1})  # TODO: fix
            raise StopIteration

    def _broadcast(
        self, current_epoch: int, current_batch_index: int, data: tuple
    ) -> dict:
        """Broadcast tensor batch to all connected consumers.

        Args:
            current_epoch: Training epoch number
            current_batch_index: Current batch index in epoch
            data: Tensor batch to transmit

        Returns:
            Dict containing packed tensors and metadata

        Logs progress every 100 batches.
        """
        payload = dict(
            data=self.pack_fn(data),
            current_epoch=current_epoch,
            current_batch_index=current_batch_index,
        )

        # self.active_payloads.append(payload)

        if current_batch_index % 100 == 0:
            logger.info(
                f"current_batch_index {current_batch_index}, "
                f"buffer size: {len(self.rb_buffer)}"
            )

        self.socket.send_pyobj(payload)
        return payload

    def _handle_acks(
        self, expected: list, current_batch_index: int, payload: dict
    ) -> None:
        """Process acknowledgements from consumers.

        Args:
            expected: List of consumer IDs to wait for
            current_batch_index: Index of batch being acknowledged
            payload: Transmitted data payload

        Times out after 10 seconds if consumers unresponsive.
        Maintains synchronization by tracking accepted batches.
        """
        while True:
            # received all Acks, can go to next batch
            if not len(expected):
                return

            if self.ack_socket.poll(10000, zmq.POLLIN):
                (
                    consumer_index,
                    batch_max,
                    accepted,
                ) = (
                    self.ack_socket.recv_multipart()
                )  # wait for consumer acknowledgement

                batch_max = int(batch_max)
                consumer_index = str(consumer_index)

                logger.info(
                    f"Consumer: {consumer_index}, "
                    f"batch count: {batch_max}, "
                    f"total batches: {self.data_loader_len}"
                )

                if consumer_index in expected:
                    expected.remove(consumer_index)

                if (
                    accepted
                    == b"1"
                    # batch_max == self.consumers[consumer_index].batch_max
                ):  #  TODO: missing safeguard
                    print(consumer_index, batch_max)
                    self.consumers[consumer_index].add_batch(batch_max, payload)

            else:
                logger.info(
                    "Timeout on Ack, assuming consumer is dead",
                    len(self.hb.consumers),
                    expected,
                )
                expected = {}

    def __len__(self) -> int:
        """Get the length of the data loader.

        Returns:
            The number of batches in the data loader.
        """
        return self.data_loader_len

    def _send_consumer_len(self) -> None:
        """Send dataset metadata to newly connected consumers.

        Transmits:
        - Total number of batches in dataset
        - Maximum allowed buffer size
        """
        logger.info("Sending data loader length")
        self.socket.send_pyobj(
            {
                "data_loader_len": self.__len__(),
                "max_buffer_size": self.consumer_max_buffer_size,
            }
        )
