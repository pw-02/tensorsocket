import logging
import threading
import time
import zmq

from torch import Tensor, device, cuda
from tornado import ioloop
from zmq.eventloop import zmqstream

from .heartbeat import HeartBeater
from .payload import TensorPayload

logger = logging.getLogger("tensorsocket")
logger.setLevel(logging.WARNING)
LOCALHOST = "tcp://*"


def process_tensor(tensor):
    if isinstance(tensor, Tensor):
        if cuda.is_available():
            tensor = tensor.to(device="cuda")
        tensor = TensorPayload(tensor)
    return tensor


def pack(data: tuple):
    return tuple((process_tensor(t) for t in data))


class TensorProducer:
    def __init__(
        self,
        data_loader: object,
        port: int = 5555,
        ack_port: int = 5556,
        heart_ports: (int, int) = (4444, 4445),
        rubber_band_pct: int = 0.02,
        pack_fn=pack,
    ):
        """
        Data loader that sends inputs and labels over tcp to training processes (consumers).

        Args:
            data_loader (object): Data loader to wrap around. Should be iterable.
            port (int, optional): Data transmission port. Defaults to 5555.
            ack_port (int, optional): Acknowledgement port. Defaults to 5556.
            heart_ports (int, int, optional): Life pulse ports. Defaults to (4444, 4445).
            rubber_band_pct (int, optional): Maximum allowed distance between consumers, in percent of training dataset size. Defaults to 0.02.
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

    def _start_heartbeat(self):
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

    def _heartbeat_monitor(self):
        while True:
            if not self._heartbeat_monitor_alive:
                break
            time.sleep(1)

    def join(self):
        self.loop.stop()
        self.heartbeat_thread.join()
        self._heartbeat_monitor_alive = False
        self.heartbeat_monitor_thread.join()

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        return self

    def __next__(self):
        return self.get_sample()

    def get_sample(self):
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
                    self.consumers[str(consumer)] = 0
                    send_len = True

            if send_len:
                self._send_consumer_len()

            # # if we are relatively early in the epoch, allow for new proc to catch up (rubberbanding)
            if (
                (self.rb_buffer)
                and ((min_batch := min(self.consumers.values())) < current_batch_index)
                and (current_batch_index < self.rb_max_len)
            ):
                current_batch_index, data = self.rb_buffer[min_batch]
            else:
                data = next(self.data_loader_iter)

                # add CPU tensors to rubberband buffer TODO: make sure not gpu
                self.rb_buffer.append((current_batch_index, data))

                # if buffer full, pop from end
                if len(self.rb_buffer) > self.rb_max_len:
                    _ = self.rb_buffer.pop(-1)

                self.index += 1

                current_batch_index, data = self.rb_buffer[min(self.consumers.values())]

            expected = [x for x in expected]
            self._broadcast(self.epoch, current_batch_index, data)
            self._handle_acks(expected)

        except StopIteration:
            self.socket.send_pyobj({"stop_iteration": 1})  # TODO: fix
            raise StopIteration

    def _broadcast(
        self,
        current_epoch: int,
        current_batch_index: int,
        data: tuple,
    ):

        payload = dict(
            data=self.pack_fn(data),
            current_epoch=current_epoch,
            current_batch_index=current_batch_index,
        )

        if current_batch_index % 100 == 0:
            logger.info(
                f"current_batch_index {current_batch_index}, "
                f"buffer size: {len(self.rb_buffer)}"
            )

        self.socket.send_pyobj(payload)

    def _handle_acks(self, expected: list):
        while True:
            # received all Acks, can go to next batch
            if not len(expected):
                return

            if self.ack_socket.poll(10000, zmq.POLLIN):
                (
                    consumer_index,
                    batch_count,
                ) = (
                    self.ack_socket.recv_multipart()
                )  # wait for consumer acknowledgement

                batch_count = int(batch_count)
                consumer_index = str(consumer_index)

                logger.info(
                    f"Consumer: {consumer_index}, "
                    f"batch count: {batch_count}, "
                    f"total batches: {self.data_loader_len}"
                )

                if consumer_index in expected:
                    if (
                        batch_count == self.consumers[consumer_index]
                    ):  #  TODO: missing safeguard
                        expected.remove(consumer_index)
                        self.consumers[consumer_index] = batch_count + 1
                    else:
                        expected.remove(consumer_index)

            else:
                logger.info(
                    "Timeout on Ack, assuming consumer is dead",
                    len(self.hb.consumers),
                    expected,
                )
                expected = {}

    def __len__(self):
        return self.data_loader_len

    def _send_consumer_len(self):
        logger.info("Sending data loader length")
        self.socket.send_pyobj({"data_loader_len": self.__len__()})
