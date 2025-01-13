# Based on https://github.com/zeromq/pyzmq/blob/main/examples/heartbeat/heartbeater.py

import time
import zmq
from typing import Set

from threading import Thread
from tornado import ioloop
from zmq.eventloop import zmqstream


class HeartBeater:
    """Monitors consumer health through periodic heartbeat messages.

    Manages pub/sub streams for heartbeat signals and tracks connected consumers.
    Detects disconnections and new connections through periodic checks.
    """

    def __init__(
        self,
        loop: ioloop.IOLoop,
        pingstream: zmqstream.ZMQStream,
        pongstream: zmqstream.ZMQStream,
        period: int = 500,
    ) -> None:
        """Initialize heartbeat monitor.

        Args:
            loop: Tornado IO loop for async operations
            pingstream: ZMQ stream for sending heartbeats
            pongstream: ZMQ stream for receiving responses
            period: Milliseconds between heartbeats
        """
        self.loop = loop
        self.period = period

        self.pingstream = pingstream
        self.pongstream = pongstream
        self.pongstream.on_recv(self.handle_pong)

        self.hearts: Set = set()
        self.responses: Set = set()
        self.lifetime = 0
        self.tic = time.monotonic()

        self.caller = ioloop.PeriodicCallback(self.beat, period)
        self.caller.start()

        self.heart_progress = dict()
        self.heart_batch_size = dict()

    def beat(self) -> None:
        """Send heartbeat and process responses.

        Checks for:
        - New consumers that have connected
        - Existing consumers that failed to respond
        - Updates tracking sets accordingly
        """
        toc = time.monotonic()
        self.lifetime += toc - self.tic
        self.tic = toc

        goodhearts = self.hearts.intersection(self.responses)
        heartfailures = self.hearts.difference(goodhearts)
        newhearts = self.responses.difference(goodhearts)

        for heart in newhearts:
            self.handle_new_heart(heart)
        for heart in heartfailures:
            self.handle_heart_failure(heart)
        self.responses = set()
        self.pingstream.send(str(self.lifetime).encode("utf-8"))

    def handle_new_heart(self, heart: bytes) -> None:
        """Register new consumer connection.

        Args:
            heart: Consumer identifier
        """
        self.hearts.add(heart)

    def handle_heart_failure(self, heart: bytes) -> None:
        """Remove failed consumer connection.

        Args:
            heart: Consumer identifier
        """
        self.hearts.remove(heart)

    def handle_pong(self, msg: list) -> None:
        """Process heartbeat response from consumer.

        Args:
            msg: [consumer_id, progress, lifetime] message
        """
        if float(msg[3]) == self.lifetime:
            self.responses.add(msg[0])
            self.heart_progress[str(msg[0])] = int.from_bytes(msg[1], "big")
            self.heart_batch_size[str(msg[0])] = int.from_bytes(msg[2], "big")
        else:
            pass

    @property
    def consumers(self):
        return [str(x) for x in self.hearts]


class Heart(Thread):
    """Consumer-side heartbeat sender.

    Runs in separate thread to maintain connection with HeartBeater.
    Sends periodic updates including processing progress.
    """

    def __init__(
        self,
        consumer=None,
        uuid="default",
        batch_size=0,
        port_in="tcp://localhost:10112",
        port_out="tcp://localhost:10113",
        *args,
        **kwargs,
    ) -> None:
        """Initialize heartbeat sender.

        Args:
            consumer: Associated consumer instance
            uuid: Unique identifier for this consumer
            port_in: Port for receiving heartbeats
            port_out: Port for sending responses
        """
        super(Heart, self).__init__(*args, **kwargs)
        self._uuid = uuid if isinstance(uuid, bytes) else str(uuid).encode("utf-8")
        self._port_in = port_in
        self._port_out = port_out

        self.consumer = consumer
        self.batch_size = batch_size

    def run(self) -> None:
        """Main heartbeat loop.

        Continuously:
        1. Listens for heartbeat pings
        2. Responds with current progress
        3. Handles connection issues
        """
        self._ctx = zmq.Context()
        self._in = self._ctx.socket(zmq.SUB)
        self._in.connect(self._port_in)
        self._in.subscribe(b"")

        self._out = self._ctx.socket(zmq.PUB)
        self._out.connect(self._port_out)

        while True:
            try:
                event = self._in.poll(timeout=3000)
                if event == 0:
                    pass  # Failed
                else:
                    item = self._in.recv_multipart()  # flags=zmq.NOBLOCK)
                    self._out.send_multipart(
                        [
                            self._uuid,
                            self.consumer.batch_count.to_bytes(8, "big"),
                            self.consumer.batch_size.to_bytes(8, "big"),
                        ]
                        + item
                    )
            except zmq.ZMQError:
                time.sleep(0.01)  # Wait a little for next item to arrive
