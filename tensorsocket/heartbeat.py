# Based on https://github.com/zeromq/pyzmq/blob/main/examples/heartbeat/heartbeater.py

import time
import zmq
from typing import Set

from threading import Thread
from tornado import ioloop
from zmq.eventloop import zmqstream


class HeartBeater:
    """A basic HeartBeater class
    pingstream: a PUB stream
    pongstream: an ROUTER stream"""

    def __init__(
        self,
        loop: ioloop.IOLoop,
        pingstream: zmqstream.ZMQStream,
        pongstream: zmqstream.ZMQStream,
        period: int = 500,
    ):
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

    def beat(self):
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

    def handle_new_heart(self, heart):
        self.hearts.add(heart)

    def handle_heart_failure(self, heart):
        self.hearts.remove(heart)

    def handle_pong(self, msg):
        if float(msg[2]) == self.lifetime:
            self.responses.add(msg[0])
            self.heart_progress[msg[0]] = msg[1]
        else:
            pass

    @property
    def consumers(self):
        return [str(x) for x in self.hearts]


class Heart(Thread):

    def __init__(
        self,
        consumer=None,  # TODO: change
        uuid="default",
        port_in="tcp://localhost:10112",
        port_out="tcp://localhost:10113",
        *args,
        **kwargs,
    ):
        super(Heart, self).__init__(*args, **kwargs)
        self._uuid = uuid if isinstance(uuid, bytes) else str(uuid).encode("utf-8")
        self._port_in = port_in
        self._port_out = port_out

        self.consumer = consumer

    def run(self):
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
                        [self._uuid, self.consumer.batch_count.to_bytes(2, "big")]
                        + item
                    )
            except zmq.ZMQError:
                time.sleep(0.01)  # Wait a little for next item to arrive
