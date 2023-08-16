# Based on https://github.com/zeromq/pyzmq/blob/main/examples/heartbeat/heartbeater.py

import time
from typing import Set

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
        if float(msg[1]) == self.lifetime:
            self.responses.add(msg[0])
        else:
            pass
