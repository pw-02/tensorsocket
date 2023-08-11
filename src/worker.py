import sys
import zmq
import time
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from zmq import devices
import uuid

class TensorConsumer:
    def __init__(self, port, ack_port, rubber_band_pct=0.1):
        self.port = port
        self.ack_port = ack_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:%s" % self.port)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.consumer_id = uuid.uuid4()

        # Ack
        self.ack_socket = self.context.socket(zmq.PUB)
        self.ack_socket.connect("tcp://localhost:%s" % self.ack_port)

        # Heartbeat
        self.dev = devices.ThreadDevice(zmq.FORWARDER, zmq.SUB, zmq.DEALER)
        self.dev.setsockopt_in(zmq.SUBSCRIBE, b"")
        self.dev.connect_in('tcp://127.0.0.1:4444')
        self.dev.connect_out('tcp://127.0.0.1:4445')
        self.dev.start()

        # Logic
        self.batch_count = 0
        self.rubber_band_pct = rubber_band_pct # how far off we allow a new process to be
                                               # from the furthest progressed process
        self.epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
        cuda_tensor_info = self.socket.recv_pyobj()
        current_epoch = cuda_tensor_info["current_epoch"]
        batch_idx = cuda_tensor_info["current_batch_index"]
        max_batch_idx = cuda_tensor_info["max_batch_index"]
        total_batches = cuda_tensor_info["total_batches"]
        inputs = cuda_tensor_info["inputs"]
        labels = cuda_tensor_info["labels"]

        if current_epoch != self.epoch:
            self.epoch = current_epoch
            self.batch_count = 0

        print(f"Epoch: {self.epoch}, Diff: {abs(self.batch_count-max_batch_idx)}, batch count: {self.batch_count}, limit: {(self.rubber_band_pct*total_batches)}")
        if abs(self.batch_count-max_batch_idx) < (self.rubber_band_pct*total_batches):
            if batch_idx == self.batch_count:
                inputs = rebuild_cuda_tensor(torch.Tensor, **inputs)
                labels = rebuild_cuda_tensor(torch.Tensor, **labels)
                batch = (inputs, labels)
                self.batch_count += 1
            else:
                batch = (None, None)
        else:
            batch = (None, None)
        self.ack_socket.send_multipart([
            bytes(str(self.consumer_id).encode("utf-8")),
            bytes(str(self.epoch).encode("utf-8")),
            bytes(str(self.batch_count).encode("utf-8")),
        ])
        return batch