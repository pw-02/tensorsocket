import sys
import zmq
import time
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

port = "5556"
ack_port = "5557"
if len(sys.argv) > 1:
    port = sys.argv[1]

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:%s" % port)
socket.setsockopt_string(zmq.SUBSCRIBE, "")

# Ack
ack_socket = context.socket(zmq.PUB)
ack_socket.connect("tcp://localhost:%s" % ack_port)


class TensorConsumer:
    def __init__(self, port, ack_port):
        self.port = port
        self.ack_port = ack_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:%s" % self.port)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Ack
        self.ack_socket = self.context.socket(zmq.PUB)
        self.ack_socket.connect("tcp://localhost:%s" % self.ack_port)

    def __iter__(self):
        return self

    def __next__(self):
        cuda_tensor_info = self.socket.recv_pyobj()
        inputs = cuda_tensor_info["inputs"]
        labels = cuda_tensor_info["labels"]

        inputs = rebuild_cuda_tensor(torch.Tensor, **inputs)
        labels = rebuild_cuda_tensor(torch.Tensor, **labels)
        batch = (inputs, labels)
        self.ack_socket.send_string("received batch!")
        return batch