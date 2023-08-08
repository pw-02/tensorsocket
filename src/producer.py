import zmq
import torch
from zmq.utils.monitor import recv_monitor_message
from zmq.eventloop import zmqstream
from zmq.eventloop.ioloop import ZMQIOLoop
loop = ZMQIOLoop()
loop.install()

class TensorProducer:
    def __init__(self, data_loader, port, ack_port, workers=1):
        self.port = port
        self.ack_port = ack_port
        self.data_loader = iter(data_loader)
        self.data_loader_len = len(data_loader)
        self.workers = workers
        self.idx = 0
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:%s" % self.port)

        # Ack
        self.ack_socket = self.context.socket(zmq.SUB)
        self.ack_socket.bind("tcp://*:%s" % self.ack_port)
        self.ack_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.ack_count = 0

        # Monitor on Ack sockets
        #self.monitor_socket = self.ack_socket.get_monitor_socket(events=zmq.EVENT_HANDSHAKE_SUCCEEDED)

        #self.wait_for_n_subscribers(workers)
        events_socket = self.ack_socket.get_monitor_socket(events=zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED | zmq.EVENT_ACCEPTED)
        self._mon_stream = zmqstream.ZMQStream(events_socket, io_loop=loop)
        out = self._mon_stream.on_recv(self._on_mon)
        print("out:", out)

    def set_workers(self, new_workers):
        self.workers = new_workers

    def __iter__(self):
        return self
    
    def wait_for_n_subscribers(self, n_subscribers):
        connections = 0
        events_socket = self.ack_socket.get_monitor_socket(events=zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED | zmq.EVENT_ACCEPTED)
        while connections < n_subscribers:
            msg = recv_monitor_message(events_socket)  # this will block until a handshake was successful
            print(msg)
            connections += 1
        print("waiting done")

    def _on_mon(self, msg):
        print("_on_mon")
        ev = zmq.utils.monitor.parse_monitor_message(msg)
        event = ev['event']
        endpoint = ev['endpoint']
        print(ev)
        return ev
    
    def __next__(self):
        if self.idx >= self.data_loader_len:
            raise StopIteration
        else:
            inputs, labels = next(self.data_loader)
            inputs.to("cuda:0")
            labels.to("cuda:0")
            inputs_payload = self._create_payload(inputs)
            labels_payload = self._create_payload(labels)
            payload = {"inputs": inputs_payload, "labels": labels_payload}

            self.socket.send_pyobj(payload)

            while True:
                #msg = recv_monitor_message(self.monitor_socket)
                #print(msg)

                _ = self.ack_socket.recv_string() # wait for worker/consumer acknowledgement
                self.ack_count += 1

                if self.ack_count == self.workers:
                    self.ack_count = 0
                    break

            self.idx += 1

    def _create_payload(self, tensor):
        storage = tensor.untyped_storage()
        (storage_device, storage_handle, storage_size_bytes, storage_offset_bytes,
                    ref_counter_handle, ref_counter_offset, event_handle, event_sync_required) = storage._share_cuda_()

        payload = {
            "dtype": tensor.dtype,
            "tensor_size": tuple(tensor.size()),
            "tensor_stride": tensor.stride(),
            "tensor_offset": tensor.storage_offset(),
            "storage_cls": type(storage),
            "storage_device": storage_device,
            "storage_handle": storage_handle,
            "storage_size_bytes": int(storage_size_bytes),
            "storage_offset_bytes":storage_offset_bytes,
            "requires_grad": False,
            "ref_counter_handle": ref_counter_handle,
            "ref_counter_offset": ref_counter_offset,
            "event_handle": event_handle,
            "event_sync_required": event_sync_required,
        }
        return payload

    def __len__(self):
        return self.data_loader_len