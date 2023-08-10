import zmq
import torch
from zmq.eventloop import zmqstream
from tornado import ioloop
from zmq_utils.heartbeat import HeartBeater
import threading
import time
import random

class TensorProducer:
    def __init__(self, dataset, loader_fn, port, ack_port, worker_count=1):
        self.port = port
        self.ack_port = ack_port
        self.dataset = dataset
        self.loader_fn = loader_fn
        self._reset_data_loader()
        self.data_loader_len = len(self.data_loader)
        self.worker_count = worker_count
        self.idx = 0
        self.context = zmq.Context()

        # Send batches via
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:%s" % self.port)

        # Ack
        self.ack_socket = self.context.socket(zmq.SUB)
        self.ack_socket.bind("tcp://*:%s" % self.ack_port)
        self.ack_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.ack_count = 0

        # Heartbeat monitor
        heartbeat_thread = threading.Thread(target=self.start_heartbeat, args=())
        heartbeat_thread.start()
        time.sleep(1)

        # Dataset logic
        self.dataset_is_reset = True
        self.workers = {} # hold batch index for each worker

    def start_heartbeat(self):
        loop = ioloop.IOLoop()
        context = zmq.Context()
        pub = context.socket(zmq.PUB)
        pub.bind('tcp://127.0.0.1:4444')
        router = context.socket(zmq.ROUTER)
        router.bind('tcp://127.0.0.1:4445')

        outstream = zmqstream.ZMQStream(pub, loop)
        instream = zmqstream.ZMQStream(router, loop)

        self.hb = HeartBeater(loop, outstream, instream)
        loop.start()

    def set_worker_count(self, new_worker_count):
        self.worker_count = new_worker_count

    def __iter__(self):
        return self

    def __next__(self):
        # end of data loader
        if self.idx >= self.data_loader_len:
            raise StopIteration
        
        # idle when no workers attached
        elif len(self.hb.hearts) == 0:
            print("No workers, waiting ...")
            time.sleep(0.5)
            # currently does not work properly.
            # dataset seed is updated and only iterator can be reset deterministically
            if not self.dataset_is_reset:
                self._reset_data_loader()
                self.dataset_is_reset = True
                self.idx = 0

        # new workers joined in
        # reset data loader and update number of workers
        elif len(self.hb.hearts) > self.worker_count:
            self._reset_data_loader()
            self.dataset_is_reset = True
            self.set_worker_count(len(self.hb.hearts))
            self.idx = 0

        else:
            start = time.time()
            if len(self.hb.hearts) != self.worker_count:
                self.set_worker_count(len(self.hb.hearts))
            inputs, labels = next(self.data_loader_iter)
            inputs = inputs.to("cuda:0")
            labels = labels.to("cuda:0")
            inputs_payload = self._create_payload(inputs)
            labels_payload = self._create_payload(labels)
            payload = {"batch_index": self.idx, "inputs": inputs_payload, "labels": labels_payload}

            self.socket.send_pyobj(payload)

            while True:
                if self.ack_socket.poll(1000, zmq.POLLIN):
                    consumer_idx, batch_count = self.ack_socket.recv_multipart() # wait for worker/consumer acknowledgement
                    #consumer_idx, batch_count = int(consumer_idx), int(batch_count)
                    print(f"Consumer: {consumer_idx}, batch count: {batch_count}")
                    self.ack_count += 1
                else:
                    print("Timeout on Ack, assuming worker is dead")
                    self.worker_count -= 1

                if self.ack_count == self.worker_count:
                    self.ack_count = 0
                    break

            self.idx += 1
            self.dataset_is_reset = False

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

    def _reset_data_loader(self):
        random.seed(1337)
        torch.manual_seed(1337)
        self.data_loader = self.loader_fn(self.dataset)
        self.data_loader_iter = iter(self.data_loader)