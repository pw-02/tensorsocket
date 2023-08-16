import zmq
import torch
from zmq.eventloop import zmqstream
from tornado import ioloop
from zmq_utils.heartbeat import HeartBeater
import threading
import time
import random

class TensorProducer:
    def __init__(self, data_loader, port, ack_port, worker_count=1, rubber_band_pct=0.2):
        self.port = port
        self.ack_port = ack_port
        self.data_loader = data_loader
        self.data_loader_len = len(self.data_loader)
        self.data_loader_iter = iter(self.data_loader)
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
        self.batch_progress = 0 # largest batch index we have made it to thus far
        self.rb_pct = rubber_band_pct # how far off we allow a new process to be
                                               # from the furthest progressed process
        self.epoch = 0

        # Rubberbanding
        self.rb_buffer = list()
        self.rb_max_len = 0
        #self.rb_max_len = (self.rb_pct*self.data_loader_len)
        self.empty_rb_buffer = False

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
            self.idx = 0
            self.batch_progress = 0
            self.epoch += 1
            self.rb_buffer = list()
            raise StopIteration
        
        # idle when no workers attached
        elif len(self.hb.hearts) == 0:
            print("No workers, waiting ...")
            time.sleep(0.5)

        else:
            current_batch_idx = self.idx
            # if we are relatively early in the epoch, allow for new proc to catch up
            if (self.worker_count > 0) and (len(self.hb.hearts) > self.worker_count) and (current_batch_idx < self.rb_max_len):
                print("Empty RB buffer")
                self.empty_rb_buffer = True

            if len(self.hb.hearts) != self.worker_count:
                self.set_worker_count(len(self.hb.hearts))

            inputs, labels = next(self.data_loader_iter)
            # add CPU tensors to rubber band buffer
            if not self.empty_rb_buffer:
                self.rb_buffer.append((current_batch_idx, inputs, labels))

            # if buffer full, pop from end
            if len(self.rb_buffer) > self.rb_max_len:
                _ = self.rb_buffer.pop(-1)

            if self.empty_rb_buffer:
                current_batch_idx, inputs, labels = self.rb_buffer.pop(0)
            inputs_gpu = inputs.to(torch.device('cuda'))
            labels_gpu = labels.to(torch.device('cuda'))

            inputs_payload = self._create_payload(inputs_gpu)
            labels_payload = self._create_payload(labels_gpu)
            payload = {"current_epoch": self.epoch, "current_batch_index": current_batch_idx, 
                       "inputs": inputs_payload, "labels": labels_payload}
            #print(f"current_batch_index {current_batch_idx}, max_batch_idx {self.batch_progress}, buffer size: {len(self.rb_buffer)}")

            self.socket.send_pyobj(payload)

            while True:
                if self.ack_socket.poll(1000, zmq.POLLIN):
                    consumer_idx, batch_count = self.ack_socket.recv_multipart() # wait for worker/consumer acknowledgement
                    #print(f"Consumer: {consumer_idx}, batch count: {batch_count}, total batches: {self.data_loader_len}")
                    self.ack_count += 1
                else:
                    print("Timeout on Ack, assuming worker is dead")
                    self.worker_count -= 1

                # received all Acks, can go to next batch
                if self.ack_count == self.worker_count:
                    self.ack_count = 0
                    break

            if not self.empty_rb_buffer:
                self.idx += 1

            if len(self.rb_buffer) == 0:
                self.empty_rb_buffer = False

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
