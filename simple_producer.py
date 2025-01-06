import torch

from tensorsocket.producer import TensorProducer

"""
A simple example on how to implement a data sender (producer) in your training script.
The TensorProducer class wraps around your data loader.
Please check out simple_consumer.py for the paired consumer script.
"""


class DummyLoader:
    def __init__(self, length=1000000):
        self.length = length
        self.id = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        a, b = self.id * torch.ones((100, 200, 10)), self.id * torch.ones((10,))

        self.id += 1
        return a, b


data_loader = DummyLoader()


producer = TensorProducer(data_loader, "5556", "5557", rubber_band_pct=0.2)

for epoch in range(10):
    for i, _ in enumerate(producer):
        if not i % 100:
            pass
producer.join()
print("finished")
