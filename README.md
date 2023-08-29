# tensorsocket
Share PyTorch tensors over ZMQ sockets

## Installation

Install as module. Should be installed in a project such as [RAD/data-sharing](https://github.com/Resource-Aware-Data-systems-RAD/data-sharing)

**From source**

From the root of the tensorsocket directory, install it with pip:

```shell
$ pip install .
```

**From PyPi**

```shell
$ pip install tensorsocket
```

## Usage

tensorsocket works by exposing batches of data, represented as PyTorch tensors, on sockets that training processes can access. This allows for minimizing redundancy of training data during collocated tasks such as hyper-parameter tuning. Training with tensorsocket builds on the concept of a producer-consumer relationship, where the following example code shows how the producer wraps around an arbitrary data loader object. As with nested epoch-batch loops, one can iterate over the producer in the same manner as iterating over a data loader.

The use of tensorsocket relies on a `TensorProducer` and `TensorConsumer`. The `TensorProducer` can be used as is, however the `TensorConsumer` needs to be embedded in a class that exposes the same functionality as a PyTorch data loader, as shown in the `SharedLoader` example class, below:

```python
# shared_data_loader.py

from tensorsocket import TensorConsumer

class SharedLoader(object):

    def __init__(self, port="5556", ack_port="5557"):
        self.consumer = TensorConsumer(port, ack_port)
        self.counter = 0

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < self.__len__():
            self.counter += 1
            return next(self.consumer)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.consumer)

```

Using the `TensorProducer` requires next to no additional implementation, apart from the original data loader. 

```python
# producer.py

data_loader = DataLoader(dataset)

producer = TensorProducer(data_loader, port="5556", ack_port="5557")

for _ in range(epochs):
        for _ in producer:
            pass
producer.join()
```

Given the `SharedLoader` with the `TensorConsumer` class, it is straightforward to modify a training script to fetch batches of data from the shared loader, rather than using the process-specific data loader, which is created for each collocated training job.

```python
# consumer.py (or train.py)
from ... import SharedLoader

...

if not use_shared_loader:
    data_loader = create_loader(...)
else:
    data_loader = SharedLoader()

...

for batch_idx, (input, target) in enumerate(data_loader):
    output = model(input)
    ...

```

## Features