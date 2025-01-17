import time

from tensorsocket.consumer import TensorConsumer

"""
A simple example on how to implement a data receiver (consumer) in your training script.
The TensorConsumer class directly replaces the data loader in the training script.
Please check out simple_producer.py for the paired producer script.
"""

consumer = TensorConsumer("5556", "5557", batch_size=16)
for i, batch in enumerate(consumer):
    (inputs, labels) = batch
    if labels != None:
        if True:
            print(f"I:{i:0>7} -", labels[0], consumer.epoch, len(labels))
            # time.sleep(0.1)
            pass
    else:
        print("Waiting ...")

print("Finished")
