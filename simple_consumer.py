import time

from tensorsocket.consumer import TensorConsumer

"""
A simple example on how to implement a data receiver (consumer) in your training script.
The TensorConsumer class directly replaces the data loader in the training script.
Please check out simple_producer.py for the paired producer script.
"""

consumer = TensorConsumer("5556", "5557")
for i, batch in enumerate(consumer):
    b, (inputs, labels) = batch
    if labels != None:
        # if not i % 100:
        if True:
            print(f"I:{i:0>7} -", b, labels[0])
            time.sleep(0.1)
            pass
            # print("\n", f"I:{i:0>7} -", labels[:5], "\n")
    else:
        print("Waiting ...")
    time.sleep(0.05)

print("Finished")
