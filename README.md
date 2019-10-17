# Priority Memory

A prioritized sampling tool for priority memory replay.

The implementation is based on sum tree, i.e. segmentation tree.

- Set the priority of each sample at anytime. 
- When you do not know the priorities of the samples, you can append 
  them to the buffer, and they will show up in the next sampling batch.
- When the buffer is full, it'll drop the samples with lowest priority.

The time complexity for sampling a batch with batch size m 
from a dataset with n samples is O(mlogn), for setting priority 
for the batch is O(mlogn).


# Usage

> pip install priority_memory

```python

from priority_memory import FastPriorReplayBuffer

buffer = FastPriorReplayBuffer(8000)
buffer.append(features=[0.1, 0.1, 0.1], prior=1)
buffer.append(features=[0.2, 0.2, 0.2], prior=2)
buffer.append(features=[0.3, 0.3, 0.3], prior=3)
buffer.append(features=[0.4, 0.4, 0.4], prior=4)
indexes, data, weights = buffer.sample_with_weights(batch_size=2)

mae = [10, 20]
buffer.set_weights(indexes, mae)

```


