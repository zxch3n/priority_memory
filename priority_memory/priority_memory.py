import numpy as np
from numpy import ndarray
import random
import warnings
from math import log, ceil
from queue import Queue
from .index_priority_queue import QNode, IndexPriorityQueue
from typing import Tuple, List, Union
from sklearn.svm import SVC


class FastPriorReplayBuffer:
    """Fast priority experience replay buffer.

    The implementation is based on sum tree, or segmentation tree.

    - Set the priority of each sample at anytime.
    - When you do not know the priority of the sample, you can append
      them to the buffer, and they will show up in the next sampling batch.
    - When the buffer is full, drop the samples with lowest priority.

    The time complexity for sampling a batch with batch size m
    from a dataset with n samples is O(mlogn), for setting priority
    for the batch is O(mlogn).

    Parameters
    ----------
    buffer_size: int, optional (default=8000)
        the maximum buffer size

    alpha: float, optional (default=1.0)
        The smooth term for setting the priority.
        `actual_p = input_p ** alpha`

    min_prior: float, optional (default=None)
        The minimum priority.

    max_prior: float, optional (default=None)
        The maximum priority.

    initial_b: float, optional (default=0.4)
        The smooth term for the sample weights.

            `sample weight = (p * n) ** (-b)`

        Where p is the sampling probability of that sample,
        n is the total sample number.

    total_episode: int, optional (default=2000)
        Total episode num.
        It is only used to get the increment for b value after every episode.

    """
    def __init__(self,
                 buffer_size: int=8000,
                 alpha: float =1.0,
                 min_prior: Union[float, None]=None,
                 max_prior: Union[float, None]=None,
                 initial_b: float=0.4,
                 total_episode: int=2000):
        self.buffer_size = buffer_size
        self.buff = SegmentTree(buffer_size, use_max_space=True)
        self.alpha = alpha
        self.min_prior = min_prior
        self.max_prior = max_prior
        self.initial_b = initial_b
        self._b = initial_b
        self.b_increment_per_episode = (1 - initial_b) / total_episode
        self._len = None

    def append(self, features: Union[List, Tuple, ndarray], prior: [float, int, None]=None):
        """
        Append a row of value to the buffer.

        :param features: a row of features.
        :param prior:
            The importance/priority of this sample.

            Prior can be None, and this sample will be put to
            a buffer queue and be appended to the next sampling
            batch when have chance.
        """
        if len(features) == 0:
            raise ValueError('Arguments must be nonzero')

        if self._len is None:
            self._len = len(features)
        elif self._len != len(features):
            raise ValueError('Arguments length must be the same')

        if prior is None:
            self.buff.put_unknown_p(features)
        else:
            self.buff.put(features, prior)

    def sample(self, batch_size: int=32) -> Tuple[ndarray, ndarray]:
        """
        Get a batch of samples with batch size with weights.

        :param batch_size: default value is 32
        :return: index_list, batch
        """
        chosen_slice, batch, prob = self.buff.sample_batch(batch_size)
        return chosen_slice, batch

    def sample_with_weights(self, batch_size=32) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Get a batch of samples with batch size with weights.

        :param batch_size: batch size, default value is 32
        :return: index_list, batch, weights
        """
        chosen_slice, batch, prob = self.buff.sample_batch(batch_size)
        n = len(self.buff)
        return chosen_slice, batch, np.power(prob * n, -self._b)

    def set_weights(self,
                    indexes: Union[List[int], ndarray],
                    priors: List[float]):
        """
        Set the weights of the samples according to the index returned by `sample_with_weights`

        :param indexes: the indexes of the samples
        :param priors: the priorities of each samples
        """
        for index, p in zip(indexes, priors):
            if self.min_prior is not None:
                p = max(self.min_prior, p)
            if self.max_prior is not None:
                p = min(self.max_prior, p)
            p = p ** self.alpha
            self.buff.set_p(index, p)

    def update_after_episode(self):
        """
        You should invoke this method after every episode is done.

        This is for increase_b method only.
        It's stupid, I know.
        """
        self.increase_b()

    def increase_b(self):
        """
        Increase the b value, which is used to smooth the loss weights value.

            sample weight = (p * n) ** (-b)

            Where p is the sampling probability of that sample,
            n is the total sample number.

        """
        self._b = min(1.0, self._b + self.b_increment_per_episode)

    def __len__(self):
        return len(self.buff)


class Node:
    def __init__(self, data, p):
        if p < 0:
            raise ValueError('p cannot be assign to negative value')
        self.dat = data
        self.p = p

    def __lt__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.p < other
        return self.p < other.p

    def __gt__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.p > other
        return self.p > other.p

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.p == other.p

    def __ne__(self, other):
        if not isinstance(other, Node):
            return False
        return not self == other

    def __repr__(self):
        return f'Node({self.dat}, {self.p})'

    def __add__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self.p + other
        return self.p + other.p

    def __radd__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self.p + other
        return self.p + other.p

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self.p / other
        return self.p / other.p

    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self.p - other
        return self.p - other.p

    def __rsub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return other - self.p
        return other.p - self.p


class SegmentTree:
    def __init__(self, max_len: int, use_max_space: bool=False):
        self.size: int = pow(2, int(ceil(log(max_len, 2))))
        if not use_max_space:
            self.max_len: int = max_len
        else:
            self.max_len: int = self.size
        self.sum: List[(float, Node)] = [0] * (2 * self.size - 1)
        self.queue: IndexPriorityQueue = IndexPriorityQueue()
        self.unknown_buff: Queue = Queue()
        self.high: int = -1

    def build(self, arr: List):
        self.high = len(arr)
        for a in arr:
            self.put(a)

    def put(self, node, priority: [float, None]=None) -> int:
        """

        :param node:
        :param priority:
        :return:
        """
        if not isinstance(node, Node):
            if priority is None:
                return self.put_unknown_p(node)
            node = Node(node, priority)

        if self.high + 1 < self.max_len:
            self.high += 1
            idx = self.high
        else:
            q_node = self.queue.get_nowait()
            old_node, idx = q_node.dat, q_node.index
            if old_node > node:
                warnings.warn('Did not push the node because the node.p is smaller than the min p')
                self.queue.put(QNode(old_node, idx))
                return idx
        self.queue.put(QNode(node, idx))
        self.set(idx, node)
        return idx

    def _avg_value(self) -> float:
        if len(self) == 0:
            return 1

        return self.sum[0] / len(self)

    def put_unknown_p(self, data) -> int:
        """
        There are times when we do not know the priority,
        and we need to put them in the next batch.

        :param data:
        :return: the index of the new item
        """
        node = Node(data, self._avg_value())
        idx = None
        v = 10
        while idx is None:
            idx = self.put(node, v)
            v += 1
        self.unknown_buff.put((idx, node))
        return idx

    def set_p(self, idx, p):
        self.queue.set_p(idx, p)
        self._set_p(idx, p)

    def _set_p(self, idx, p):
        pos = idx + self.size - 1
        self.sum[pos].p = p
        while True:
            pos = (pos - 1) // 2
            self.sum[pos] = self.sum[2 * pos + 1] + self.sum[2 * pos + 2]
            if pos == 0:
                break

    def set(self, idx, val):
        self.queue.set(idx, val)
        self._set(idx, val)

    def _set(self, idx, val):
        pos = idx + self.size - 1
        self.sum[pos] = val
        while True:
            pos = (pos - 1) // 2
            self.sum[pos] = self.sum[2 * pos + 1] + self.sum[2 * pos + 2]
            if pos == 0:
                break

    def _get_prob(self, idx):
        return self.sum[idx + self.size - 1].p / self.sum[0]

    def _sample_batch(self, batch_size) -> List:
        return [self.sample()[1].dat for _ in range(batch_size)]

    def _sample_from_unknown_buff(self, slices: (list, None), data: (list, None), prob: (list, None), batch_size: int):
        while not self.unknown_buff.empty():
            idx, node = self.unknown_buff.get_nowait()
            slices.append(idx)
            data.append(node.dat)
            prob.append(self._get_prob(idx))
            batch_size -= 1
            if batch_size == 0:
                break
        return batch_size

    def _sample_lowest_p(self,
                         slices: (list, None),
                         data: (list, None),
                         prob: (list, None),
                         batch_size: int):
        """
        Sometime when the buffer size is too large, the value may be out dated,
        we may need to refresh the priority of the samples before dropping them.

        Because we always throwing away the sample with lowest priority,
        we only need to sample a small part of those value.

        TODO: add configurable setting
        :param slices: the output indexes of
        :param data: the output data
        :param prob: the ouput probability of the sample
        :param batch_size: sampling batch size
        :return:
        """
        if batch_size == 0:
            return 0

        if self.high + 1 >= self.max_len:
            q_node = self.queue.top()
            idx, node = q_node.index, q_node.dat
            prob.append(self._get_prob(idx))
            data.append(node.dat)
            slices.append(idx)
            batch_size -= 1
        return batch_size

    def sample_batch(self, batch_size) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Sample a batch of sample with their index and probability.

        :param batch_size:
        :return:
            slices: indexes of the batch. You can set the sample priority according to the indexes latter.
            data: a batch of features ndarray with size [batch_size, feature_size]
            prob: the probability of each sample
        """
        slices = []
        data = []
        prob = []
        batch_size = self._sample_from_unknown_buff(slices, data, prob, batch_size)
        batch_size = self._sample_lowest_p(slices, data, prob, batch_size)
        if batch_size <= 0:
            warnings.warn('Batch size may be too small. Only took samples from the unknown buff.')

        for i in range(batch_size):
            idx, node = self.sample()
            prob.append(self._get_prob(idx))
            slices.append(idx)
            data.append(node.dat)

        data = np.array(data)
        prob = np.array(prob)
        slices = np.array(slices)
        return slices, data, prob

    def _sample_node_batch(self, batch_size) -> List[Node]:
        return [self.sample()[1] for _ in range(batch_size)]

    def sample(self) -> Tuple[int, Union[Node, float]]:
        random_v = random.random() * self.sum[0]
        pos, node = self._find_leaf(random_v)
        return pos, node

    def _find_leaf(self, v):
        pos = 0
        left_pos = pos * 2 + 1
        right_pos = left_pos + 1
        while right_pos < 2 * self.size - 1:
            if v > self.sum[left_pos]:
                v -= self.sum[left_pos]
                pos = right_pos
            else:
                pos = left_pos
            left_pos = pos * 2 + 1
            right_pos = left_pos + 1
        return pos - self.size + 1, self.sum[pos]

    def __len__(self):
        return self.high + 1

    def top(self):
        node = self.queue.top()
        return node.dat
