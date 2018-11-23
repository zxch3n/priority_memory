from priority_memory.priority_memory import SegmentTree, Node, FastPriorReplayBuffer
import random
import time
import numpy as np


def test_buffer_0():
    buff = FastPriorReplayBuffer(100, 3)
    data = gen_data(n=20, n_f=10)
    data_str = [','.join((str(x) for x in d)) for d in data]
    for d in data:
        buff.append(d)
        if len(buff) < 10:
            continue
        slices, values = buff.sample()
        for a in values:
            s = ','.join((str(x) for x in a))
            assert s in data_str


def test_buffer_set_prior():
    buff = FastPriorReplayBuffer(100, 3)
    data = gen_data(n=20, n_f=10)
    data_str = [','.join((str(x) for x in d)) for d in data]
    for d in data:
        buff.append(d)
        if len(buff) < 10:
            continue
        slices, values = buff.sample()
        buff.set_weights(slices, [5 for _ in range(len(slices))])
        for a in values:
            s = ','.join((str(x) for x in a))
            assert s in data_str


def segment_tree_prob_test():
    buff = SegmentTree(10)
    for i in range(10):
        buff.put(Node(i, 1))
    for i in range(10):
        b = buff._get_prob(i)
        assert abs(b - 0.1) < 1e-6


def segment_tree_prob_test_1():
    n = 100
    buff = SegmentTree(n)
    data = [random.random() for _ in range(n)]
    for i in range(n):
        buff.put(Node(i, data[i]))
    s = sum(data)
    data = [x/s for x in data]
    for i in range(n):
        assert abs(buff._get_prob(i) - data[i]) < 1e-6


def test_segment_tree_dist_1():
    buff = SegmentTree(5)
    for i in range(5):
        idx = buff.put(Node(i, 1))
        assert idx == i
    for i in range(5):
        buff.set_p(i, i)
    arr = np.array(buff._sample_batch(30000))
    for i in range(5):
        rate = (arr == i).mean()
        assert abs(rate - i/10) < 1e-2, f'Actual rate = {rate} != {i/10}'


def test_segment_tree_0():
    buff = SegmentTree(5)
    buff.put(Node(-3, 0.1))
    buff.put(Node(0, 0.001))
    buff.put(Node(1, 1))
    buff.put(Node(2, 1))
    buff.put(Node(3, 1))
    buff.put(Node(-4, 0.1))
    buff.put(Node(4, 1))
    buff.put(Node(5, 1))
    arr = buff._sample_batch(10000)
    for a in arr:
        assert a > 0

    for i in range(1, 6):
        assert i in arr


def test_segment_tree_basic():
    buff = SegmentTree(2)
    buff.put(Node(-5, 0.1))
    buff.put(Node(2, 1))
    buff.put(Node(1, 1))
    arr = buff._sample_batch(100)
    for a in arr:
        assert a > 0

    for i in range(1, 3):
        assert i in arr


def test_segment_tree_1():
    buff = SegmentTree(4)
    buff.put(Node(104, 9))
    buff.put(Node(-3, 0.1))
    buff.put(Node(0, 0.001))
    buff.put(Node(1, 1))
    buff.put(Node(103, 10))
    buff.put(Node(3, 1))
    buff.put(Node(-4, 0.1))
    buff.put(Node(102, 100))
    buff.put(Node(5, 1))
    buff.put(Node(101, 99))
    arr = buff._sample_batch(1000)
    for a in arr:
        assert a > 100

    for i in range(101, 105):
        assert i in arr


def test_segment_tree_update():
    buff = SegmentTree(5)
    for i in range(5):
        buff.put(i, i)
    for i in range(5):
        assert buff.sum[i + buff.size - 1].dat == i
    buff.set(0, Node(6, 6))
    assert buff.sum[buff.size - 1].dat == 6


def test_segment_tree_dist():
    buff = SegmentTree(5)
    for i in range(5):
        buff.put(Node(i, i))
    arr = np.array(buff._sample_batch(30000))
    for i in range(5):
        rate = (arr == i).mean()
        assert abs(rate - i/10) < 1e-2, f'Actual rate = {rate} != {i/10}'


def test_segment_tree_sample_with_slice():
    buff = SegmentTree(100)
    for i in range(5000):
        buff.put(Node(i, i))
    slices, dat, prob = buff.sample_batch(10000)
    for s in slices:
        assert s is not None
        buff.set_p(s, 1)
    slices, dat, prob = buff.sample_batch(10000)
    for s in slices:
        assert s is not None


def test_segment_tree_query():
    buff = SegmentTree(5)
    for i in range(5):
        buff.put(Node(i, i))
    arr = np.array(buff._sample_batch(11000))
    for i in range(5):
        rate = (arr == i).mean()
        assert abs(rate - i/10) < 2e-2, f'Actual rate = {rate} != {i/10}'


def segment_tree_perf_test():
    n = int(1e5)
    buff = SegmentTree(n)
    for i in range(n):
        buff.put(Node(i, i))
    start = time.time()
    for i in range(20):
        arr = np.array(buff._sample_batch(100))
    used = time.time() - start
    assert used < 0.1


def gen_data(n=100, n_f=10):
    return [[random.random() for j in range(n_f)] for _ in range(n)]


