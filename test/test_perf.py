import numpy as np
import random
import time
from priority_memory import FastPriorReplayBuffer


def create_data(size=100000):
    dat = np.arange(0, size)
    p = [random.random() for _ in range(size)]
    s = sum(p)
    p = [_p / s for _p in p]
    return dat, p


def test_np_sample():
    size = 100000
    n_iter = 100
    batch_size = 320
    dat, p = create_data(size)
    start = time.time()
    for i in range(n_iter):
        np.random.choice(dat, batch_size, True, p)
    np_used = time.time() - start
    print(f'numpy.random.choice used {np_used}s')

    buff = FastPriorReplayBuffer(buffer_size=size)
    for _d, _p in zip(dat, p):
        buff.append((_d,), _p)
    start = time.time()
    for i in range(n_iter):
        buff.sample_with_weights(32)
    buff_used = time.time() - start
    print(f'priority_memory.FastPriorReplayBuffer used {buff_used}s')

    assert np_used / 10 > buff_used
    assert buff_used < 0.1


