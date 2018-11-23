from priority_memory.index_priority_queue import QNode, MyList, IndexPriorityQueue
from priority_memory.priority_memory import Node, FastPriorReplayBuffer
from queue import PriorityQueue
import random


def test_basic_put_get():
    q = IndexPriorityQueue()
    for i in range(10):
        q.put_nowait(QNode(i, index=i))
    for i in range(10):
        node = q.get_nowait()
        assert node.dat == i
        assert node.index == i


def test_priority():
    q = IndexPriorityQueue()
    for i in range(11):
        q.put_nowait(QNode(10 - i, index=i))
    for i in range(11):
        node = q.get_nowait()
        assert node.dat == i
        assert node.index == 10 - i


def test_set_priority():
    q = IndexPriorityQueue()
    for i in range(100):
        q.put_nowait(QNode(random.random(), index=i))
    for i in range(100):
        q.set(i, i)
    for i in range(100):
        node = q.get_nowait()
        assert node.dat == i
        assert node.index == i


def put_get_interleave_test():
    q = IndexPriorityQueue()
    t_q = PriorityQueue()
    for i in range(100):
        node = QNode(random.random(), index=i)
        q.put_nowait(node)
        t_q.put_nowait(node)

    for i in range(100):
        if random.random() < 0.5:
            node = q.get_nowait()
            t_node = t_q.get_nowait()
            assert node.dat == t_node.dat and node.index == t_node.index
        else:
            node = QNode(random.random(), index=i + 100)
            q.put_nowait(node)
            t_q.put_nowait(node)

    while not q.empty():
        node = q.get_nowait()
        t_node = t_q.get_nowait()
        assert node.dat == t_node.dat and node.index == t_node.index


def put_get_set_interleave_test():
    q = IndexPriorityQueue()
    data = []
    for i in range(1000):
        data.append(QNode(random.random(), i))

    for d in data:
        q.put_nowait(d)

    for i in range(100):
        item = q.get_nowait()
        data.remove(item)

    for i in range(len(data)):
        data[i] = QNode(random.random(), data[i].index)
        q.set(data[i].index, data[i].dat)

    data.sort()
    for d in data:
        node = q.get_nowait()
        assert d.dat == node.dat
        assert d.index == node.index


def set_p_test():
    q = IndexPriorityQueue()
    data = []
    for i in range(1000):
        node = QNode(Node(random.random(), p=i), index=i)
        data.append(node)
        q.put_nowait(node)

    for d in data:
        d.dat.p = random.random()

    data.sort()
    for d in data:
        q.set_p(d.index, d.dat.p)

    i = 0
    while not q.empty():
        node = q.get_nowait()
        assert node.dat.dat == data[i].dat.dat
        assert node.dat.p == data[i].dat.p
        i += 1






