from queue import Queue
from typing import List


class QNode:
    def __init__(self, dat, index):
        self.dat = dat
        self.index = index

    def __repr__(self):
        return f'QNode({self.dat}, index={self.index})'

    def __lt__(self, other):
        return self.dat < other.dat

    def __eq__(self, other):
        return self.dat == other.dat

    def __ne__(self, other):
        return self.dat != other.dat

    def __hash__(self):
        return hash(self.dat)


def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def siftup(heap, pos):
    endpos = len(heap)
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, 0, pos)


def heappop(heap):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        siftup(heap, 0)
        return returnitem
    return lastelt


def heappush(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)


class MyList:
    def __init__(self, maxsize):
        self.data: List[QNode] = []
        self.maxsize = maxsize
        self.old_new_mapping = {}

    def __len__(self):
        return len(self.data)

    def __setitem__(self, key: int, value: QNode):
        self.data[key] = value
        self.old_new_mapping[value.index] = key

    def __getitem__(self, key):
        return self.data[key]

    def append(self, item):
        assert isinstance(item, QNode)
        # if item.index in self.old_new_mapping:
        #     raise ValueError(f'Index of {item} already exists.')
        self.old_new_mapping[item.index] = len(self.data)
        self.data.append(item)

    def pop(self):
        node = self.data.pop()
        old_index = node.index
        del self.old_new_mapping[old_index]
        return node

    def get_new_index(self, key: int):
        return self.old_new_mapping[key]


class IndexPriorityQueue(Queue):
    def _init(self, maxsize):
        self.queue = MyList(maxsize)

    def _qsize(self):
        return len(self.queue)

    def _put(self, item):
        assert isinstance(item, QNode)
        heappush(self.queue, item)

    def _get(self):
        return heappop(self.queue)

    def set(self, old_index: int, value):
        new_index = self.queue.get_new_index(old_index)
        self.queue[new_index] = QNode(value, old_index)
        siftup(self.queue, new_index)

    def set_p(self, old_index: int, p):
        new_index = self.queue.get_new_index(old_index)
        node = self.queue[new_index]
        node.dat.p = p
        self.queue[new_index] = node
        siftup(self.queue, new_index)

    def top(self):
        return self.queue[0]


