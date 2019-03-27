from heapq import _heapify_max as heapify_max
from heapq import _siftdown_max, _siftup_max, nlargest

__all__ = ['heappush_max', 'heapreplace_max', 'heappushpop_max',
           'heappop_max', 'heapreplace_max', 'heapify_max', 'nlargest']


def heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt


def heapreplace_max(heap, item):
    """Maxheap version of a heappop followed by a heappush."""
    returnitem = heap[0]    # raises appropriate IndexError if heap is empty
    heap[0] = item
    _siftup_max(heap, 0)
    return returnitem


def heappush_max(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown_max(heap, 0, len(heap) - 1)


def heappushpop_max(heap, item):
    """Fast version of a heappush followed by a heappop."""
    if heap and heap[0] > item:
        # if item >= heap[0], it will be popped immediately after pushed
        item, heap[0] = heap[0], item
        _siftup_max(heap, 0)
    return item


try:
    from heapq import _heappop_max as heappop_max
except ImportError:
    pass

try:
    from heapq import _heapreplace_max as heapreplace_max
except ImportError:
    pass