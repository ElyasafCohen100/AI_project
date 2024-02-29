'''
implements a priority queue using a minimum heap
the heap is represented by a list of the items
every item is a list whose first item is the priority.
the parent of index i is in index (i-1)//2
the left child of index i is in index 2i+1
the right side of index i is in index 2i+2
'''


def create():
    return []  # returns a priority queue that contains s


def is_empty(f):
    return f == []  # returns true iff f is empty list


def size(f):
    return len(f)


def insert(f, s):
    # inserts  s to the frontier
    f.append(s)  # inserts the s as the last item
    i = len(f) - 1  # i gets its index

    # move the item with smallest value to the root
    while i > 0 and f[i][0] < f[(i - 1) // 2][0]:  # while item i's value is smaller than the value of his father, swap!
        # the next three lines swap i and its parent
        t = f[i]
        f[i] = f[(i - 1) // 2]
        f[(i - 1) // 2] = t
        i = (i - 1) // 2  # i moves upwards


def remove(f):  # removes and return the root of heap f
    if is_empty(f):  # underflow
        return 0
    s = f[0]  # stores the root that should be returned
    f[0] = f[len(f) - 1]  # the last leaf becomes the root
    del f[-1]  # deletes the last leaf
    heapify(f, 0)  # fixes the heap
    return s


def heapify(f, i):  # fixes the heap by rolling down from index i
    # compares f[i] with its children
    # if f[i] is bigger than at least one of its children
    # f[i] and its biggest child are swapped
    minSon = i  # defines i as minSon
    if 2 * i + 1 < len(f) and f[2 * i + 1][0] < f[minSon][0]:  # if f[i] has a left son
        # and its left son is smaller than f[i]
        minSon = 2 * i + 1  # defines the left son as minSon
    if 2 * i + 2 < len(f) and f[2 * i + 2][0] < f[minSon][0]:  # if f[i] has a right son
        # and its right son is smaller than f[minSon]
        minSon = 2 * i + 2  # defines the right son as minSon
    if minSon != i:  # if f[i] is bigger than one of its sons
        t = f[minSon]  # swap f[i] with the smaller son
        f[minSon] = f[i]
        f[i] = t
        heapify(f, minSon)  # repeats recursively

