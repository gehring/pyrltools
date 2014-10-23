import numpy as np

class bvector(object):
    def __init__(self, indices):
        self.i = indices

def dot(self, a, x):
    return np.sum(a[x.i])

def matmult(self, A, x):
    return np.sum(A[:,x.i], axis=1)

def addto(self, a, x):
    a[x.i] += 1

class trace(object):
    def __init__(self, size, threshold = 1e-3):
        self.data = np.zeros(size)
        self.i = set()
        self.thres = threshold

    def clear(self):
        self.data[self.i] = 0
        self.i = set()

    def add(self, x):
        if isinstance(x, bvector):
            self.data[x.i] += 1
            self.i.update(x.i)
        else:
            self.data += x
            self.i = set(np.nonzero(self.data))

    def scaleaddto(self, c, x):
        ind = list(self.i)
        x[ind] += c*self.data[ind]

    def scale(self, c):
        ind = np.array(list(self.i))
        self.data[ind] *= c
        self.i.difference_update(ind[np.nonzero(self.data[ind] < self.thres)])

