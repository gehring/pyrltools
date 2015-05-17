import theano
# theano.config.optimizer='None'
# theano.config.compute_test_value='raise'
# theano.config.profile = True

from rltools.theanotools import Theano_Tiling, Theano_UNH
from rltools.representation import TileCoding, UNH

import numpy as np
import timeit

dim = 6
k = 1000

indices = [np.arange(dim)]
s_range = [np.zeros(dim).astype('float32'), np.ones(dim).astype('float32')]
ntiles = [10]
ntilings = [10]

tunh = Theano_UNH(dim, 10000)
unh = UNH(10000)
unh.rndseq = tunh.rndseq.copy()

phi1 =Theano_Tiling(indices, ntiles, ntilings, [tunh], s_range, True)
phi2 = TileCoding(indices, ntiles, ntilings, [unh], s_range, True)

x = np.random.rand(dim).astype('float32')
X = np.random.rand(k,dim).astype('float32')

# print np.all(phi1(x) == phi2(x))
# print phi1(X[:2,:])
# print phi2(X[:2,:])


def testone1():
    phi1(x)

def testone2():
    phi2(x)

def test1():
    for s in X:
        phi1(s)

def test2():
    phi2(X)

def test3():
    phi1(X)


print 'testone1'
print timeit.timeit("testone1()", setup = "from __main__ import testone1", number = k)

print 'testone2'
print timeit.timeit("testone2()", setup = "from __main__ import testone2", number = k)


print 'test1'
print timeit.timeit("test1()", setup = "from __main__ import test1", number = 4)

print 'test2'
print timeit.timeit("test2()", setup = "from __main__ import test2", number = 4)
#
print 'test3'
print timeit.timeit("test3()", setup = "from __main__ import test3", number = 4)

