import theano
from rltools.theanotools import Theano_NRBF_Projector
from rltools.representation import NRBFCoding
import numpy as np

import timeit

dim = 2
n_rbf = 4

c = np.random.rand(n_rbf,dim).astype('float32')
w = (np.random.rand(n_rbf,dim)+ 0.1).astype('float32')

rbf= Theano_NRBF_Projector(c, w)
phi = NRBFCoding(w,c)



testd = np.random.rand(5, dim).astype(theano.config.floatX)
r1, r2 = rbf( testd), phi(testd)
print np.allclose(r1, r2)
print (r1-r2)[(np.abs(r1-r2) > 0.00001).nonzero()]


def test1():
    rbf( testd)
    
def test2():
    phi(testd)
print timeit.timeit("test1()", setup = "from __main__ import test1", number = 300)
print timeit.timeit("test2()", setup = "from __main__ import test2", number = 300)


s = np.random.rand(2,dim).astype('float32')
dx = np.random.rand(2,dim).astype('float32') + 0.1
dphi = rbf.getdphids(s, dx)
print dphi.shape
print dphi

print (rbf((s+dx/10000).astype('float32')) - rbf(s))*10000