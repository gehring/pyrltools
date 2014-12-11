import theano
from rltools.theanotools import Theano_RBF_Projector
from rltools.representation import NRBFCoding
import numpy as np

import timeit

dim = 20
n_rbf = 1000
k = 100

c = np.random.rand(n_rbf,dim).astype('float32')
w = (np.random.rand(n_rbf,dim)+ 0.1).astype('float32')

rbf= Theano_RBF_Projector(c, w, normalized = True)
phi = NRBFCoding(w,c)



testd1 = np.random.rand(k, dim).astype(theano.config.floatX)
testd2 = np.random.rand(k, dim).astype(theano.config.floatX)

p1= phi(testd1)
r1, r2 = rbf( testd1), np.hstack((p1, np.ones((p1.shape[0],1))))
print np.allclose(r1, r2)
print (r1-r2)[(np.abs(r1-r2) > 0.00001).nonzero()]


def test1():
    rbf( testd1)
    
def test2():
    phi(testd2)
print timeit.timeit("test1()", setup = "from __main__ import test1", number = 2)
print timeit.timeit("test2()", setup = "from __main__ import test2", number = 2)


# s = np.random.rand(2,dim).astype('float32')
# dx = np.random.rand(2,dim).astype('float32') + 0.1
# dphi = rbf.getdphids(s, dx)
# print dphi.shape
# print dphi
# 
# print (rbf((s+dx/10000).astype('float32')) - rbf(s))*10000