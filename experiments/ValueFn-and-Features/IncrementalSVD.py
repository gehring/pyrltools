from rltools.mathtools import iSVD
import numpy as np

data = [ (np.random.rand(10), np.random.rand(6)) for i in range(10)]

isvd = iSVD( 6, shape = (10,6))

A = sum( [ a[:,None] * b[None,:] for a,b in data])

for a,b in data:
    isvd.update(a, b)

U,S,V = isvd.get_decomp()
Ut, St, Vt = np.linalg.svd(A, full_matrices = False)
# print np.allclose(A, U.dot(np.diag(S)).dot(V.T))
print np.allclose(A, U.dot(np.diag(S)).dot(V))
print U.dot(U.T)
print V.dot(V.T)
print S
print St
# print A - U.dot(np.diag(S)).dot(V)
