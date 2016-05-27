# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 17:07:53 2016

@author: cgehri
"""

from rltools.model import CompressedModel, CompressedMatrix
import numpy as np


def ridge_reg( A, b, lamb):
    U,S,Vt = np.linalg.svd(A, full_matrices = False)
    
    S = S/ (S**2 + lamb)
    return Vt.T.dot(np.diag(S).dot(U.T.dot(b)))

A = [np.random.rand(100,200) for a in [0,1]]
B = [np.random.rand(100,200) for a in [0,1]]
r = [np.random.rand(100) for a in [0,1]]

lamb = 0.2
model = CompressedModel(dim = 200,
                        num_actions = 2,
                        max_rank = 200,
                        lamb = lamb,
                        Xa_t = A,
                        Xa_tp1 = B,
                        Ra = r,
                        keep_all_samples=True,
                        compress_end_states=False,
                        initialize_uniform=False)
                        
U,S,V = model.Ma[0]
print 'svd is close to original', np.allclose(U.dot(np.diag(S).dot(V.T)), A[0])

Kab, Da, wa, Va, Ua = model.generate_embedded_model()

Ub, Sb, Vbt = np.linalg.svd(A[0], full_matrices = False)
print 'S close:', np.allclose(S, Sb)
print 'Kab close:', np.allclose( Vbt.dot(B[0].T.dot(Ub)), Kab[0,0])
print 'Da close:', np.allclose(Da[0], Sb/(Sb**2 + lamb))
print 'wa close:', np.allclose(r[0].dot(Ub), wa[0])
print 'Va close:', np.allclose(Va[0], Vbt.T)

vec = np.random.rand(200)
print wa[0].dot(Da[0]* (Kab[0,0].dot(Da[0] *Va[0].T.dot(vec))))

F = ridge_reg(A[0], B[0], lamb = lamb).T
theta = ridge_reg(A[0], r[0], lamb = lamb)

print theta.dot(F.dot(vec))

print "testing compressed matrix:"

M = CompressedMatrix(200, 200, True)
M.add_rows(A[0])
Ahat = M.get_updated_full_matrix()
print np.allclose(Ahat, A[0])