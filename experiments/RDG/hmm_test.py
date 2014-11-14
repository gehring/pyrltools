from rltools.HMM import DiscreteHMM, generate_sequence
from itertools import product, izip
import matplotlib.pyplot as plt
import numpy as np
from rltools.RDG import RDG
import pickle
import scipy.sparse
import scipy.linalg
from pulp import *
import sys
import cvxopt.solvers
from cvxopt import spmatrix


def generateLP(X, Y):
    n=X.shape[0]

    prob = LpProblem("Prob Solve", LpMaximize)
    
    rows = [str(i) for i in xrange(n)]
    columns = rows
    
    p = LpVariable.dicts("theta", (rows, columns), 0.0, 1.0, cat = 'Continuous')
    alpha = LpVariable("alpha", 0, None)

    prob += alpha
    
    for c in columns:
        prob += lpSum([p[r][c] for r in rows]) == 1,""
        
    for i in xrange(X.shape[1]):
        for j in xrange(Y.shape[0]):
            prob += LpAffineExpression([(p[rows[j]][columns[k]], X[k,j])
                                        for k in xrange(X.shape[0])]) <= alpha
                                        
    return prob, p

def getQP(X, Y):
    n = X.shape[0]
    m = Y.shape[0]
    xxt = np.zeros((n*m, n*m))
    yx = np.zeros(n*m)
    
    
    toolbar_width = 40
    interval = int(X.shape[1]/toolbar_width) + 1
    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\r[") # return to start of line, after '['

    for i in xrange(X.shape[1]):
        x = scipy.sparse.block_diag(([X[:,i]]*m))
        
        xxti = x.T.dot(x)
        ri = np.repeat(np.arange(xxti.shape[0]), np.diff(xxti.indptr))
        xxt[ri,xxti.indices] += xxti.data
        
#         xxt += x.T.dot(x)
        yx += x.T.dot((Y[:,i].reshape((-1,1))))[:,0]
        if i%interval == 0:
            # update the bar
            sys.stdout.write("-")
            sys.stdout.flush()

    sys.stdout.write("\n")
    
#     A = scipy.linalg.block_diag(*([np.ones(m)]*n))
    A = spmatrix(np.ones(m*n), [j for i in range(n) for j in [i]*m], range(n*m))
    b = np.ones(n)
#     G = np.vstack((np.eye(n*m), -np.eye(n*m)))
    G = spmatrix( [1]*(n*m) + [-1]*(n*m), range(n*m*2), range(n*m)*2)
    h = np.hstack((np.ones(n*m), np.zeros(n*m)))
    return (cvxopt.matrix(xxt/X.shape[1]), 
            cvxopt.matrix(yx/X.shape[1]), 
            G, 
            cvxopt.matrix(h), 
            A, 
            cvxopt.matrix(b))

def generate_RDG_seq(obs_seq, rdg):
    rdg.reset()
    states = np.zeros((rdg.size, obs_seq.size+1))
    for i, o in enumerate(obs_seq):
        states[rdg.getState(),i] = 1
        rdg.update(o)
    states[rdg.getState(),i] = 1   
    return states

def generate_many_seq(seq_length, number_seq, rdg):
    for i in xrange(number_seq):
        obs, states = generate_sequence(hmm, seq_length)
        rdg_states = generate_RDG_seq(obs, rdg)
        bs = hmm.filter_all(obs, p0)
        yield rdg_states, bs
        

size = 20
weight = 2
# random walk on a chain with some probability of staying in place
T = np.array([[weight, 1] + [0]*(size-2)]
             + [ [0]*i + [1,weight,1] + [0]*(size-i-3) for i in xrange(size-2)]
             + [[0]*(size-2) + [1,weight]], dtype='float')

weight = 1
out_weight= 0.4
# random observations around the current state
O = np.array([[weight, 1, 1] + [out_weight]*(size-3)]
             + [[1, weight, 1, 1] + [out_weight]*(size-4)]
             + [ [out_weight]*i + [1, 1,weight,1, 1] + [out_weight]*(size-i-5) for i in xrange(size-4)]
             + [[out_weight]*(size-4) + [1, 1,weight, 1]]
             + [[out_weight]*(size-3) + [1, 1,weight]], dtype='float')


T = T/ np.sum(T, 1)[:,None]
O = O/ np.sum(O, 1)[:,None]
p0 = np.ones(size)/size
hmm = DiscreteHMM(p0, T, O)

seq_length = 30

num_graphs = 40
num_nodes = 20
rdg = RDG(num_graphs, num_nodes, num_obs=size)
 
# with open('solution.data', 'rb') as f:
#     rdg, w, hmm = pickle.load(f)
 
rdg_states, bs = zip(*generate_many_seq(seq_length, 100, rdg))
 
print "generating QP..."
# qp = getQP(np.hstack(rdg_states), np.hstack(bs))
qp = getQP(np.hstack(bs), np.hstack(bs))
  
print 'problem generated! Now solving!'
sol = cvxopt.solvers.qp(*qp)
print 'LP solver finished!'
  
w = np.array(sol['x']).reshape((size, -1), order='F')
with open('solution.data', 'wb') as f:
    pickle.dump((rdg, w, hmm), f)

obs, states = generate_sequence(hmm, seq_length)
rdg_states = generate_RDG_seq(obs, rdg)
bs = hmm.filter_all(obs, p0)

plt.figure(1)
plt.subplot((211))
plt.imshow(bs)

plt.subplot((212))
# plt.imshow(w.dot(rdg_states))
plt.imshow(w.dot(bs))
plt.show()

# def getLPMinMax(X, Y):
#     n = X.shape[0]+1
#     A = np.zeros((n,Y.size*2 + 1))
#     b = np.zeros(Y.size*2 + 1)
#     c = np.zeros(n)
#     c[-1] = 1.0
#     for i,j in product(xrange(Y.shape[0]), xrange(Y.shape[1])):
#         c_index = index((i,j), Y.shape)
#         c_index2 = c_index + Y.size
#         A[c_index, -1] = -1
#         A[c_index2, -1] = -1
#         for k in xrange(n-1):
#             v_index = index((j,k), (n-1, n-1))
#             A[c_index, v_index] = -X[k,i]
#             b[c_index] = -Y[i,j]
#             
#             A[c_index2, v_index] = X[k,i]
#             b[c_index2] = Y[i,j]
